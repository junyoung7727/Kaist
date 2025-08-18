"""
에포크 레벨 데이터 캐싱 시스템 v2
첫 번째 에포크에서 처리된 데이터를 저장하고 이후 에포크에서 재사용
버전 관리와 메타데이터 검증을 포함한 안정적인 캐싱 시스템
"""

import os
import pickle
import hashlib
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import torch


class EpochCache:
    """에포크 레벨 데이터 캐싱 시스템 v2"""
    
    CACHE_VERSION = "2.0"  # 캐시 버전
    METADATA_FILE = "cache_metadata.json"  # 메타데이터 파일명
    
    def __init__(self, cache_dir: str = "cache", max_cache_size_gb: float = 2.0):
        # 절대 경로로 변환하여 명확한 위치 지정
        if not os.path.isabs(cache_dir):
            cache_dir = os.path.abspath(cache_dir)
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size_bytes = int(max_cache_size_gb * 1024 * 1024 * 1024)
        self.metadata_path = self.cache_dir / self.METADATA_FILE
        
        # 메모리 캐시 (빠른 접근용)
        self._memory_cache: Dict[str, Any] = {}
        self._cache_stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'saves': 0,
            'invalidations': 0
        }
        
        # 캐시 초기화 및 버전 검증
        self._initialize_cache()
        
    def _initialize_cache(self):
        """캐시 초기화 및 버전 검증 - 구버전 캐시 완전 제거"""
        try:
            # 구버전 캐시 파일들 먼저 정리
            self._cleanup_old_version_cache()
            
            # 기존 메타데이터 로드
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # 버전 호환성 검사
                if metadata.get('version') != self.CACHE_VERSION:
                    self._clear_all_cache()
                    self._create_metadata()
            else:
                # 새로운 메타데이터 생성
                self._create_metadata()
                
        except Exception as e:
            # 오류 발생 시 캐시 초기화
            self._clear_all_cache()
            self._create_metadata()
    
    def _create_metadata(self):
        """메타데이터 파일 생성"""
        metadata = {
            'version': self.CACHE_VERSION,
            'created_at': time.time(),
            'cache_entries': {}
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📝 캐시 메타데이터 생성 완료: {self.metadata_path}")
    
    def _update_metadata(self, cache_key: str, cache_info: Dict[str, Any]):
        """메타데이터 업데이트"""
        try:
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {'version': self.CACHE_VERSION, 'cache_entries': {}}
            
            metadata['cache_entries'][cache_key] = cache_info
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  메타데이터 업데이트 실패: {e}")
    
    def _generate_dataset_hash(self, dataloader) -> str:
        """데이터로더의 고유 해시 생성 (개선된 버전)"""
        # 기본 데이터셋 정보
        dataset_info = {
            'dataset_size': len(dataloader.dataset),
            'batch_size': dataloader.batch_size,
            'shuffle': getattr(dataloader, 'shuffle', False),
            'cache_version': self.CACHE_VERSION  # 버전 포함
        }
        
        # 안전한 첫 번째 배치 샘플링
        try:
            # 데이터로더 상태 보존을 위한 임시 이터레이터 생성
            temp_iter = iter(dataloader)
            first_batch = next(temp_iter)
            
            if isinstance(first_batch, dict):
                # 배치 구조 정보 추가
                batch_keys = sorted(first_batch.keys())
                dataset_info['batch_structure'] = batch_keys
                
                # 회로 정보가 있다면 추가
                if 'circuit_specs' in first_batch:
                    specs = first_batch['circuit_specs'][:3]  # 처음 3개만
                    try:
                        gate_counts = [len(spec.gates) if hasattr(spec, 'gates') else 0 for spec in specs]
                        dataset_info['sample_gate_counts'] = gate_counts
                    except:
                        pass
                        
                # 패딩 정보 추가 (padded_421 관련)
                if 'input_sequence' in first_batch:
                    seq_shape = first_batch['input_sequence'].shape if hasattr(first_batch['input_sequence'], 'shape') else None
                    if seq_shape:
                        dataset_info['sequence_shape'] = list(seq_shape)
                        
        except Exception as e:
            print(f"⚠️  데이터셋 해시 생성 중 오류 (무시됨): {e}")
            dataset_info['sampling_error'] = str(e)
            
        # 안정적인 해시 생성
        data_str = json.dumps(dataset_info, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, dataset_hash: str, epoch: int) -> Path:
        """캐시 파일 경로 생성 (버전 포함)"""
        return self.cache_dir / f"epoch_{epoch}_{dataset_hash}_v{self.CACHE_VERSION.replace('.', '_')}.pkl"
    
    def has_cached_epoch(self, dataloader, epoch: int) -> bool:
        """캐시된 에포크 데이터가 있는지 확인 (구버전 캐시 자동 제거)"""
        try:
            dataset_hash = self._generate_dataset_hash(dataloader)
            cache_key = f"{dataset_hash}_epoch_{epoch}"
            
            # 메모리 캐시 확인
            if cache_key in self._memory_cache:
                return True
                
            # 디스크 캐시 확인 (현재 버전만)
            cache_path = self._get_cache_path(dataset_hash, epoch)
            if cache_path.exists():
                # 메타데이터에서 유효성 확인
                return self._validate_cache_entry(cache_key, cache_path)
            
            return False
            
        except Exception:
            return False
    
    def _validate_cache_entry(self, cache_key: str, cache_path: Path) -> bool:
        """캐시 엔트리 유효성 검증 - 구버전 캐시 완전 차단"""
        try:
            # 파일명에서 버전 정보 확인
            if not cache_path.name.endswith(f"_v{self.CACHE_VERSION.replace('.', '_')}.pkl"):
                # 구버전 캐시 파일은 조용히 삭제
                self._invalidate_cache_file(cache_path)
                return False
                
            if not self.metadata_path.exists():
                return False
                
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # 메타데이터 버전 확인
            if metadata.get('version') != self.CACHE_VERSION:
                self._invalidate_cache_file(cache_path)
                return False
            
            # 메타데이터에 엔트리가 있는지 확인
            if cache_key not in metadata.get('cache_entries', {}):
                self._invalidate_cache_file(cache_path)
                return False
            
            # 파일 존재 여부 재확인
            if not cache_path.exists():
                return False
            
            return True
            
        except Exception as e:
            # 오류 발생 시 조용히 무효화
            self._invalidate_cache_file(cache_path)
            return False
    
    def _invalidate_cache_file(self, cache_path: Path):
        """개별 캐시 파일 무효화"""
        try:
            if cache_path.exists():
                cache_path.unlink()
                self._cache_stats['invalidations'] += 1
        except Exception as e:
            print(f"⚠️  캐시 파일 삭제 실패: {e}")
    
    def clear_cache(self):
        """모든 캐시 삭제 (사용자 호출용)"""
        self._clear_all_cache()
        print("🧹 모든 캐시가 삭제되었습니다.")
    
    def _cleanup_old_version_cache(self):
        """구버전 캐시 파일들 조용히 삭제"""
        try:
            current_version_suffix = f"_v{self.CACHE_VERSION.replace('.', '_')}.pkl"
            deleted_count = 0
            
            # 구버전 캐시 파일 패턴 검사
            old_patterns = [
                "*_padded_*.pkl",  # 구버전 padded 패턴
                "*_v1_*.pkl",      # v1 버전
                "*_421.pkl"        # 특정 길이 패턴
            ]
            
            for pattern in old_patterns:
                for cache_file in self.cache_dir.glob(pattern):
                    try:
                        cache_file.unlink(missing_ok=True)
                        deleted_count += 1
                        print(f"⚠️  캐시 무효화: 구버전 메타데이터 구조 - {cache_file.name}")
                    except Exception:
                        pass
            
            # 현재 버전이 아닌 파일들 삭제
            for cache_file in self.cache_dir.glob("*.pkl"):
                if not cache_file.name.endswith(current_version_suffix):
                    try:
                        cache_file.unlink(missing_ok=True)
                        deleted_count += 1
                    except Exception:
                        pass
                
        except Exception:
            pass  # 오류도 조용히 무시
    
    def _clear_all_cache(self):
        """모든 캐시 삭제 (내부 사용)"""
        # 메모리 캐시 삭제
        self._memory_cache.clear()
        
        try:
            # 디스크 캐시 삭제 (모든 패턴)
            deleted_count = 0
            cache_patterns = ["*.pkl", "*.json"]
            
            for pattern in cache_patterns:
                for cache_file in self.cache_dir.glob(pattern):
                    try:
                        cache_file.unlink(missing_ok=True)
                        deleted_count += 1
                    except Exception as e:
                        # 파일 삭제 실패 시 조용히 무시
                        pass
            
            # 메타데이터 파일 삭제
            if self.metadata_path.exists():
                try:
                    self.metadata_path.unlink()
                except Exception:
                    pass
                
        except Exception:
            pass  # 오류 조용히 처리
    
    def save_epoch_data(self, dataloader, epoch: int, processed_batches: List[Dict[str, Any]]):
        """에포크 처리 데이터 저장"""
        dataset_hash = self._generate_dataset_hash(dataloader)
        cache_key = f"{dataset_hash}_epoch_{epoch}"
        cache_path = self._get_cache_path(dataset_hash, epoch)
        
        try:
            # 디렉토리 존재 확인 및 생성
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 메모리 캐시에 저장
            self._memory_cache[cache_key] = processed_batches
            
            # 디스크에 저장
            with open(cache_path, 'wb') as f:
                pickle.dump(processed_batches, f)
            
            self._cache_stats['saves'] += 1
            
            # 캐시 크기 관리
            self._cleanup_old_cache()
            
            print(f" 에포크 {epoch} 데이터 캐시 저장 완료 ({len(processed_batches)} 배치)")
            
        except Exception as e:
            print(f"에포크 캐시 저장 실패: {e}")
            # 오류 발생 시 메모리에서도 제거
            if cache_key in self._memory_cache:
                del self._memory_cache[cache_key]
    
    def load_epoch_data(self, dataloader, epoch: int) -> Optional[List[Dict[str, Any]]]:
        """캐시된 에포크 데이터 로드"""
        dataset_hash = self._generate_dataset_hash(dataloader)
        cache_key = f"{dataset_hash}_epoch_{epoch}"
        
        # 메모리 캐시 확인
        if cache_key in self._memory_cache:
            self._cache_stats['memory_hits'] += 1
            print(f" 에포크 {epoch} 데이터 메모리 캐시 히트!")
            return self._memory_cache[cache_key]
        
        # 디스크 캐시 확인
        cache_path = self._get_cache_path(dataset_hash, epoch)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    processed_batches = pickle.load(f)
                
                # 메모리 캐시에도 저장
                self._memory_cache[cache_key] = processed_batches
                
                self._cache_stats['disk_hits'] += 1
                print(f"📂 에포크 {epoch} 데이터 디스크 캐시 히트!")
                return processed_batches
                
            except Exception as e:
                # 캐시 파일이 손상된 경우 삭제
                self._invalidate_cache_file(cache_path)
        
        self._cache_stats['misses'] += 1
        return None
    
    def _cleanup_old_cache(self):
        """오래된 캐시 파일 삭제 (크기 제한 기반)"""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            if not cache_files:
                return
            
            # 파일 크기 계산 (안전한 방식)
            total_size = 0
            valid_files = []
            for f in cache_files:
                try:
                    if f.exists():
                        total_size += f.stat().st_size
                        valid_files.append(f)
                except (OSError, FileNotFoundError):
                    # 파일에 접근할 수 없는 경우 스킵
                    continue
            
            if total_size > self.max_cache_size_bytes:
                # 오래된 파일부터 삭제 (수정 시간 기준)
                valid_files.sort(key=lambda f: f.stat().st_mtime)
                
                deleted_size = 0
                for cache_file in valid_files:
                    if total_size - deleted_size <= self.max_cache_size_bytes * 0.8:
                        break
                    
                    try:
                        if cache_file.exists():
                            file_size = cache_file.stat().st_size
                            cache_file.unlink(missing_ok=True)
                            deleted_size += file_size
                    except Exception:
                        # 개별 파일 삭제 실패 시 계속
                        continue
                        
        except Exception:
            pass  # 전체 오류 조용히 무시
    
    def clear_cache(self):
        """모든 캐시 삭제"""
        self._memory_cache.clear()
        
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            print("🧹 모든 캐시가 삭제되었습니다.")
        except Exception as e:
            print(f"캐시 삭제 실패: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        total_requests = sum(self._cache_stats.values()) - self._cache_stats['saves']
        hit_rate = 0.0
        
        if total_requests > 0:
            total_hits = self._cache_stats['memory_hits'] + self._cache_stats['disk_hits']
            hit_rate = (total_hits / total_requests) * 100
        
        return {
            **self._cache_stats,
            'hit_rate_percent': hit_rate,
            'total_requests': total_requests,
            'memory_cache_size': len(self._memory_cache)
        }
    
    def print_cache_stats(self):
        """캐시 통계 출력"""
        stats = self.get_cache_stats()
        print(f" 에포크 캐시 통계:")
        print(f"   메모리 히트: {stats['memory_hits']}")
        print(f"   디스크 히트: {stats['disk_hits']}")
        print(f"   미스: {stats['misses']}")
        print(f"   저장: {stats['saves']}")
        print(f"   히트율: {stats['hit_rate_percent']:.1f}%")
        print(f"   메모리 캐시 크기: {stats['memory_cache_size']}")


# 캐시 테스트 메인 코드
if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    print("🧪 캐시 시스템 테스트 시작")
    print("=" * 50)
    
    # 테스트용 더미 데이터셋 생성
    class DummyDataset:
        def __init__(self, size=100):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            return {
                'input_sequence': torch.randn(421),  # padded_421 시뮬레이션
                'target_actions': torch.randint(0, 10, (50,)),
                'attention_mask': torch.ones(421),
                'action_prediction_mask': torch.ones(50),
                'circuit_specs': [{'gates': ['H', 'CNOT', 'RZ'] * (idx % 5 + 1)}]
            }
    
    # 테스트 데이터로더 생성
    dataset = DummyDataset(size=50)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # 캐시 인스턴스 생성
    cache = EpochCache(cache_dir="test_cache", max_cache_size_gb=0.1)
    
    print("\n1️⃣ 캐시 초기 상태 확인")
    print(f"캐시 디렉토리: {cache.cache_dir}")
    print(f"메타데이터 파일: {cache.metadata_path}")
    cache.print_cache_stats()
    
    # 테스트용 더미 배치 데이터
    dummy_batches = [
        {'batch_id': i, 'data': f'batch_{i}_data', 'processed': True}
        for i in range(5)
    ]
    
    print("\n2️⃣ 캐시 저장 테스트")
    epoch = 1
    has_cache_before = cache.has_cached_epoch(dataloader, epoch)
    print(f"에포크 {epoch} 캐시 존재 여부 (저장 전): {has_cache_before}")
    
    # 캐시 저장
    cache.save_epoch_data(dataloader, epoch, dummy_batches)
    
    has_cache_after = cache.has_cached_epoch(dataloader, epoch)
    print(f"에포크 {epoch} 캐시 존재 여부 (저장 후): {has_cache_after}")
    
    print("\n3️⃣ 캐시 로드 테스트")
    loaded_data = cache.load_epoch_data(dataloader, epoch)
    if loaded_data:
        print(f"✅ 캐시 로드 성공! 배치 수: {len(loaded_data)}")
        print(f"첫 번째 배치: {loaded_data[0]}")
    else:
        print("❌ 캐시 로드 실패")
    
    print("\n4️⃣ 메모리 캐시 테스트")
    # 메모리에서 다시 로드
    loaded_data_memory = cache.load_epoch_data(dataloader, epoch)
    if loaded_data_memory:
        print("✅ 메모리 캐시에서 로드 성공!")
    
    print("\n5️⃣ 다른 에포크 테스트")
    epoch2 = 2
    dummy_batches2 = [
        {'batch_id': i, 'data': f'epoch2_batch_{i}_data', 'processed': True}
        for i in range(3)
    ]
    cache.save_epoch_data(dataloader, epoch2, dummy_batches2)
    
    print("\n6️⃣ 캐시 통계 확인")
    cache.print_cache_stats()
    
    print("\n7️⃣ 메타데이터 파일 확인")
    if cache.metadata_path.exists():
        with open(cache.metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"메타데이터 버전: {metadata.get('version')}")
        print(f"캐시 엔트리 수: {len(metadata.get('cache_entries', {}))}")
        for key, info in metadata.get('cache_entries', {}).items():
            print(f"  - {key}: {info.get('batch_count')}개 배치, {info.get('file_size', 0)/1024:.1f}KB")
    
    print("\n8️⃣ 캐시 무효화 테스트")
    # 다른 데이터로더로 테스트 (다른 해시 생성)
    dataset_different = DummyDataset(size=30)  # 다른 크기
    dataloader_different = DataLoader(dataset_different, batch_size=8, shuffle=False)
    
    has_cache_different = cache.has_cached_epoch(dataloader_different, epoch)
    print(f"다른 데이터로더의 에포크 {epoch} 캐시 존재 여부: {has_cache_different}")
    
    print("\n9️⃣ 캐시 정리 테스트")
    print("캐시 정리 전:")
    cache_files_before = list(cache.cache_dir.glob("*.pkl"))
    print(f"캐시 파일 수: {len(cache_files_before)}")
    
    # 캐시 정리 (크기 제한을 매우 작게 설정)
    cache.max_cache_size_bytes = 1024  # 1KB로 제한
    cache._cleanup_old_cache()
    
    print("캐시 정리 후:")
    cache_files_after = list(cache.cache_dir.glob("*.pkl"))
    print(f"캐시 파일 수: {len(cache_files_after)}")
    
    print("\n🔟 최종 통계")
    cache.print_cache_stats()
    
    print("\n🧹 테스트 정리")
    cache.clear_cache()
    
    print("\n✅ 캐시 시스템 테스트 완료!")
    print("=" * 50)
