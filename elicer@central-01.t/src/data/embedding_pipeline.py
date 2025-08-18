"""
Embedding Pipeline Module
CircuitSpec -> Grid Encoder -> Decision Transformer Embedding
"""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import hashlib
import json
import pickle
from pathlib import Path
import threading
import time
import sys
from utils.debug_utils import debug_print

# 🚀 DEBUG 모드 설정
DEBUG_MODE = False  # 성능 최적화를 위해 기본값 False

# 임포트 경로 문제 해결
try:
    # 절대 경로 시도
    from .quantum_circuit_dataset import CircuitSpec
    from ..encoding.grid_graph_encoder import GridGraphEncoder
    from ..encoding.Decision_Transformer_Embed import QuantumGateSequenceEmbedding
except ImportError:
    # 상대 경로 시도
    try:
        from .quantum_circuit_dataset import CircuitSpec
        from ..encoding.grid_graph_encoder import GridGraphEncoder
        from ..encoding.Decision_Transformer_Embed import QuantumGateSequenceEmbedding
    except ImportError:
        # 로컬 경로 시도
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from data.quantum_circuit_dataset import CircuitSpec
        from encoding.grid_graph_encoder import GridGraphEncoder
        from encoding.Decision_Transformer_Embed import QuantumGateSequenceEmbedding

# 🎆 NEW: 게이트 레지스트리 싱글톤 임포트
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))
from gates import QuantumGateRegistry

@dataclass
class BatchMetadata:
    """배치 메타데이터 캐시용 데이터 클래스"""
    circuit_ids: List[str]
    num_qubits: List[int]
    num_gates: List[int]
    batch_size: int
    timestamp: float

@dataclass
class EmbeddingConfig:
    """임베딩 설정"""
    d_model: int = 512
    n_gate_types: int = None  # NEW: gate vocab 싱글톤에서 자동 설정
    n_qubits: int = 10
    max_seq_len: int = 1000
    max_time_steps: int = 50

    def __post_init__(self):
        """초기화 후 gate 수를 싱글톤에서 가져오기"""
        if self.n_gate_types is None:
            self.n_gate_types = QuantumGateRegistry.get_singleton_gate_count()
            print(f" EmbeddingConfig: Using gate vocab singleton, n_gate_types = {self.n_gate_types}")


class EmbeddingPipeline:
    """🚀 완전한 임베딩 파이프라인 (캐싱 시스템 통합)"""
    
    def __init__(self, config: EmbeddingConfig, enable_cache: bool = True):
        self.config = config
        self.enable_cache = enable_cache
        
        # 🚀 캐싱 시스템 초기화
        self._memory_cache = {}
        self._cache_access_order = []
        self._cache_lock = threading.RLock()
        self._max_cache_size = 1000
        self._cache_stats = {'hits': 0, 'misses': 0, 'total': 0}
        
        # 🚀 NEW: 배치 메타데이터 캐시 초기화
        self._batch_metadata_cache = {}
        self._batch_cache_access_order = []
        self._max_batch_cache_size = 800
        self._batch_cache_stats = {'hits': 0, 'misses': 0, 'total': 0}
        
        # 게이트 레지스트리 싱글톤 초기화
        self.gate_registry = QuantumGateRegistry()
        
        # 게이트 vocab 초기화
        self.gate_vocab = self.gate_registry.get_gate_vocab()
        
        # 게이트 수 확인 및 설정 동기화
        actual_gate_count = len(self.gate_vocab)
        if self.config.n_gate_types != actual_gate_count:
            print(f" Config mismatch: expected {self.config.n_gate_types}, got {actual_gate_count}")
            self.config.n_gate_types = actual_gate_count
        print(f" 🚀 EmbeddingPipeline initialized with {actual_gate_count} gate types (Cache: {enable_cache})")
        
        # Grid Encoder 초기화
        self.grid_encoder = GridGraphEncoder()
        
        # Decision Transformer Embedding 초기화
        self.dt_embedding = QuantumGateSequenceEmbedding(
            d_model=config.d_model,
            n_gate_types=config.n_gate_types,
            n_qubits=config.n_qubits,
            max_seq_len=config.max_seq_len
        )
    
    def _generate_cache_key(self, circuit_spec: CircuitSpec) -> str:
        """🚀 회로 스펙으로부터 고유한 캐시 키 생성"""
        # 🚀 GateOperation 객체들을 직렬화 가능한 형태로 변환
        serializable_gates = []
        for gate in circuit_spec.gates:
            if hasattr(gate, '__dict__'):
                # GateOperation 객체인 경우 딕셔너리로 변환
                gate_dict = {
                    'gate_type': getattr(gate, 'gate_type', str(gate)),
                    'qubits': getattr(gate, 'qubits', []),
                    'parameters': getattr(gate, 'parameters', [])
                }
                serializable_gates.append(gate_dict)
            else:
                # 이미 직렬화 가능한 형태인 경우
                serializable_gates.append(str(gate))
        
        circuit_data = {
            'circuit_id': circuit_spec.circuit_id,
            'gates': serializable_gates,
            'num_qubits': circuit_spec.num_qubits,
            'depth': circuit_spec.depth,
            'd_model': self.config.d_model,
            'max_seq_len': self.config.max_seq_len
        }
        data_str = json.dumps(circuit_data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, torch.Tensor]]:
        """🚀 캐시에서 데이터 조회"""
        if not self.enable_cache:
            return None
            
        with self._cache_lock:
            self._cache_stats['total'] += 1
            
            if cache_key in self._memory_cache:
                cached_data = self._memory_cache[cache_key]
                
                # 🚀 NEW: 캐시된 데이터에 SAR 기반 메타데이터가 없으면 캐시 무효화 (호환성 보장)
                needs_invalidation = False
                
                # num_gates가 없거나 sar_sequence_len이 없으면 무효화
                if 'num_gates' not in cached_data or 'sar_sequence_len' not in cached_data:
                    needs_invalidation = True
                    print(f"⚠️  캐시 무효화: SAR 메타데이터 누락 - {cache_key}")
                
                # original_gate_count가 없으면 무효화 (새로운 구조)
                elif 'original_gate_count' not in cached_data:
                    needs_invalidation = True
                    print(f"⚠️  캐시 무효화: 구버전 메타데이터 구조 - {cache_key}")
                
                if needs_invalidation:
                    # 오래된 캐시 데이터 - 재계산 필요
                    del self._memory_cache[cache_key]
                    if cache_key in self._cache_access_order:
                        self._cache_access_order.remove(cache_key)
                    self._cache_stats['misses'] += 1
                    return None
                
                # 캐시 히트
                self._update_cache_access(cache_key)
                self._cache_stats['hits'] += 1
                return cached_data
            
            # 캐시 미스
            self._cache_stats['misses'] += 1
            return None
    
    def _put_to_cache(self, cache_key: str, data: Dict[str, torch.Tensor]) -> None:
        """🚀 데이터를 캐시에 저장"""
        if not self.enable_cache:
            return
            
        with self._cache_lock:
            # 캐시 크기 제한 확인
            if len(self._memory_cache) >= self._max_cache_size:
                # LRU 정책: 가장 오래된 항목 제거
                oldest_key = self._cache_access_order.pop(0)
                del self._memory_cache[oldest_key]
            
            self._memory_cache[cache_key] = data
            self._update_cache_access(cache_key)
    
    def _update_cache_access(self, cache_key: str) -> None:
        """🚀 캐시 접근 순서 업데이트 (LRU)"""
        if cache_key in self._cache_access_order:
            self._cache_access_order.remove(cache_key)
        self._cache_access_order.append(cache_key)
    
    def process_single_circuit(self, circuit_spec: CircuitSpec) -> Dict[str, torch.Tensor]:
        """🚀 단일 회로 처리 (캐싱 적용)"""
        
        # 🚀 캐시 키 생성
        cache_key = self._generate_cache_key(circuit_spec)
        
        # 🚀 캐시에서 조회 시도
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        # 🚀 캐시 미스 - 새로 계산
        start_time = time.time()
        
        # 1. Grid Encoder로 회로 인코딩
        encoded_data = self.grid_encoder.encode(circuit_spec)
        
        # 2. 인코딩된 데이터를 그리드 매트릭스로 변환
        grid_matrix_data = self.grid_encoder.to_grid_matrix(encoded_data)
        
        # 3. 🚀 NEW: 순수 게이트 수 기반 간단한 Decision Transformer Embedding 적용
        original_gate_count = len(circuit_spec.gates)
        # 패딩 길이는 배치 처리 시점에서 결정되므로 여기서는 None
        dt_results = self.dt_embedding.process_grid_matrix_data_simple(grid_matrix_data, original_gate_count, max_seq_len=None)
        
        # 4. 메타데이터 추가 (이미 올바른 게이트 수 기반으로 생성됨)
        
        # SAR 시퀀스 길이는 디버깅용으로만 보존
        sar_sequence_len = dt_results.get('sar_sequence_len', original_gate_count * 3)
        if hasattr(sar_sequence_len, 'item'):  # 텐서인 경우 스칼라로 변환
            sar_sequence_len = sar_sequence_len.item()
        
        dt_results.update({
            'circuit_id': circuit_spec.circuit_id,
            'num_qubits': circuit_spec.num_qubits,
            'num_gates': original_gate_count,  # 🔧 실제 액션 수 (원래 게이트 수) 사용
            'original_gate_count': original_gate_count,  # 원본 게이트 수
            'sar_sequence_len': sar_sequence_len  # SAR 시퀀스 길이 (디버깅용)
        })
        
        compute_time = time.time() - start_time
        
        # 🚀 결과를 캐시에 저장
        self._put_to_cache(cache_key, dt_results)
        
        # 🔇 DEBUG: print(f"⚡ 새로 계산: {circuit_spec.circuit_id} ({compute_time:.3f}s)")
        
        return dt_results
    
    def process_batch(self, circuit_specs: List[CircuitSpec]) -> Dict[str, torch.Tensor]:
        """🚀 배치 처리 (캐싱 및 메모리 효율성 적용)"""
        batch_size = len(circuit_specs)
        
        if batch_size == 0:
            return {}
        
        debug_print(f"🚀 배치 처리 시작: {batch_size}개 회로")
        
        # 🚀 NEW: 배치 내 최대 시퀀스 길이 계산 (패딩용)
        max_gate_count = max(len(spec.gates) for spec in circuit_specs)
        max_seq_len = max_gate_count * 3 + 1  # SAR 패턴 + EOS
        debug_print(f"🚀 배치 최대 시퀀스 길이: {max_seq_len} (최대 게이트 수: {max_gate_count})")
        
        # 각 회로를 개별 처리 (최대 길이 전달)
        batch_results = []
        for i, circuit_spec in enumerate(circuit_specs):
            debug_print(f"  회로 {i+1}/{batch_size} 처리 중... (게이트 수: {len(circuit_spec.gates)})")
            result = self._process_single_circuit_with_padding(circuit_spec, max_seq_len)
            batch_results.append(result)
        
        # 3. 배치 결과 합치기 (최대 게이트 수 기준 패딩)
        return self._combine_batch_results_simple(batch_results, max_gate_count)
    
    def _combine_batch_results_simple(self, batch_results: List[Dict[str, torch.Tensor]], max_gate_count: int) -> Dict[str, torch.Tensor]:
        """🚀 NEW: 순수 게이트 수 기반 간단한 배치 결과 합치기"""
        
        if not batch_results:
            raise ValueError("❌ 빈 배치 결과입니다!")
        
        batch_size = len(batch_results)
        combined = {}
        
        # 1. 메타데이터 수집
        meta_keys = ['circuit_id', 'num_qubits', 'num_gates']
        for key in meta_keys:
            combined[key] = [result[key] for result in batch_results]
        
        # 2. 배치 내 최대 SAR 시퀀스 길이 계산
        max_sar_len = max_gate_count * 3 + 1  # EOS 포함
        
        print(f"🚀 배치 패딩: 최대 게이트 수 {max_gate_count} → 최대 SAR 길이 {max_sar_len}")
        
        # 3. 텐서 패딩 및 스택
        tensor_keys = ['input_sequence', 'attention_mask', 'action_prediction_mask', 'target_actions', 'target_qubits', 'target_params']
        
        for key in tensor_keys:
            tensors = [result[key] for result in batch_results]
            
            # 최대 길이로 패딩
            padded_tensors = []
            for tensor in tensors:
                current_len = tensor.shape[0]
                if current_len < max_sar_len:
                    # 패딩 필요
                    if key == 'input_sequence':
                        # [seq_len, d_model] → [max_sar_len, d_model]
                        pad_len = max_sar_len - current_len
                        padding = torch.zeros(pad_len, tensor.shape[1], device=tensor.device, dtype=tensor.dtype)
                        padded_tensor = torch.cat([tensor, padding], dim=0)
                    elif key == 'attention_mask':
                        # [seq_len, seq_len] → [max_sar_len, max_sar_len]
                        padded_tensor = torch.zeros(max_sar_len, max_sar_len, device=tensor.device, dtype=torch.bool)
                        padded_tensor[:current_len, :current_len] = tensor
                    elif key == 'action_prediction_mask':
                        # [seq_len] → [max_sar_len]
                        padded_tensor = torch.zeros(max_sar_len, device=tensor.device, dtype=torch.bool)
                        padded_tensor[:current_len] = tensor
                    elif key == 'target_qubits':
                        # [actual_gate_count, 2] → [max_sar_len, 2]
                        pad_len = max_sar_len - current_len
                        padding = torch.full((pad_len, tensor.shape[1]), -1, device=tensor.device, dtype=tensor.dtype)
                        padded_tensor = torch.cat([tensor, padding], dim=0)
                    elif key in ['target_actions', 'target_params']:
                        # [actual_gate_count] → [max_sar_len]
                        pad_len = max_sar_len - current_len
                        if key == 'target_actions':
                            padding = torch.full((pad_len,), -1, device=tensor.device, dtype=tensor.dtype)
                        else:  # target_params
                            padding = torch.zeros(pad_len, device=tensor.device, dtype=tensor.dtype)
                        padded_tensor = torch.cat([tensor, padding], dim=0)
                    else:
                        # Default padding for other tensors
                        pad_len = max_sar_len - current_len
                        if len(tensor.shape) == 1:
                            padding = torch.zeros(pad_len, device=tensor.device, dtype=tensor.dtype)
                        else:
                            padding_shape = (pad_len,) + tensor.shape[1:]
                            padding = torch.zeros(padding_shape, device=tensor.device, dtype=tensor.dtype)
                        padded_tensor = torch.cat([tensor, padding], dim=0)
                else:
                    padded_tensor = tensor
                
                padded_tensors.append(padded_tensor)
            
            # 배치 차원으로 스택
            combined[key] = torch.stack(padded_tensors, dim=0)
        
        # 4. 개별 회로별 액션 마스크 조정
        if 'action_prediction_mask' in combined and 'num_gates' in combined:
            action_mask = combined['action_prediction_mask']  # [batch_size, max_sar_len]
            gate_counts = combined['num_gates']
            
            debug_print(f"🔧 액션 마스크 조정: {action_mask.shape}")
            
            # 각 회로별로 실제 게이트 수만큼만 액션 위치를 True로 설정
            for b in range(batch_size):
                actual_gates = gate_counts[b]
                actual_sar_len = actual_gates * 3 + 1
                
                # 전체 마스크를 False로 초기화
                action_mask[b] = False
                
                # 실제 액션 위치만 True로 설정 (1::3 패턴)
                for i in range(actual_gates):
                    action_pos = i * 3 + 1  # 1, 4, 7, 10...
                    if action_pos < max_sar_len:
                        action_mask[b, action_pos] = True
            
            combined['action_prediction_mask'] = action_mask
            
            # 검증
            total_true = action_mask.sum().item()
            expected_true = sum(gate_counts)
            debug_print(f"✅ 액션 마스크 검증: True 수 {total_true}, 예상 {expected_true}")
            
            if total_true != expected_true:
                raise ValueError(f"❌ 액션 마스크 불일치! True 수: {total_true}, 예상: {expected_true}")
        
        return combined
    
    def _combine_batch_results(self, batch_results: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """🚀 엄격한 검증을 통한 배치 결과 합치기"""
        if not batch_results:
            raise ValueError("❌ CRITICAL ERROR: 배치 결과가 비어있습니다!")
        
        batch_size = len(batch_results)
        combined = {}
        
        # 🚀 필수 메타데이터 검증
        required_meta_keys = ['circuit_id', 'num_qubits', 'num_gates']
        for key in required_meta_keys:
            if key not in batch_results[0]:
                raise ValueError(f"❌ CRITICAL ERROR: 필수 메타데이터 '{key}'가 누락되었습니다!")
        
        # 🚀 모든 회로에서 메타데이터 일관성 검증
        for idx, result in enumerate(batch_results):
            for key in required_meta_keys:
                if key not in result:
                    raise ValueError(f"❌ CRITICAL ERROR: 회로 {idx}에서 메타데이터 '{key}'가 누락되었습니다!")
                
                # 🚀 데이터 타입 및 값 검증
                if key == 'num_qubits':
                    if not isinstance(result[key], int) or result[key] <= 0:
                        raise ValueError(f"❌ CRITICAL ERROR: 회로 {idx}의 num_qubits가 잘못되었습니다: {result[key]}")
                elif key == 'num_gates':
                    if not isinstance(result[key], int) or result[key] <= 0:
                        raise ValueError(f"❌ CRITICAL ERROR: 회로 {idx}의 num_gates가 잘못되었습니다: {result[key]}")
        
        # 텐서 데이터 합치기
        tensor_keys = ['input_sequence', 'attention_mask', 'action_prediction_mask', 
                      'state_embedded', 'action_embedded', 'reward_embedded', 
                      'target_actions', 'target_qubits', 'target_params']
        
        for key in tensor_keys:
            if key in batch_results[0]:
                # 패딩을 통한 배치 처리 (가변 길이 지원)
                tensors = [result[key] for result in batch_results]
                
                # 🔍 215 길이 미스터리 해결을 위한 디버깅
                if key == 'action_prediction_mask':
                    debug_print(f"🔍 === 215 길이 미스터리 추적 ===")
                    debug_print(f"   키: {key}")
                    debug_print(f"   텐서 개수: {len(tensors)}")
                    for i, tensor in enumerate(tensors[:5]):  # 처음 5개만
                        debug_print(f"   텐서 {i} 형태: {tensor.shape}")
                    
                    # 최대 길이 계산
                    if tensors:
                        max_len_in_batch = max(t.shape[-1] if t.dim() > 0 else 0 for t in tensors)
                        debug_print(f"   배치 내 최대 길이: {max_len_in_batch}")
                
                combined[key] = self._pad_and_stack_tensors(tensors)
        
        # 🚀 엄격한 메타데이터 수집 및 검증
        meta_keys = ['circuit_id', 'num_qubits', 'num_gates', 'episode_time_len', 'sar_sequence_len']
        for key in meta_keys:
            if key in batch_results[0]:
                meta_list = [result[key] for result in batch_results]
                
                # 🚀 메타데이터 완전성 검증
                if len(meta_list) != batch_size:
                    raise ValueError(f"❌ CRITICAL ERROR: 메타데이터 '{key}' 길이 불일치! 예상: {batch_size}, 실제: {len(meta_list)}")
                
                # 🚀 None 값 검증
                if any(item is None for item in meta_list):
                    none_indices = [i for i, item in enumerate(meta_list) if item is None]
                    raise ValueError(f"❌ CRITICAL ERROR: 메타데이터 '{key}'에서 None 값 발견! 인덱스: {none_indices}")
                
                combined[key] = meta_list
        
        # 🔧 액션 마스크 배치별 조정 (메타데이터 수집 후 실행)
        debug_print(f"🔍 메타데이터 수집 후 키 확인: {list(combined.keys())}")
        if 'action_prediction_mask' in combined and 'num_gates' in combined:
            debug_print(f"🔧 액션 마스크 배치별 조정 시작")
            debug_print(f"   원본 마스크 형태: {combined['action_prediction_mask'].shape}")
            debug_print(f"   num_gates: {combined['num_gates']}")
            
            adjusted_mask, actual_action_counts = self._adjust_action_mask_for_batch(
                combined['action_prediction_mask'], 
                combined['num_gates']
            )
            combined['action_prediction_mask'] = adjusted_mask
            
            # 🔧 메타데이터를 실제 액션 수로 업데이트 (시퀀스 길이 제한 반영)
            debug_print(f"🔧 메타데이터 num_gates 업데이트:")
            debug_print(f"   원본 num_gates 합: {sum(combined['num_gates'])}")
            debug_print(f"   실제 액션 수 합: {actual_action_counts}")
            
            # 각 회로별로 실제 사용된 액션 수로 업데이트
            actual_gates_per_circuit = []
            action_idx = 0
            for batch_idx, original_gates in enumerate(combined['num_gates']):
                # 해당 회로의 실제 액션 수 계산
                sar_seq_len = original_gates * 3
                actual_seq_len = min(sar_seq_len + 1, combined['action_prediction_mask'].shape[1])
                actual_actions = len(torch.arange(1, actual_seq_len, 3))
                actual_gates_per_circuit.append(actual_actions)
                debug_print(f"   회로 {batch_idx}: {original_gates} → {actual_actions} 게이트")
            
            combined['num_gates'] = actual_gates_per_circuit
        else:
            debug_print(f"⚠️ 액션 마스크 조정 건너뜀:")
            debug_print(f"   action_prediction_mask 존재: {'action_prediction_mask' in combined}")
            debug_print(f"   num_gates 존재: {'num_gates' in combined}")
        
        # 🚀 최종 일관성 검증 (디버그 모드에서만)
        if DEBUG_MODE and 'num_qubits' in combined and 'num_gates' in combined:
            total_expected_gates = sum(combined['num_gates'])
            
            # target_actions 텐서가 있다면 크기 검증
            if 'target_actions' in combined:
                actual_total_elements = combined['target_actions'].numel()
                print(f"🔍 배치 검증: 예상 게이트 수 {total_expected_gates}, 실제 요소 수 {actual_total_elements}")
        
        # 배치 생성 완료 로그도 디버그 모드에서만
        if DEBUG_MODE:
            print(f"✅ 배치 생성 완료: {batch_size}개 회로, 큐빗 수 {combined['num_qubits']}, 게이트 수 {combined['num_gates']}")
        
        # 🚀 NEW: 배치 메타데이터 캐시에 저장
        if self.enable_cache and 'circuit_id' in combined and 'num_qubits' in combined and 'num_gates' in combined:
            self._cache_batch_metadata(combined)
        
        return combined
    
    def _process_single_circuit_with_padding(self, circuit_spec: CircuitSpec, max_seq_len: int) -> Dict[str, torch.Tensor]:
        """🚀 단일 회로 처리 (패딩 길이 지정)"""
        
        # 🚀 캐시 키 생성
        cache_key = self._generate_cache_key(circuit_spec)
        
        # 🚀 캐시에서 조회 시도 (패딩 길이 포함)
        padded_cache_key = f"{cache_key}_padded_{max_seq_len}"
        cached_result = self._get_from_cache(padded_cache_key)
        if cached_result is not None:
            return cached_result
        
        # 🚀 캐시 미스 - 새로 계산
        start_time = time.time()
        
        # 1. Grid Encoder로 회로 인코딩
        encoded_data = self.grid_encoder.encode(circuit_spec)
        
        # 2. 인코딩된 데이터를 그리드 매트릭스로 변환
        grid_matrix_data = self.grid_encoder.to_grid_matrix(encoded_data)
        
        # 3. 🚀 NEW: 패딩 길이를 포함한 Decision Transformer Embedding 적용
        original_gate_count = len(circuit_spec.gates)
        dt_results = self.dt_embedding.process_grid_matrix_data_simple(grid_matrix_data, original_gate_count, max_seq_len)
        
        # 4. 메타데이터 추가
        dt_results.update({
            'circuit_id': circuit_spec.circuit_id,
            'num_qubits': circuit_spec.num_qubits,
            'num_gates': original_gate_count,
            'episode_time_len': dt_results.get('episode_time_len', original_gate_count),
            'sar_sequence_len': original_gate_count * 3
        })
        
        # 🚀 캐시에 저장
        processing_time = time.time() - start_time
        self._put_to_cache(padded_cache_key, dt_results)
        
        debug_print(f"  회로 {circuit_spec.circuit_id} 처리 완료 (패딩: {max_seq_len}, 시간: {processing_time:.3f}초)")
        
        return dt_results
    
    def _generate_batch_cache_key(self, circuit_ids: List[str]) -> str:
        """🚀 배치 메타데이터 캐시 키 생성"""
        # 회로 ID 리스트를 정렬하여 순서에 무관한 키 생성
        sorted_ids = sorted(circuit_ids)
        key_string = "|".join(sorted_ids)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _cache_batch_metadata(self, batch_data: Dict[str, Any]) -> None:
        """🚀 배치 메타데이터 캐시에 저장"""
        if not self.enable_cache:
            return
            
        circuit_ids = batch_data['circuit_id']
        num_qubits = batch_data['num_qubits']
        num_gates = batch_data['num_gates']
        
        cache_key = self._generate_batch_cache_key(circuit_ids)
        
        with self._cache_lock:
            # 배치 메타데이터 생성
            metadata = BatchMetadata(
                circuit_ids=circuit_ids.copy(),
                num_qubits=num_qubits.copy(),
                num_gates=num_gates.copy(),
                batch_size=len(circuit_ids),
                timestamp=time.time()
            )
            
            # 캐시에 저장
            self._batch_metadata_cache[cache_key] = metadata
            
            # LRU 관리
            if cache_key in self._batch_cache_access_order:
                self._batch_cache_access_order.remove(cache_key)
            self._batch_cache_access_order.append(cache_key)
            
            # 캐시 크기 제한
            while len(self._batch_metadata_cache) > self._max_batch_cache_size:
                oldest_key = self._batch_cache_access_order.pop(0)
                del self._batch_metadata_cache[oldest_key]
    
    def get_cached_batch_metadata(self, circuit_ids: List[str]) -> Optional[BatchMetadata]:
        """🚀 캐시된 배치 메타데이터 조회"""
        if not self.enable_cache:
            return None
            
        cache_key = self._generate_batch_cache_key(circuit_ids)
        
        with self._cache_lock:
            self._batch_cache_stats['total'] += 1
            
            if cache_key in self._batch_metadata_cache:
                # 캐시 히트
                self._batch_cache_stats['hits'] += 1
                
                # LRU 업데이트
                self._batch_cache_access_order.remove(cache_key)
                self._batch_cache_access_order.append(cache_key)
                
                return self._batch_metadata_cache[cache_key]
            else:
                # 캐시 미스
                self._batch_cache_stats['misses'] += 1
                return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """🚀 캐시 통계 반환 (임베딩 + 배치 메타데이터)"""
        with self._cache_lock:
            hit_rate = self._cache_stats['hits'] / max(self._cache_stats['total'], 1) * 100
            batch_hit_rate = self._batch_cache_stats['hits'] / max(self._batch_cache_stats['total'], 1) * 100
            
            return {
                'cache_enabled': self.enable_cache,
                # 임베딩 캐시 통계
                'embedding_cache': {
                    'total_requests': self._cache_stats['total'],
                    'cache_hits': self._cache_stats['hits'],
                    'cache_misses': self._cache_stats['misses'],
                    'hit_rate_percent': hit_rate,
                    'cache_size': len(self._memory_cache),
                },
                # 🚀 NEW: 배치 메타데이터 캐시 통계
                'batch_metadata_cache': {
                    'total_requests': self._batch_cache_stats['total'],
                    'cache_hits': self._batch_cache_stats['hits'],
                    'cache_misses': self._batch_cache_stats['misses'],
                    'hit_rate_percent': batch_hit_rate,
                    'cache_size': len(self._batch_metadata_cache),
                },
                'max_cache_size': self._max_cache_size
            }
    
    def clear_cache(self) -> None:
        # 🔧 강제 캐시 리셋 (액션 마스크 배치별 조정 반영)
        if self.enable_cache:
            print("🔧 강제 캐시 리셋: 액션 마스크 배치별 조정 로직 추가 - 각 회로의 실제 길이에 맞는 마스크 생성")
            self._memory_cache.clear()
            self._batch_metadata_cache.clear()
            self._cache_access_order.clear()
            self._cache_stats = {'hits': 0, 'misses': 0, 'total': 0}
            self._batch_cache_access_order.clear()
            self._batch_cache_stats = {'hits': 0, 'misses': 0, 'total': 0}
            
            # NEW: 배치 메타데이터 캐시 초기화
            self._batch_metadata_cache.clear()
            self._batch_cache_access_order.clear()
            self._batch_cache_stats = {'hits': 0, 'misses': 0, 'total': 0}
            
            print("모든 캐시가 초기화되었습니다 (임베딩 + 배치 메타데이터).")
            print("🚀 모든 캐시가 초기화되었습니다 (임베딩 + 배치 메타데이터).")
    
    def print_cache_stats(self) -> None:
        """🚀 캐시 통계 출력 (임베딩 + 배치 메타데이터)"""
        stats = self.get_cache_stats()
        print(f"\n🚀 캐시 통계:")
        print(f"   - 캐시 활성화: {stats['cache_enabled']}")
        
        # 임베딩 캐시 통계
        embedding_stats = stats['embedding_cache']
        print(f"\n   📊 임베딩 캐시:")
        print(f"      - 총 요청: {embedding_stats['total_requests']}")
        print(f"      - 캐시 히트: {embedding_stats['cache_hits']}")
        print(f"      - 캐시 미스: {embedding_stats['cache_misses']}")
        print(f"      - 히트율: {embedding_stats['hit_rate_percent']:.1f}%")
        print(f"      - 캐시 크기: {embedding_stats['cache_size']}/{stats['max_cache_size']}")
        
        # 🚀 NEW: 배치 메타데이터 캐시 통계
        batch_stats = stats['batch_metadata_cache']
        print(f"\n   🎯 배치 메타데이터 캐시:")
        print(f"      - 총 요청: {batch_stats['total_requests']}")
        print(f"      - 캐시 히트: {batch_stats['cache_hits']}")
        print(f"      - 캐시 미스: {batch_stats['cache_misses']}")
        print(f"      - 히트율: {batch_stats['hit_rate_percent']:.1f}%")
        print(f"      - 캐시 크기: {batch_stats['cache_size']}/{self._max_batch_cache_size}")
    
    def _pad_and_stack_tensors(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        서로 다른 크기의 텐서들을 패딩하여 배치로 합치기
        
        Args:
            tensors: 서로 다른 크기의 텐서 리스트
        
        Returns:
            패딩된 배치 텐서
        """
        if not tensors:
            return torch.tensor([])
        
        # 최대 크기 계산
        max_dims = []
        for dim in range(len(tensors[0].shape)):
            max_size = max(tensor.shape[dim] for tensor in tensors)
            max_dims.append(max_size)
            
            # 🔍 시퀀스 길이 차원 디버깅 (마지막 차원)
            if dim == len(tensors[0].shape) - 1:
                debug_print(f"🔍 === 시퀀스 길이 차원 분석 ===")
                debug_print(f"   차원 {dim} (시퀀스 길이)")
                debug_print(f"   최대 크기: {max_size}")
                debug_print(f"   개별 텐서 길이들:")
                for i, tensor in enumerate(tensors[:10]):  # 처음 10개만
                    debug_print(f"     텐서 {i}: {tensor.shape[dim]}")
                if len(tensors) > 10:
                    debug_print(f"     ... (총 {len(tensors)}개 텐서)")
                debug_print(f"   모든 길이: {[t.shape[dim] for t in tensors]}")
        
        # Boolean 텐서 여부 확인
        is_bool_tensor = tensors[0].dtype == torch.bool
        
        # 패딩된 텐서 리스트 생성
        padded_tensors = []
        for tensor in tensors:
            # 패딩 크기 계산
            padding = []
            for dim in reversed(range(len(tensor.shape))):
                pad_size = max_dims[dim] - tensor.shape[dim]
                padding.extend([0, pad_size])  # (left, right) padding
            
            # 패딩 적용
            if any(p > 0 for p in padding):
                # Boolean 마스크의 경우 False(0)로 패딩, 다른 텐서는 0으로 패딩
                pad_value = False if is_bool_tensor else 0
                padded_tensor = torch.nn.functional.pad(tensor, padding, value=pad_value)
            else:
                padded_tensor = tensor
            
            padded_tensors.append(padded_tensor)
        
        # 배치 차원으로 합치기
        return torch.stack(padded_tensors, dim=0)
    
    def _adjust_action_mask_for_batch(self, action_mask: torch.Tensor, num_gates_list: List[int]) -> torch.Tensor:
        """
        배치별로 액션 마스크를 각 회로의 실제 길이에 맞게 조정
        
        Args:
            action_mask: [batch_size, max_seq_len] 패딩된 액션 마스크
            num_gates_list: 각 회로의 실제 게이트 수 리스트
        
        Returns:
            조정된 액션 마스크
        """
        debug_print(f"🔧 === 액션 마스크 배치별 조정 함수 시작 ===")
        debug_print(f"   action_mask 타입: {type(action_mask)}")
        debug_print(f"   action_mask 형태: {action_mask.shape if hasattr(action_mask, 'shape') else 'No shape'}")
        debug_print(f"   action_mask 내용: {action_mask}")
        debug_print(f"   num_gates_list: {num_gates_list}")
        
        # 형태 검증 및 수정
        if not hasattr(action_mask, 'shape'):
            raise ValueError(f"❌ action_mask가 텐서가 아닙니다: {type(action_mask)}")
        
        # 3차원인 경우 2차원으로 변환 (중간 차원 제거)
        if len(action_mask.shape) == 3:
            debug_print(f"   3차원 액션 마스크를 2차원으로 변환: {action_mask.shape}")
            action_mask = action_mask.squeeze(1)  # 중간 차원 제거

        
        batch_size, max_seq_len = action_mask.shape
        debug_print(f"   배치 크기: {batch_size}, 최대 시퀀스 길이: {max_seq_len}")
        
        adjusted_mask = torch.zeros_like(action_mask)
        total_expected_actions = 0
        
        for batch_idx, num_gates in enumerate(num_gates_list):
            # SAR 구조에서 실제 시퀀스 길이 계산
            # num_gates는 실제 게이트 수이므로, SAR 시퀀스 길이는 num_gates * 3
            sar_seq_len = num_gates * 3
            actual_seq_len = min(sar_seq_len + 1, max_seq_len)  # EOS 토큰 포함, 최대 길이 제한
            
            # 액션 위치만 True로 설정 (1::3 패턴)
            action_positions = torch.arange(1, actual_seq_len, 3, device=action_mask.device)
            action_positions = action_positions[action_positions < max_seq_len]  # 범위 내 위치만
            
            if len(action_positions) > 0:
                adjusted_mask[batch_idx, action_positions] = True
                total_expected_actions += len(action_positions)
            
            debug_print(f"   회로 {batch_idx}: gates={num_gates}, sar_len={sar_seq_len}, actions={len(action_positions)}")
        
        debug_print(f"🔧 액션 마스크 배치별 조정 완료:")
        debug_print(f"   원본 True 수: {action_mask.sum().item()}")
        debug_print(f"   조정 후 True 수: {adjusted_mask.sum().item()}")
        debug_print(f"   예상 True 수: {sum(num_gates_list)} (메타데이터 합)")
        debug_print(f"   실제 액션 수: {total_expected_actions}")
        
        return adjusted_mask, total_expected_actions


def create_embedding_pipeline(config: EmbeddingConfig = None, enable_cache: bool = True) -> EmbeddingPipeline:
    """🚀 임베딩 파이프라인 팩토리 함수 (캐싱 지원)"""
    if config is None:
        config = EmbeddingConfig()
    
    # 🚀 캐싱이 활성화된 임베딩 파이프라인 생성
    pipeline = EmbeddingPipeline(config, enable_cache=enable_cache)
    
    if enable_cache:
        print(f"🚀 임베딩 파이프라인 생성 완료! (캐싱: {enable_cache}, 최대 캐시 크기: {pipeline._max_cache_size})")
    
    return pipeline


# 사용 예시
if __name__ == "__main__":
    from quantum_circuit_dataset import DatasetManager
    
    # 데이터셋 로딩
    manager = DatasetManager("OAT_Model/dataset/unified_batch_experiment_results_with_circuits.json")
    circuit_specs = manager.parse_circuits()
    
    # 임베딩 파이프라인 생성
    config = EmbeddingConfig(d_model=256)  # NEW: 싱글톤에서 가져온 gate 수로 임베딩 레이어 초기화
    pipeline = create_embedding_pipeline(config)
    
    # 단일 회로 테스트
    if circuit_specs:
        print("Testing single circuit processing...")
        result = pipeline.process_single_circuit(circuit_specs[0])
        
        print(f"Circuit ID: {result['circuit_id']}")
        print(f"Input sequence shape: {result['input_sequence'].shape}")
        print(f"Attention mask shape: {result['attention_mask'].shape}")
        
        # 배치 테스트
        print("\nTesting batch processing...")
        batch_result = pipeline.process_batch(circuit_specs[:3])
        
        if batch_result:
            print(f"Batch input sequence shape: {batch_result['input_sequence'].shape}")
            print(f"Batch attention mask shape: {batch_result['attention_mask'].shape}")
