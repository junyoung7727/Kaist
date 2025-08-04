import os
import json
from typing import Dict, Any, List, Optional
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit
from qiskit.result import Result
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# 환경변수에서 IBM 토큰 가져오기 (또는 직접 입력)
IBM_TOKEN = os.getenv('IBM_TOKEN') or "YOUR_IBM_TOKEN_HERE"

class IBMJobRetriever:
    """IBM Quantum job 결과 가져오기 클래스"""
    
    def __init__(self, token: str = None):
        """
        초기화
        
        Args:
            token: IBM Quantum 토큰
        """
        self.token = token or IBM_TOKEN
        self.service = None
        self.initialize()
    
    def initialize(self):
        """IBM Quantum 서비스 초기화"""
        try:
            # 먼저 저장된 계정이 있는지 확인
            try:
                self.service = QiskitRuntimeService()
                print("✅ 저장된 IBM Quantum 계정 사용")
            except Exception:
                # 저장된 계정이 없으면 토큰으로 초기화
                if self.token == "YOUR_IBM_TOKEN_HERE":
                    print("❌ IBM 토큰이 설정되지 않았습니다.")
                    print("다음 중 하나를 수행하세요:")
                    print("1. 환경변수 설정: export IBM_TOKEN='your_token_here'")
                    print("2. 코드에서 직접 설정: IBM_TOKEN = 'your_token_here'")
                    print("3. 또는 QiskitRuntimeService.save_account('your_token_here') 실행")
                    raise ValueError("IBM 토큰이 설정되지 않음")
                
                self.service = QiskitRuntimeService(
                    channel="ibm_quantum_platform",
                    token=self.token
                )
                print("✅ 토큰으로 IBM Quantum 서비스 초기화 완료")
                
        except Exception as e:
            print(f"❌ IBM Quantum 서비스 초기화 실패: {e}")
            print("\n해결 방법:")
            print("1. IBM Quantum 토큰 확인: https://quantum.ibm.com/")
            print("2. 계정 저장: QiskitRuntimeService.save_account('your_token')")
            print("3. 환경변수 설정: export IBM_TOKEN='your_token'")
            raise
    
    def get_job_info(self, job_id: str) -> Dict[str, Any]:
        """
        Job 기본 정보 가져오기
        
        Args:
            job_id: IBM Quantum job ID
            
        Returns:
            Job 정보 딕셔너리
        """
        try:
            job = self.service.job(job_id)
            
            # Job 상태 및 기본 정보
            info = {
                'job_id': job_id,
                'status': job.status().name,
                'creation_date': str(job.creation_date),
                'backend': job.backend().name if hasattr(job, 'backend') else 'Unknown',
                'num_circuits': len(job.circuits()) if hasattr(job, 'circuits') else 'Unknown'
            }
            
            # 추가 메타데이터
            if hasattr(job, 'tags'):
                info['tags'] = job.tags()
            
            print(f"📋 Job 정보:")
            for key, value in info.items():
                print(f"   {key}: {value}")
                
            return info
            
        except Exception as e:
            print(f"❌ Job 정보 가져오기 실패: {e}")
            return {}
    
    def get_job_results(self, job_id: str, save_to_file: bool = True) -> Optional[Dict[str, Any]]:
        """
        Job 결과 가져오기
        
        Args:
            job_id: IBM Quantum job ID
            save_to_file: 결과를 파일로 저장할지 여부
            
        Returns:
            Job 결과 딕셔너리
        """
        try:
            job = self.service.job(job_id)
            
            # Job 상태 확인
            status = job.status()
            print(f"🔍 Job {job_id} 상태: {status.name}")
            
            if status.name != 'DONE':
                print(f"⚠️  Job이 완료되지 않았습니다. 현재 상태: {status.name}")
                return None
            
            # 결과 가져오기
            print("📥 결과 가져오는 중...")
            result = job.result()
            
            # 결과 정리
            results_data = {
                'job_id': job_id,
                'backend': job.backend().name if hasattr(job, 'backend') else 'Unknown',
                'creation_date': str(job.creation_date),
                'num_circuits': len(result.results),
                'results': []
            }
            
            # 각 회로별 결과 처리
            for i, circuit_result in enumerate(result.results):
                circuit_data = {
                    'circuit_index': i,
                    'shots': circuit_result.shots,
                    'success': circuit_result.success,
                    'counts': circuit_result.data.counts if hasattr(circuit_result.data, 'counts') else None,
                    'memory': circuit_result.data.memory if hasattr(circuit_result.data, 'memory') else None
                }
                results_data['results'].append(circuit_data)
            
            print(f"✅ {len(result.results)}개 회로 결과 가져오기 완료")
            
            # 파일 저장
            if save_to_file:
                filename = f"job_results_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(results_data, f, indent=2, ensure_ascii=False)
                print(f"💾 결과 저장됨: {filename}")
            
            return results_data
            
        except Exception as e:
            print(f"❌ Job 결과 가져오기 실패: {e}")
            return None
    
    def get_job_circuits(self, job_id: str, save_to_file: bool = True) -> Optional[List[QuantumCircuit]]:
        """
        Job에서 사용된 회로들 가져오기
        
        Args:
            job_id: IBM Quantum job ID
            save_to_file: 회로를 파일로 저장할지 여부
            
        Returns:
            QuantumCircuit 리스트
        """
        try:
            print(f"🔄 Job {job_id} 정보 가져오는 중... (대용량 job의 경우 시간이 걸릴 수 있습니다)")
            job = self.service.job(job_id)
            print("✅ Job 정보 가져오기 완료")
            
            # 회로 가져오기
            if hasattr(job, 'circuits'):
                circuits = job.circuits()
                print(f"🔧 {len(circuits)}개 회로 가져오기 완료")
                
                # 파일 저장 (96개만)
                if save_to_file:
                    # 96개 회로만 선택
                    circuits_to_save = circuits[:96]
                    print(f"📝 {len(circuits_to_save)}개 회로를 하나의 파일에 저장")
                    
                    # 하나의 파일에 모든 회로 저장
                    filename = f"circuits_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.qasm"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"// IBM Quantum Job: {job_id}\n")
                        f.write(f"// Total circuits: {len(circuits_to_save)}\n")
                        f.write(f"// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        
                        for i, circuit in enumerate(circuits_to_save):
                            f.write(f"// Circuit {i:04d}\n")
                            f.write(circuit.qasm())
                            f.write("\n\n")
                    
                    print(f"💾 회로 저장됨: {filename}")
                    
                    # 처음 3개 회로만 다이어그램으로 저장
                    for i in range(min(3, len(circuits_to_save))):
                        try:
                            fig_filename = f"circuit_{job_id}_{i:04d}.png"
                            circuits_to_save[i].draw(output='mpl', filename=fig_filename)
                            plt.close()
                        except:
                            pass  # 다이어그램 저장 실패해도 계속 진행
                
                return circuits
            else:
                print("⚠️  이 job에서는 회로 정보를 가져올 수 없습니다.")
                return None
                
        except Exception as e:
            print(f"❌ Job 회로 가져오기 실패: {e}")
            return None
    
    def analyze_job_statistics(self, job_id: str) -> Dict[str, Any]:
        """
        Job 결과 통계 분석
        
        Args:
            job_id: IBM Quantum job ID
            
        Returns:
            통계 정보 딕셔너리
        """
        try:
            results_data = self.get_job_results(job_id, save_to_file=False)
            if not results_data:
                return {}
            
            stats = {
                'total_circuits': results_data['num_circuits'],
                'total_shots': 0,
                'success_rate': 0,
                'unique_outcomes': set(),
                'most_common_outcome': None,
                'outcome_distribution': {}
            }
            
            successful_circuits = 0
            all_counts = {}
            
            for result in results_data['results']:
                if result['success']:
                    successful_circuits += 1
                
                if result['counts']:
                    stats['total_shots'] += result['shots']
                    
                    # 결과 통합
                    for outcome, count in result['counts'].items():
                        stats['unique_outcomes'].add(outcome)
                        all_counts[outcome] = all_counts.get(outcome, 0) + count
            
            stats['success_rate'] = successful_circuits / results_data['num_circuits'] * 100
            
            if all_counts:
                stats['most_common_outcome'] = max(all_counts, key=all_counts.get)
                stats['outcome_distribution'] = dict(sorted(all_counts.items(), 
                                                           key=lambda x: x[1], reverse=True)[:10])
            
            print(f"📊 Job 통계:")
            print(f"   총 회로 수: {stats['total_circuits']}")
            print(f"   총 샷 수: {stats['total_shots']}")
            print(f"   성공률: {stats['success_rate']:.1f}%")
            print(f"   고유 결과 수: {len(stats['unique_outcomes'])}")
            print(f"   가장 빈번한 결과: {stats['most_common_outcome']}")
            
            return stats
            
        except Exception as e:
            print(f"❌ 통계 분석 실패: {e}")
            return {}


def main():
    """메인 실행 함수"""
    print("🚀 IBM Qiskit Job 결과 가져오기 시작")
    print("=" * 50)
    
    # Job ID 설정 (여기에 실제 job ID 입력)
    JOB_ID = "d2830hhogaas73ctdju0"  # 사용자가 제공한 job ID
    
    try:
        # Job 가져오기 객체 생성
        retriever = IBMJobRetriever()
    
        # 3. Job 회로 가져오기
        print("\n3️⃣ Job 회로 가져오기")
        circuits = retriever.get_job_circuits(JOB_ID)
        

        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")


if __name__ == "__main__":
    main()