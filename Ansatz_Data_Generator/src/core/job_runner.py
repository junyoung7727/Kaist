#!/usr/bin/env python3
"""
Job Runner 모듈 - IBM 백엔드에서의 작업 실행 로직을 처리합니다.
"""

import sys
import os
import time
import gc
from typing import Dict, List, Any, Optional, Union, Tuple
from tqdm import tqdm

# 현재 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qiskit import transpile
from qiskit_ibm_runtime import SamplerV2 as Sampler
from src.backends import IBMBackendManager
from src.config import config


def run_mega_job(qiskit_circuits, circuit_metadata, ibm_backend, shots=None, circuit_shot_requirements=None):
    """
    IBM 백엔드에서 대량 회로를 한 번의 job으로 실행 (진짜 배치 실행)
    
    Args:
        qiskit_circuits (list): Qiskit 회로 목록
        circuit_metadata (list): 회로 메타데이터 목록
        ibm_backend (IBMBackendManager): IBM 백엔드 관리 객체
        shots (int): 기본 회로당 샷 수
        circuit_shot_requirements (list): 각 회로별 필요 샷 수 목록 (선택사항)
        
    Returns:
        tuple: (결과 객체, 실행 시간(초), 회로 메타데이터)
    """
    from src.utils.quantum_utils import calculate_error_rates_mega, calculate_robust_fidelity_mega
    
    # 샷 수 처리: config에서 가져오고, 제공되지 않으면 기본값 사용
    if shots is None:
        # 속성 접근 방식 사용
        shots = config.ibm_backend.default_shots
        
    from src.utils.quantum_utils import calculate_error_rates_mega, calculate_robust_fidelity_mega
    
    if not qiskit_circuits:
        print("⚠️ 실행할 회로가 없습니다.")
        return None, 0, circuit_metadata
    
    # 회로별 샷 수 결정
    if circuit_shot_requirements and len(circuit_shot_requirements) == len(qiskit_circuits):
        total_shots = sum(circuit_shot_requirements)
        print(f"\n🚀 IBM 백엔드에서 {len(qiskit_circuits)}개 회로를 한 번의 배치 job으로 실행 시작")
        print(f"   회로별 개별 샷 수: Config 설정에 따라 다름")
        print(f"   배치 총 실행 수: {total_shots:,}")
    else:
        total_shots = len(qiskit_circuits) * shots
        print(f"\n🚀 IBM 백엔드에서 {len(qiskit_circuits)}개 회로를 한 번의 배치 job으로 실행 시작")
        print(f"   회로당 고정 샷 수: {shots:,}")
        print(f"   배치 총 실행 수: {total_shots:,}")
    
    print(f"   예상 데이터 품질: {'🟢 높음' if total_shots/len(qiskit_circuits) >= 1024 else '🟡 보통' if total_shots/len(qiskit_circuits) >= 512 else '🔴 낮음'}")
    
    start_time = time.time()
    
    try:
        # IBM 백엔드에서 배치 실행
        print("🚀 IBM 백엔드에서 배치 job 제출 중...")
        
        # 각 회로별 샷 수 설정
        if circuit_shot_requirements:
            # 회로별 다른 샷 수 (현재 IBM API는 모든 회로에 동일한 샷 수만 지원)
            # 평균 샷 수 사용
            avg_shots = int(sum(circuit_shot_requirements) / len(circuit_shot_requirements))
            print(f"   회로별 평균 샷 수: {avg_shots}")
        else:
            avg_shots = shots
        
        # IBM Runtime Sampler 사용
        if not hasattr(ibm_backend, 'backend') or ibm_backend.backend is None:
            print("⚠️ 유효한 IBM 백엔드가 연결되어 있지 않습니다.")
            return None, time.time() - start_time, circuit_metadata
            
        sampler = Sampler(mode=ibm_backend.backend)
        
        # 배치 실행
        print(f"   {len(qiskit_circuits)}개 회로를 {avg_shots} 샷으로 실행 중...")
        print(f"   백엔드: {ibm_backend.name}")
        job = sampler.run(qiskit_circuits, shots=avg_shots)
        
        print(f"   Job ID: {job.job_id()}")
        print("   결과 기다리는 중...")
        
        # 결과 대기
        result = job.result()
        
        print("✅ 배치 실행 완료!")
        
        execution_time = time.time() - start_time
        print(f"   실행 시간: {execution_time:.2f}초")
        
        # 메모리 정리
        gc.collect()
        
        return result, execution_time, circuit_metadata
        
    except Exception as e:
        print(f"❌ 배치 실행 실패: {str(e)}")
        execution_time = time.time() - start_time
        return None, execution_time, circuit_metadata


def calculate_optimal_shots_and_batching(total_circuits: int, target_total_shots: int = 8000000, max_executions: int = 10000000):
    """
    Config 설정 기반 최적 샷 수 및 배치 분할 계산
    
    Args:
        total_circuits (int): 총 회로 수
        target_total_shots (int): 목표 총 샷 수 (참고용)
        max_executions (int): IBM 제한 최대 실행 수
        
    Returns:
        dict: 배치 분할 정보
    """
    # 가능한 샷 수 옵션들 - config에서 가져오거나 기본값 사용
    shot_options = getattr(config.ibm_backend, 'shot_options', [128, 256, 512, 1024, 2048, 4096, 8192])
    
    # 초기 최적 샷 수: 목표 총 샷 수 / 총 회로 수
    ideal_shots_per_circuit = target_total_shots / total_circuits
    
    # 가장 가까운 샷 수 옵션 찾기
    optimal_shots = min(shot_options, key=lambda x: abs(x - ideal_shots_per_circuit))
    
    # 총 실행 수
    total_executions = total_circuits * optimal_shots
    
    # 실행 수가 IBM 제한을 초과하는지 확인
    if total_executions > max_executions:
        print(f"⚠️ 경고: 총 실행 수({total_executions:,})가 IBM 제한({max_executions:,})을 초과합니다.")
        print("   샷 수를 줄이는 중...")
        
        # 가능한 가장 큰 샷 수 찾기
        for shots in sorted(shot_options, reverse=True):
            if total_circuits * shots <= max_executions:
                optimal_shots = shots
                total_executions = total_circuits * shots
                print(f"   조정된 샷 수: {optimal_shots}")
                break
    
    # 배치 크기 계산 (IBM 배치 제한: 300개 회로)
    max_batch_size = getattr(config.ibm_backend, 'max_batch_size', 300)
    
    if total_circuits <= max_batch_size:
        batch_count = 1
        batch_sizes = [total_circuits]
    else:
        batch_count = (total_circuits + max_batch_size - 1) // max_batch_size
        batch_sizes = [max_batch_size] * (batch_count - 1)
        remainder = total_circuits - (batch_count - 1) * max_batch_size
        batch_sizes.append(remainder)
    
    # 결과 반환
    result = {
        "optimal_shots": optimal_shots,
        "total_circuits": total_circuits,
        "total_executions": total_executions,
        "batch_count": batch_count,
        "batch_sizes": batch_sizes,
        "expected_data_quality": "높음" if optimal_shots >= 1024 else "보통" if optimal_shots >= 512 else "낮음"
    }
    
    print("\n📊 배치 실행 계획:")
    print(f"   총 회로 수: {total_circuits:,}")
    print(f"   최적 샷 수/회로: {optimal_shots:,}")
    print(f"   총 실행 수: {total_executions:,}")
    print(f"   배치 수: {batch_count}")
    print(f"   배치 크기: {batch_sizes}")
    print(f"   예상 데이터 품질: {result['expected_data_quality']}")
    
    return result


def run_mega_expressibility_batch(circuit_metadata_list, ibm_backend):
    """
    모든 회로의 표현력 계산을 위한 메가 배치 처리
    모든 회로의 모든 파라미터 샘플에 대한 클래식 쉐도우 회로를 효율적으로 처리
    메모리 사용 최적화를 위해 배치 단위 생성 및 실행
    
    Args:
        circuit_metadata_list (List[Dict]): 모든 회로의 메타데이터 리스트
        ibm_backend (IBMBackendManager): IBM 백엔드 관리 객체
        
    Returns:
        Dict[int, Dict]: 회로 인덱스별 표현력 계산 결과
    """
    from src.calculators.expressibility.ibm import IBMExpressibilityCalculator
    import random
    import gc  # 가비지 컬렉션 명시적 제어용
    
    print(f"\n🚀 메가 배치 표현력 계산 시작 ({len(circuit_metadata_list)}개 회로)")
    
    # 표현력 계산기 초기화
    expressibility_calculator = IBMExpressibilityCalculator()
    
    # 결과 저장용 딕셔너리
    circuit_results = {}
    
    # Shadow 파라미터 설정
    S = config.expressibility.n_samples  # 파라미터 샘플 수
    shadow_size = config.expressibility.shadow_measurements  # Shadow 크기
    
    print(f"📊 설정: {S}개 파라미터 샘플 × {shadow_size}개 쉐도우 측정")
    
    # 각 회로에 대해 모든 쉐도우 회로 생성
    for circuit_idx, circuit_info in enumerate(circuit_metadata_list):
        try:
            base_circuit = circuit_info.get("qiskit_circuit")
            if not base_circuit:
                print(f"⚠️ 회로 {circuit_idx}: qiskit_circuit 없음, 건너뜀")
                circuit_results[circuit_idx] = {
                    "expressibility_value": float('nan'),
                    "method": "skipped_no_base_circuit",
                    "error": "Qiskit circuit not found in metadata"
                }
                continue
                
            n_qubits = base_circuit.num_qubits
            print(f"  🔄 회로 {circuit_idx+1}/{len(circuit_metadata_list)}: {n_qubits}큐빗, {S}개 샘플 처리 중...")
            
            # 각 파라미터 샘플에 대해 쉐도우 회로 생성
            for param_idx in range(S):
                # 쉐도우 회로 생성
                shadow_circuit, bases_used = expressibility_calculator._create_shadow_circuit(
                    base_circuit, n_qubits
                )
                
                # 전체 리스트에 추가
                all_shadow_circuits.append(shadow_circuit)
                circuit_mapping.append((circuit_idx, param_idx, bases_used))
                
        except Exception as e:
            print(f"⚠️ 회로 {circuit_idx} 쉐도우 생성 오류: {str(e)}")
            circuit_results[circuit_idx] = {
                "expressibility_value": float('nan'),
                "method": "failed_shadow_generation",
                "error": str(e)
            }
    
    total_circuits = len(all_shadow_circuits)
    print(f"🎯 총 {total_circuits}개 쉐도우 회로 생성 완료")
    
    if total_circuits == 0:
        print("❌ 실행할 쉐도우 회로가 없습니다")
        return circuit_results
    
    # 메가 배치 실행
    print(f"⚡ 메가 배치 실행 중... ({total_circuits}개 회로)")
    try:
        # 배치 크기 설정
        max_batch_size = config.ibm_backend.max_batch_size
        
        # 전체 회로를 여러 작은 배치로 나누기
        batches = [all_shadow_circuits[i:i + max_batch_size] 
                  for i in range(0, len(all_shadow_circuits), max_batch_size)]
        
        print(f"\n🚀 {total_circuits}개 회로 실행 중 (shots={shadow_size})...")
        print(f"  {len(batches)}개 배치로 나누어 실행 (각 배치 최대 {max_batch_size}개 회로)")
        
        # 각 배치 결과를 저장할 리스트
        batch_results = []
        
        # 배치별로 회로 실행
        for batch_idx, batch_circuits in enumerate(batches):
            batch_start = batch_idx * max_batch_size
            batch_end = min(batch_start + len(batch_circuits), total_circuits)
            
            # 배치 실행 상태 출력
            print(f"  ⏳ 배치 {batch_idx+1}/{len(batches)} 실행 중... (회로 {batch_start+1}-{batch_end}/{total_circuits})")
            
            # 배치 실행
            results = ibm_backend.run_circuits(batch_circuits, shots=shadow_size)
            
            if results and len(results) == len(batch_circuits):
                print(f"  ✅ 배치 {batch_idx+1}/{len(batches)} 완료! ({len(results)}개 결과)")
                # 결과를 전체 결과 리스트에 추가
                batch_results.extend(results)
            else:
                print(f"  ❌ 배치 {batch_idx+1} 실행 실패 또는 결과 수 불일치")
                
        # 전체 결과 확인
        if not batch_results or len(batch_results) != total_circuits:
            print(f"❌ 배치 실행 실패 또는 결과 수 불일치")
            return circuit_results
            
        print(f"✅ 모든 배치 실행 완료, 결과 처리 중... (총 {len(batch_results)}개 회로)")
        
        # 결과를 회로별로 그룹화
        circuit_shadow_data = {}  # circuit_idx -> List[shadow_data]
        
        for result_idx, (circuit_idx, param_idx, bases_used) in enumerate(circuit_mapping):
            if circuit_idx not in circuit_shadow_data:
                circuit_shadow_data[circuit_idx] = []
                
            try:
                # 결과에서 카운트 추출
                result_dict = batch_results[result_idx]
                counts = result_dict.get('counts', {})
                
                if not counts:
                    print(f"⚠️ 결과 {result_idx}: 카운트 없음")
                    continue
                
                # 메타데이터에서 n_qubits 가져오기
                circuit_info = circuit_metadata_list[circuit_idx]
                base_circuit = circuit_info.get("qiskit_circuit")
                n_qubits = base_circuit.num_qubits if base_circuit else 0
                
                # Classical Shadow 데이터로 변환
                shadow_data = expressibility_calculator.convert_ibm_to_classical_shadow(
                    counts, bases_used, n_qubits, shadow_size
                )
                circuit_shadow_data[circuit_idx].append(shadow_data)
                
            except Exception as e:
                print(f"⚠️ 결과 {result_idx} 처리 오류: {str(e)}")
        
        # 각 회로별로 표현력 계산 완료
        print(f"🔮 표현력 값 계산 중...")
        for circuit_idx, shadow_data_list in circuit_shadow_data.items():
            try:
                circuit_info = circuit_metadata_list[circuit_idx]
                base_circuit = circuit_info.get("qiskit_circuit")
                n_qubits = base_circuit.num_qubits if base_circuit else 0
                
                # Shadow 데이터로부터 표현력 계산
                estimated_moments = expressibility_calculator.estimate_pauli_expectations_from_shadows(
                    shadow_data_list, n_qubits
                )
                
                # 거리 계산 (실제 vs 추정)
                distance = expressibility_calculator.calculate_distance_from_haar_random(
                    estimated_moments, n_qubits
                )
                
                # 표현력 값 계산
                expressibility_value = 1.0 - distance
                
                circuit_results[circuit_idx] = {
                    "expressibility_value": expressibility_value,
                    "method": "classical_shadow_mega_batch",
                    "distance_from_haar": distance,
                    "samples_used": len(shadow_data_list),
                    "pauli_moments": estimated_moments
                }
                
            except Exception as e:
                print(f"⚠️ 회로 {circuit_idx} 표현력 계산 오류: {str(e)}")
                circuit_results[circuit_idx] = {
                    "expressibility_value": float('nan'),
                    "method": "failed_expressibility_calculation",
                    "error": str(e)
                }
        
        print(f"✅ 메가 배치 표현력 계산 완료 ({len(circuit_results)}개 회로)")
        
    except Exception as e:
        print(f"❌ 메가 배치 실행 실패: {str(e)}")
        # 모든 회로에 대해 실패 결과 설정
        for circuit_idx in range(len(circuit_metadata_list)):
            if circuit_idx not in circuit_results:
                circuit_results[circuit_idx] = {
                    "expressibility_value": float('nan'),
                    "method": "failed_mega_batch_execution",
                    "error": str(e)
                }
    
    return circuit_results
