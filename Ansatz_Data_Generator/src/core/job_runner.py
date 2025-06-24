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
from src.core.circuit_operations import create_inverse_circuit, calculate_fidelity_from_counts


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
        tuple: (결과 리스트, 실행 시간(초), 회로 메타데이터)
    """
    # 샷 수 처리: config에서 가져오고, 제공되지 않으면 기본값 사용
    if shots is None:
        shots = config.ibm_backend.default_shots
    
    if not qiskit_circuits:
        print("⚠️ 실행할 회로가 없습니다.")
        return None, 0, circuit_metadata
    
    # 회로별 샷 수 결정
    if circuit_shot_requirements and len(circuit_shot_requirements) == len(qiskit_circuits):
        total_shots = sum(circuit_shot_requirements)
        print(f"\n🚀 IBM 백엔드에서 {len(qiskit_circuits)}개 회로를 한 번의 배치 job으로 실행 시작")
        print(f"   회로별 개별 샷 수: Config 설정에 따라 다름")
        print(f"   배치 총 실행 수: {total_shots:,}")
        # 평균 샷 수 사용 (IBM API는 모든 회로에 동일한 샷 수만 지원)
        avg_shots = int(sum(circuit_shot_requirements) / len(circuit_shot_requirements))
    else:
        total_shots = len(qiskit_circuits) * shots
        print(f"\n🚀 IBM 백엔드에서 {len(qiskit_circuits)}개 회로를 한 번의 배치 job으로 실행 시작")
        print(f"   회로당 고정 샷 수: {shots:,}")
        print(f"   배치 총 실행 수: {total_shots:,}")
        avg_shots = shots
    
    print(f"   예상 데이터 품질: {'🟢 높음' if total_shots/len(qiskit_circuits) >= 1024 else '🟡 보통' if total_shots/len(qiskit_circuits) >= 512 else '🔴 낮음'}")
    
    start_time = time.time()
    
    try:
        # 백엔드 관리자의 run_circuits 메소드 사용 (이미 올바른 파싱 로직 포함)
        print("🚀 IBM 백엔드에서 배치 job 제출 중...")
        print(f"   {len(qiskit_circuits)}개 회로를 {avg_shots} 샷으로 실행 중...")
        print(f"   백엔드: {ibm_backend.name}")
        
        # 백엔드 관리자의 run_circuits 메소드 사용
        results = ibm_backend.run_circuits(qiskit_circuits, shots=avg_shots)
        
        if results is None:
            print("❌ 배치 실행 실패")
            execution_time = time.time() - start_time
            return None, execution_time, circuit_metadata
        
        print("✅ 배치 실행 완료!")
        execution_time = time.time() - start_time
        print(f"   실행 시간: {execution_time:.2f}초")
        
        # 메모리 정리
        gc.collect()
        
        return results, execution_time, circuit_metadata
        
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
    메모리 효율적인 배치 단위 처리로 성능 최적화
    """
    from src.calculators.expressibility.ibm import IBMExpressibilityCalculator
    import gc
    
    print(f"\n🚀 메가 배치 표현력 계산 시작 ({len(circuit_metadata_list)}개 회로)")
    
    # 초기화
    expressibility_calculator = IBMExpressibilityCalculator()
    circuit_results = {}
    
    # 설정값
    S = config.expressibility.n_samples
    shadow_size = config.expressibility.shadow_measurements
    max_batch_size = config.ibm_backend.max_batch_size
    
    # 배치 크기 계산
    meta_per_batch = max(1, max_batch_size // S)
    meta_batches = [circuit_metadata_list[i:i + meta_per_batch] 
                   for i in range(0, len(circuit_metadata_list), meta_per_batch)]
    
    print(f"📊 설정: {S}개 샘플 × {shadow_size}개 측정, {len(meta_batches)}개 배치")
    
    # 배치별 처리
    for batch_idx, meta_batch in enumerate(meta_batches):
        print(f"\n🔄 배치 {batch_idx+1}/{len(meta_batches)} 처리 중...")
        
        try:
            # 1. 쉐도우 회로 생성
            shadow_circuits, circuit_mapping = _generate_shadow_circuits(
                meta_batch, batch_idx * meta_per_batch, expressibility_calculator, S, circuit_results
            )
            
            if not shadow_circuits:
                print("  ⏩ 생성된 회로 없음, 다음 배치로")
                continue
            
            # 2. 회로 실행
            batch_results = _execute_circuits_batch(
                shadow_circuits, ibm_backend, shadow_size, max_batch_size, batch_idx
            )
            
            if not batch_results:
                print("  ❌ 실행 실패, 다음 배치로")
                continue
            
            # 3. 결과 처리 및 표현력 계산
            _process_results_and_calculate_expressibility(
                batch_results, circuit_mapping, expressibility_calculator, 
                circuit_metadata_list, shadow_size, circuit_results
            )
            
            # 4. 메모리 정리
            del shadow_circuits, circuit_mapping, batch_results
            gc.collect()
            
            print(f"✅ 배치 {batch_idx+1} 완료")
            
        except Exception as e:
            print(f"❌ 배치 {batch_idx+1} 처리 오류: {str(e)}")
    
    print(f"✅ 메가 배치 완료 ({len(circuit_results)}개 회로)")
    return circuit_results


def _generate_shadow_circuits(meta_batch, batch_start_idx, expressibility_calculator, S, circuit_results):
    """쉐도우 회로 생성"""
    shadow_circuits = []
    circuit_mapping = []
    
    for batch_offset, circuit_info in enumerate(meta_batch):
        circuit_idx = batch_start_idx + batch_offset
        
        try:
            base_circuit = circuit_info.get("qiskit_circuit")
            if not base_circuit:
                circuit_results[circuit_idx] = {
                    "expressibility_value": float('nan'),
                    "method": "skipped_no_base_circuit",
                    "error": "Qiskit circuit not found"
                }
                continue
            
            n_qubits = base_circuit.num_qubits
            print(f"  📝 회로 {circuit_idx}: {n_qubits}큐빗, {S}개 샘플")
            
            # 각 파라미터 샘플에 대해 쉐도우 회로 생성
            for param_idx in range(S):
                shadow_circuit, bases_used = expressibility_calculator._create_shadow_circuit(
                    base_circuit, n_qubits
                )
                shadow_circuits.append(shadow_circuit)
                circuit_mapping.append((circuit_idx, param_idx, bases_used, n_qubits))
                
        except Exception as e:
            print(f"⚠️ 회로 {circuit_idx} 생성 오류: {str(e)}")
            circuit_results[circuit_idx] = {
                "expressibility_value": float('nan'),
                "method": "failed_shadow_generation",
                "error": str(e)
            }
    
    print(f"  🎯 {len(shadow_circuits)}개 쉐도우 회로 생성 완료")
    return shadow_circuits, circuit_mapping


def _execute_circuits_batch(shadow_circuits, ibm_backend, shadow_size, max_batch_size, batch_idx):
    """회로 배치 실행"""
    execute_batches = [shadow_circuits[i:i + max_batch_size] 
                      for i in range(0, len(shadow_circuits), max_batch_size)]
    
    print(f"  🚀 {len(shadow_circuits)}개 회로를 {len(execute_batches)}개 실행 배치로 분할")
    
    all_results = []
    for exec_idx, batch_circuits in enumerate(execute_batches):
        print(f"    ⏳ 실행 배치 {exec_idx+1}/{len(execute_batches)}")
        
        results = ibm_backend.run_circuits(batch_circuits, shots=shadow_size)
        
        if results and len(results) == len(batch_circuits):
            print(f"    ✅ 실행 배치 {exec_idx+1} 완료 ({len(results)}개)")
            all_results.extend(results)
        else:
            print(f"    ❌ 실행 배치 {exec_idx+1} 실패")
            return None
    
    return all_results


def _process_results_and_calculate_expressibility(batch_results, circuit_mapping, expressibility_calculator, 
                                                circuit_metadata_list, shadow_size, circuit_results):
    """결과 처리 및 표현력 계산"""
    import json
    import os
    from datetime import datetime
    
    # 디버깅용 데이터 저장을 위한 딕셔너리
    debug_data = {
        "timestamp": datetime.now().isoformat(),
        "batch_info": {
            "total_results": len(batch_results),
            "circuit_mapping_count": len(circuit_mapping),
            "shadow_size": shadow_size
        },
        "circuits": {}
    }
    
    # 회로별 쉐도우 데이터 수집
    circuit_shadow_data = {}
    
    for i, result in enumerate(batch_results):
        if i >= len(circuit_mapping):
            continue
            
        circuit_idx, param_idx, bases_used, n_qubits = circuit_mapping[i]
        
        if not isinstance(result, dict) or "counts" not in result:
            print(f"    ⚠️ 회로 {circuit_idx} 결과 오류")
            continue
        
        if circuit_idx not in circuit_shadow_data:
            circuit_shadow_data[circuit_idx] = []
        
        # classical shadow 변환
        counts = result["counts"]
        shadow_data = expressibility_calculator.convert_ibm_to_classical_shadow(
            counts, bases_used, n_qubits, shadow_size
        )
        circuit_shadow_data[circuit_idx].extend(shadow_data)
    
    # 각 회로의 표현력 계산
    print(f"  🔮 표현력 계산 중...")
    for circuit_idx, shadow_data_list in circuit_shadow_data.items():
        try:
            circuit_info = circuit_metadata_list[circuit_idx]
            base_circuit = circuit_info.get("qiskit_circuit")
            n_qubits = base_circuit.num_qubits if base_circuit else 0
            
            # 디버깅: shadow 데이터 구조 확인
            print(f"  🔍 회로 {circuit_idx}: {len(shadow_data_list)}개 shadow 샘플")
            if len(shadow_data_list) > 0:
                print(f"    첫 번째 샘플 타입: {type(shadow_data_list[0])}")
                if isinstance(shadow_data_list[0], dict):
                    print(f"    첫 번째 샘플 키: {list(shadow_data_list[0].keys())}")
            
            # 표현력 계산
            estimated_moments = expressibility_calculator.estimate_pauli_expectations_from_shadows(
                shadow_data_list, n_qubits
            )
            
            # 디버깅 데이터에 최종 결과도 추가
            debug_data["circuits"][f"circuit_{circuit_idx}"] = {
                "n_qubits": n_qubits,
                "shadow_samples": len(shadow_data_list),
                "estimated_moments": estimated_moments.tolist() if hasattr(estimated_moments, 'tolist') else estimated_moments,
                "estimated_moments_shape": str(estimated_moments.shape) if hasattr(estimated_moments, 'shape') else str(type(estimated_moments)),
                "estimated_moments_stats": {
                    "mean": float(estimated_moments.mean()) if hasattr(estimated_moments, 'mean') else None,
                    "std": float(estimated_moments.std()) if hasattr(estimated_moments, 'std') else None,
                    "min": float(estimated_moments.min()) if hasattr(estimated_moments, 'min') else None,
                    "max": float(estimated_moments.max()) if hasattr(estimated_moments, 'max') else None
                }
            }
            
            distance = expressibility_calculator.calculate_distance_from_haar_random(
                estimated_moments, n_qubits, config.expressibility.distance_metric
            )
            
            expressibility_value = 1.0 - distance
            
            # 디버깅 데이터에 최종 결과도 추가
            debug_data["circuits"][f"circuit_{circuit_idx}"]["distance_from_haar"] = float(distance)
            debug_data["circuits"][f"circuit_{circuit_idx}"]["expressibility_value"] = float(expressibility_value)
            
            circuit_results[circuit_idx] = {
                "expressibility_value": expressibility_value,
                "method": "classical_shadow_mega_batch",
                "distance_from_haar": distance,
                "samples_used": len(shadow_data_list)
            }
            
        except Exception as e:
            print(f"    ⚠️ 회로 {circuit_idx} 계산 오류: {str(e)}")
            
            # 디버깅 데이터에 오류 정보도 저장
            debug_data["circuits"][f"circuit_{circuit_idx}"] = {
                "error": str(e),
                "n_qubits": n_qubits if 'n_qubits' in locals() else 0,
                "shadow_samples": len(shadow_data_list) if 'shadow_data_list' in locals() else 0
            }
            
            circuit_results[circuit_idx] = {
                "expressibility_value": float('nan'),
                "method": "calculation_error",
                "error": str(e)
            }
    
    # 디버깅 데이터를 JSON 파일로 저장
    try:
        debug_dir = "experiments/debug"
        os.makedirs(debug_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_file = f"{debug_dir}/pauli_expectations_debug_{timestamp}.json"
        
        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, indent=2, ensure_ascii=False)
        
        print(f"  🐛 디버깅 데이터 저장: {debug_file}")
        
    except Exception as e:
        print(f"  ⚠️ 디버깅 데이터 저장 실패: {str(e)}")


def run_analysis_job(circuit_metadata_list, ibm_backend):
    """
    양자 회로 분석 작업을 실행합니다. 두 단계로 진행됩니다:
    1. 피델리티 계산 (역회로 U†U를 실행하여 |0...0> 측정 확률 확인)
    2. 표현력 계산 (파라미터 샘플별 회로 실행 및 classical shadow 분석)
    
    Args:
        circuit_metadata_list (list): 회로 메타데이터 목록
        ibm_backend (IBMBackendManager): IBM 백엔드 관리 객체
        
    Returns:
        dict: 분석 결과 (회로별 피델리티 및 표현력 포함)
    """
    print(f"\n🚀 양자 회로 분석 작업 시작 ({len(circuit_metadata_list)}개 회로)")
    
    # 결과 저장 딕셔너리
    analysis_results = {}
    
    # ===== 1단계: 피델리티 계산 =====
    fidelity_results = run_fidelity_batch(circuit_metadata_list, ibm_backend)
    
    # 분석 결과에 피델리티 정보 추가
    for circuit_idx, fidelity_data in fidelity_results.items():
        analysis_results[circuit_idx] = {
            "fidelity": fidelity_data.get("fidelity_value", 0.0),
            "fidelity_method": fidelity_data.get("method", ""),
            "circuit_info": circuit_metadata_list[circuit_idx] if circuit_idx < len(circuit_metadata_list) else None
        }
    
    # ===== 2단계: 표현력 계산 ====
    # 피델리티가 임계값 이상인 회로만 필터링
    filtered_circuits = []
    filtered_indices = []
    
    for idx, circuit_info in enumerate(circuit_metadata_list):
        filtered_circuits.append(circuit_info)
        filtered_indices.append(idx)
    
    print(f"\n🔍 피델리티 필터링: {len(filtered_circuits)}/{len(circuit_metadata_list)} 회로 선택")
    
    # 필터링된 회로에 대해 표현력 계산
    if filtered_circuits:
        expressibility_results = run_mega_expressibility_batch(filtered_circuits, ibm_backend)
        
        # 원래 인덱스로 표현력 결과 매핑
        for result_idx, (circuit_idx, expr_data) in enumerate(expressibility_results.items()):
            if result_idx < len(filtered_indices):
                original_idx = filtered_indices[result_idx]
                if original_idx in analysis_results:
                    analysis_results[original_idx]["expressibility"] = expr_data.get("expressibility_value", 0.0)
                    analysis_results[original_idx]["expressibility_method"] = expr_data.get("method", "")
                    analysis_results[original_idx]["distance_from_haar"] = expr_data.get("distance_from_haar", 1.0)
    
    print(f"\n✅ 양자 회로 분석 작업 완료: {len(analysis_results)}개 회로 처리됨")
    return analysis_results


def run_fidelity_batch(circuit_metadata_list, ibm_backend):
    """
    모든 회로에 대한 피델리티 계산 배치 실행
    각 회로에 대해 역회로(U†U)를 생성하고 실행하여 |0...0> 상태의 확률 측정
    
    Args:
        circuit_metadata_list (list): 회로 메타데이터 목록
        ibm_backend (IBMBackendManager): IBM 백엔드 관리 객체
        
    Returns:
        dict: 피델리티 결과
    """
    print(f"\n🔄 피델리티 계산 시작 ({len(circuit_metadata_list)}개 회로)")
    
    # 결과 저장 딕셔너리
    fidelity_results = {}
    
    # 역회로 생성
    fidelity_circuits = []
    circuit_indices = []
    
    for idx, circuit_info in enumerate(circuit_metadata_list):
        try:
            # 역회로 (U†U) 생성
            inverse_circuit = create_inverse_circuit(circuit_info)
            
            # 메타데이터 추가 (피델리티 회로 표시)
            inverse_circuit.metadata = {
                "original_circuit_idx": idx,
                "circuit_type": "fidelity_check",
                "n_qubits": inverse_circuit.num_qubits
            }
            
            fidelity_circuits.append(inverse_circuit)
            circuit_indices.append(idx)
            
        except Exception as e:
            print(f"⚠️ 회로 {idx} 역회로 생성 오류: {str(e)}")
            fidelity_results[idx] = {
                "fidelity_value": float('nan'),
                "method": "failed_inverse_generation",
                "error": str(e)
            }
    
    # 역회로 배치 실행
    if fidelity_circuits:
        print(f"🚀 {len(fidelity_circuits)}개 역회로 실행...")
        # 피델리티 측정엔 더 많은 샷 수 필요
        fidelity_shots = config.ibm_backend.default_shots
        
        result, exec_time, _ = run_mega_job(
            fidelity_circuits, 
            [circuit.metadata for circuit in fidelity_circuits], 
            ibm_backend, 
            shots=fidelity_shots
        )
        
        # 결과 처리
        if result:
            print(f"✅ 역회로 실행 완료 ({exec_time:.2f}초)")
            
            try:
                # 결과는 이제 list of dictionaries 형태 (각 dict에 'counts' 키 포함)
                for i, result_dict in enumerate(result):
                    if i >= len(circuit_indices):
                        continue
                        
                    try:
                        # 측정 결과에서 counts 추출
                        counts = result_dict['counts']
                        n_qubits = fidelity_circuits[i].num_qubits
                        
                        # 피델리티 계산 (|0...0> 상태의 확률)
                        zero_state = '0' * n_qubits
                        prob_zero = counts.get(zero_state, 0) / sum(counts.values()) if counts else 0
                        
                        original_idx = circuit_indices[i]
                        circuit_metadata_list[original_idx]['fidelity'] = prob_zero
                        
                        # fidelity_results 딕셔너리에도 결과 저장
                        fidelity_results[original_idx] = {
                            "fidelity_value": prob_zero,
                            "method": "inverse_circuit_zero_state",
                            "shots": fidelity_shots,
                            "n_qubits": n_qubits
                        }
                        
                        print(f"   회로 {original_idx+1}: 피델리티 = {prob_zero:.4f}")
                        
                    except Exception as e:
                        print(f"   ⚠️ 회로 {i+1} 피델리티 계산 오류: {str(e)}")
                        original_idx = circuit_indices[i]
                        circuit_metadata_list[original_idx]['fidelity'] = 0.0
                        
                        # fidelity_results 딕셔너리에도 오류 상태 저장
                        fidelity_results[original_idx] = {
                            "fidelity_value": 0.0,
                            "method": "calculation_error",
                            "error": str(e)
                        }
            except Exception as e:
                print(f"❌ 결과 처리 중 오류 발생: {str(e)}")
                # 모든 회로에 대해 오류 표시
                for idx in circuit_indices:
                    if idx not in fidelity_results:
                        fidelity_results[idx] = {
                            "fidelity_value": float('nan'),
                            "method": "result_processing_failure",
                            "error": str(e)
                        }
        else:
            print("❌ 역회로 실행 실패")
    
    print(f"✅ 피델리티 계산 완료 ({len(fidelity_results)}개 회로)")
    return fidelity_results
