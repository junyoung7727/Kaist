#!/usr/bin/env python3
"""
결과 처리 모듈 - 양자 회로 실행 결과 처리 및 분석 로직을 처리합니다.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import gc
from tqdm import tqdm

# 표현력 계산 모듈 임포트
from src.calculators.expressibility.ibm import IBMExpressibilityCalculator

# 현재 스크립트의 상위 디렉토리를 시스템 경로에 추가 (모듈 임포트를 위함)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import config # Ensure config is imported

from data_manager import save_experiment_hdf5


def process_mega_results(result, circuit_metadata, execution_time, ibm_backend):
    """
    메가 잡(Mega job) 결과 처리 - 실제 측정 데이터를 분석합니다.
    
    Args:
        result: IBM 백엔드에서 실행된 결과 객체입니다.
        circuit_metadata (List[Dict]): 각 회로에 대한 메타데이터 목록입니다.
        execution_time (float): 전체 잡 실행에 소요된 시간(초)입니다.
        
    Returns:
        List[Dict]: 각 회로별로 처리된 결과의 목록입니다.
    """
    from src.utils.quantum_utils import (
        calculate_error_rates_mega,
        calculate_robust_fidelity_mega,
        calculate_measurement_statistics
    )
    from src.core.job_runner import run_mega_expressibility_batch
    
    print(f"\n🔬 메가 잡 결과 처리 시작 ({len(circuit_metadata)}개 회로)")
    
    # 메가 배치 표현력 계산 - 모든 회로에 대해 한 번에 실행
    print("🚀 메가 배치 표현력 계산 실행 중...")
    mega_expressibility_results = run_mega_expressibility_batch(circuit_metadata, ibm_backend)
    print(f"✅ 메가 배치 표현력 계산 완료 ({len(mega_expressibility_results)}개 결과)")
    
    all_results = []
    
    # 각 회로별로 결과 처리
    print("📊 회로별 결과 분석 중...")
    
    for circuit_idx, circuit_result in enumerate(tqdm(result, desc="회로 처리")):
        try:
            # 메타데이터 가져오기
            metadata = circuit_metadata[circuit_idx] if circuit_idx < len(circuit_metadata) else {}
            n_qubits = metadata.get('n_qubits', 0)
            
            # 측정 결과 가져오기
            if hasattr(circuit_result, 'data'):
                if hasattr(circuit_result.data(), '__iter__'):
                    counts = circuit_result.data()[0].get("meas", {})
                else:
                    counts = circuit_result.data().get("meas", {})
            elif hasattr(circuit_result, 'get_counts'):
                counts = circuit_result.get_counts()
            else:
                counts = getattr(circuit_result, 'counts', {})
            
            if not counts:
                print(f"⚠️ 회로 {circuit_idx}: 측정 결과 없음")
                continue
            
            # 비트 문자열 길이 정규화
            total_counts = sum(counts.values())
            processed_counts = {}
            
            for bit_str, count in counts.items():
                if len(bit_str) > n_qubits:
                    bit_str = bit_str[:n_qubits]
                elif len(bit_str) < n_qubits:
                    bit_str = bit_str.zfill(n_qubits)
                
                if bit_str in processed_counts:
                    processed_counts[bit_str] += count
                else:
                    processed_counts[bit_str] = count
            
            # 0 상태(zero state) 확률을 기반으로 단순 피델리티를 계산합니다.
            zero_state = '0' * n_qubits
            zero_count = processed_counts.get(zero_state, 0)
            zero_state_probability = zero_count / total_counts if total_counts > 0 else 0
            
            # 다양한 오류율 지표를 계산합니다.
            error_rates = calculate_error_rates_mega(
                processed_counts,
                n_qubits,
                total_counts
            )
            
            # Robust 피델리티를 계산합니다.
            robust_fidelity = calculate_robust_fidelity_mega(
                processed_counts,
                n_qubits,
                total_counts
            )
            
            # 측정 결과에 대한 추가 통계치를 계산합니다.
            measurement_stats = calculate_measurement_statistics(
                processed_counts,
                n_qubits
            )
            
            # 메가 배치에서 미리 계산된 표현력 결과 가져오기
            expressibility_result = mega_expressibility_results.get(circuit_idx, {
                "expressibility_value": float('nan'),
                "method": "not_calculated",
                "error": "Not found in mega batch results"
            })
                
            # 계산된 모든 지표를 포함하는 실행 결과 딕셔너리를 구성합니다.
            execution_result = {
                "zero_state_probability": zero_state_probability,
                "measurement_counts": processed_counts,
                "measured_states": total_counts,
                "error_rates": error_rates,
                "robust_fidelity": robust_fidelity,
                "measurement_statistics": measurement_stats,
                "expressibility": expressibility_result,  # 메가 배치 결과 사용
                "execution_metadata": {
                    "circuit_index": circuit_idx,
                    "execution_time": execution_time,
                    "backend_name": ibm_backend.name,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # 메타데이터와 실행 결과를 결합
            complete_result = {**metadata, **execution_result}
            all_results.append(complete_result)
            
        except Exception as e:
            print(f"⚠️ 회로 {circuit_idx} 처리 중 오류: {str(e)}")
            # 오류 발생 시에도 기본 결과 구조 유지
            error_result = {
                "circuit_index": circuit_idx,
                "error": str(e),
                "zero_state_probability": float('nan'),
                "measurement_counts": {},
                "measured_states": 0,
                "error_rates": {},
                "robust_fidelity": float('nan'),
                "measurement_statistics": {},
                "expressibility": {
                    "expressibility_value": float('nan'),
                    "method": "processing_error",
                    "error": str(e)
                },
                "execution_metadata": {
                    "circuit_index": circuit_idx,
                    "execution_time": execution_time,
                    "backend_name": ibm_backend.name,
                    "timestamp": datetime.now().isoformat(),
                    "processing_error": True
                }
            }
            if circuit_idx < len(circuit_metadata):
                error_result.update(circuit_metadata[circuit_idx])
            all_results.append(error_result)
    
    print(f"✅ 메가 잡 결과 처리 완료 ({len(all_results)}개 결과)")
    return all_results


def save_mega_results(all_results, training_circuits):
    """
    메가 잡(Mega job) 결과를 파일에 저장합니다.
    
    Args:
        all_results (List[Dict]): `process_mega_results` 함수에서 반환된 처리된 결과 목록입니다.
        training_circuits (List[Dict]): 훈련에 사용된 회로 목록입니다 (선택 사항).
        
    Returns:
        Dict: 저장된 파일 경로 등 결과 정보를 담은 딕셔너리입니다.
    """
    print("\n💾 실험 결과 저장 중...")
    
    # 파일명에 사용될 현재 시각 타임스탬프를 생성합니다.
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 결과 파일을 저장할 디렉토리를 확인하고, 없으면 생성합니다.
    results_dir = "experiments/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # JSON 파일에 저장할 전체 데이터 구조를 준비합니다.
    result_data = {
        "experiment_type": "mega_job",
        "timestamp": timestamp,
        "circuit_count": len(all_results),
        "results": all_results
    }
    
    if training_circuits:
        result_data["training_circuits"] = training_circuits
    
    # 모든 결과 데이터를 JSON 형식으로 저장합니다.
    file_prefix = config.get('experiment_file_prefix', 'mega_job')
    json_filename = f"{results_dir}/{file_prefix}_results_{timestamp}.json"
    try:
        with open(json_filename, 'w') as f:
            json.dump(result_data, f, indent=2)
        print(f"   JSON 파일로 저장 완료: {json_filename}")
    except Exception as e:
        print(f"⚠️ JSON 저장 실패: {str(e)}")
    
    # 주요 결과 지표를 요약하여 CSV 파일로 저장합니다.
    try:
        summary_list = []
        for result in all_results:
            execution_result = result.get("execution_result", {})
            circuit_properties = result.get("circuit_properties", {})
            
            # CSV에 저장할 주요 회로 정보를 추출합니다.
            row = {
                "circuit_id": result.get("circuit_id", -1),
                "config_group": result.get("config_group", ""),
                "n_qubits": result.get("n_qubits", 0),
                "depth": result.get("depth", 0),
                "two_qubit_ratio_target": result.get("two_qubit_ratio_target", 0),
                "zero_state_prob": execution_result.get("zero_state_probability", 0),
                "robust_fidelity": execution_result.get("robust_fidelity", 0),
            }
            
            # 표현력 지표 추가
            expressibility = execution_result.get("expressibility", {})
            if isinstance(expressibility, dict):
                # 기본 표현력 점수 및 엔트로피
                row["expressibility_score"] = expressibility.get("expressibility_score", None)
                row["expressibility_entropy"] = expressibility.get("entropy", None)
                
                # 추가적인 거리 기반 표현력 측정 지표들
                distance_metrics = expressibility.get("distance_metrics", {})
                if isinstance(distance_metrics, dict):
                    for metric_name, value in distance_metrics.items():
                        row[f"expressibility_{metric_name}"] = value
            
            summary_list.append(row)
        
        # 추출된 요약 정보를 Pandas DataFrame으로 변환하여 CSV 파일로 저장합니다.
        if summary_list:
            df = pd.DataFrame(summary_list)
            # file_prefix is defined above for the json filename
            csv_filename = f"{results_dir}/{file_prefix}_summary_{timestamp}.csv"
            df.to_csv(csv_filename, index=False)
            print(f"   CSV 요약 저장 완료: {csv_filename}")
        
    except Exception as e:
        print(f"⚠️ CSV 요약 저장 실패: {str(e)}")
    
    # 모든 결과 데이터를 HDF5 형식으로 저장합니다 (대용량 데이터에 적합).
    try:
        # file_prefix is defined above for the json filename
        hdf5_filename = f"{results_dir}/{file_prefix}_data_{timestamp}.h5"
        save_experiment_hdf5(all_results, hdf5_filename)
        print(f"   HDF5 데이터 저장 완료: {hdf5_filename}")
    except Exception as e:
        print(f"⚠️ HDF5 저장 실패: {str(e)}")
    
    return {
        "timestamp": timestamp,
        "json_file": json_filename,
        "csv_file": f"{results_dir}/mega_job_summary_{timestamp}.csv",
        "hdf5_file": f"{results_dir}/mega_job_data_{timestamp}.h5"
    }


def analyze_two_qubit_ratio_results(all_results):
    """
    2큐빗 게이트 비율에 따른 실험 결과를 분석합니다.
    
    Args:
        all_results (List[Dict]): `process_mega_results` 함수에서 반환된 처리된 결과 목록입니다.
        
    Returns:
        pd.DataFrame: 분석 결과를 담은 Pandas DataFrame입니다.
    """
    print("\n📊 2큐빗 게이트 비율별 결과 분석 중...")
    
    # DataFrame 생성을 위해 분석에 필요한 데이터를 결과 목록에서 추출합니다.
    analysis_data = []
    
    for result in all_results:
        try:
            execution_result = result.get("execution_result", {})
            
            # 핵심 메트릭(지표)을 추출합니다.
            row = {
                "circuit_id": result.get("circuit_id", -1),
                "config_group": result.get("config_group", ""),
                "n_qubits": result.get("n_qubits", 0),
                "depth": result.get("depth", 0),
                "two_qubit_ratio_target": result.get("two_qubit_ratio_target", 0),
                "zero_state_prob": execution_result.get("zero_state_probability", 0),
                "robust_fidelity": execution_result.get("robust_fidelity", 0),
            }
            
            # 표현력 지표 추가
            expressibility = execution_result.get("expressibility", {})
            if isinstance(expressibility, dict):
                # 기본 표현력 점수 및 엔트로피
                row["expressibility_score"] = expressibility.get("expressibility_score", None)
                row["expressibility_entropy"] = expressibility.get("entropy", None)
                
                # 추가적인 거리 기반 표현력 측정 지표들
                distance_metrics = expressibility.get("distance_metrics", {})
                if isinstance(distance_metrics, dict):
                    for metric_name, value in distance_metrics.items():
                        row[f"expressibility_{metric_name}"] = value
            
            analysis_data.append(row)
            
        except Exception as e:
            continue
    
    if not analysis_data:
        print("⚠️ 분석할 데이터가 없습니다.")
        return None
    
    # 추출된 분석용 데이터로 Pandas DataFrame을 생성합니다.
    df = pd.DataFrame(analysis_data)
    
    # 큐빗 수, 회로 깊이, 2큐빗 게이트 비율별로 그룹화하여 통계를 계산합니다.
    print("\n📈 2큐빗 게이트 비율별 피델리티 및 표현력 분석:")
    
    try:
        # 그룹별 통계 (추가 표현력 메트릭 포함)
        metric_columns = ['zero_state_prob', 'robust_fidelity', 'expressibility_score', 'expressibility_entropy']
        
        # DataFrame에 동적으로 추가된 표현력 메트릭 컬럼들을 찾습니다.
        distance_metrics_columns = [col for col in df.columns if col.startswith('expressibility_') 
                                   and col not in ['expressibility_score', 'expressibility_entropy']]
        if distance_metrics_columns:
            metric_columns.extend(distance_metrics_columns)
        
        # 각 메트릭별로 어떤 통계 함수(평균, 표준편차 등)를 적용할지 정의합니다.
        agg_dict = {}
        for col in metric_columns:
            if col == 'zero_state_prob':
                agg_dict[col] = ['mean', 'std', 'count']
            else:
                agg_dict[col] = ['mean', 'std']
        
        grouped = df.groupby(['n_qubits', 'depth', 'two_qubit_ratio_target']).agg(agg_dict)
        
        # Pandas DataFrame 출력 형식을 설정합니다.
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 120)
        pd.set_option('display.precision', 4)
        
        print("\n🔍 그룹별 성능 통계:")
        print(grouped)
        
        # 보고서 파일명에 사용될 현재 시각 타임스탬프를 생성합니다.
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 분석 보고서를 저장할 디렉토리를 설정합니다.
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        
        # 분석 결과를 CSV 파일로 저장합니다.
        csv_filename = f"{report_dir}/two_qubit_ratio_analysis_{timestamp}.csv"
        grouped.to_csv(csv_filename)
        print(f"\n✅ 분석 결과 저장 완료: {csv_filename}")
        
    except Exception as e:
        print(f"⚠️ 분석 중 오류 발생: {str(e)}")
    
    return df
