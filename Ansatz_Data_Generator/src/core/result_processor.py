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


def make_json_serializable(obj):
    """
    재귀적으로 객체를 JSON 직렬화 가능한 형태로 변환합니다.
    QuantumCircuit 등의 객체는 문자열 표현으로 변환됩니다.
    """
    if hasattr(obj, '__dict__'):
        # QuantumCircuit 등의 복잡한 객체
        if hasattr(obj, 'name') and hasattr(obj, 'num_qubits'):
            return {
                "type": "QuantumCircuit",
                "name": getattr(obj, 'name', 'unnamed'),
                "num_qubits": getattr(obj, 'num_qubits', 0),
                "depth": getattr(obj, 'depth', lambda: 0)(),
                "size": getattr(obj, 'size', lambda: 0)()
            }
        else:
            # 다른 객체들은 딕셔너리로 변환
            try:
                return {k: make_json_serializable(v) for k, v in obj.__dict__.items() 
                       if not k.startswith('_')}
            except:
                return str(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    elif hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    else:
        # 직렬화할 수 없는 객체는 문자열로 변환
        return str(obj)


def process_mega_results(analysis_results, circuit_metadata, execution_time, ibm_backend):
    """
    메가 잡(Mega job) 분석 결과 처리 - 피델리티 및 표현력 분석 결과를 처리합니다.
    
    Args:
        analysis_results (Dict): run_analysis_job에서 반환된 분석 결과
            (회로별 피델리티와 표현력 값이 포함됨)
        circuit_metadata (List[Dict]): 각 회로에 대한 메타데이터 목록입니다.
        execution_time (float): 전체 작업 실행에 소요된 시간(초)입니다.
        ibm_backend: IBM 백엔드 관리 객체
        
    Returns:
        List[Dict]: 각 회로별로 처리된 결과의 목록입니다.
    """
    from src.utils.quantum_utils import (
        calculate_error_rates_mega,
        calculate_robust_fidelity_mega,
        calculate_measurement_statistics
    )
    
    print(f"\n🔬 메가 잡 분석 결과 처리 시작 ({len(circuit_metadata)}개 회로)")
    
    # 피델리티 및 표현력 계산 결과는 이미 analysis_results에 포함됨
    print(f"📊 분석 결과 포맷팅 중... ({len(analysis_results)}개 회로 데이터)")
    
    all_results = []
    
    # 회로 인덱스별로 결과 처리
    print("📊 회로별 결과 통합 중...")
    
    # 모든 회로 메타데이터를 순회
    for circuit_idx, metadata in enumerate(tqdm(circuit_metadata, desc="회로 처리")):
        try:
            # 회로가 분석 결과에 없으면 건너뜀
            if circuit_idx not in analysis_results:
                print(f"⚠️ 회로 {circuit_idx}: 분석 결과 없음")
                continue
            
            # 회로 분석 결과 가져오기
            circuit_analysis = analysis_results[circuit_idx]
            
            # 메타데이터 기본 정보 추출
            n_qubits = metadata.get('n_qubits', 0)
            depth = metadata.get('depth', 0)
            circuit_name = metadata.get('name', f"circuit_{circuit_idx}")
            gate_counts = metadata.get('gate_counts', {})
            circuit_type = metadata.get('circuit_type', 'unknown')
            
            # 피델리티 정보 추출
            fidelity = circuit_analysis.get('fidelity', 0.0)
            fidelity_method = circuit_analysis.get('fidelity_method', 'not_available')
            
            # 표현력 정보 추출 (있을 경우)
            expressibility = circuit_analysis.get('expressibility', 0.0)
            expressibility_method = circuit_analysis.get('expressibility_method', 'not_available')
            distance_from_haar = circuit_analysis.get('distance_from_haar', 1.0)
            
            # 측정 통계 및 오류율 계산에 필요한 기본 값들
            total_counts = 0
            processed_counts = {}
            zero_state_probability = 0.0
            
            # 피델리티 값을 통해 모델링된 피델리티 지표 계산
            error_rates = {
                "gate_error_rate": 1.0 - fidelity if isinstance(fidelity, (int, float)) else 1.0,
                "circuit_error_probability": 1.0 - fidelity if isinstance(fidelity, (int, float)) else 1.0
            }
            
            # 강화 피델리티 - 업데이트된 시스템에서는 직접 측정된 값 사용
            robust_fidelity = fidelity
            
            # 측정 통계 - 레거시 호환성을 위해 임의의 값 사용
            measurement_statistics = {
                "entropy": 0.0,
                "unique_states": 1
            }
            
            # 표현력 정보 구성
            circuit_expressibility = {
                "value": expressibility if isinstance(expressibility, (int, float)) else 0.0,
                "method": expressibility_method,
                "distance_from_haar": distance_from_haar
            }
            
            # 피델리티 계산 및 추가 분석
            output_result = {
                "circuit_index": circuit_idx,
                "gate_metrics": error_rates,
                "fidelity": {
                    "simple": fidelity,  # 이제 직접 피델리티 값 사용
                    "robust": robust_fidelity,
                    "method": fidelity_method
                },
                "expressibility": circuit_expressibility,
                "measurement_statistics": measurement_statistics,
                "execution_metadata": {
                    "circuit_index": circuit_idx,
                    "execution_time": execution_time,
                    "backend_name": ibm_backend.name,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # 메타데이터에서 추가 정보 추출
            if circuit_idx < len(circuit_metadata):
                circuit_meta = circuit_metadata[circuit_idx]
                output_result.update(circuit_meta)
                
                # 특별히 중요한 정보는 별도로 그룹화
                output_result["additional_metrics"] = {
                    "depth": depth,
                    "width": n_qubits
                }
            
            # 최종 결과에 추가
            all_results.append(output_result)
            
        except Exception as e:
            print(f"⚠️ 회로 {circuit_idx} 처리 오류: {str(e)}")
            error_result = {
                "circuit_index": circuit_idx,
                "fidelity": {
                    "simple": float('nan'),
                    "robust": float('nan'),
                    "method": "processing_error"
                },
                "expressibility": {
                    "value": float('nan'),
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
    
    # 빈 결과 체크
    if not all_results:
        print("⚠️ 저장할 결과가 없습니다.")
        return {"error": "No results to save"}
    
    # 파일명에 사용될 현재 시각 타임스탬프를 생성합니다.
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 결과 파일을 저장할 디렉토리를 확인하고, 없으면 생성합니다.
    results_dir = "experiments/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 결과 데이터 구조를 준비합니다.
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
        # JSON 직렬화 가능한 형태로 변환
        serializable_data = make_json_serializable(result_data)
        with open(json_filename, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        print(f"   JSON 파일로 저장 완료: {json_filename}")
    except Exception as e:
        print(f"⚠️ JSON 저장 실패: {str(e)}")
    
    # 주요 결과 지표를 요약하여 CSV 파일로 저장합니다.
    try:
        summary_list = []
        for result in all_results:
            # 안전한 값 추출 함수
            def safe_get(obj, key, default=None):
                if isinstance(obj, dict):
                    return obj.get(key, default)
                elif hasattr(obj, key):
                    return getattr(obj, key, default)
                else:
                    return default
            
            # 새로운 결과 구조에서 정보 추출
            fidelity = result.get("fidelity")
            expressibility = result.get("expressibility")
            additional_metrics = result.get("additional_metrics", {})
            
            # CSV에 저장할 주요 회로 정보를 추출합니다.
            row = {
                "circuit_index": result.get("circuit_index", -1),
                "config_group": result.get("config_group", ""),
                "n_qubits": result.get("n_qubits", 0),
                "depth": result.get("depth", 0) or safe_get(additional_metrics, "depth", 0),
                "two_qubit_ratio_target": result.get("two_qubit_ratio_target", 0),
            }
            
            # 피델리티 값 안전하게 추출
            if isinstance(fidelity, dict):
                row["zero_state_prob"] = fidelity.get("simple", 0)
                row["robust_fidelity"] = fidelity.get("robust", 0)
                row["fidelity_method"] = fidelity.get("method", "unknown")
            elif isinstance(fidelity, (int, float)):
                row["zero_state_prob"] = fidelity
                row["robust_fidelity"] = fidelity
                row["fidelity_method"] = "simple_value"
            else:
                row["zero_state_prob"] = 0
                row["robust_fidelity"] = 0
                row["fidelity_method"] = "unknown"
            
            # 표현력 지표 추가
            if isinstance(expressibility, dict):
                # 기본 표현력 값
                row["expressibility_score"] = expressibility.get("value", None)
                row["expressibility_method"] = expressibility.get("method", "unknown")
                row["distance_from_haar"] = expressibility.get("distance_from_haar", 1.0)
                
                # 추가적인 표현력 측정 지표들이 있다면 추가
                for metric_name, value in expressibility.items():
                    if metric_name not in ["value", "method", "distance_from_haar"]:
                        if isinstance(value, (int, float, str, bool)):
                            row[f"expressibility_{metric_name}"] = value
            elif isinstance(expressibility, (int, float)):
                row["expressibility_score"] = expressibility
                row["expressibility_method"] = "simple_value"
                row["distance_from_haar"] = 1.0
            else:
                row["expressibility_score"] = None
                row["expressibility_method"] = "unknown"
                row["distance_from_haar"] = 1.0
            
            summary_list.append(row)
        
        # 추출된 요약 정보를 Pandas DataFrame으로 변환하여 CSV 파일로 저장합니다.
        if summary_list:
            df = pd.DataFrame(summary_list)
            csv_filename = f"{results_dir}/{file_prefix}_summary_{timestamp}.csv"
            df.to_csv(csv_filename, index=False)
            print(f"   CSV 요약 저장 완료: {csv_filename}")
        
    except Exception as e:
        print(f"⚠️ CSV 요약 저장 실패: {str(e)}")
        print(f"   오류 세부사항: {type(e).__name__}")
    
    # 트랜스포머 친화적 형식으로 저장
    try:
        transformer_data = create_transformer_friendly_format(all_results)
        transformer_filename = f"{results_dir}/{file_prefix}_transformer_{timestamp}.json"
        with open(transformer_filename, 'w') as f:
            json.dump(transformer_data, f, indent=2)
        print(f"   트랜스포머 형식 저장 완료: {transformer_filename}")
    except Exception as e:
        print(f"⚠️ 트랜스포머 형식 저장 실패: {str(e)}")
    
    # HDF5 저장 (간소화된 버전)
    try:
        if summary_list:  # CSV 데이터가 성공적으로 생성된 경우만
            df = pd.DataFrame(summary_list)
            hdf5_filename = f"{results_dir}/{file_prefix}_data_{timestamp}.h5"
            df.to_hdf(hdf5_filename, key='circuit_data', mode='w')
            print(f"   HDF5 데이터 저장 완료: {hdf5_filename}")
    except Exception as e:
        print(f"⚠️ HDF5 저장 실패: {str(e)}")
    
    return {
        "timestamp": timestamp,
        "json_file": json_filename,
        "csv_file": f"{results_dir}/{file_prefix}_summary_{timestamp}.csv",
        "hdf5_file": f"{results_dir}/{file_prefix}_data_{timestamp}.h5",
        "transformer_file": f"{results_dir}/{file_prefix}_transformer_{timestamp}.json"
    }


def create_transformer_friendly_format(all_results):
    """
    트랜스포머 모델 훈련에 적합한 형식으로 데이터 변환
    """
    transformer_data = []
    
    for result in all_results:
        try:
            # 기본 회로 메타데이터
            circuit_record = {
                "circuit_id": result.get("circuit_index", -1),
                "n_qubits": result.get("n_qubits", 4),
                "depth": result.get("depth", 1),
                "two_qubit_ratio": result.get("two_qubit_ratio_target", 0.0),
            }
            
            # 회로 토큰화 (게이트 시퀀스) - 최상위 레벨에서 추출
            gates = result.get("gates", [])
            wires_list = result.get("wires_list", [])
            params = result.get("params", [])
            
            # 간단한 토큰화: gate_type:qubit_indices 형식
            circuit_tokens = []
            for i, gate in enumerate(gates):
                if i < len(wires_list):
                    wires = wires_list[i]
                    if isinstance(wires, list):
                        wire_str = "_".join(map(str, wires))
                    else:
                        wire_str = str(wires)
                    token = f"{gate}:{wire_str}"
                    circuit_tokens.append(token)
            
            circuit_record["circuit_tokens"] = circuit_tokens
            circuit_record["circuit_sequence"] = " ".join(circuit_tokens)
            
            # 타겟 레이블 (회귀 예측용)
            fidelity = result.get("fidelity")
            expressibility = result.get("expressibility")
            
            # 피델리티 값 추출
            if isinstance(fidelity, dict):
                circuit_record["target_fidelity"] = fidelity.get("simple", 0.0)
            elif isinstance(fidelity, (int, float)):
                circuit_record["target_fidelity"] = float(fidelity)
            else:
                circuit_record["target_fidelity"] = 0.0
            
            # 표현력 값 추출
            if isinstance(expressibility, dict):
                circuit_record["target_expressibility"] = expressibility.get("value", 0.0)
            elif isinstance(expressibility, (int, float)):
                circuit_record["target_expressibility"] = float(expressibility)
            else:
                circuit_record["target_expressibility"] = 0.0
            
            # 2큐빗 게이트 비율 계산
            two_qubit_gates = sum(1 for gate in gates if gate.lower() in ['cx', 'cy', 'cz', 'cnot'])
            total_gates = len(gates)
            actual_two_qubit_ratio = two_qubit_gates / max(total_gates, 1)
            circuit_record["actual_two_qubit_ratio"] = actual_two_qubit_ratio
            
            transformer_data.append(circuit_record)
            
        except Exception as e:
            print(f"⚠️ 회로 {result.get('circuit_index', 'unknown')} 트랜스포머 형식 변환 실패: {e}")
            continue
    
    return {
        "format": "transformer_friendly",
        "description": "Circuit data formatted for transformer model training",
        "total_circuits": len(transformer_data),
        "features": [
            "circuit_id", "n_qubits", "depth", "two_qubit_ratio", 
            "circuit_tokens", "circuit_sequence"
        ],
        "targets": [
            "target_fidelity", "target_expressibility", "actual_two_qubit_ratio"
        ],
        "data": transformer_data
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
            # 새로운 결과 구조에서 정보 추출
            fidelity = result.get("fidelity", {})
            expressibility = result.get("expressibility", {})
            additional_metrics = result.get("additional_metrics", {})
            
            # 핵심 메트릭(지표)를 추출합니다.
            row = {
                "circuit_index": result.get("circuit_index", -1),
                "config_group": result.get("config_group", ""),
                "n_qubits": result.get("n_qubits", 0),
                "depth": result.get("depth", 0) or additional_metrics.get("depth", 0),
                "two_qubit_ratio_target": result.get("two_qubit_ratio_target", 0),
                "zero_state_prob": fidelity.get("simple", 0),
                "robust_fidelity": fidelity.get("robust", 0),
                "fidelity_method": fidelity.get("method", "unknown")
            }
            
            # 표현력 지표 추가
            if isinstance(expressibility, dict):
                # 기본 표현력 값
                row["expressibility_score"] = expressibility.get("value", None)
                row["expressibility_method"] = expressibility.get("method", "unknown")
                row["distance_from_haar"] = expressibility.get("distance_from_haar", 1.0)
                
                # 추가적인 표현력 측정 지표들이 있다면 추가
                for metric_name, value in expressibility.items():
                    if metric_name not in ["value", "method", "distance_from_haar"]:
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
