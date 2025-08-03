"""
엔트로피 기반 표현력(Expressibility) 계산 모듈

이 모듈은 측정 결과의 엔트로피를 기반으로 표현력을 계산하는 함수들을 제공합니다.
"""

import numpy as np
import time
from scipy.stats import entropy
from typing import Dict, List, Any, Optional, Tuple, Union

# 내부 모듈 임포트
from src.calculators.expressibility.base import calculate_measurement_entropy
from src.config import config # Ensure config is imported


def calculate_entropy_expressibility(measurement_counts: Dict[str, int], n_qubits: int, n_bins: Optional[int] = None) -> Dict[str, Any]:
    """
    측정 결과의 엔트로피를 기반으로 한 표현력 계산
    
    이 함수는 양자 회로의 측정 결과로부터 샤논 엔트로피를 계산하여
    양자 회로의 표현력을 평가합니다. 또한 가능한 경우 각도 엔트로피도 계산합니다.
    
    Args:
        measurement_counts (Dict[str, int]): 측정 결과 {비트열: 카운트} 형식의 딕셔너리
        n_qubits (int): 큐빗 수
        n_bins (int): 히스토그램 구간 수 (각도 엔트로피 계산에 사용)
        
    Returns:
        Dict[str, Any]: 엔트로피 기반 표현력 결과를 포함한 딕셔너리:
            - expressibility_value: 측정 엔트로피 값
            - measurement_entropy: 측정 엔트로피 값 (expressibility_value와 동일)
            - angle_entropy: 각도 엔트로피 값 (가능한 경우)
            - method: 사용된 방법론 ('measurement_entropy')
            - n_qubits: 큐빗 수
            - measured_states: 측정된 고유 상태 수
    """
    start_time = time.time()
    
    # 측정 엔트로피 계산
    entropy_value = calculate_measurement_entropy(measurement_counts)
    
    # 측정된 상태 수
    measured_states = len(measurement_counts)
    
    # 최대 가능 엔트로피 (이론적 한계)
    max_entropy = n_qubits  # 큐빗당 1비트의 엔트로피가 이론적 최대
    
    # 측정 결과 → 벡터 변환 (각도 엔트로피 계산용)
    bit_vectors = []
    weights = []
    
    for bitstring, count in measurement_counts.items():
        # 비트열을 정수 배열로 변환
        if len(bitstring) < n_qubits:
            # 부족한 비트는 0으로 채움
            bitstring = bitstring.zfill(n_qubits)
            
        bit_vector = np.array([int(b) for b in bitstring])
        bit_vectors.append(bit_vector)
        weights.append(count)
    
    # 각도 엔트로피 계산 (벡터가 충분히 있을 경우)
    angle_entropy_value = None
    if n_bins is None:
        n_bins = config.get("expressibility", {}).get("entropy_angle_bins", 10)
    if len(bit_vectors) >= 2:
        try:
            angle_entropy_value = calculate_angle_entropy(
                bit_vectors, weights, n_bins
            )
        except Exception as e:
            print(f"⚠️ 각도 엔트로피 계산 오류: {str(e)}")
    
    # 실행 시간
    execution_time = time.time() - start_time
    
    result = {
        "expressibility_value": float(entropy_value),  # 이 값을 메인 표현력 값으로 사용
        "measurement_entropy": float(entropy_value),
        "max_entropy": float(max_entropy),
        "normalized_entropy": float(entropy_value / max_entropy) if max_entropy > 0 else 0.0,
        "angle_entropy": float(angle_entropy_value) if angle_entropy_value is not None else None,
        "method": "measurement_entropy",
        "n_qubits": n_qubits,
        "measured_states": measured_states,
        "total_measurements": sum(measurement_counts.values()),
        "execution_time": execution_time
    }
    
    return result


def entropy_based_expressibility(bit_strings: List[str], frequencies: List[int], n_bins: Optional[int] = None) -> Dict[str, Any]:
    """
    측정 결과의 엔트로피 기반 표현력 계산
    
    이 함수는 측정 결과의 엔트로피를 계산하여 양자 회로의 표현력을 평가합니다.
    
    Args:
        bit_strings: 측정된 비트스트링들의 리스트
        frequencies: 각 비트스트링의 측정 빈도 리스트
        n_bins: 히스토그램 구간 수 (사용하지 않음, 호환성 유지용)
        
    Returns:
        Dict[str, Any]: 엔트로피 기반 표현력 결과
            - total_entropy: 전체 측정 엔트로피
            - method: 사용된 방법론 ('measurement_entropy')
    """
    # 측정 결과 딕셔너리 생성
    measurement_counts = {}
    for bit_string, freq in zip(bit_strings, frequencies):
        measurement_counts[bit_string] = freq
    
    # 비트스트링 길이로 큐빗 수 추정
    n_qubits = max(len(bs) for bs in bit_strings)
    
    # 엔트로피 계산
    # 여기서 n_bins는 None일 수 있으므로 calculate_entropy_expressibility에서 처리
    # calculate_entropy_expressibility는 config에서 entropy_angle_bins 값을 가져오게 됨
    result = calculate_entropy_expressibility(measurement_counts, n_qubits, n_bins)
    
    return {
        "total_entropy": result["measurement_entropy"],
        "method": "measurement_entropy"
    }


def calculate_angle_entropy(vectors: List[np.ndarray], weights: List[int], n_bins: int) -> float:
    """
    벡터 간 각도 분포의 엔트로피
    
    Args:
        vectors (List[np.ndarray]): 벡터 리스트
        weights (List[int]): 각 벡터의 가중치
        n_bins (int): 히스토그램 구간 수
        
    Returns:
        float: 각도 엔트로피
    """
    angles = []
    total_weight = sum(weights)
    weights_normalized = [w / total_weight for w in weights]
    
    # 모든 벡터 쌍 간의 각도 계산
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            v1 = vectors[i]
            v2 = vectors[j]
            
            # 내적으로 코사인 각도 계산
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                cos_angle = dot_product / (norm1 * norm2)
                # 수치 안정성을 위한 클리핑
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                
                # 가중치 계산
                weight = weights_normalized[i] * weights_normalized[j]
                angles.extend([angle] * int(weight * 1000))  # 가중치에 비례하여 추가
    
    # 각도 분포의 히스토그램
    hist, _ = np.histogram(angles, bins=n_bins, range=(0, np.pi), density=True)
    
    # 엔트로피 계산
    angle_entropy = entropy(hist, base=2)
    
    return angle_entropy


def calculate_entropy_expressibility_from_ibm_results(measurement_counts: Dict[str, int], n_qubits: int) -> Dict[str, Any]:
    """
    IBM 측정 결과로부터 엔트로피 기반 표현력 계산
    
    이 함수는 IBM 양자 컴퓨터에서 얻은 측정 결과를 바탕으로
    양자 회로의 표현력을 측정 엔트로피 및 각도 엔트로피를 사용하여 평가합니다.
    
    Args:
        measurement_counts (Dict[str, int]): 측정 결과 {'00': count, '01': count, ...} 형식의 딕셔너리
        n_qubits (int): 큐빗 수
        
    Returns:
        Dict[str, Any]: 엔트로피 기반 표현력 결과를 포함한 딕셔너리:
            - expressibility_value: 측정 엔트로피 값
            - measurement_entropy: 측정 엔트로피 값 (expressibility_value와 동일)
            - angle_entropy: 각도 엔트로피 값 (가능한 경우)
            - method: 사용된 방법론 ('measurement_entropy')
            - n_qubits: 큐빗 수
            - measured_states: 측정된 고유 상태 수
            - total_measurements: 총 측정 횟수
            - run_time: 실행 시간(초)
    """
    start_time = time.time()
    
    # 측정 결과 필터링 (기저 표기가 있는 경우)
    filtered_counts = {}
    for bitstring, count in measurement_counts.items():
        # 기저 표기가 있는 경우 (예: "00_X0_Z1") 기저 정보 제거
        if "_" in bitstring:
            bits = bitstring.split("_")[0]
            filtered_counts[bits] = filtered_counts.get(bits, 0) + count
        else:
            filtered_counts[bitstring] = filtered_counts.get(bitstring, 0) + count
    
    # 엔트로피 기반 표현력 계산
    result = calculate_entropy_expressibility(filtered_counts, n_qubits)
    
    # 실행 시간 추가
    execution_time = time.time() - start_time
    result["run_time"] = execution_time
    
    return result
