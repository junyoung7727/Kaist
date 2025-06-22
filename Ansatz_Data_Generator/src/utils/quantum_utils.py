#!/usr/bin/env python3
"""
양자 계산 유틸리티 모듈 - 양자 회로 실행 관련 유틸리티 함수를 제공합니다.
"""

import sys
import os
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

# 현재 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def calculate_error_rates_mega(counts, n_qubits, total_counts):
    """
    메가잡용 측정 결과에서 오류율 계산
    
    Args:
        counts (dict): 측정 결과 카운트 {비트열: 카운트}
        n_qubits (int): 큐빗 수
        total_counts (int): 총 측정 횟수
        
    Returns:
        dict: 오류율 정보
    """
    # 올바른 상태는 |0...0>
    zero_state = '0' * n_qubits
    zero_count = counts.get(zero_state, 0)
    zero_probability = zero_count / total_counts if total_counts > 0 else 0
    
    # 오류 상태 처리
    error_states = {}
    error_prob_by_distance = {}
    
    for state, count in counts.items():
        if state != zero_state:
            distance = hamming_distance_mega(state, zero_state)
            prob = count / total_counts
            
            error_states[state] = {
                "count": count,
                "probability": prob,
                "distance": distance
            }
            
            if distance in error_prob_by_distance:
                error_prob_by_distance[distance] += prob
            else:
                error_prob_by_distance[distance] = prob
    
    # 허용 오류 기준의 로버스트 오류율
    threshold = get_error_threshold_mega(n_qubits)
    robust_error_states = {
        state: info for state, info in error_states.items()
        if info["distance"] <= threshold and info["distance"] > 0
    }
    
    robust_error_prob = sum(info["probability"] for info in robust_error_states.values())
    
    # 평균 해밍 거리 (오류 정도)
    total_hamming = sum(info["distance"] * info["probability"] for info in error_states.values())
    avg_hamming = total_hamming / (1 - zero_probability) if 1 - zero_probability > 0 else 0
    
    return {
        "correct_prob": zero_probability,
        "error_prob": 1 - zero_probability,
        "robust_error_prob": robust_error_prob,
        "avg_hamming_distance": avg_hamming,
        "error_prob_by_distance": error_prob_by_distance
    }


def calculate_robust_fidelity_mega(counts, n_qubits, total_counts):
    """
    메가잡용 Robust Fidelity 계산 (노이즈 허용)
    
    Args:
        counts (dict): 측정 결과 카운트 {비트열: 카운트}
        n_qubits (int): 큐빗 수
        total_counts (int): 총 측정 횟수
        
    Returns:
        float: Robust Fidelity (0~1 사이)
    """
    # 올바른 상태는 |0...0>
    zero_state = '0' * n_qubits
    zero_count = counts.get(zero_state, 0)
    
    # 허용 오류 임계값
    threshold = get_error_threshold_mega(n_qubits)
    
    # 임계값 이내의 오류 상태들 카운트
    robust_count = zero_count
    
    for state, count in counts.items():
        if state != zero_state:
            distance = hamming_distance_mega(state, zero_state)
            if distance <= threshold:
                robust_count += count
    
    # Robust Fidelity 계산
    robust_fidelity = robust_count / total_counts if total_counts > 0 else 0
    
    return robust_fidelity


def get_error_threshold_mega(n_qubits):
    """
    메가잡용 큐빗 수에 따른 허용 오류 비트 수 계산
    
    Args:
        n_qubits (int): 큐빗 수
        
    Returns:
        int: 허용 오류 비트 수
    """
    if n_qubits <= 5:
        return 0  # 5큐빗 이하는 오류 허용 안함
    elif n_qubits <= 10:
        return 1  # 10큐빗 이하는 1비트 오류 허용
    elif n_qubits <= 20:
        return 2  # 20큐빗 이하는 2비트 오류 허용
    else:
        return 3  # 그 이상은 3비트 오류 허용


def hamming_distance_mega(state1, state2):
    """
    메가잡용 두 비트 문자열 간의 해밍 거리 계산
    
    Args:
        state1 (str): 첫 번째 비트 문자열
        state2 (str): 두 번째 비트 문자열
        
    Returns:
        int: 해밍 거리 (다른 비트 수)
    """
    if len(state1) != len(state2):
        # 길이가 다르면 최소 길이까지만 비교
        min_len = min(len(state1), len(state2))
        return sum(s1 != s2 for s1, s2 in zip(state1[:min_len], state2[:min_len])) + abs(len(state1) - len(state2))
    else:
        return sum(s1 != s2 for s1, s2 in zip(state1, state2))


def calculate_measurement_statistics(counts, n_qubits):
    """
    실제 측정 결과에서만 나오는 통계적 특성
    
    Args:
        counts (dict): 측정 결과 카운트 {비트열: 카운트}
        n_qubits (int): 큐빗 수
        
    Returns:
        dict: 측정 통계 정보
    """
    if not counts:
        return {
            "entropy": 0,
            "significant_states": 0,
            "majority_state_prob": 0,
            "zero_state_prob": 0
        }
    
    # 총 측정 수
    total_counts = sum(counts.values())
    
    # 상태별 확률 분포
    probabilities = {state: count / total_counts for state, count in counts.items()}
    
    # 엔트로피 계산 (비트 단위)
    entropy = -sum(p * np.log2(p) for p in probabilities.values() if p > 0)
    
    # 중요 상태 수 (1% 이상 확률)
    significant_states = sum(1 for p in probabilities.values() if p >= 0.01)
    
    # 가장 많이 측정된 상태 확률
    majority_state_prob = max(probabilities.values()) if probabilities else 0
    
    # |0...0> 상태 확률
    zero_state = '0' * n_qubits
    zero_state_prob = probabilities.get(zero_state, 0)
    
    # 결과
    return {
        "entropy": entropy,
        "max_entropy": n_qubits,  # 최대 가능 엔트로피
        "entropy_ratio": entropy / n_qubits if n_qubits > 0 else 0,
        "significant_states": significant_states,
        "majority_state_prob": majority_state_prob,
        "zero_state_prob": zero_state_prob
    }


def create_circuit_embedding(circuit_sequence, max_length=100):
    """
    회로 시퀀스를 트랜스포머 입력용 임베딩으로 변환 - 개선된 버전
    
    Args:
        circuit_sequence (dict): 회로 시퀀스 정보
        max_length (int): 최대 임베딩 길이
        
    Returns:
        dict: 회로 임베딩 정보
    """
    # 게이트 타입 사전
    gate_dict = {
        "H": 1, "X": 2, "Y": 3, "Z": 4, "S": 5, "T": 6,
        "RX": 7, "RY": 8, "RZ": 9, "CNOT": 10, "CZ": 11,
        "SWAP": 12
    }
    
    gates = circuit_sequence.get("gates", [])
    wires_list = circuit_sequence.get("wires_list", [])
    params_idx = circuit_sequence.get("params_idx", [])
    params = circuit_sequence.get("params", [])
    
    # 시퀀스 길이 제한
    seq_len = min(len(gates), max_length)
    
    # 임베딩 배열 초기화
    gate_ids = np.zeros(max_length, dtype=int)
    qubit1_ids = np.zeros(max_length, dtype=int)
    qubit2_ids = np.zeros(max_length, dtype=int)
    has_param = np.zeros(max_length, dtype=int)
    param_values = np.zeros(max_length, dtype=float)
    
    # 임베딩 생성
    for i in range(seq_len):
        gate = gates[i]
        wires = wires_list[i] if i < len(wires_list) else []
        
        # 게이트 ID
        gate_ids[i] = gate_dict.get(gate, 0)
        
        # 큐빗 ID
        if wires:
            qubit1_ids[i] = wires[0] + 1  # 1부터 시작하는 인덱스
            if len(wires) > 1:
                qubit2_ids[i] = wires[1] + 1
        
        # 파라미터 정보
        param_idx = i in params_idx
        has_param[i] = 1 if param_idx else 0
        
        if param_idx:
            param_pos = params_idx.index(i)
            if param_pos < len(params):
                param_values[i] = params[param_pos]
    
    # 어텐션 마스크 (실제 시퀀스 길이까지만 1, 나머지는 0)
    attention_mask = np.zeros(max_length, dtype=int)
    attention_mask[:seq_len] = 1
    
    return {
        "gate_ids": gate_ids.tolist(),
        "qubit1_ids": qubit1_ids.tolist(),
        "qubit2_ids": qubit2_ids.tolist(),
        "has_param": has_param.tolist(),
        "param_values": param_values.tolist(),
        "attention_mask": attention_mask.tolist(),
        "seq_len": seq_len
    }
