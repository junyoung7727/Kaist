#!/usr/bin/env python3
"""
피델리티 계산 모듈

양자 회로의 피델리티를 계산하는 순수한 수학적 로직입니다.
백엔드에 무관하게 작동하며, 실행 결과만을 사용합니다.
"""

from typing import Dict, List, Optional
import numpy as np
from execution.executor import ExecutionResult
from config import default_config


class ErrorFidelityCalculator:
    """
    에러 피델리티 계산기
    
    양자 회로의 피델리티를 다양한 방법으로 계산합니다.
    백엔드 구현에 전혀 의존하지 않습니다.
    """
    
    @staticmethod
    def calculate_from_counts(counts: Dict[str, int], num_qubits: int) -> float:
        """
        측정 결과 카운트로부터 피델리티를 계산합니다.
        
        피델리티는 |00...0⟩ 상태의 측정 확률로 정의됩니다.
        
        Args:
            counts: 측정 결과 카운트 딕셔너리
            num_qubits: 큐빗 수
            
        Returns:
            피델리티 값 (0.0 ~ 1.0)
        """
        if not counts:
            return 0.0
        
        cleaned_counts = {}
        for key, value in counts.items():
            cleaned_key = str(key.replace(' ', ''))
            cleaned_counts[cleaned_key] = value
        counts = cleaned_counts
    

        # 전체 샷 수 계산
        total_shots = default_config.shots
        if total_shots == 0:
            return 0.0
        
        # |00...0⟩ 상태의 카운트 (모든 큐빗이 0인 상태)
        zero_state = '0' * num_qubits
        zero_counts = counts.get(zero_state, 0)
        
        print(counts)
        print(num_qubits)
        print(counts[zero_state])
        print(counts.get('00000000'))
        print(zero_counts)
        print(total_shots)

        # 피델리티 = P(|00...0⟩)
        fidelity = zero_counts / total_shots
        return float(fidelity)
    
    @staticmethod
    def calculate_from_execution_result(result: ExecutionResult, num_qubits: int) -> float:
        """
        실행 결과로부터 피델리티를 계산합니다.
        
        Args:
            result: 회로 실행 결과
            num_qubits: 큐빗 수
            
        Returns:
            피델리티 값 (0.0 ~ 1.0)
        """
        if not result.success:
            return 0.0
        
        return ErrorFidelityCalculator.calculate_from_counts(result.counts, num_qubits)


def calculate_error_fidelity(counts: Dict[str, int], num_qubits: int) -> float:
    """피델리티 계산 편의 함수"""
    return ErrorFidelityCalculator.calculate_from_counts(counts, num_qubits)


def calculate_error_fidelity_from_result(result: ExecutionResult, num_qubits: int) -> float:
    """실행 결과로부터 피델리티 계산 편의 함수"""
    return ErrorFidelityCalculator.calculate_from_execution_result(result, num_qubits)
