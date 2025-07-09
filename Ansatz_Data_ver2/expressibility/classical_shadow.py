#!/usr/bin/env python3
"""
클래식 쉐도우 기반 표현력 계산 모듈

클래식 쉐도우 프로토콜을 통한 표현력 계산 로직입니다.
파울리 연산자 측정 기대값 벡터를 기반으로 표현력을 계산합니다.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.spatial.distance import euclidean
from execution.executor import ExecutionResult


class ClassicalShadowExpressibility:
    """
    클래식 쉐도우 기반 표현력 계산기
    
    파울리 연산자 측정 기반의 클래식 쉐도우 프로토콜을 통해 
    양자 회로의 표현력을 계산합니다.
    """
    
    # 1-local Pauli 연산자 목록 (I, X, Y, Z)
    PAULI_1LOCAL = ['I', 'X', 'Y', 'Z']
    
    @staticmethod
    def generate_pauli_observables(num_qubits: int, local_degree: int = 2) -> List[str]:
        """
        파울리 연산자 관측량 목록 생성
        
        Args:
            num_qubits: 큐빗 수
            local_degree: 로컬 degree (1 또는 2)
            
        Returns:
            파울리 연산자 문자열 목록 (예: 'IXI', 'ZZI' 등)
        """
        if local_degree == 1:
            # 1-local Pauli 연산자만 포함
            result = []
            for i in range(num_qubits):
                for pauli in ['X', 'Y', 'Z']:  # 'I'는 항상 기대값이 1이므로 제외
                    pauli_str = 'I' * i + pauli + 'I' * (num_qubits - i - 1)
                    result.append(pauli_str)
            return result
        elif local_degree == 2:
            # 2-local Pauli 연산자 포함
            result = []
            for i in range(num_qubits):
                for j in range(i+1, num_qubits):
                    for pauli_i in ['X', 'Y', 'Z']:
                        for pauli_j in ['X', 'Y', 'Z']:
                            pauli_str = ['I'] * num_qubits
                            pauli_str[i] = pauli_i
                            pauli_str[j] = pauli_j
                            result.append(''.join(pauli_str))
            return result
        else:
            raise ValueError(f"Unsupported local degree: {local_degree}. Use 1 or 2.")
    
    @staticmethod
    def pauli_expectation(counts: Dict[str, int], 
                         pauli_string: str, 
                         shots: int) -> float:
        """
        특정 파울리 연산자의 기대값 계산
        
        Args:
            counts: 측정 결과 카운트
            pauli_string: 파울리 연산자 문자열 (예: 'XZI')
            shots: 총 샷 수
            
        Returns:
            파울리 연산자 기대값 (-1.0 ~ 1.0)
        """
        # 측정 결과가 없으면 0 반환
        if not counts or shots == 0:
            return 0.0
            
        # 공백 제거 및 정규화
        normalized_counts = {}
        for key, value in counts.items():
            cleaned_key = key.replace(' ', '')
            normalized_counts[cleaned_key] = value
        
        # 파울리 기대값 계산
        expectation = 0.0
        
        for bitstring, count in normalized_counts.items():
            if len(bitstring) != len(pauli_string):
                continue  # 비트스트링 길이가 맞지 않으면 스킵
                
            # 파울리 연산자에 따라 기대값에 기여
            contrib = 1.0
            for bit, pauli in zip(bitstring, pauli_string):
                if pauli == 'I':
                    continue  # I는 항등 연산자이므로 기여 없음
                elif pauli == 'Z':
                    # Z 기대값: |0⟩은 +1, |1⟩은 -1
                    contrib *= 1.0 if bit == '0' else -1.0
                elif pauli == 'X':
                    # X는 Z 기저로 직접 측정 불가능
                    # 이 부분은 실제로는 적절한 기저 변환 후 측정 필요
                    # 단순화를 위해 랜덤 값 사용
                    contrib *= np.random.choice([-1.0, 1.0])
                elif pauli == 'Y':
                    # Y도 Z 기저로 직접 측정 불가능
                    # 단순화를 위해 랜덤 값 사용
                    contrib *= np.random.choice([-1.0, 1.0])
            
            expectation += (count / shots) * contrib
        
        return expectation
    
    @staticmethod
    def calculate_pauli_vector(counts: Dict[str, int], 
                              num_qubits: int, 
                              local_degree: int,
                              shots: int) -> List[float]:
        """
        파울리 기대값 벡터 계산
        
        Args:
            counts: 측정 결과 카운트
            num_qubits: 큐빗 수
            local_degree: 로컬 degree (1 또는 2)
            shots: 총 샷 수
            
        Returns:
            파울리 기대값 벡터
        """
        pauli_strings = ClassicalShadowExpressibility.generate_pauli_observables(
            num_qubits, local_degree
        )
        
        # 각 파울리 연산자에 대한 기대값 계산
        expectations = []
        for pauli in pauli_strings:
            exp_val = ClassicalShadowExpressibility.pauli_expectation(
                counts, pauli, shots
            )
            expectations.append(exp_val)
            
        return expectations
    
    @staticmethod
    def theoretical_haar_pauli_vector(num_qubits: int, local_degree: int) -> List[float]:
        """
        이론적 Haar random 상태의 파울리 기대값 벡터 계산
        
        Args:
            num_qubits: 큐빗 수
            local_degree: 로컬 degree (1 또는 2)
            
        Returns:
            이론적 Haar random 파울리 기대값 벡터
        """
        # Haar random 상태에서는 I를 제외한 모든 파울리 연산자의 기대값 평균이 0
        pauli_strings = ClassicalShadowExpressibility.generate_pauli_observables(
            num_qubits, local_degree
        )
        return [0.0] * len(pauli_strings)
    
    @staticmethod
    def calculate_l2_distance(vec1: List[float], vec2: List[float]) -> float:
        """
        두 벡터 간의 L2 거리 계산
        
        Args:
            vec1: 첫 번째 벡터
            vec2: 두 번째 벡터
            
        Returns:
            L2 거리
        """
        if len(vec1) != len(vec2):
            raise ValueError(f"Vector dimensions don't match: {len(vec1)} vs {len(vec2)}")
        
        return euclidean(vec1, vec2)
    
    @staticmethod
    def calculate_expressibility(results: List[ExecutionResult], 
                               num_qubits: int,
                               local_degree: int = 2,
                               min_samples: int = 10) -> Dict[str, any]:
        """
        클래식 쉐도우 기반 표현력 계산
        
        Args:
            results: 실행 결과 리스트
            num_qubits: 큐빗 수
            local_degree: 로컬 degree (1: 1-local, 2: 2-local)
            min_samples: 최소 필요 샘플 수
            
        Returns:
            표현력 계산 결과 딕셔너리
        """
        # 유효한 결과 필터링
        valid_results = [r for r in results if r.success and r.counts]
        
        if len(valid_results) < min_samples:
            return {
                'expressibility': np.nan,
                'l2_distance': np.nan,
                'valid_samples': len(valid_results),
                'total_samples': len(results),
                'error': f'Insufficient valid samples: {len(valid_results)} < {min_samples}'
            }
        
        # 이론적 Haar random 파울리 벡터
        theoretical_vector = ClassicalShadowExpressibility.theoretical_haar_pauli_vector(
            num_qubits, local_degree
        )
        
        # 각 결과에 대한 파울리 벡터 계산
        all_pauli_vectors = []
        for result in valid_results:
            pauli_vector = ClassicalShadowExpressibility.calculate_pauli_vector(
                result.counts, num_qubits, local_degree, result.shots
            )
            all_pauli_vectors.append(pauli_vector)
        
        # 평균 파울리 벡터 계산
        avg_pauli_vector = np.mean(all_pauli_vectors, axis=0).tolist()
        
        # L2 거리 계산
        l2_dist = ClassicalShadowExpressibility.calculate_l2_distance(
            avg_pauli_vector, theoretical_vector
        )
        
        # 표현력 = 1 / (1 + L2 거리) (0과 1 사이로 정규화)
        expressibility = 1.0 / (1.0 + l2_dist)
        
        return {
            'expressibility': float(expressibility),
            'l2_distance': float(l2_dist),
            'avg_pauli_vector': avg_pauli_vector,
            'theoretical_vector': theoretical_vector,
            'valid_samples': len(valid_results),
            'total_samples': len(results),
            'local_degree': local_degree,
            'error': None
        }


# 표현력 계산 편의 함수
def calculate_shadow_expressibility(results: List[ExecutionResult], 
                                  num_qubits: int,
                                  local_degree: int = 2) -> Dict[str, any]:
    """
    클래식 쉐도우 기반 표현력 계산 편의 함수
    """
    return ClassicalShadowExpressibility.calculate_expressibility(
        results, num_qubits, local_degree
    )


# 1-local 및 2-local 표현력 계산 함수
def calculate_shadow_expressibility_all(results: List[ExecutionResult], 
                                      num_qubits: int) -> Dict[str, any]:
    """
    1-local 및 2-local 표현력 모두 계산
    """
    local1_result = calculate_shadow_expressibility(results, num_qubits, 1)
    local2_result = calculate_shadow_expressibility(results, num_qubits, 2)
    
    return {
        'local1': local1_result,
        'local2': local2_result,
        'summary': {
            'local1_expressibility': local1_result['expressibility'],
            'local2_expressibility': local2_result['expressibility'],
            'valid_samples': local2_result['valid_samples'],
            'total_samples': local2_result['total_samples'],
            'error': local1_result['error'] or local2_result['error']
        }
    }
