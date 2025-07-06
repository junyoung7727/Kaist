#!/usr/bin/env python3
"""
표현력(Expressibility) 계산 모듈

양자 회로의 표현력을 계산하는 순수한 수학적 로직입니다.
백엔드에 무관하게 작동하며, 피델리티 결과만을 사용합니다.
다양한 divergence 측정 방법(KL, JS, L2)을 지원합니다.
"""

from typing import List, Dict, Optional, Tuple, Any 
import numpy as np
from scipy.stats import kstest, entropy
from scipy.spatial.distance import euclidean
from execution.executor import ExecutionResult
from core.error_fidelity import ErrorFidelityCalculator


class Divergence_Expressibility:
    """
    표현력 계산기
    
    양자 회로의 표현력을 다양한 분포 비교 방법을 통해 계산합니다.
    지원하는 방법: KL Divergence, JS Divergence, L2 Norm, KS Test
    백엔드 구현에 전혀 의존하지 않습니다.
    """
    
    @staticmethod
    def calculate_from_fidelities(fidelities: List[float], 
                                 num_qubits: int,
                                 min_samples: int = 100) -> Dict[str, float]:
        """
        피델리티 리스트로부터 표현력을 계산합니다.
        
        Args:
            fidelities: 피델리티 값들의 리스트
            num_qubits: 큐빗 수
            min_samples: 최소 필요 샘플 수
            
        Returns:
            표현력 계산 결과 딕셔너리
        """
        # 유효한 피델리티 필터링 (0 <= fidelity <= 1)
        valid_fidelities = [f for f in fidelities if 0.0 <= f <= 1.0 and not np.isnan(f)]
        
        if len(valid_fidelities) < min_samples:
            return {
                'expressibility': np.nan,
                'ks_statistic': np.nan,
                'p_value': np.nan,
                'valid_samples': len(valid_fidelities),
                'total_samples': len(fidelities),
                'error': f'Insufficient valid samples: {len(valid_fidelities)} < {min_samples}'
            }
        
        # 이론적 분포 (Haar random 분포)
        # d차원 Hilbert 공간에서 Haar random 상태의 피델리티 분포
        d = 2 ** num_qubits
        
        def haar_fidelity_cdf(x):
            """Haar random 상태의 피델리티 누적분포함수"""
            if x <= 0:
                return 0.0
            elif x >= 1:
                return 1.0
            else:
                # F(x) = 1 - (1-x)^(d-1) for 0 <= x <= 1
                return 1.0 - np.power(1.0 - x, d - 1)
        
        # 실제 피델리티 분포 히스토그램 계산
        try:
            # 히스토그램 빈 설정 (0-1 사이를 100개 구간으로 분할)
            bins = 100
            bin_edges = np.linspace(0, 1, bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # 실제 피델리티 히스토그램
            hist_fidelities, _ = np.histogram(valid_fidelities, bins=bin_edges, density=True)
            
            # 이론적 Haar 분포 히스토그램 (bin 중심에서 PDF 계산)
            d = 2 ** num_qubits
            hist_haar = [(d-1) * (1-x)**(d-2) for x in bin_centers]  # Haar PDF: (d-1)(1-x)^(d-2)
            hist_haar = hist_haar / np.sum(hist_haar)  # 정규화
            
            # 분포에 0이 있으면 작은 값으로 대체 (발산 방지)
            epsilon = 1e-10
            hist_fidelities = np.maximum(hist_fidelities, epsilon)
            hist_haar = np.maximum(hist_haar, epsilon)
            
            # 모든 히스토그램 정규화
            hist_fidelities = hist_fidelities / np.sum(hist_fidelities)
            hist_haar = hist_haar / np.sum(hist_haar)
            
            # 1. KL Divergence 계산 (Kullback-Leibler)
            kl_divergence = entropy(hist_fidelities, hist_haar)
            
            # 2. JS Divergence 계산 (Jensen-Shannon)
            m = 0.5 * (hist_fidelities + hist_haar)
            js_divergence = 0.5 * (entropy(hist_fidelities, m) + entropy(hist_haar, m))
            
            # 3. L2 Norm 계산
            l2_norm = euclidean(hist_fidelities, hist_haar)
            
            # 4. Kolmogorov-Smirnov 테스트 수행
            ks_statistic, p_value = kstest(valid_fidelities, haar_fidelity_cdf)
            
            # 표현력 계산 (각 측정치에 대해 1에 가까울수록 Haar에 가까움)
            # KL과 L2는 값이 작을수록 유사하므로 역수 관계 이용
            expr_kl = 1.0 / (1.0 + kl_divergence)
            expr_js = 1.0 - js_divergence  # JS는 [0,1] 범위를 가지므로
            expr_l2 = 1.0 / (1.0 + l2_norm)
            expr_ks = 1.0 - ks_statistic
            
            # 모든 표현력 평균 계산
            expr_avg = np.mean([expr_kl, expr_js, expr_l2, expr_ks])
            
            return {
                'expressibility': float(expr_avg),  # 평균 표현력
                'expressibility_kl': float(expr_kl),
                'expressibility_js': float(expr_js),
                'expressibility_l2': float(expr_l2),
                'expressibility_ks': float(expr_ks),
                'kl_divergence': float(kl_divergence),
                'js_divergence': float(js_divergence),
                'l2_norm': float(l2_norm),
                'ks_statistic': float(ks_statistic),
                'p_value': float(p_value),
                'valid_samples': len(valid_fidelities),
                'total_samples': len(fidelities),
                'hist_fidelities': hist_fidelities.tolist(),
                'hist_haar': hist_haar.tolist(),
                'bin_centers': bin_centers.tolist(),
                'error': None
            }
            
        except Exception as e:
            return {
                'expressibility': np.nan,
                'expressibility_kl': np.nan,
                'expressibility_js': np.nan,
                'expressibility_l2': np.nan,
                'expressibility_ks': np.nan,
                'kl_divergence': np.nan,
                'js_divergence': np.nan,
                'l2_norm': np.nan,
                'ks_statistic': np.nan,
                'p_value': np.nan,
                'valid_samples': len(valid_fidelities),
                'total_samples': len(fidelities),
                'error': str(e)
            }
    
    @staticmethod
    def calculate_from_execution_results(results: List[ExecutionResult], 
                                       num_qubits: int,
                                       min_samples: int = 100) -> Dict[str, float]:
        """
        실행 결과 리스트로부터 표현력을 계산합니다.
        
        Args:
            results: 실행 결과 리스트
            num_qubits: 큐빗 수
            min_samples: 최소 필요 샘플 수
            
        Returns:
            표현력 계산 결과 딕셔너리
        """
        # 각 결과로부터 피델리티 계산
        fidelities = []
        for result in results:
            if result.success and result.counts:
                fidelity = ErrorFidelityCalculator.calculate_from_counts(result.counts, num_qubits)
                fidelities.append(fidelity)
        
        return Divergence_Expressibility.calculate_from_fidelities(
            fidelities, num_qubits, min_samples
        )
    
    @staticmethod
    def generate_haar_random_fidelities(num_samples: int, num_qubits: int) -> List[float]:
        """
        비교용 Haar random 피델리티 샘플을 생성합니다.
        
        Args:
            num_samples: 생성할 샘플 수
            num_qubits: 큐빗 수
            
        Returns:
            Haar random 피델리티 리스트
        """
        d = 2 ** num_qubits
        
        # Haar random 분포에서 샘플링
        # F^(-1)(u) = 1 - (1-u)^(1/(d-1)) where u ~ Uniform(0,1)
        uniform_samples = np.random.uniform(0, 1, num_samples)
        haar_fidelities = 1.0 - np.power(1.0 - uniform_samples, 1.0 / (d - 1))
        
        return haar_fidelities.tolist()
    
    @staticmethod
    def compare_with_haar_random(fidelities: List[float], 
                               num_qubits: int,
                               num_haar_samples: int = 1000) -> Dict[str, Any]:
        """
        실제 피델리티와 Haar random 피델리티를 비교합니다.
        
        Args:
            fidelities: 실제 피델리티 리스트
            num_qubits: 큐빗 수
            num_haar_samples: 비교용 Haar random 샘플 수
            
        Returns:
            비교 결과 딕셔너리
        """
        # Haar random 샘플 생성
        haar_fidelities = ExpressibilityCalculator.generate_haar_random_fidelities(
            num_haar_samples, num_qubits
        )
        
        # 각각의 표현력 계산
        actual_result = ExpressibilityCalculator.calculate_from_fidelities(
            fidelities, num_qubits
        )
        haar_result = ExpressibilityCalculator.calculate_from_fidelities(
            haar_fidelities, num_qubits
        )
        
        return {
            'actual': actual_result,
            'haar_random': haar_result,
            'comparison': {
                'expressibility_diff': actual_result['expressibility'] - haar_result['expressibility'],
                'ks_statistic_diff': actual_result['ks_statistic'] - haar_result['ks_statistic']
            }
        }


def calculate_expressibility(fidelities: List[float], num_qubits: int) -> Dict[str, float]:
    """
    표현력 계산 편의 함수
    """
    return Divergence_Expressibility.calculate_from_fidelities(fidelities, num_qubits)


def calculate_expressibility_from_results(results: List[Any], num_qubits: int) -> Dict[str, float]:
    """
    실행 결과로부터 표현력 계산 편의 함수
    """
    return Divergence_Expressibility.calculate_from_execution_results(results, num_qubits)


def analyze_circuit_expressibility(circuit_results: List[Any], num_qubits: int, num_random_params: int = 30) -> Dict[str, Any]:
    """
    회로 표현력 분석 함수

    단일 회로에 대해 다수의 랜덤 파라미터 값으로 실행하고
    그 결과의 fidelity 분포를 분석하여 표현력을 계산합니다.

    Args:
        circuit_results: 회로 실행 결과 리스트
        num_qubits: 큐빗 수
        num_random_params: 랜덤 파라미터 샘플 수

    Returns:
        표현력 분석 결과 딕셔너리
    """
    # 피델리티 계산
    fidelities = []

    for result in circuit_results:
        if result.success and result.counts:
            fidelity = ErrorFidelityCalculator.calculate_from_counts(result.counts, num_qubits)
            fidelities.append(fidelity)

    # 표현력 계산
    expr_result = calculate_expressibility(fidelities, num_qubits)

    # 결과 데이터에 메타데이터 추가
    expr_result['num_random_params'] = num_random_params
    expr_result['num_qubits'] = num_qubits

    return expr_result
