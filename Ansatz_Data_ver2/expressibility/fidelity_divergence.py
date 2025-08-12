#!/usr/bin/env python3
"""
표현력(Expressibility) 계산 모듈

양자 회로의 표현력을 계산하는 순수한 수학적 로직입니다.
백엔드에 무관하게 작동하며, 피델리티 결과만을 사용합니다.
다양한 divergence 측정 방법(KL, JS, L2)을 지원합니다.
"""

from typing import List, Dict, Optional, Tuple, Any, Union 
import numpy as np
from scipy.stats import kstest, entropy
from scipy.spatial.distance import euclidean
from execution.executor import ExecutionResult
from expressibility.statevector_fidelity import StatevectorFidelityCalculator
from core.error_fidelity import ErrorFidelityCalculator
from core.circuit_interface import CircuitSpec
from core.random_circuit_generator import create_random_parameterized_samples
from expressibility.statevector_fidelity import StatevectorFidelityCalculator
from matplotlib import pyplot as plt
from config import ExperimentConfig




class Divergence_Expressibility:
    """
    표현력 계산기
    
    양자 회로의 표현력을 다양한 분포 비교 방법을 통해 계산합니다.
    지원하는 방법: KL Divergence, JS Divergence, L2 Norm, KS Test
    백엔드 구현에 전혀 의존하지 않습니다.
    """
    @staticmethod
    def generate_haar_random_fidelities(num_qubits: int, num_pairs: int) -> List[float]:
        """
        비교용 Haar random 피델리티 샘플을 생성합니다.
        
        수학적 근거:
        - 두 Haar random 상태 간 피델리티는 Beta(1, d-1) 분포
        - PDF: P(F) = (d-1)(1-F)^(d-2)
        - CDF: P(F≤f) = 1 - (1-f)^(d-1)
        - Inverse CDF: F = 1 - (1-u)^(1/(d-1))
        
        Args:
            num_samples: 생성할 샘플 수
            num_qubits: 큐빗 수
            
        Returns:
            Haar random 피델리티 리스트
        """
        # 힐베르트 공간 차원
        d = 2 ** num_qubits
        
        # [0,1] 균등분포에서 샘플링
        uniform_samples = np.random.uniform(0, 1, num_pairs)
        
        # 수치적 안정성을 위해 log scale에서 계산
        # (1-u)^(1/(d-1)) = exp(log(1-u) / (d-1))
        
        # 1-u가 0에 가까워질 때를 위한 안전장치
        one_minus_u = 1.0 - uniform_samples
        one_minus_u = np.clip(one_minus_u, 1e-16, 1.0)  # underflow 방지
        
        # 로그 스케일 계산
        log_term = np.log(one_minus_u)  # 이 부분이 빠져있었음!
        log_exponent = log_term / (d - 1)
        
        # 피델리티 계산: F = 1 - (1-u)^(1/(d-1))
        haar_fidelities = 1.0 - np.exp(log_exponent)
        
        # 결과값 범위 확인 및 제한
        haar_fidelities = np.clip(haar_fidelities, 0.0, 1.0)
        
        # 소수점 제한하여 정밀도 문제 방지
        haar_fidelities = np.round(haar_fidelities, 8)

        return haar_fidelities.tolist()


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
        # numpy 배열 처리를 위해 안전하게 변환
        if isinstance(fidelities, np.ndarray):
            fidelities = fidelities.tolist()
        
        valid_fidelities = []
        for f in fidelities:
            f_val = float(f)
            if 0.0 <= f_val <= 1.0 and not np.isnan(f_val):
                valid_fidelities.append(f_val)
        
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
            """Haar random 상태의 피델리틴 누적분포함수"""
            # scalar 값만 처리 (간단하게)
            x = float(x)  # 확실히 scalar로 변환
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
    def plot_overlapping_histograms(circuit_hist, haar_hist, title="Distribution Comparison", bins=None):
        """
        미리 계산된 히스토그램 데이터를 시각화
        
        Parameters:
        -----------
        circuit_hist : array-like
            회로 분포 히스토그램 데이터 (이미 계산된 빈도)
        haar_hist : array-like
            Haar 랜덤 히스토그램 데이터 (이미 계산된 빈도)
        title : str
            그래프 제목
        bins : array-like, optional
            빈 경계값들. 제공되지 않으면 0과 1 사이에 균등하게 생성됨
        """
        # 빈 경계값 설정 (제공되지 않은 경우)
        if bins is None:
            bins = np.linspace(0, 1, len(circuit_hist) + 1)
        
        # 빈 중앙값 계산 (x축 위치용)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_width = bins[1] - bins[0]
        
        # 플롯 설정
        plt.figure(figsize=(10, 6))
        
        # 바 차트로 히스토그램 표시 (겹쳐서)
        plt.bar(bin_centers, circuit_hist, width=bin_width*0.8, alpha=0.7, 
                color='skyblue', label='Circuit Distribution', edgecolor='navy', linewidth=0.8)
        
        plt.bar(bin_centers, haar_hist, width=bin_width*0.8, alpha=0.7,
                color='lightcoral', label='Haar Random Distribution', edgecolor='darkred', linewidth=0.8)
        
        # 스타일링
        plt.xlabel('Fidelity', fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 축 범위 조정 - 피델리티는 0-1 범위
        plt.xlim(0, 1)
        
        # y축도 적절하게 조정
        max_height = max(np.max(circuit_hist), np.max(haar_hist))
        plt.ylim(0, max_height * 1.1)  # 최대값보다 10% 더 여유있게
        
        # 배경색 설정
        plt.gca().set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def batch_circuit(circuit_specs: List[CircuitSpec], exp_config: ExperimentConfig, num_samples: int = 10, batch_manager=None):
        qc_idxs = []
        for circuit_spec in circuit_specs:
            qc_idx = Divergence_Expressibility.calculate_from_circuit_specs_divergence_hardware(circuit_spec, exp_config, num_samples, batch_manager)
            qc_idxs.append(qc_idx)
            print(qc_idx)
        return qc_idxs

    @staticmethod
    def calculate_from_circuit_specs_divergence_hardware(circuit_spec: CircuitSpec, exp_config: ExperimentConfig, num_samples: int = 50, batch_manager=None) -> Union[Dict[str, float], List[int]]:
        """
        하드웨어 실행 결과로부터 클래식 쉐도우 기반 표현력 계산
        
        Args:
            circuit_spec: 기본 회로 사양 (파라미터화된 회로)
            exp_config: 실험 설정
            num_samples: 생성할 랜덤 파라미터 샘플 수
            
        Returns:
            표현력 계산 결과 딕셔너리
        """
        from expressibility.swap_test_fidelity import SwapTestFidelityEstimator
        
        try:
            # IBM 실행자 생성
            executor = exp_config.executor
            Swaptest = SwapTestFidelityEstimator(executor, exp_config)
            if batch_manager:
                # 배치 모드: 회로만 수집
                indices = Swaptest.generate_pairwise_fidelities(
                    circuit_spec=circuit_spec,
                    num_samples=num_samples,
                    shots_per_measurement=1024,
                    batch_manager=batch_manager
                )
                return indices
            else:
                # 기존 모드: 직접 실행
                # 클래식 쉐도우 기반 페어와이즈 피델리티 계산
                fidelities = Swaptest.generate_pairwise_fidelities(
                    circuit_spec=circuit_spec,
                    num_samples=num_samples,
                    shots_per_measurement=1024
                )
                
                return Divergence_Expressibility._cal_fidelity_divergence(fidelities, circuit_spec.num_qubits)
            
        except Exception as e:
            return {
                "expressibility": np.nan,
                "kl_divergence": np.nan,
                "js_divergence": np.nan,
                "wasserstein_distance": np.nan,
                "total_samples": 0,
                "error": f"Hardware execution failed: {str(e)}"
            }
    @staticmethod
    def calculate_from_circuit_specs_divergence_list(circuit_specs: List[CircuitSpec], 
                                    num_samples: int = 50) -> Dict[str, float]:
        results = []
        for circuit_spec in circuit_specs:
            results.append(Divergence_Expressibility.calculate_from_circuit_specs_divergence_simulator(circuit_spec, num_samples))
        return results
    
    @staticmethod
    def calculate_from_circuit_specs_divergence_simulator(circuit_spec: CircuitSpec, 
                                    num_samples: int = 50) -> Dict[str, float]:
        """
        회로 사양으로부터 표현력을 계산합니다.
        상태벡터 시뮬레이터를 사용하여 랜덤 파라미터화된 회로 간의 피델리티를 계산하고
        Haar 랜덤 분포와 비교합니다.
        
        Args:
            circuit_spec: 기본 회로 사양 (파라미터화된 회로)
            num_qubits: 큐빗 수
            num_samples: 생성할 랜덤 파라미터 샘플 수
            
        Returns:
            표현력 계산 결과 딕셔너리
        """
        # 랜덤 파라미터화된 회로에서 페어와이즈 피델리티 계산
        fidelities = StatevectorFidelityCalculator.generate_pairwise_fidelities(
            circuit_spec, num_samples)
        return Divergence_Expressibility._cal_fidelity_divergence(fidelities, circuit_spec.num_qubits)

    @staticmethod
    def _cal_fidelity_divergence(fidelities: List[float], num_qubits: int):
        try:
         # Haar 랜덤 피델리티 생성 (비교용)
            num_pairs = len(fidelities)
            haar_fidelities = Divergence_Expressibility.generate_haar_random_fidelities(num_qubits, num_pairs)
            
            # 히스토그램 계산을 위한 빈(bin) 정의 - 충분한 해상도 제공
            num_bins = max(50, min(100, num_pairs * 10))  # 적응적 빈 수, 최소 50개
            bins = np.linspace(0, 1, num_bins)
            
            # 히스토그램 계산
            circuit_hist, _ = np.histogram(fidelities, bins=bins, density=True)
            haar_hist, _ = np.histogram(haar_fidelities, bins=bins, density=True)
            
            # 히스토그램 정규화
            circuit_hist = circuit_hist / np.sum(circuit_hist)
            haar_hist = haar_hist / np.sum(haar_hist)
            
            # 0이 아닌 값만 고려하여 KL 다이버전스 계산
            # 범위 제한 강화 - 너무 큰 값이나 작은 값 방지
            circuit_hist = np.clip(circuit_hist, 1e-10, 1e10)  # 0과 극단적 큰 값 방지
            haar_hist = np.clip(haar_hist, 1e-10, 1e10)  # 0과 극단적 큰 값 방지
            #Divergence_Expressibility.plot_overlapping_histograms(circuit_hist, haar_hist)
            
            try:
                # KL 다이버전스 계산: KL(circuit || haar) & KL(haar || circuit)
                kl_div_circuit_haar = float(entropy(circuit_hist, haar_hist))
                # 값 제한 - 수치 안정성 보장
                kl_div_circuit_haar = min(kl_div_circuit_haar, 1e6)  # 너무 큰 값 방지
                
                kl_div_haar_circuit = float(entropy(haar_hist, circuit_hist))
                kl_div_haar_circuit = min(kl_div_haar_circuit, 1e6)  # 너무 큰 값 방지
                
                # JS 다이버전스 계산
                m_dist = 0.5 * (circuit_hist + haar_hist)
                js_div = float(0.5 * entropy(circuit_hist, m_dist) + 0.5 * entropy(haar_hist, m_dist))
                js_div = min(js_div, 1e6)  # 너무 큰 값 방지
                
                # L2 노름 계산
                l2_dist = float(euclidean(circuit_hist, haar_hist) / np.sqrt(len(circuit_hist)))
                l2_dist = min(l2_dist, 1e6)  # 너무 큰 값 방지
                
                # 표현력 지표 (1 / (1 + KL)) - 값이 1에 가까울수록 Haar와 유사
                expressibility = float(1.0 / (1.0 + kl_div_circuit_haar))
            except Exception as e:
                print(f"다이버전스 계산 중 오류: {e}")
                kl_div_circuit_haar = 1e6
                kl_div_haar_circuit = 1e6
                js_div = 1e6
                l2_dist = 1e6
                expressibility = 0.0  # 오류 발생 시 최악의 표현력으로 설정
            
            return {
                "expressibility": expressibility,
                "kl_divergence": kl_div_circuit_haar,
                "kl_divergence_reverse": kl_div_haar_circuit,
                "js_divergence": js_div,
                "l2_norm": l2_dist,
                "valid_samples": len(fidelities),
            }
        except Exception as e:
            return {
                "expressibility": float('nan'),
                "kl_divergence": float('nan'),
                "js_divergence": float('nan'),
                "l2_norm": float('nan'),
                "valid_samples": 0,
                "error": str(e)
            }
        
           
    
    @staticmethod
    def expressibility_l2_norm(fidelities: List[float], 
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

        num_pairs = len(fidelities)
        # Haar random 샘플 생성
        haar_fidelities = Divergence_Expressibility.generate_haar_random_fidelities(
            num_qubits, num_pairs
        )
        
        # 각각의 표현력 계산
        actual_result = Divergence_Expressibility.calculate_from_fidelities(
            fidelities, num_qubits
        )
        haar_result = Divergence_Expressibility.calculate_from_fidelities(
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


def calculate_experiment_expressibility(circuit_spec: CircuitSpec, num_qubits: int) -> Dict[str, float]:
    """
    스펙객체로부터 표현력을 계산하는 함수
    """
    return Divergence_Expressibility.calculate_from_circuit_specs_divergence(circuit_spec, num_qubits)


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
    expr_result = Divergence_Expressibility.calculate_from_fidelities(fidelities, num_qubits)

    # 결과 데이터에 메타데이터 추가
    expr_result['num_random_params'] = num_random_params
    expr_result['num_qubits'] = num_qubits

    return expr_result
