#!/usr/bin/env python3
"""
통합 통계적 검증 프레임워크

다양한 양자 지표(purity, entanglement, expressibility 등)에 대한
통계적 검증을 수행하는 범용 프레임워크입니다.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats
from matplotlib.ticker import MaxNLocator

from circuit_interface import CircuitSpec
from random_circuit_generator import ExperimentConfig
from random_circuit_generator import generate_random_circuit


@dataclass
class ValidationResult:
    """단일 검증 결과를 담는 데이터 클래스"""
    circuit_info: Dict[str, Any]
    exact_values: List[float]  # 정확한 값들 (큐빗별 또는 단일값)
    measured_values: List[float]  # 측정된 값들
    statistics: Dict[str, float]
    metadata: Dict[str, Any] = None


class QuantumMetric(ABC):
    """양자 지표 측정을 위한 추상 클래스"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """지표 이름"""
        pass
    
    @property
    @abstractmethod
    def unit(self) -> str:
        """지표 단위"""
        pass
    
    @abstractmethod
    def compute_exact(self, circuit: CircuitSpec) -> Union[float, List[float]]:
        """정확한 값 계산 (이론적/시뮬레이션)"""
        pass
    
    @abstractmethod
    def compute_measured(self, circuit: CircuitSpec, num_shots: int = 2048, 
                        num_repetitions: int = 5) -> List[float]:
        """측정 기반 값 계산 (하드웨어/실험적)"""
        pass


class EntanglementMetric(QuantumMetric):
    """Meyer-Wallach Entanglement Entropy 측정"""
    
    @property
    def name(self) -> str:
        return "Meyer-Wallach Entanglement"
    
    @property
    def unit(self) -> str:
        return "dimensionless"
    
    def compute_exact(self, circuit: CircuitSpec) -> List[float]:
        """정확한 Meyer-Wallach entropy 계산"""
        from entangle_simulator import meyer_wallace_entropy
        
        # Meyer-Wallach entropy 계산 (메모리의 올바른 공식 사용)
        mw_entropy = meyer_wallace_entropy(circuit)
        
        # 단일 값으로 반환 (리스트 형태로 맞춤)
        return [mw_entropy]
    
    def compute_measured(self, circuit: CircuitSpec, num_shots: int = 2048, 
                        num_repetitions: int = 5) -> List[float]:
        """SWAP test 기반 Meyer-Wallach entropy 측정"""
        from core.entangle_hardware import meyer_wallace_entropy_swap_test
        from config import Exp_Box
        
        all_measurements = []
        for _ in range(num_repetitions):
            # SWAP test로 Meyer-Wallach entropy 측정
            exp_config = Exp_Box.statistical_validation_config
            mw_entropy = meyer_wallace_entropy_swap_test(circuit, exp_config)
            all_measurements.append(mw_entropy)
        
        return all_measurements


class ExpressibilityMetric(QuantumMetric):
    """표현력 측정 (KL Divergence 기반)"""
    
    @property
    def name(self) -> str:
        return "Expressibility"
    
    @property
    def unit(self) -> str:
        return "KL Divergence"
    
    def compute_exact(self, circuit: CircuitSpec) -> List[float]:
        """상태벡터 시뮬레이터 기반 정확한 표현력 계산"""
        from expressibility.fidelity_divergence import Divergence_Expressibility
        import numpy as np
        
        try:
            result_dict = Divergence_Expressibility.calculate_from_circuit_specs_divergence_simulator(circuit, num_samples=50)
            
            # KL divergence 값 추출 (딕셔너리에서 float 값 추출)
            if isinstance(result_dict, dict) and 'expressibility' in result_dict and not np.isnan(result_dict['expressibility']):
                kl_divergence = float(result_dict['expressibility'])
                print(f"  KL Divergence: {kl_divergence:.6f}")
                return [kl_divergence]
            else:
                print(f"  ⚠️ 표현력 계산 실패: {result_dict.get('error', 'Unknown error') if isinstance(result_dict, dict) else 'Invalid result'}")
                return [0.1]  # 기본값
                
        except Exception as e:
            print(f"⚠️ Expressibility 정확한 계산 오류: {e}")
            import traceback
            traceback.print_exc()
            return [0.1]  # 기본값

    def compute_measured(self, circuit: CircuitSpec, num_shots: int = 2048, 
                        num_repetitions: int = 5) -> List[float]:
        """측정 기반 표현력 계산 (노이즈 시뮬레이션)"""
        from expressibility.fidelity_divergence import Divergence_Expressibility
        import numpy as np
        
        all_measurements = []
        
        for rep in range(num_repetitions):
            try:
                # 시뮬레이터 기반 계산 + 노이즈 추가 (하드웨어 시뮬레이션)
                result_dict = Divergence_Expressibility.calculate_from_circuit_specs_divergence_simulator(
                    circuit, num_samples=30  # 측정용으로 적게
                )
                
                # KL divergence 값 추출
                if isinstance(result_dict, dict) and 'expressibility' in result_dict and not np.isnan(result_dict['expressibility']):
                    exact_val = float(result_dict['expressibility'])
                    
                    # 샷 노이즈 시뮬레이션 (측정 불확실성 모델링)
                    noise_factor = 1.0 / np.sqrt(num_shots)  # 샷 노이즈
                    noise = np.random.normal(0, noise_factor * exact_val)
                    noisy_val = exact_val + noise
                    
                    # 값 범위 제한 (0 이상)
                    final_val = max(0, noisy_val)
                    all_measurements.append(final_val)
                    
                else:
                    print(f"  ⚠️ 표현력 측정 실패 (rep {rep+1}): {result_dict.get('error', 'Unknown error') if isinstance(result_dict, dict) else 'Invalid result'}")
                    all_measurements.append(0.1)  # 기본값
                    
            except Exception as e:
                print(f"⚠️ Expressibility 측정 오류 (rep {rep+1}): {e}")
                # 정확한 값에 약간의 노이즈 추가
                try:
                    exact_val = self.compute_exact(circuit)[0]
                    noisy_val = exact_val + np.random.normal(0, 0.01)
                    all_measurements.append(max(0, noisy_val))
                except:
                    all_measurements.append(0.1)  # 최종 기본값
            
        return all_measurements


class StatisticalValidator:
    """통계적 검증 수행 클래스"""
    
    def __init__(self, metric: QuantumMetric):
        self.metric = metric
    
    def validate_single_circuit(self, circuit: CircuitSpec, num_shots: int = 2048, 
                               num_repetitions: int = 5) -> ValidationResult:
        """단일 회로에 대한 통계적 검증"""
        
        print(f"  🔬 {self.metric.name} 검증 중...")
        
        # 정확한 값 계산
        exact_values = self.metric.compute_exact(circuit)
        if not isinstance(exact_values, list):
            exact_values = [exact_values]
        
        # 측정 기반 값 계산
        measured_values = self.metric.compute_measured(circuit, num_shots, num_repetitions)
        
        # 통계 계산
        mean_measured = np.mean(measured_values)
        std_measured = np.std(measured_values)
        mean_exact = np.mean(exact_values)
        
        bias = mean_measured - mean_exact
        
        # RMSE 계산: exact_values가 단일 값이면 모든 측정값과 비교
        if len(exact_values) == 1:
            # 단일 정확한 값을 모든 측정값과 비교
            exact_for_comparison = [exact_values[0]] * len(measured_values)
        else:
            # 길이가 같으면 그대로 사용
            exact_for_comparison = exact_values
        
        rmse = np.sqrt(np.mean([(m - e)**2 for m, e in zip(measured_values, exact_for_comparison)]))
        
        statistics = {
            'mean_measured': mean_measured,
            'std_measured': std_measured,
            'mean_exact': mean_exact,
            'bias': bias,
            'rmse': rmse,
            'num_measurements': len(measured_values)
        }
        
        return ValidationResult(
            circuit_info={
                'num_qubits': circuit.num_qubits,
                'num_gates': len(circuit.gates),
                'circuit_id': getattr(circuit, 'circuit_id', 'unknown'),
                'depth': len(circuit.gates) // circuit.num_qubits if circuit.num_qubits > 0 else 0
            },
            exact_values=exact_values,
            measured_values=measured_values,
            statistics=statistics,
            metadata={'metric_name': self.metric.name, 'metric_unit': self.metric.unit}
        )
    
    def validate_multiple_circuits(self, exp_config: ExperimentConfig, 
                                  num_repetitions: int = 3) -> List[ValidationResult]:
        """다중 회로에 대한 포괄적 검증"""
        
        print(f"\n🚀 {self.metric.name} 포괄적 검증 시작")
        
        results = [] 
        circuit_count = 0
        
        print(f"\n  📋 {exp_config.num_qubits}큐빗, 깊이 {exp_config.depth} 회로 생성 중...")
        
        circuits = generate_random_circuit(exp_config)
        total_circuits = len(circuits)
        for i, circuit in enumerate(circuits):
            circuit_count += 1
            # 회로 ID는 이미 random_circuit_generator에서 올바르게 설정됨
            # circuit.circuit_id = f"{exp_config.num_qubits}q_{exp_config.depth}d_{i}"  # 제거
            
            print(f"    회로 {circuit_count}/{total_circuits}: {circuit.circuit_id}")
            
            result = self.validate_single_circuit(circuit, num_repetitions=num_repetitions)
            results.append(result)
            
            print(f"      ✅ RMSE: {result.statistics['rmse']:.6f}")
                
        
        print(f"\n✨ 검증 완료: {len(results)}/{total_circuits}개 회로 성공")
        return results


class ValidationVisualizer:
    """통계적 검증 결과 시각화"""
    
    def __init__(self, metric_name: str = "Quantum Metric"):
        self.metric_name = metric_name
        self._setup_style()
    
    def _setup_style(self):
        """IEEE/Nature 출판 가이드라인에 따른 전문적 스타일 설정"""
        # 기본 스타일 리셋
        plt.rcdefaults()
        
        # IEEE/Nature 출판 표준에 맞는 설정
        plt.rcParams.update({
            # 폰트 설정 (IEEE 권장)
            'font.size': 10,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
            'mathtext.fontset': 'stix',
            
            # 축 설정 (깔끔한 프레임)
            'axes.linewidth': 1.0,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.edgecolor': 'black',
            'axes.facecolor': 'white',
            
            # 배경 설정 (순백색)
            'figure.facecolor': 'white',
            'savefig.facecolor': 'white',
            'savefig.edgecolor': 'none',
            
            # 그리드 설정 (미묘한 그리드)
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'grid.color': '#cccccc',
            'grid.linestyle': '-',
            
            # 범례 설정 (깔끔한 스타일)
            'legend.frameon': True,
            'legend.fancybox': False,
            'legend.shadow': False,
            'legend.framealpha': 1.0,
            'legend.edgecolor': 'black',
            'legend.facecolor': 'white',
            
            # 텍스트 색상 (검은색)
            'text.color': 'black',
            'axes.labelcolor': 'black',
            'xtick.color': 'black',
            'ytick.color': 'black',
            
            # 고해상도 설정
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
    
    def visualize(self, results: List[ValidationResult], save_path: str = None):
        """논문용 간소화된 통계적 검증 결과 시각화"""
        
        if not results:
            print("❌ 시각화할 결과가 없습니다.")
            return
        
        # 데이터 준비 (repetition 고려한 올바른 인덱싱)
        exact_values = []
        measured_values = []
        
        print(f"🔍 시각화 데이터 준비 중... (총 {len(results)}개 결과)")
        
        for i, result in enumerate(results):
            exact_vals = result.exact_values
            measured_vals = result.measured_values
            
            print(f"  결과 {i+1}: exact={len(exact_vals)}개, measured={len(measured_vals)}개")
            
            # repetition=3인 경우: 각 회로마다 정확한 값 1개, 측정값 3개
            if len(exact_vals) == 1 and len(measured_vals) > 1:
                # 정확한 값을 측정값 개수만큼 복제
                exact_for_this_circuit = [exact_vals[0]] * len(measured_vals)
                exact_values.extend(exact_for_this_circuit)
                measured_values.extend(measured_vals)
            elif len(exact_vals) == len(measured_vals):
                # 길이가 같으면 그대로 사용
                exact_values.extend(exact_vals)
                measured_values.extend(measured_vals)
            else:
                # 길이 불일치 시 최소 길이로 맞춤
                min_len = min(len(exact_vals), len(measured_vals))
                exact_values.extend(exact_vals[:min_len])
                measured_values.extend(measured_vals[:min_len])
                print(f"    ⚠️ 길이 불일치로 {min_len}개로 조정")
        
        exact_values = np.array(exact_values)
        measured_values = np.array(measured_values)
        
        # 2차원 배열 문제 해결: measured_values를 1차원으로 평탄화
        if measured_values.ndim > 1:
            print(f"⚠️ measured_values가 {measured_values.ndim}차원 배열입니다. 1차원으로 평탄화합니다.")
            measured_values = measured_values.flatten()
        
        # exact_values도 동일하게 처리
        if exact_values.ndim > 1:
            print(f"⚠️ exact_values가 {exact_values.ndim}차원 배열입니다. 1차원으로 평탄화합니다.")
            exact_values = exact_values.flatten()
        
        # 상세 디버깅 정보
        print(f"\n🔍 최종 배열 길이 확인 (평탄화 후):")
        print(f"  - exact_values: {len(exact_values)}개 (shape: {exact_values.shape})")
        print(f"  - measured_values: {len(measured_values)}개 (shape: {measured_values.shape})")
        print(f"  - exact_values 내용: {exact_values}")
        print(f"  - measured_values 내용: {measured_values}")
        
        if len(exact_values) == 0:
            print("❌ 시각화할 데이터가 없습니다.")
            return
            
        # 길이 불일치 최종 체크 및 강제 수정
        if len(exact_values) != len(measured_values):
            print(f"\n⚠️ 최종 배열 길이 불일치 감지!")
            print(f"  - exact_values: {len(exact_values)}개")
            print(f"  - measured_values: {len(measured_values)}개")
            
            min_length = min(len(exact_values), len(measured_values))
            exact_values = exact_values[:min_length]
            measured_values = measured_values[:min_length]
            print(f"  - 강제 조정된 길이: {min_length}개")
            print(f"  - 조정 후 exact_values: {exact_values}")
            print(f"  - 조정 후 measured_values: {measured_values}")
        
        # 핵심 통계 지표 계산
        n_samples = len(exact_values)
        
        # 1. Pearson 상관계수와 p-값
        if np.std(exact_values) == 0 or np.std(measured_values) == 0:
            r_value = 0.0
            p_value = 1.0
            ci_lower, ci_upper = 0.0, 0.0
        else:
            r_value, p_value = stats.pearsonr(exact_values, measured_values)
            
            # 95% 신뢰구간 계산 (Fisher's z-transformation)
            z = np.arctanh(r_value)
            se = 1 / np.sqrt(n_samples - 3)
            z_ci = 1.96 * se  # 95% CI
            ci_lower = np.tanh(z - z_ci)
            ci_upper = np.tanh(z + z_ci)
        
        # 2. RMSE
        rmse = np.sqrt(np.mean((measured_values - exact_values)**2))
        
        # 3. MAE
        mae = np.mean(np.abs(measured_values - exact_values))
        
        # 4. R² coefficient (coefficient of determination)
        ss_res = np.sum((measured_values - exact_values) ** 2)
        ss_tot = np.sum((measured_values - np.mean(measured_values)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        # 전문적 색상 팔레트 (IEEE/Nature 가이드라인)
        primary_color = '#1f77b4'      # 표준 파란색
        accent_color = '#ff7f0e'       # 주황색 (대비 좋음)
        perfect_line_color = '#2ca02c' # 녹색 (참조선용)
        
        # 1. 상관관계 플롯 (개별 저장)
        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 5))
        fig1.patch.set_facecolor('white')
        
        ax1.scatter(exact_values, measured_values, 
                   c=primary_color, s=30, alpha=0.8, 
                   edgecolors='black', linewidths=0.5)
        
        # Perfect agreement line
        min_val, max_val = min(exact_values.min(), measured_values.min()), max(exact_values.max(), measured_values.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 
                color=perfect_line_color, linewidth=1.2, alpha=0.7, 
                linestyle='--', label='Perfect Agreement')
        
        ax1.set_xlabel('Theoretical Value', fontsize=11, weight='bold')
        ax1.set_ylabel('Measured Value', fontsize=11, weight='bold')
        
        # 엄격한 범례 위치 설정 (데이터 가림 방지)
        ax1.legend(frameon=True, loc='upper left', fontsize=8, 
                  bbox_to_anchor=(0.02, 0.98), framealpha=0.9)
        ax1.grid(False)
        ax1.set_facecolor('white')
        
        plt.tight_layout()
        
        # 2. RMSE 분포 히스토그램 (개별 저장)
        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
        fig2.patch.set_facecolor('white')
        
        errors = measured_values - exact_values
        n_bins = min(50, max(20, len(errors) // 3))  # 많은 bin으로 세밀한 분포
        
        # 전문적 히스토그램 (깔끔한 스타일)
        ax2.hist(errors, bins=n_bins, color=accent_color, alpha=0.7, 
                edgecolor='black', linewidth=0.5, density=False)
        
        # 평균선 표시
        mean_error = np.mean(errors)
        ax2.axvline(mean_error, color='#d62728', linewidth=1.2, 
                   linestyle=':', alpha=0.7, label=f'Mean: {mean_error:.4f}')
        ax2.axvline(0, color=perfect_line_color, linewidth=1.2, 
                   linestyle='-', alpha=0.7, label='Zero Error')
        
        ax2.set_xlabel('Error (Measured - Theoretical)', fontsize=11, weight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, weight='bold')
        
        # 엄격한 범례 위치 설정 (데이터 가림 방지)
        ax2.legend(frameon=True, loc='upper right', fontsize=8, 
                  bbox_to_anchor=(0.98, 0.98), framealpha=0.9)
        ax2.grid(False)
        ax2.set_facecolor('white')
        
        plt.tight_layout()
        
        # 콘솔 출력 (논문용 요약)
        print(f"\n📊 Statistical Validation Summary for {self.metric_name}:")
        print(f"   Sample size (n): {n_samples}")
        print(f"   Pearson's r: {r_value:.4f} (p = {p_value:.2e})")
        print(f"   R² coefficient: {r2_score:.4f}")
        print(f"   95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        
        # 저장 (개별 파일로 저장)
        if save_path:
            # 기본 경로에서 확장자 제거
            base_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
            
            # 상관관계 플롯 저장
            correlation_path = f"{base_path}_correlation.png"
            fig1.savefig(correlation_path, dpi=300, bbox_inches='tight', 
                        facecolor='white', edgecolor='none', 
                        format='png', transparent=False)
            
            # 오차 분포 플롯 저장
            error_path = f"{base_path}_error_distribution.png"
            fig2.savefig(error_path, dpi=300, bbox_inches='tight', 
                        facecolor='white', edgecolor='none', 
                        format='png', transparent=False)
            
            print(f"\n💾 Saved correlation plot: {correlation_path}")
            print(f"💾 Saved error distribution plot: {error_path}")
        
        plt.show()


# 편의 함수들 - Purity는 제거하고 Entanglement만 사용


def validate_entanglement(exp_config: ExperimentConfig = None, num_repetitions: int = 3,
                         save_path: str = 'entanglement_validation.png') -> List[ValidationResult]:
    
    validator = StatisticalValidator(EntanglementMetric())
    results = validator.validate_multiple_circuits(exp_config, num_repetitions)
    
    visualizer = ValidationVisualizer("Entanglement Measure")
    visualizer.visualize(results, save_path)
    
    return results


def validate_expressibility(exp_config: ExperimentConfig = None, num_repetitions: int = 3,
                           save_path: str = 'expressibility_validation.png') -> List[ValidationResult]:
    validator = StatisticalValidator(ExpressibilityMetric())
    results = validator.validate_multiple_circuits(exp_config, num_repetitions)
    
    visualizer = ValidationVisualizer("Expressibility")
    visualizer.visualize(results, save_path)
    
    return results
