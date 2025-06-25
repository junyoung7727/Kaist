#!/usr/bin/env python3
"""
시뮬레이터 기반 표현력(Expressibility) 계산 모듈

이 모듈은 양자 시뮬레이터를 사용한 상태벡터 및 피델리티 기반 표현력 계산 기능을 제공합니다.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Union

# 내부 모듈 임포트
from src.calculators.expressibility.base import ExpressibilityCalculatorBase
from src.config import config
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity


def haar_fidelity_distribution(n_qubits, F_values):
    """Haar random state의 이론적 피델리티 분포"""
    N = 2**n_qubits
    return (N - 1) * (1 - F_values)**(N - 2)


def calculate_kl_divergence(measured_hist, theoretical_hist, bin_widths):
    """KL divergence 계산"""
    epsilon = 1e-10
    
    # 0인 값들을 epsilon으로 대체
    measured_hist = np.maximum(measured_hist, epsilon)
    theoretical_hist = np.maximum(theoretical_hist, epsilon)
    
    # 정규화
    measured_hist = measured_hist / np.sum(measured_hist * bin_widths)
    theoretical_hist = theoretical_hist / np.sum(theoretical_hist * bin_widths)
    
    # KL divergence 계산
    kl_div = np.sum(measured_hist * np.log(measured_hist / theoretical_hist) * bin_widths)
    
    return kl_div


def create_parametrized_circuit_from_info(circuit_info, n_qubits, random_params=True):
    """circuit_info를 기반으로 매개변수화된 양자회로 생성"""
    qc = QuantumCircuit(n_qubits)
    
    gates = circuit_info.get("gates", [])
    wires_list = circuit_info.get("wires_list", [])
    params = circuit_info.get("params", [])
    params_idx = circuit_info.get("params_idx", [])
    
    # 랜덤 매개변수 생성
    if random_params and len(params) > 0:
        random_param_values = np.random.uniform(0, 2*np.pi, len(params))
    else:
        random_param_values = params
    
    param_counter = 0
    
    for i, gate in enumerate(gates):
        if i >= len(wires_list):
            break
            
        wires = wires_list[i]
        
        try:
            if gate.lower() == 'h':
                qc.h(wires[0])
            elif gate.lower() == 'x':
                qc.x(wires[0])
            elif gate.lower() == 'y':
                qc.y(wires[0])
            elif gate.lower() == 'z':
                qc.z(wires[0])
            elif gate.lower() == 'rx':
                if param_counter < len(random_param_values):
                    qc.rx(random_param_values[param_counter], wires[0])
                    param_counter += 1
            elif gate.lower() == 'ry':
                if param_counter < len(random_param_values):
                    qc.ry(random_param_values[param_counter], wires[0])
                    param_counter += 1
            elif gate.lower() == 'rz':
                if param_counter < len(random_param_values):
                    qc.rz(random_param_values[param_counter], wires[0])
                    param_counter += 1
            elif gate.lower() == 'cx' and len(wires) >= 2:
                qc.cx(wires[0], wires[1])
            elif gate.lower() == 'cy' and len(wires) >= 2:
                qc.cy(wires[0], wires[1])
            elif gate.lower() == 'cz' and len(wires) >= 2:
                qc.cz(wires[0], wires[1])
        except Exception:
            # 잘못된 게이트나 큐빗 인덱스는 무시
            continue
    
    return qc


def _calculate_statevector_expressibility(circuit_info: Dict[str, Any], num_samples: Optional[int] = None, num_bins: Optional[int] = None) -> Dict[str, Any]:
    """
    상태벡터 기반 표현력 계산 (개선된 버전)
    
    Args:
        circuit_info: 회로 정보 딕셔너리
        num_samples: 샘플 수 (기본값: config에서 가져옴)
        num_bins: 히스토그램 빈 수 (기본값: config에서 가져옴)
    
    Returns:
        표현력 결과 딕셔너리
    """
    n_qubits = circuit_info.get("n_qubits", 4)
    
    # 설정값 가져오기
    if num_samples is None:
        num_samples = getattr(config.simulator, "fidelity_shots", 256)
    if num_bins is None:
        num_bins = getattr(config.simulator, "fidelity_kl_num_bins", 100)
    
    # 피델리티 분포 생성 (같은 회로 구조, 다른 매개변수)
    fidelities = []
    
    for i in range(num_samples):
        try:
            # 두 개의 서로 다른 랜덤 매개변수로 회로 생성
            qc1 = create_parametrized_circuit_from_info(circuit_info, n_qubits, random_params=True)
            qc2 = create_parametrized_circuit_from_info(circuit_info, n_qubits, random_params=True)
            
            # 상태 계산
            state1 = Statevector.from_instruction(qc1)
            state2 = Statevector.from_instruction(qc2)
            
            # 피델리티 계산
            fidelity = state_fidelity(state1, state2)
            fidelities.append(fidelity)
            
        except Exception as e:
            # 오류가 발생한 샘플은 건너뛰기
            continue
    
    if len(fidelities) < 10:  # 너무 적은 샘플
        return {
            "expressibility_value": float('nan'),
            "method": "statevector_fidelity_kl",
            "n_qubits": n_qubits,
            "num_samples": len(fidelities),
            "error": "Too few valid samples generated"
        }
    
    # 히스토그램 구성
    hist_measured, bin_edges = np.histogram(fidelities, bins=num_bins, range=(0, 1), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = np.diff(bin_edges)
    
    # 이론적 Haar 분포 계산
    haar_theoretical = haar_fidelity_distribution(n_qubits, bin_centers)
    
    # KL divergence 계산
    kl_divergence = calculate_kl_divergence(hist_measured, haar_theoretical, bin_widths)
    
    return {
        "expressibility_value": float(kl_divergence),
        "method": "statevector_fidelity_kl",
        "n_qubits": n_qubits,
        "num_samples": len(fidelities),
        "num_bins": num_bins,
        "histogram_measured": list(map(float, hist_measured)),
        "histogram_haar": list(map(float, haar_theoretical)),
        "bin_centers": list(map(float, bin_centers)),
        "fidelities": list(map(float, fidelities)),
    }


class SimulatorExpressibilityCalculator(ExpressibilityCalculatorBase):
    """
    시뮬레이터 기반 표현력 계산기
    
    상태벡터와 피델리티를 사용한 표현력 계산을 구현합니다.
    """
    
    def __init__(self, seed=None):
        """
        시뮬레이터 표현력 계산기 초기화
        
        Args:
            seed (Optional[int]): 난수 생성기 시드
        """
        super().__init__(seed=seed)
        self.config = config

    def calculate_expressibility(self, circuit_info: Dict[str, Any], num_samples: Optional[int] = None, 
                               num_bins: Optional[int] = None) -> Dict[str, Any]:
        """
        양자 회로의 표현력 계산 - 상태벡터/피델리티 기반 방법
        
        Args:
            circuit_info (Dict[str, Any]): 회로 정보
            num_samples (Optional[int]): 피델리티 계산을 위한 샘플 수
            num_bins (int): 히스토그램 빈 수
            
        Returns:
            Dict[str, Any]: 표현력 계산 결과
        """
        start_time = time.time()
        
        # 상태벡터 기반 표현력 계산 (Haar 분포와의 KL 발산 사용)
        result = _calculate_statevector_expressibility(circuit_info, num_samples, num_bins)
        
        # 실행 시간 추가
        execution_time = time.time() - start_time
        result["execution_time"] = execution_time
        
        print(f"✅ 상태벡터 기반 표현력 계산 완료: {result['expressibility_value']:.6f}")
        
        return result
        
    # 클래식 섀도우 관련 메서드 제거됨 - ibm.py 모듈에서 구현
