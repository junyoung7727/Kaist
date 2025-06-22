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
from qiskit.quantum_info import Statevector
from qiskit import transpile



def _calculate_statevector_expressibility(circuit_info: Dict[str, Any], num_samples: Optional[int] = None, num_bins: Optional[int] = None) -> Dict[str, Any]:
    """
    상태벡터 기반 표현력 계산 (피델리티 분포의 KL 발산)
    
    Args:
        circuit_info (Dict[str, Any]): 회로 정보
        num_samples (Optional[int]): 샘플 수
        num_bins (int): 히스토그램 빈 수
        
    Returns:
        Dict[str, Any]: 표현력 계산 결과
    """
    from src.core.circuit_base import QuantumCircuitBase
    
    n_qubits = circuit_info.get("n_qubits")
    
    # qubit 제한
    if n_qubits > config.simulator.max_fidelity_qubits:
        print(f"⚠️ 큐빗 수 {n_qubits} > max_fidelity_qubits, 제한: {config.simulator.max_fidelity_qubits}")
        n_qubits = config.simulator.max_fidelity_qubits
        
    # 샷 수 관련 설정 - 표준화된 직접 속성 접근 방식 사용
    if num_samples is None:
        num_samples = config.simulator.fidelity_shots
    if num_bins is None:
        num_bins = getattr(config.simulator, "fidelity_kl_num_bins", 100)  # 직접 속성 접근
        
    # 상태벡터 수집
    states = []
    for _ in range(num_samples):
        qc = QuantumCircuitBase().build_qiskit_circuit(circuit_info, n_qubits)
        qc = transpile(qc, basis_gates=None)
        state = Statevector.from_instruction(qc)
        states.append(state.data)
        
    # 피델리티 분포 생성
    pairs = num_samples
    fidelities = []
    for _ in range(pairs):
        i, j = np.random.choice(len(states), size=2, replace=False)
        fidelity = np.abs(np.sum(np.conj(states[i]) * states[j]))**2
        fidelities.append(fidelity)
        
    # Haar 랜덤 분포 (이론값)
    haar_fidelities = []
    haar_n_qubits = n_qubits
    for _ in range(pairs):
        fid = np.random.beta(1, 2**haar_n_qubits - 1)
        haar_fidelities.append(fid)
        
    # 히스토그램 구성
    hist_measured, bin_edges = np.histogram(fidelities, bins=num_bins, range=(0, 1), density=True)
    hist_haar, _ = np.histogram(haar_fidelities, bins=bin_edges, density=True)
    
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + bin_widths / 2
    
    # KL 발산 계산
    eps = 1e-10
    kl_div = np.sum(hist_measured * np.log((hist_measured + eps) / (hist_haar + eps)) * bin_widths)
    
    return {
        "expressibility_value": float(kl_div),
        "method": "statevector_fidelity_kl",
        "n_qubits": n_qubits,
        "num_samples": num_samples,
        "num_bins": num_bins,
        "histogram_measured": list(map(float, hist_measured)),
        "histogram_haar": list(map(float, hist_haar)),
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
