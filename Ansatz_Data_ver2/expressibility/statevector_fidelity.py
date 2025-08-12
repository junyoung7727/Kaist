#!/usr/bin/env python3
"""
상태벡터 기반 피델리티 계산 모듈

상태벡터 시뮬레이터를 사용하여 회로 간의 피델리티를 계산하는 로직입니다.
두 상태 간의 피델리티는 내적의 절대값 제곱(|⟨ψ₁|ψ₂⟩|²)으로 계산됩니다.
"""

from typing import List, Dict, Tuple, Any
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, state_fidelity
from core.circuit_interface import CircuitSpec
from core.qiskit_circuit import QiskitQuantumCircuit
from core.random_circuit_generator import create_random_parameterized_samples
import matplotlib.pyplot as plt

class StatevectorFidelityCalculator:
    """
    상태벡터 기반 피델리티 계산기
    
    상태벡터 시뮬레이터를 사용하여 양자 회로 간의 피델리티를 계산합니다.
    피델리티는 |⟨ψ₁|ψ₂⟩|²로 정의됩니다.
    """
    
    @staticmethod
    def calculate_statevector(circuit_spec: CircuitSpec) -> np.ndarray:
        """
        회로 사양으로부터 상태벡터 계산
        
        Args:
            circuit_spec: 회로 사양
            
        Returns:
            np.ndarray: 상태벡터
        """
        # Qiskit 회로 생성
        qiskit_circuit = QiskitQuantumCircuit(circuit_spec).build()
        try:
            # 상태벡터 시뮬레이터로 실행
            qc = qiskit_circuit._qiskit_circuit
            statevector = Statevector(qc)
            #qc.draw('mpl')
            #plt.show()
            
            return statevector
        except Exception as e:
            print(f"Statevector 계산 중 오류 발생: {e}")
            return None
        
    @staticmethod
    def calculate_pairwise_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
        """
        두 상태벡터 간의 피델리티 계산 (|⟨ψ₁|ψ₂⟩|²)
        
        Args:
            state1: 첫 번째 상태벡터
            state2: 두 번째 상태벡터
            
        Returns:
            float: 피델리티 (0~1 사이 값)
        """
        # Qiskit의 state_fidelity 함수는 내적의 절대값 제곱을 계산
        return np.round(state_fidelity(state1, state2), 4)
    
    @staticmethod
    def generate_pairwise_fidelities(circuit_spec: CircuitSpec, num_samples: int) -> List[float]:
        """
        하나의 회로에서 랜덤 파라미터화된 샘플들 간의 페어와이즈 피델리티 계산
        
        Args:
            circuit_spec: 기본 회로 사양
            num_samples: 생성할 샘플 수
            
        Returns:
            List[float]: 페어와이즈 피델리티 리스트
        """
        # 랜덤 파라미터화된 회로 샘플 생성
        samples = create_random_parameterized_samples(circuit_spec, num_samples)
        
        # 각 샘플의 상태벡터 계산
        statevectors = []
        for sample in samples:
            sv = StatevectorFidelityCalculator.calculate_statevector(sample)
            statevectors.append(sv)
        
        # 페어와이즈 피델리티 계산
        fidelities = []
        for i in range(len(statevectors)):
            for j in range(i+1, len(statevectors)):
                fidelity = StatevectorFidelityCalculator.calculate_pairwise_fidelity(
                    statevectors[i], statevectors[j]
                )
                fidelities.append(fidelity)
                #print(f"Fidelity between sample {i} and {j}: {fidelity}")
        # if fidelities.count(1.0) > 1:
            # print(circuit_spec.circuit_id)
            # QiskitQuantumCircuit(circuit_spec).build().qiskit_circuit.draw('mpl')
            # plt.show()

        return fidelities