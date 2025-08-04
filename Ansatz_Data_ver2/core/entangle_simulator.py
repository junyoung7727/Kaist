import numpy as np
import os
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from typing import Dict, List, Any, Union, Optional
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, entropy, partial_trace, purity
from core.grid_graph_encoder import create_simple_circuit_example
from core.circuit_interface import CircuitSpec
from core.qiskit_circuit import QiskitQuantumCircuit
from qiskit_aer import AerSimulator

def meyer_wallace_entropy(circuit: CircuitSpec) -> float:
    """ Meyer-Wallace Entropy 계산
    
    각 큐빗에 대한 축소된 밀도 행렬의 순도(purity)를 계산하고,
    이를 바탕으로 Meyer-Wallace 얽힘 엔트로피를 계산합니다.
    
    Args:
        circuit: 양자 회로 사양
        
    Returns:
        Meyer-Wallace 얽힘 엔트로피 (0에서 1 사이의 값)
    """
    qiskit_circuit = QiskitQuantumCircuit(circuit).build()
    qc = qiskit_circuit.qiskit_circuit
    statevector_data = Statevector(qc)
    full_rho = DensityMatrix(statevector_data)
    
    sum_purities = 0
    n = qiskit_circuit.num_qubits
    
    for j in range(n):
        # 매 반복마다 원본 density matrix에서 시작하여 j번째 큐빗만 남김
        reduced_rho_j = partial_trace(full_rho, [j])
        purity_j = reduced_rho_j.purity().real  # 복소수의 실수 부분만 사용
        sum_purities += purity_j

    # 기존 구현: 잘못된 공식 사용 (메모리에서 수정된 공식 확인)
    # meyer_wallace_entropy = 2 * (1 - (1/n) * sum_purities)
    
    # 올바른 Meyer-Wallach 공식
    average_purity = sum_purities / n
    meyer_wallace_entropy = 2 * (1 - average_purity)
    return meyer_wallace_entropy
    
def debug_meyer_wallace():
    """Meyer-Wallace 얽힘 측정 디버깅 함수"""
    circuit = create_simple_circuit_example()
    print(f"테스트 회로: {circuit.num_qubits}큐빗, {len(circuit.gates)}게이트")
    
    # 회로 상태 출력
    qiskit_circuit = QiskitQuantumCircuit(circuit).build()
    qc = qiskit_circuit.qiskit_circuit
    #print(f"회로 구조:\n{qc}")
    
    # 상태벡터 계산
    statevector = Statevector(qc)
    # print(f"\n상태벡터 크기: {len(statevector.data)}")
    full_rho = DensityMatrix(statevector)
    
    # 각 큐빗에 대한 축소된 밀도 행렬 확인
    n = circuit.num_qubits
    sum_purities = 0
    
    vec = []
    #print("\n각 큐빗의 축소된 밀도 행렬:")
    for j in range(n):
        # j번째 큐빗만 남기고 나머지 트레이스
        reduced_rho_j = partial_trace(full_rho, [j])
        purity_j = reduced_rho_j.purity().real
        print(f"\n큐빗 {j}의 축소된 밀도 행렬:")
        print(reduced_rho_j)
        print(f"순도(purity): {purity_j}")
        sum_purities += purity_j
    #print(f"\n각 큐빗의 순도 합: {sum_purities}")
    # 최종 계산
    meyer_wallace_value = (1 - (1/n) * sum_purities)
    #print(f"\n최종 Meyer-Wallace 얽힘 측정값: {meyer_wallace_value}")
    return meyer_wallace_value

#debug_meyer_wallace()