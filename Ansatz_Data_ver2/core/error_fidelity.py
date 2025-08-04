#!/usr/bin/env python3
"""
피델리티 계산 모듈

양자 회로의 피델리티를 계산하는 순수한 수학적 로직입니다.
백엔드에 무관하게 작동하며, 실행 결과만을 사용합니다.
"""

from typing import Dict, List, Optional, Union, TYPE_CHECKING
import numpy as np
from config import ExperimentConfig
from core.circuit_interface import CircuitSpec
from core.inverse import InverseCircuitGenerator
from core.qiskit_circuit import QiskitQuantumCircuit

if TYPE_CHECKING:
    from execution.executor import ExecutionResult



class ErrorFidelityCalculator:
    """
    에러 피델리티 계산기
    
    양자 회로의 피델리티를 다양한 방법으로 계산합니다.
    백엔드 구현에 전혀 의존하지 않습니다.
    """
    
    @staticmethod
    def calculate_from_counts(counts: Dict[str, int], num_qubits: int, shots: int) -> float:
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
        total_shots = shots
        if total_shots == 0:
            return 0.0
        
        # |00...0⟩ 상태의 카운트 (모든 큐빗이 0인 상태)
        zero_state = '0' * num_qubits
        zero_counts = counts.get(zero_state, 0)
        
        print(counts)

        # 피델리티 = P(|00...0⟩)
        fidelity = zero_counts / total_shots
        return float(fidelity)
    
    @staticmethod
    def calculate_from_execution_result(result: 'ExecutionResult', num_qubits: int, shots: int) -> float:
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
        
        return ErrorFidelityCalculator.calculate_from_counts(result.counts, num_qubits, shots)

def run_error_fidelity(circuit_specs: List[CircuitSpec], exp_config: ExperimentConfig, batch_manager=None) -> Union[float, List[int]]:
    """
    피델리티 계산을 위한 실행 함수
    
    Args:
        circuit_specs: 회로 사양 리스트
        exp_config: 실험 설정
        batch_manager: 배치 관리자 (None이면 기존 모드)
        
    Returns:
        배치 모드: 배치 인덱스 리스트
        기존 모드: 피델리티 값
    """
    from typing import Union
    
    # 역회로 생성
    inverse_circuits = []
    qiskit_circuits = []
    
    for circuit_spec in circuit_specs:
        # 역회로 스펙 생성
        inverse_circuit_spec = InverseCircuitGenerator.create_inverse_spec(circuit_spec)
        inverse_circuits.append(inverse_circuit_spec)
        
        # Qiskit 회로로 변환
        qiskit_circuit = QiskitQuantumCircuit(inverse_circuit_spec)
        qiskit_circuit.build() 
        
        # 측정 추가
        qiskit_circuit.add_measurements()
        qiskit_circuits.append(qiskit_circuit.qiskit_circuit)
    
    if batch_manager:
        # 배치 모드: 회로만 수집
        metadata = {
            "task": "fidelity",
            "circuit_count": len(circuit_specs)
        }
        indices = batch_manager.collect_task_circuits(
            "fidelity", qiskit_circuits, circuit_specs, metadata
        )
        return indices
    else:
        executor = exp_config.executor
        results = executor.execute_circuits(qiskit_circuits)
        
        # 피델리티 계산
        fidelities = []
        for result, circuit_spec in zip(results, circuit_specs):
            fidelity = ErrorFidelityCalculator.calculate_from_execution_result(
                result, circuit_spec.num_qubits, exp_config.shots
            )
            fidelities.append(fidelity)
        
        # 평균 피델리티 반환 (기존 동작 유지)
        return sum(fidelities) / len(fidelities) if fidelities else 0.0
    
def calculate_error_fidelity(counts: Dict[str, int], num_qubits: int, shots: int) -> float:
    """피델리티 계산 편의 함수"""
    return ErrorFidelityCalculator.calculate_from_counts(counts, num_qubits, shots)


def calculate_error_fidelity_from_result(result: 'ExecutionResult', num_qubits: int, shots: int) -> float:
    """실행 결과로부터 피델리티 계산 편의 함수"""
    return ErrorFidelityCalculator.calculate_from_execution_result(result, num_qubits, shots)
