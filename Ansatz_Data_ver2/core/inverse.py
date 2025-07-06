#!/usr/bin/env python3
"""
역회로 생성 모듈

양자 회로의 역회로(inverse circuit)를 생성하는 순수한 수학적 로직입니다.
백엔드에 무관하게 작동하며, 추상 회로 인터페이스만 사용합니다.
"""

from typing import List
from core.circuit_interface import AbstractQuantumCircuit, CircuitSpec, GateOperation, CircuitBuilder
from core.gates import gate_registry


class InverseCircuitGenerator:
    """
    역회로 생성기
    
    주어진 회로의 수학적 역회로를 생성합니다.
    백엔드 구현에 전혀 의존하지 않습니다.
    """
    
    @staticmethod
    def create_inverse_spec(original_spec: CircuitSpec) -> CircuitSpec:
        """
        원본 회로 사양으로부터 역회로 사양을 생성합니다.
        
        Args:
            original_spec: 원본 회로 사양
            
        Returns:
            역회로 사양
        """
        # 게이트를 역순으로 배치
        reversed_gates = original_spec.gates[::-1]
        
        # 각 게이트를 역게이트로 변환
        inverse_gates = []
        for gate in reversed_gates:
            inverse_gate = InverseCircuitGenerator._create_inverse_gate(gate)
            inverse_gates.append(inverse_gate)
        
        # 역회로 사양 생성
        inverse_name = f"inverse_{original_spec.name}" if original_spec.name else None
        return CircuitSpec(
            num_qubits=original_spec.num_qubits,
            gates=inverse_gates,
            name=inverse_name
        )
    
    @staticmethod
    def _create_inverse_gate(gate: GateOperation) -> GateOperation:
        """
        단일 게이트의 역게이트를 생성합니다.
        
        Args:
            gate: 원본 게이트 연산
            
        Returns:
            역게이트 연산
        """
        # 역게이트 이름 가져오기
        inverse_name = gate_registry.get_inverse_gate_name(gate.name)
        
        # 역게이트 파라미터 계산
        inverse_parameters = []
        if gate.parameters:
            inverse_parameters = gate_registry.get_inverse_parameters(
                gate.name, gate.parameters
            )
        
        return GateOperation(
            name=inverse_name,
            qubits=gate.qubits.copy(),  # 큐빗은 동일
            parameters=inverse_parameters
        )
    
    @staticmethod
    def create_fidelity_circuit_spec(original_spec: CircuitSpec) -> CircuitSpec:
        """
        피델리티 측정을 위한 결합 회로 사양을 생성합니다.
        
        결합 회로는 다음과 같은 구조입니다:
        |0⟩ → [Reset] → [Original Circuit] → [Inverse Circuit] → [Measure]
        
        Args:
            original_spec: 원본 회로 사양
            
        Returns:
            피델리티 측정용 결합 회로 사양
        """
        builder = CircuitBuilder()
        builder.set_qubits(original_spec.num_qubits)
        builder.set_name(f"fidelity_{original_spec.name}")
        
        # 1. 모든 큐빗을 |0⟩ 상태로 초기화 (Reset 게이트)
        for qubit in range(original_spec.num_qubits):
            builder.add_gate("reset", [qubit])
        
        # 2. 원본 회로의 게이트들 추가
        for gate in original_spec.gates:
            builder.add_gate(gate.name, gate.qubits, gate.parameters)
        
        # 3. 배리어 추가 (시각적 구분용)
        builder.add_gate("barrier", list(range(original_spec.num_qubits)))
        
        # 4. 역회로의 게이트들 추가
        inverse_spec = InverseCircuitGenerator.create_inverse_spec(original_spec)
        for gate in inverse_spec.gates:
            builder.add_gate(gate.name, gate.qubits, gate.parameters)
        
        return builder.build_spec()


def create_inverse_circuit_spec(original_spec: CircuitSpec) -> CircuitSpec:
    """역회로 사양 생성 편의 함수"""
    return InverseCircuitGenerator.create_inverse_spec(original_spec)


def create_fidelity_circuit_spec(original_spec: CircuitSpec) -> CircuitSpec:
    """피델리티 측정 회로 사양 생성 편의 함수"""
    return InverseCircuitGenerator.create_fidelity_circuit_spec(original_spec)
