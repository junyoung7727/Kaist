#!/usr/bin/env python3
"""
양자 게이트 정의 및 레지스트리

모든 양자 게이트의 정의와 역게이트 매핑을 관리합니다.
백엔드에 무관하게 순수한 수학적 정의만 포함합니다.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np


@dataclass
class GateOperation:
    """단일 게이트 연산을 나타내는 데이터 클래스"""
    name: str                    # 게이트 이름 (예: 'h', 'cx', 'rx')
    qubits: List[int]           # 적용할 큐빗 인덱스
    parameters: List[float] = None  # 파라미터 (파라메트릭 게이트용)
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = []

class GateType(Enum):
    """게이트 타입 분류"""
    SINGLE_QUBIT = "single_qubit"
    TWO_QUBIT = "two_qubit"
    MULTI_QUBIT = "multi_qubit"
    PARAMETRIC = "parametric"
    TWO_QUBIT_PARAMETRIC = "two_qubit_parametric"


@dataclass
class GateDefinition:
    """게이트 정의"""
    name: str
    gate_type: GateType
    num_qubits: int
    num_parameters: int = 0
    inverse_name: Optional[str] = None
    is_self_inverse: bool = False
    parameter_signs: Optional[List[int]] = None  # 역게이트 시 파라미터 부호 변경
    description: str = ""
    
    def __post_init__(self):
        if self.parameter_signs is None and self.num_parameters > 0:
            # 기본적으로 회전 게이트는 부호 반전
            self.parameter_signs = [-1] * self.num_parameters


class QuantumGateRegistry:
    """
    양자 게이트 레지스트리 - 싱글톤 패턴
    
    모든 게이트 정의와 역게이트 매핑을 중앙에서 관리합니다.
    """
    
    _instance = None
    _gates: Dict[str, GateDefinition] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_gates()
        return cls._instance
    
    def _initialize_gates(self):
        """기본 게이트들을 등록합니다"""
        
        # 단일 큐빗 게이트 (비파라메트릭)
        single_qubit_gates = [
            GateDefinition("h", GateType.SINGLE_QUBIT, 1, is_self_inverse=True, 
                          description="Hadamard gate"),
            GateDefinition("x", GateType.SINGLE_QUBIT, 1, is_self_inverse=True,
                          description="Pauli-X gate"),
            GateDefinition("y", GateType.SINGLE_QUBIT, 1, is_self_inverse=True,
                          description="Pauli-Y gate"),
            GateDefinition("z", GateType.SINGLE_QUBIT, 1, is_self_inverse=True,
                          description="Pauli-Z gate"),
            GateDefinition("s", GateType.SINGLE_QUBIT, 1, inverse_name="sdg",
                          description="S gate (phase gate)"),
            GateDefinition("sdg", GateType.SINGLE_QUBIT, 1, inverse_name="s",
                          description="S dagger gate"),
            GateDefinition("t", GateType.SINGLE_QUBIT, 1, inverse_name="tdg",
                          description="T gate"),
            GateDefinition("tdg", GateType.SINGLE_QUBIT, 1, inverse_name="t",
                          description="T dagger gate"),
        ]
        
        # 단일 큐빗 파라메트릭 게이트
        parametric_single_gates = [
            GateDefinition("rx", GateType.PARAMETRIC, 1, 1, is_self_inverse=True,
                          description="Rotation around X-axis"),
            GateDefinition("ry", GateType.PARAMETRIC, 1, 1, is_self_inverse=True,
                          description="Rotation around Y-axis"),
            GateDefinition("rz", GateType.PARAMETRIC, 1, 1, is_self_inverse=True,
                          description="Rotation around Z-axis"),
            GateDefinition("p", GateType.PARAMETRIC, 1, 1, is_self_inverse=True,
                          description="Phase gate"),
            # 현재 Qiskit에서 지원하는 파라메트릭 게이트만 유지
            # u1, u2, u3는 최신 Qiskit에서 데프리케이트됨
        ]
        
        # 2큐빗 게이트
        two_qubit_gates = [
            GateDefinition("cx", GateType.TWO_QUBIT, 2, is_self_inverse=True,
                          description="CNOT gate"),
            GateDefinition("cy", GateType.TWO_QUBIT, 2, is_self_inverse=True,
                          description="Controlled-Y gate"),
            GateDefinition("cz", GateType.TWO_QUBIT, 2, is_self_inverse=True,
                          description="Controlled-Z gate"),
            GateDefinition("swap", GateType.TWO_QUBIT, 2, is_self_inverse=True,
                          description="SWAP gate"),
        ]
        
        # 2큐빗 파라메트릭 게이트
        parametric_two_gates = [
            GateDefinition("crx", GateType.TWO_QUBIT_PARAMETRIC, 2, 1, is_self_inverse=True,
                          description="Controlled rotation around X-axis"),
            GateDefinition("cry", GateType.TWO_QUBIT_PARAMETRIC, 2, 1, is_self_inverse=True,
                          description="Controlled rotation around Y-axis"),
            GateDefinition("crz", GateType.TWO_QUBIT_PARAMETRIC, 2, 1, is_self_inverse=True,
                          description="Controlled rotation around Z-axis"),
        ]
        
        # 모든 게이트 등록
        all_gates = (single_qubit_gates + parametric_single_gates + 
                    two_qubit_gates + parametric_two_gates)
        
        for gate_def in all_gates:
            self._gates[gate_def.name] = gate_def
    
    def get_gate(self, name: str) -> Optional[GateDefinition]:
        """게이트 정의 반환"""
        return self._gates.get(name.lower())
    
    def get_inverse_gate_name(self, name: str) -> str:
        """역게이트 이름 반환"""
        gate_def = self.get_gate(name)
        if not gate_def:
            raise ValueError(f"Unknown gate: {name}")
        
        if gate_def.is_self_inverse:
            return name
        elif gate_def.inverse_name:
            return gate_def.inverse_name
        else:
            # 기본적으로 자기 자신이 역게이트
            return name
    
    def get_inverse_parameters(self, name: str, parameters: List[float]) -> List[float]:
        """역게이트의 파라미터 반환"""
        gate_def = self.get_gate(name)
        if not gate_def or not gate_def.parameter_signs:
            return parameters
        
        inverse_params = []
        for param, sign in zip(parameters, gate_def.parameter_signs):
            inverse_params.append(param * sign)
        
        return inverse_params
    
    def is_parametric_gate(self, name: str) -> bool:
        """파라메트릭 게이트인지 확인"""
        gate_def = self.get_gate(name)
        return gate_def is not None and gate_def.num_parameters > 0
    
    def get_required_parameters(self, name: str) -> int:
        """필요한 파라미터 수 반환"""
        gate_def = self.get_gate(name)
        return gate_def.num_parameters if gate_def else 0
    
    def get_required_qubits(self, name: str) -> int:
        """필요한 큐빗 수 반환"""
        gate_def = self.get_gate(name)
        return gate_def.num_qubits if gate_def else 1
    
    def list_gates(self) -> List[str]:
        """등록된 모든 게이트 이름 반환"""
        return list(self._gates.keys())
    
    def register_custom_gate(self, gate_def: GateDefinition):
        """사용자 정의 게이트 등록"""
        self._gates[gate_def.name] = gate_def
    
    def get_gate_type(self, name: str) -> str:
        gate_def = self.get_gate(name)
        return gate_def.gate_type if gate_def else None

    def get_gate_operation_info(self, name : str, info : str) -> int:
        gate_def = self.get_gate(name)
        return getattr(gate_def if gate_def else None, info)


# 전역 게이트 레지스트리 인스턴스
gate_registry = QuantumGateRegistry()


def get_gate_operation_info(gate : GateOperation, info : str) -> int:
    return gate_registry.get_gate_operation_info(gate.name, info)

# 편의 함수들
def get_gate_info(name: str) -> Optional[GateDefinition]:
    """게이트 정보 반환"""
    return gate_registry.get_gate(name)


def get_inverse_gate(name: str) -> str:
    """역게이트 이름 반환"""
    return gate_registry.get_inverse_gate_name(name)
    

def get_inverse_parameters(name: str, parameters: List[float]) -> List[float]:
    """역게이트 파라미터 반환"""
    return gate_registry.get_inverse_parameters(name, parameters)


def is_parametric(name: str) -> bool:
    """파라메트릭 게이트 여부 확인"""
    return gate_registry.is_parametric_gate(name)


def validate_gate_operation(name: str, qubits: List[int], parameters: List[float] = None) -> bool:
    """게이트 연산 유효성 검사"""
    gate_def = gate_registry.get_gate(name)
    if not gate_def:
        return False
    
    # 큐빗 수 검사
    if len(qubits) != gate_def.num_qubits:
        return False
    
    # 파라미터 수 검사
    param_count = len(parameters) if parameters else 0
    if param_count != gate_def.num_parameters:
        return False
    
    return True
