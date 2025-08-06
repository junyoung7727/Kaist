#!/usr/bin/env python3
"""
추상 양자 회로 인터페이스

모든 백엔드에서 공통으로 사용할 수 있는 통합된 회로 인터페이스를 제공합니다.
백엔드별 구현 세부사항은 완전히 숨겨집니다.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "quantumcommon"))

from gates import GateOperation

@dataclass
class CircuitSpec:
    """회로 사양을 정의하는 데이터 클래스"""
    num_qubits: int
    gates: List[GateOperation]
    circuit_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_qubits": self.num_qubits,
            "gates": [g.to_dict() for g in self.gates],
            "circuit_id": self.circuit_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CircuitSpec':
        """딕셔너리에서 CircuitSpec 객체를 생성"""
        gates = [GateOperation.from_dict(gate_dict) for gate_dict in data.get("gates", [])]
        return cls(
            num_qubits=data.get("num_qubits", 0),
            gates=gates,
            circuit_id=data.get("circuit_id", "")
        )


class AbstractQuantumCircuit(ABC):
    """
    모든 백엔드에서 공통으로 사용할 추상 양자 회로 인터페이스
    
    이 인터페이스를 통해 시뮬레이터든 IBM 하드웨어든 동일하게 처리됩니다.
    """
    
    def __init__(self, spec: CircuitSpec):
        self._spec = spec
        self._built = False
    
    @property
    def spec(self) -> CircuitSpec:
        """회로 사양 반환"""
        return self._spec
    
    @property
    def num_qubits(self) -> int:
        """큐빗 수 반환"""
        return self._spec.num_qubits
    
    @property
    def name(self) -> str:
        """회로 이름 반환 (circuit_id 동일)"""
        return self.circuit_id
        
    @property
    def circuit_id(self) -> str:
        """회로 식별자 반환"""
        return self._spec.circuit_id
    
    @abstractmethod
    def build(self) -> 'AbstractQuantumCircuit':
        """회로를 실제로 구성합니다"""
        pass
    
    @abstractmethod
    def compose(self, other: 'AbstractQuantumCircuit') -> 'AbstractQuantumCircuit':
        """다른 회로와 결합합니다"""
        pass
    
    @abstractmethod
    def inverse(self) -> 'AbstractQuantumCircuit':
        """역회로를 생성합니다"""
        pass
    
    @abstractmethod
    def reset_all_qubits(self) -> 'AbstractQuantumCircuit':
        """모든 큐빗을 |0⟩ 상태로 초기화합니다"""
        pass
    
    @abstractmethod
    def add_measurements(self) -> 'AbstractQuantumCircuit':
        """측정을 추가합니다"""
        pass
    
    @abstractmethod
    def add_gate(self, gate: GateOperation) -> 'AbstractQuantumCircuit':
        """게이트를 추가합니다"""
        pass

    @abstractmethod
    def _add_gate(self, gate: GateOperation) -> 'AbstractQuantumCircuit':
        """Qiskit 회로에 게이트 추가"""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """회로를 딕셔너리로 직렬화합니다"""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AbstractQuantumCircuit':
        """딕셔너리에서 회로를 역직렬화합니다"""
        # name과 circuit_id 둘 다 지원하도록 처리 (하위 호환성)
        circuit_id = data.get('circuit_id', data.get('name'))
        spec = CircuitSpec(
            num_qubits=data['num_qubits'],
            gates=[GateOperation.from_dict(g) for g in data['gates']],
            circuit_id=circuit_id
        )
        return cls(spec)


class CircuitBuilder:
    """
    백엔드 무관한 회로 빌더
    
    이 클래스는 어떤 백엔드를 사용하는지 전혀 모릅니다.
    """
    
    def __init__(self):
        self._gates: List[GateOperation] = []
        self._num_qubits = 0
        self._circuit_id = None
    
    def set_qubits(self, num_qubits: int) -> 'CircuitBuilder':
        """큐빗 수 설정"""
        self._num_qubits = num_qubits
        return self
    
    def set_circuit_id(self, circuit_id: str) -> 'CircuitBuilder':
        """회로 식별자 설정"""
        self._circuit_id = circuit_id
        return self
    
    def add_gate(self, name: str, qubits: Union[int, List[int]], 
                 parameters: Optional[List[float]] = None) -> 'CircuitBuilder':
        """게이트 추가"""
        if isinstance(qubits, int):
            qubits = [qubits]
        
        gate = GateOperation(name=name, qubits=qubits, parameters=parameters or [])
        self._gates.append(gate)
        
        # 큐빗 수 자동 업데이트
        max_qubit = max(qubits) if qubits else -1
        if max_qubit >= self._num_qubits:
            raise ValueError("Exceeding qubit index")
        
        return self
    
    def build_spec(self) -> CircuitSpec:
        """회로 사양 생성"""
        return CircuitSpec(
            num_qubits=self._num_qubits,
            gates=self._gates.copy(),
            circuit_id=self._circuit_id
        )
    
    def clear(self) -> 'CircuitBuilder':
        """빌더 초기화"""
        self._gates.clear()
        self._num_qubits = 0
        self._circuit_id = None
        return self


# 편의 함수들
def create_circuit_spec(num_qubits: int, gates: List[GateOperation], 
                       circuit_id: Optional[str] = None) -> CircuitSpec:
    """회로 사양을 생성하는 편의 함수"""
    return CircuitSpec(num_qubits=num_qubits, gates=gates, circuit_id=circuit_id)


def gate(name: str, qubits: Union[int, List[int]], 
         parameters: Optional[List[float]] = None) -> GateOperation:
    """게이트 연산을 생성하는 편의 함수"""
    if isinstance(qubits, int):
        qubits = [qubits]
    return GateOperation(name=name, qubits=qubits, parameters=parameters or [])
