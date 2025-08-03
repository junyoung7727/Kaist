from qiskit import QuantumCircuit
from core.circuit_interface import AbstractQuantumCircuit, CircuitSpec
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))
from gates import GateType, GateOperation, gate_registry
from qiskit.circuit import ClassicalRegister
from typing import Dict, Any, Optional, List  


class QiskitQuantumCircuit(AbstractQuantumCircuit):
    """Qiskit 기반 양자 회로 구현"""
    
    def __init__(self, spec: CircuitSpec):
        super().__init__(spec)
        self._qiskit_circuit = None
    
    def build(self) -> 'QiskitQuantumCircuit':
        """Qiskit 회로를 실제로 구성합니다"""
        if self._built:
            return self
        
        # Qiskit 회로 생성
        self._qiskit_circuit = QuantumCircuit(self.num_qubits)
        self._qiskit_circuit.name = self.circuit_id
        
        # 게이트들을 Qiskit 회로에 추가
        for gate in self._spec.gates:
            self._add_gate(gate)
        
        self._built = True
        return self

    @staticmethod
    def from_qiskit_circuit(qiskit_circuit: QuantumCircuit) -> 'QiskitQuantumCircuit':
        """Qiskit 회로를 QiskitQuantumCircuit으로 변환"""
        # 빈 CircuitSpec으로 인스턴스 생성
        empty_spec = CircuitSpec(
            num_qubits=qiskit_circuit.num_qubits,
            gates=[],
            circuit_id=qiskit_circuit.name or "imported_circuit"
        )
        
        # QiskitQuantumCircuit 인스턴스 생성
        wrapper = QiskitQuantumCircuit(empty_spec)
        wrapper._qiskit_circuit = qiskit_circuit
        wrapper._built = True
        return wrapper
    
    def _add_gate(self, gate: GateOperation) -> 'QiskitQuantumCircuit':
        """Qiskit 회로에 게이트 추가"""
        name = gate.name.lower()
        qubits = gate.qubits
        params = gate.parameters or []

        gate_def = gate_registry.get_gate(name)
        if gate_def:
            gate_type = gate_def.gate_type
            num_params = gate_def.num_parameters
        
        # 특수 게이트 처리
        if name == 'reset':
            for qubit in qubits:
                self._qiskit_circuit.reset(qubit)
        elif name == 'barrier':
            self._qiskit_circuit.barrier(qubits)
        elif name == 'measure':
            # 측정은 별도로 처리
            pass
        # 단일 큐빗 게이트
        elif gate_type == GateType.SINGLE_QUBIT:
            getattr(self._qiskit_circuit, name)(qubits[0])
        # 파라메트릭 단일 큐빗 게이트
        elif gate_type == GateType.PARAMETRIC:
            getattr(self._qiskit_circuit, name)(*params[:num_params], qubits[0])
        # 2큐빗 게이트
        elif gate_type == GateType.TWO_QUBIT:
            getattr(self._qiskit_circuit, name)(qubits[0], qubits[1])
        # 파라메트릭 2큐빗 게이트
        elif gate_type == GateType.TWO_QUBIT_PARAMETRIC:
            getattr(self._qiskit_circuit, name)(*params[:num_params], qubits[0], qubits[1])
        # 3큐빗 게이트
        elif gate_type == GateType.THREE_QUBIT:
            getattr(self._qiskit_circuit, name)(qubits[0], qubits[1], qubits[2])
        else:
            raise ValueError(f"Unsupported gate: {name}")

    def add_gate(self, gate: GateOperation) -> 'QiskitQuantumCircuit':
        """게이트를 회로에 추가합니다"""
        if not self._built:
            self.build()
        
        self._add_gate(gate)
        return self
    
    def compose(self, other: 'QiskitQuantumCircuit') -> 'QiskitQuantumCircuit':
        """다른 회로와 결합합니다"""
        if not self._built:
            self.build()
        if not other._built:
            other.build()
        
        # 새로운 회로 생성
        combined_spec = CircuitSpec(
            num_qubits=max(self.num_qubits, other.num_qubits),
            gates=self._spec.gates + other._spec.gates,
            name=f"{self.name}_composed_{other.name}"
        )
        
        return QiskitQuantumCircuit(combined_spec).build()
    
    def inverse(self) -> 'QiskitQuantumCircuit':
        """역회로를 생성합니다"""
        from core.inverse import create_inverse_circuit_spec
        inverse_spec = create_inverse_circuit_spec(self._spec)
        return QiskitQuantumCircuit(inverse_spec).build()
    
    def reset_all_qubits(self) -> 'QiskitQuantumCircuit':
        """모든 큐빗을 |0⟩ 상태로 초기화합니다"""
        if not self._built:
            self.build()
        
        for qubit in range(self.num_qubits):
            self._qiskit_circuit.reset(qubit)
        
        return self
    
    def add_measurements(self) -> 'QiskitQuantumCircuit':
        """측정을 추가합니다"""

        # 이미 측정이 포함되어 있는지 확인
        has_measurements = False
        for instruction in self._qiskit_circuit.data:
            if instruction.operation.name == 'measure':
                has_measurements = True
                break
                
        if not has_measurements:
            # 모든 큐빗 측정
            self._qiskit_circuit.measure_all()
            
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """회로를 딕셔너리로 직렬화합니다"""
        return {
            'spec': {
                'num_qubits': self._spec.num_qubits,
                'gates': [
                    {
                        'name': gate.name,
                        'qubits': gate.qubits,
                        'parameters': gate.parameters
                    }
                    for gate in self._spec.gates
                ],
                'name': self._spec.name
            },
            'built': self._built
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QiskitQuantumCircuit':
        """딕셔너리에서 회로를 역직렬화합니다"""
        spec_data = data['spec']
        gates = [
            GateOperation(
                name=gate_data['name'],
                qubits=gate_data['qubits'],
                parameters=gate_data['parameters']
            )
            for gate_data in spec_data['gates']
        ]
        
        spec = CircuitSpec(
            num_qubits=spec_data['num_qubits'],
            gates=gates,
            name=spec_data['name']
        )
        
        circuit = cls(spec)
        if data['built']:
            circuit.build()
        
        return circuit
    
    @property
    def qiskit_circuit(self) -> QuantumCircuit:
        """내부 Qiskit 회로 반환"""
        if not self._built:
            self.build()
        return self._qiskit_circuit