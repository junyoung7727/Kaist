#!/usr/bin/env python3
"""
시뮬레이터 실행자 구현

Qiskit AerSimulator를 사용한 양자 회로 시뮬레이션 실행자입니다.
추상 실행자 인터페이스를 구현하며, 시뮬레이터 관련 로직만 포함합니다.
"""

import time
from typing import List, Dict, Any
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile

from execution.executor import AbstractQuantumExecutor, ExecutionResult, register_executor
from config import default_config, ExperimentConfig
from core.circuit_interface import AbstractQuantumCircuit, CircuitSpec
from core.qiskit_circuit import QiskitQuantumCircuit


@register_executor('simulator')
class SimulatorExecutor(AbstractQuantumExecutor):
    """
    시뮬레이터 실행자
    
    Qiskit AerSimulator를 사용하여 양자 회로를 시뮬레이션합니다.
    """
    
    def __init__(self):
        super().__init__()
        self._config = default_config
        self._simulator = None
    
    def initialize(self) -> bool:
        """시뮬레이터 초기화"""
        try:
            # AerSimulator 생성 (노이즈 없음)
            self._simulator = AerSimulator(method='statevector')
            self._initialized = True
            return True
        except Exception as e:
            print(f"Simulator initialization failed: {e}")
            return False

    def run(self, circuits: List[CircuitSpec],experiment_config : ExperimentConfig):
        """
        실험 실행
        
        Args:
            experiment_config: 실험 설정
            
        Returns:
            실행 결과
        """
        
        if isinstance(circuits, CircuitSpec):
            circuits = QiskitQuantumCircuit(circuits).build()
        elif isinstance(circuits, list):
            circuits = [QiskitQuantumCircuit(circuit).build() for circuit in circuits]

        return self.execute_circuits(circuits, experiment_config)
    
    def execute_circuit(self, qiskit_circuit: QiskitQuantumCircuit, exp_config: ExperimentConfig) -> ExecutionResult:
        """단일 회로 실행"""
        if not self._initialized:
            self.initialize()
        
        start_time = time.time()
        
        try:
            # 측정레이어가 없다면 측정 추가
            qiskit_circuit.add_measurements()
            
            # 회로 트랜스파일
            transpiled = transpile(qiskit_circuit._qiskit_circuit, self._simulator)
            
            # 실행
            job = self._simulator.run(transpiled, shots=exp_config.shots)
            result = job.result()
            counts = result.get_counts()
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                counts=counts,
                shots=exp_config.shots,
                execution_time=execution_time,
                backend_info=self.get_backend_info(exp_config),
                circuit_id=qiskit_circuit.name,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                counts={},
                shots=exp_config.shots,
                execution_time=execution_time,
                backend_info=self.get_backend_info(exp_config),
                circuit_id=qiskit_circuit.name,
                success=False,
                error_message=str(e)
            )
    
    def execute_circuits(self, circuits: List[QiskitQuantumCircuit], exp_config: ExperimentConfig) -> List[ExecutionResult]:
        """다중 회로 배치 실행"""
        results = []
        for circuit in circuits:
            result = self.execute_circuit(circuit, exp_config)
            results.append(result)
        return results
    
    def get_backend_info(self, exp_config: ExperimentConfig) -> Dict[str, Any]:
        """백엔드 정보 반환"""
        return {
            'backend_type': 'simulator',
            'backend_name': 'AerSimulator',
            'method': 'statevector',
            'noise_model': None,
            'shots': exp_config.shots
        }
    
    async def cleanup(self):
        """리소스 정리"""
        self._simulator = None
        self._initialized = False


def create_qiskit_circuit(spec: CircuitSpec) -> QiskitQuantumCircuit:
    """Qiskit 회로 생성 편의 함수"""
    return QiskitQuantumCircuit(spec).build()
