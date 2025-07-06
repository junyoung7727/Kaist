#!/usr/bin/env python3
"""
IBM 양자 하드웨어 실행자 구현

IBM Quantum 서비스를 사용한 양자 회로 실행자입니다.
추상 실행자 인터페이스를 구현하며, IBM 관련 로직만 포함합니다.
"""

import time
from typing import List, Dict, Any, Optional
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options
from qiskit.compiler import transpile

from execution.executor import AbstractQuantumExecutor, ExecutionResult, register_executor
from config import default_config, ExperimentConfig
from core.qiskit_circuit import QiskitQuantumCircuit
from core.circuit_interface import AbstractQuantumCircuit, CircuitSpec


@register_executor('ibm')
class IBMExecutor(AbstractQuantumExecutor):
    """
    IBM 양자 하드웨어 실행자
    
    IBM Quantum 서비스를 사용하여 실제 양자 하드웨어에서 회로를 실행합니다.
    """
    
    def __init__(self):
        super().__init__()
        self._config = default_config
        self._service = None
        self._backend = None
        self._sampler = None
        self._backend_name = None
    
    def initialize(self, exp_config: ExperimentConfig) -> bool:
        """IBM 서비스 초기화"""
        try:
            # IBM Quantum 서비스 초기화
            self._service = QiskitRuntimeService(token=self._config.ibm_token)
            
            # 사용 가능한 백엔드 중 가장 적합한 것 선택
            self._backend = self._service.least_busy(
                operational=True,
                simulator=False,
                min_num_qubits=2
            )
            self._backend_name = self._backend.name
            
            # Sampler 초기화
            options = Options()
            options.execution.shots = exp_config.shots
            options.optimization_level = exp_config.optimization_level
            
            self._sampler = Sampler(backend=self._backend, options=options)
            
            self._initialized = True
            print(f"IBM backend initialized: {self._backend_name}")
            return True
            
        except Exception as e:
            print(f"IBM initialization failed: {e}")
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
            self.initialize(exp_config)
        
        start_time = time.time()
        
        try:
            # 측정 추가
            qiskit_circuit.add_measurements()
            
            # 백엔드에 맞게 트랜스파일
            transpiled = transpile(
                qiskit_circuit._qiskit_circuit, 
                self._backend,
                optimization_level=exp_config.optimization_level
            )
            
            # IBM Quantum에서 실행
            job = self._sampler.run([transpiled])
            result = job.result()
            
            # 결과 처리
            counts = result[0].data.meas.get_counts()
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                counts=counts,
                shots=self.exp_config.shots,
                execution_time=execution_time,
                backend_info=self.get_backend_info(),
                circuit_id=qiskit_circuit.name,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                counts={},
                shots=self.exp_config.shots,
                execution_time=execution_time,
                backend_info=self.get_backend_info(),
                circuit_id=qiskit_circuit.name,
                success=False,
                error_message=str(e)
            )
    
    def execute_circuits(self, qiskit_circuits: List[QiskitQuantumCircuit], exp_config: ExperimentConfig) -> List[ExecutionResult]:
        """다중 회로 배치 실행"""
        if not self._initialized:
            self.initialize(exp_config)
        
        start_time = time.time()
        
        try:
            # 모든 회로를 Qiskit 회로로 변환
            qiskit_circuits = []
            for circuit in qiskit_circuits:
                circuit.add_measurements()
                qiskit_circuits.append(circuit.qiskit_circuit)
            
            # 백엔드에 맞게 트랜스파일
            transpiled_circuits = transpile(
                qiskit_circuits,
                self._backend,
                optimization_level=self.exp_config.optimization_level
            )
            
            # 배치 실행
            job = self._sampler.run(transpiled_circuits)
            results = job.result()
            
            # 결과 처리
            execution_results = []
            total_time = time.time() - start_time
            avg_time = total_time / len(qiskit_circuits)
            
            for i, result in enumerate(results):
                counts = result.data.meas.get_counts()
                
                execution_results.append(ExecutionResult(
                    counts=counts,
                    shots=self.exp_config.shots,
                    execution_time=avg_time,
                    backend_info=self.get_backend_info(),
                    circuit_id=qiskit_circuits[i].name,
                    success=True
                ))
            
            return execution_results
            
        except Exception as e:
            # 실패 시 개별 실행으로 폴백
            print(f"Batch execution failed, falling back to individual execution: {e}")
            results = []
            for circuit in circuits:
                result = self.execute_circuit(circuit)
                results.append(result)
            return results
    
    def get_backend_info(self) -> Dict[str, Any]:
        """백엔드 정보 반환"""
        if not self._backend:
            return {
                'backend_type': 'ibm',
                'backend_name': 'unknown',
                'status': 'not_initialized'
            }
        
        try:
            status = self._backend.status()
            configuration = self._backend.configuration()
            
            return {
                'backend_type': 'ibm',
                'backend_name': self._backend_name,
                'status': status.status_msg,
                'pending_jobs': status.pending_jobs,
                'num_qubits': configuration.num_qubits,
                'coupling_map': configuration.coupling_map,
                'basis_gates': configuration.basis_gates,
                'shots': self.exp_config.shots
            }
        except Exception as e:
            return {
                'backend_type': 'ibm',
                'backend_name': self._backend_name,
                'status': f'error: {e}',
                'shots': self.exp_config.shots
            }
    
    async def cleanup(self):
        """리소스 정리"""
        if self._sampler:
            # Sampler 세션 종료
            try:
                self._sampler.close()
            except:
                pass
        
        self._service = None
        self._backend = None
        self._sampler = None
        self._initialized = False
    
    def get_available_backends(self) -> List[str]:
        """사용 가능한 IBM 백엔드 목록 반환"""
        if not self._service:
            return []
        
        try:
            backends = self._service.backends(operational=True, simulator=False)
            return [backend.name for backend in backends]
        except Exception as e:
            print(f"Failed to get available backends: {e}")
            return []
    
    def set_backend(self, backend_name: str) -> bool:
        """특정 백엔드 설정"""
        if not self._service:
            return False
        
        try:
            self._backend = self._service.backend(backend_name)
            self._backend_name = backend_name
            
            # Sampler 재초기화
            options = Options()
            options.execution.shots = self.exp_config.shots
            options.optimization_level = self.exp_config.optimization_level
            
            self._sampler = Sampler(backend=self._backend, options=options)
            return True
            
        except Exception as e:
            print(f"Failed to set backend {backend_name}: {e}")
            return False
