#!/usr/bin/env python3
"""
IBM 양자 하드웨어 실행자 구현

IBM Quantum 서비스를 사용한 양자 회로 실행자입니다.
추상 실행자 인터페이스를 구현하며, IBM 관련 로직만 포함합니다.
"""

import time
from typing import List, Dict, Any, Optional
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.compiler import transpile
from qiskit import QuantumCircuit

from execution.executor import AbstractQuantumExecutor, ExecutionResult, register_executor
from config import default_config, ExperimentConfig
from core.qiskit_circuit import QiskitQuantumCircuit
from core.circuit_interface import AbstractQuantumCircuit, CircuitSpec
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager



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
            self._service = QiskitRuntimeService(
                channel="ibm_quantum_platform",
                token=self._config.ibm_token
            )
            self.exp_config = exp_config
            # 사용 가능한 백엔드 중 가장 적합한 것 선택
            self._backend = self._service.least_busy(
                operational=True,
                simulator=False,
                min_num_qubits=2
            )
            self._backend_name = self._backend.name
            
            # Sampler 초기화
            self.pm = generate_preset_pass_manager(backend=self._backend, optimization_level=1)
            self._sampler = Sampler(mode=self._backend)
            self._sampler.options.default_shots = self.exp_config.shots
            
            self._initialized = True
            print(f"IBM backend initialized: {self._backend_name}")
            return True
            
        except Exception as e:
            print(f"IBM initialization failed: {e}")
            return False

    def run(self, circuits, exp_config : ExperimentConfig):
        """
        실험 실행
        
        Args:
            experiment_config: 실험 설정
            
        Returns:
            실행 결과
        """
        
        if isinstance(circuits, CircuitSpec):
            circuits = QiskitQuantumCircuit(circuits).build()
            # 측정 추가
            circuits.add_measurements()
            return self.execute_circuit(circuits._qiskit_circuit, exp_config)

        elif isinstance(circuits, list):
            # CircuitSpec 리스트를 QuantumCircuit 리스트로 변환
            qiskit_circuits = []
            for circuit_spec in circuits:
                qc = QiskitQuantumCircuit(circuit_spec).build()
                qc.add_measurements()
                qiskit_circuits.append(qc._qiskit_circuit)
            return self.execute_circuits(qiskit_circuits, exp_config)
    
    def execute_circuit(self, qiskit_circuit: QuantumCircuit, exp_config: ExperimentConfig) -> ExecutionResult:
        """단일 회로 실행"""
        if not self._initialized:
            self.initialize(exp_config)
        
        start_time = time.time()
        
        try: 
            # 클래식 레지스터 수 계산 (원래 큐빗 수가 아님!)
            classical_bits = sum(creg.size for creg in qiskit_circuit.cregs)
            
            # 백엔드에 맞게 트랜스파일
            transpiled_circuit = self.pm.run(qiskit_circuit)
            
            # IBM Quantum에서 실행
            job = self._sampler.run([transpiled_circuit])
            result = job.result()
            
            # 결과 처리 - 클래식 레지스터 수만큼만 자르기
            raw_counts = result[0].data.meas.get_counts()
            counts = self._truncate_counts_to_original_qubits(raw_counts, classical_bits)

            
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
    
    def execute_circuits(self, qiskit_circuits: List[QuantumCircuit], exp_config: ExperimentConfig) -> List[ExecutionResult]:
        """
        여러 회로를 배치로 실행 (IBM 페이로드 제한 고려 자동 분할)
        
        Args:
            qiskit_circuits: 실행할 Qiskit 회로 리스트
            exp_config: 실험 설정
            
        Returns:
            실행 결과 리스트
        """
        if not qiskit_circuits:
            return []
            
        print(f"\n🚀 IBM Quantum 배치 실행 시작: {len(qiskit_circuits)}개 회로")
        
        # IBM 샷 수 제한 고려 자동 분할
        max_circuits_per_batch = self._calculate_max_batch_size(qiskit_circuits, exp_config)
        
        if len(qiskit_circuits) <= max_circuits_per_batch:
            # 단일 배치로 처리 가능
            return self._execute_single_batch(qiskit_circuits, exp_config)
        else:
            # 다중 배치로 분할 처리
            return self._execute_multiple_batches(qiskit_circuits, exp_config, max_circuits_per_batch)
    
    def _calculate_max_batch_size(self, qiskit_circuits: List[QuantumCircuit], exp_config: ExperimentConfig) -> int:
        """
        IBM 1천만 샷 제한을 고려한 최대 배치 크기 계산
        """
        if not qiskit_circuits:
            return 1000
            
        # IBM 샷 수 제한 (1천만 샷/배치)
        max_shots_per_batch = 10_000_000
        shots_per_circuit = exp_config.shots
        
        # 샷 수 기준 최대 회로 수 계산
        max_circuits_by_shots = max_shots_per_batch // shots_per_circuit
        
        # 수학적 페이로드 크기 계산 (실제 회로 복잡도 기반)
        sample_size = min(10, len(qiskit_circuits))
        sample_circuits = qiskit_circuits[:sample_size]
        
        # 회로당 평균 페이로드 크기 계산 (바이트)
        total_payload_size = 0
        for circuit in sample_circuits:
            # 기본 회로 메타데이터: ~500 바이트
            circuit_payload = 500
            
            # 큐빗당 ~50 바이트 (레지스터 정보)
            circuit_payload += circuit.num_qubits * 50
            
            # 게이트당 ~100 바이트 (게이트 타입, 파라미터, 큐빗 인덱스)
            circuit_payload += len(circuit.data) * 100
            
            # 2큐빗 게이트는 추가 복잡도 (+50 바이트)
            two_qubit_gates = sum(1 for gate, qubits, _ in circuit.data if len(qubits) == 2)
            circuit_payload += two_qubit_gates * 50
            
            total_payload_size += circuit_payload
        
        avg_circuit_payload = total_payload_size / sample_size
        
        # IBM Quantum 페이로드 제한: ~100MB (100,000,000 바이트)
        # 안전 마진 80% 적용: 80,000,000 바이트
        max_payload_bytes = 80_000_000
        max_circuits_by_payload = int(max_payload_bytes / avg_circuit_payload)
        
        # 최소 100개, 최대 50,000개로 제한 (상식적 범위)
        max_circuits_by_payload = max(100, min(50000, max_circuits_by_payload))
        
        # 두 제한 중 더 작은 값 사용
        max_batch_size = min(max_circuits_by_shots, max_circuits_by_payload)
        
        print(f"📊 배치 크기 계산:")
        print(f"  - 샷 수 기준: {shots_per_circuit:,}샷/회로 → 최대 {max_circuits_by_shots:,}개")
        print(f"  - 최종 배치 크기: {max_batch_size:,}개/배치")
        
        return max_batch_size
    
    def _execute_single_batch(self, qiskit_circuits: List[QuantumCircuit], exp_config: ExperimentConfig) -> List[ExecutionResult]:
        """
        단일 배치 실행
        """
        # 초기화 확인
        if not self._initialized:
            self.initialize(exp_config)
            
        print(f"📎 단일 배치 실행: {len(qiskit_circuits)}개 회로")
        start_time = time.time()
        
        # 원래 회로들의 클래식 레지스터 수 저장 (트랜스파일 전)
        original_classical_bits = [sum(creg.size for creg in circuit.cregs) for circuit in qiskit_circuits]

        # 트랜스파일
        transpiled_circuits = self._transpile_circuits(qiskit_circuits)
        
        # IBM Quantum에서 실행
        # job = self._sampler.run(transpiled_circuits)
        # results = job.result()

        #테스트용 코드
        results = []
        from qiskit_aer import AerSimulator
        sim = AerSimulator()
        result = sim.run(transpiled_circuits).result()
        for i in range(len(transpiled_circuits)):
            results.append(result.get_counts(i))
        
        # 결과 처리
        execution_results = self._process_batch_results(results, qiskit_circuits, original_classical_bits, exp_config, start_time)
        
        print(f"✅ 단일 배치 완료: {len(execution_results)}개 결과")
        return execution_results
    
    def _execute_multiple_batches(self, qiskit_circuits: List[QuantumCircuit], exp_config: ExperimentConfig, max_batch_size: int) -> List[ExecutionResult]:
        """
        다중 배치로 분할 실행
        """
        total_circuits = len(qiskit_circuits)
        num_batches = (total_circuits + max_batch_size - 1) // max_batch_size
        
        print(f"🔄 다중 배치 실행: {total_circuits}개 회로를 {num_batches}개 배치로 분할 (최대 {max_batch_size}개/배치)")
        
        all_results = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * max_batch_size
            end_idx = min(start_idx + max_batch_size, total_circuits)
            batch_circuits = qiskit_circuits[start_idx:end_idx]
            
            print(f"\n📦 배치 {batch_idx + 1}/{num_batches}: {len(batch_circuits)}개 회로 (인덱스 {start_idx}-{end_idx-1})")
            
            try:
                batch_results = self._execute_single_batch(batch_circuits, exp_config)
                all_results.extend(batch_results)
                print(f"✅ 배치 {batch_idx + 1} 완료")
            except Exception as e:
                print(f"❌ 배치 {batch_idx + 1} 실패: {e}")
                # 실패한 배치에 대해 빈 결과 추가
                for i in range(len(batch_circuits)):
                    all_results.append(ExecutionResult(
                        counts={},
                        shots=exp_config.shots,
                        execution_time=0.0,
                        backend_info=self.get_backend_info(exp_config),
                        circuit_id=batch_circuits[i].name or f"circuit_{start_idx + i}",
                        success=False,
                        error_message=str(e)
                    ))
        
        print(f"\n🎉 모든 배치 완료: {len(all_results)}개 결과")
        return all_results
    
    def _transpile_circuits(self, qiskit_circuits: List[QuantumCircuit]) -> List[QuantumCircuit]:
        """
        회로 트랜스파일 (대량 처리 최적화)
        """
        if len(qiskit_circuits) > 1000:
            print(f"🔧 대량 트랜스파일 시작: {len(qiskit_circuits)}개 회로")
            transpiled_circuits = []
            for i, circuit in enumerate(qiskit_circuits):
                transpiled_circuit = self.pm.run(circuit)
                transpiled_circuits.append(transpiled_circuit)
                if (i + 1) % 500 == 0:
                    print(f"  진행률: {i + 1}/{len(qiskit_circuits)} ({(i + 1)/len(qiskit_circuits)*100:.1f}%)")
            return transpiled_circuits
        else:
            # 소량의 회로는 한번에 처리
            return [self.pm.run(circuit) for circuit in qiskit_circuits]
    
    def _process_batch_results(self, results, qiskit_circuits: List[QuantumCircuit], original_classical_bits: List[int], exp_config: ExperimentConfig, start_time: float) -> List[ExecutionResult]:
        """
        배치 실행 결과 처리
        """
        execution_results = []
        total_time = time.time() - start_time
        avg_time = total_time / len(qiskit_circuits)
        
        # results는 단일 Result 객체, 각 회로의 결과는 인덱스로 접근
        for i, result in enumerate(results):
            #raw_counts = result.data.meas.get_counts() #원래코드임, 지우지 말것것
            raw_counts = result #테스트용 지우지 말것것
            # 각 회로의 원래 클래식 레지스터 수만큼만 자르기
            counts = self._truncate_counts_to_original_qubits(raw_counts, original_classical_bits[i])
            
            execution_results.append(ExecutionResult(
                counts=counts,
                shots=exp_config.shots,
                execution_time=avg_time,
                backend_info=self.get_backend_info(exp_config),
                circuit_id=qiskit_circuits[i].name or f"circuit_{i}",
                success=True
            ))
        return execution_results

    def get_backend_info(self, exp_config) -> Dict[str, Any]:
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
                'shots': exp_config.shots
            }
        except Exception as e:
            return {
                'backend_type': 'ibm',
                'backend_name': self._backend_name,
                'status': f'error: {e}',
                'shots': exp_config.shots
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

    def _truncate_counts_to_original_qubits(self, counts, original_classical_bits):
        """
        비트스트링을 원래 회로의 클래식 레지스터 수만큼만 자르기
        
        중요: Qiskit에서 나중에 추가된 클래식 레지스터는 비트스트링의 앞쪽에 위치합니다.
        따라서 원래 회로의 클래식 레지스터 수만큼만 앞에서 자르면 됩니다.
        
        Args:
            counts: IBM 실행 결과 카운트
            original_num_qubits: 원래 회로의 클래식 레지스터 수 (원래 큐빗 수가 아님!)
            
        Returns:
            자른 카운트 딕셔너리
        """
        truncated_counts = {}
        for key, value in counts.items():
            # 원래 회로에서 추가한 클래식 레지스터 수만큼만 앞에서 자르기
            # SWAP test의 경우: ancilla 레지스터 1개만 추가되므로 1비트만 자름
            # 일반 회로의 경우: 원래 큐빗 수만큼 자름
            truncated_key = key[:original_classical_bits]
            truncated_counts[truncated_key] = truncated_counts.get(truncated_key, 0) + value
        return truncated_counts
