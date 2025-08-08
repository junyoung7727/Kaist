"""
SWAP test 기반 Meyer-Wallach entropy 측정 (하드웨어 호환)
"""

import numpy as np
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from qiskit_aer import AerSimulator
from typing import Dict, List, Any, Union, Optional
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from core.circuit_interface import CircuitSpec
sys.path.append(str(pathlib.Path(__file__).parent.parent / 'quantum_commmon'))
from gates import GateOperation
from core.qiskit_circuit import QiskitQuantumCircuit
from config import ExperimentConfig

def meyer_wallace_entropy_swap_test(circuits: Union[CircuitSpec, List[CircuitSpec]], exp_config: ExperimentConfig, batch_manager=None) -> Union[List[float], List[int]]:
    """
    SWAP test 기반 Meyer-Wallach entropy 측정 (배치 모드)
    
    Args:
        circuits: 양자 회로 사양 리스트
        exp_config: 실험 설정
        
    Returns:
        Meyer-Wallach entropy 리스트
    """
    
    num_shots = exp_config.entangle_shots

    if isinstance(circuits, CircuitSpec):
        circuits = [circuits]
    
    if batch_manager:
        # 배치 모드: 모든 SWAP test 회로 수집
        all_swap_circuits = []
        circuit_specs = []
        circuit_qubit_mapping = []
        
        for circuit_idx, circuit in enumerate(circuits):
            n_qubits = circuit.num_qubits
            if n_qubits < 2:
                continue
                
            for target_qubit in range(n_qubits):
                swap_circuit = _create_swap_test_circuit(circuit, target_qubit)
                all_swap_circuits.append(swap_circuit)
                circuit_specs.append(circuit)
                circuit_qubit_mapping.append((circuit_idx, target_qubit, n_qubits))
        
        metadata = {
            "task": "entanglement",
            "circuit_mapping": circuit_qubit_mapping,
            "total_circuits": len(circuits)
        }
        indices = batch_manager.collect_task_circuits(
            "entanglement", all_swap_circuits, circuit_specs, metadata
        )
        return indices
    else:
        # 기존 모드: 직접 실행 (하위 호환성)
        # 진정한 배치 처리: 모든 회로의 모든 큐빗 SWAP test를 한 번에 실행
        all_swap_jobs = []
        circuit_qubit_mapping = []
        
        # 1단계: 모든 SWAP test 회로 준비
        for circuit_idx, circuit in enumerate(circuits):
            n_qubits = circuit.num_qubits
            if n_qubits < 2:
                continue
                
            for target_qubit in range(n_qubits):
                swap_circuit = _create_swap_test_circuit(circuit, target_qubit)
                all_swap_jobs.append(swap_circuit)
                circuit_qubit_mapping.append((circuit_idx, target_qubit, n_qubits))
    
    # 2단계: 모든 SWAP test를 한 번에 실행
    print(f"  한 번에 실행할 SWAP test: {len(all_swap_jobs)}개")
    batch_results = _execute_swap_batch(all_swap_jobs, num_shots, exp_config)
    
    # 3단계: 결과를 회로별로 매핑
    results = []
    circuit_purities = {}
    
    for i, (circuit_idx, target_qubit, n_qubits) in enumerate(circuit_qubit_mapping):
        if circuit_idx not in circuit_purities:
            circuit_purities[circuit_idx] = []
        
        purity = batch_results[i]
        circuit_purities[circuit_idx].append(max(0.0, min(1.0, purity)))
    
    # 4단계: Meyer-Wallach entropy 계산
    for circuit_idx, circuit in enumerate(circuits):
        n_qubits = circuit.num_qubits
        
        if n_qubits < 2:
            results.append(0.0)
        else:
            purities = circuit_purities.get(circuit_idx, [])
            if purities:
                average_purity = sum(purities) / len(purities)
                mw_entropy = 2 * (1 - average_purity)
                results.append(mw_entropy)
            else:
                results.append(0.0)
    
    print(f"✅ 얽힘도 배치 측정 완료: {len(results)}개 결과")
    return results


def _create_swap_test_circuit(circuit: CircuitSpec, target_qubit: int):
    """
    단일 SWAP test 회로 생성
    
    Args:
        circuit: 원본 양자 회로
        target_qubit: 대상 큐빗
        
    Returns:
        SWAP test를 위한 Qiskit 회로
    """
    n_qubits = circuit.num_qubits
    total_qubits = 2 * n_qubits + 1  # 두 복사본 + ancilla
    ancilla_idx = total_qubits - 1
    
    # 빈 회로 생성
    empty_spec = CircuitSpec(
        num_qubits=total_qubits,
        gates=[],
        circuit_id=f"swap_test_{circuit.circuit_id}_{target_qubit}"
    )
    base_qc_wrapper = QiskitQuantumCircuit(empty_spec)
    
    # 첫 번째 복사본에 원본 회로 적용
    _apply_circuit_to_qubits(base_qc_wrapper, circuit, 0)
    
    # 두 번째 복사본에 원본 회로 적용
    _apply_circuit_to_qubits(base_qc_wrapper, circuit, n_qubits)
    
    # Qiskit 회로 가져오기
    swap_qc = base_qc_wrapper.build()._qiskit_circuit
    creg_ancilla = ClassicalRegister(1, f'c_ancilla_{target_qubit}')
    swap_qc.add_register(creg_ancilla)
    
    # SWAP test 프로토콜
    swap_qc.h(ancilla_idx)
    swap_qc.cswap(ancilla_idx, target_qubit, target_qubit + n_qubits)
    swap_qc.h(ancilla_idx)
    
    # ancilla 측정
    swap_qc.measure(ancilla_idx, creg_ancilla[0])
    
    return swap_qc


def _execute_swap_batch(swap_circuits, num_shots, exp_config):
    """
    모든 SWAP test 회로를 배치로 실행
    
    Args:
        swap_circuits: SWAP test 회로 리스트 (QuantumCircuit 객체들)
        num_shots: 측정 횟수
        exp_config: 실험 설정
        
    Returns:
        purity 값 리스트
    """
    from execution.executor import QuantumExecutorFactory
    from core.circuit_interface import CircuitSpec, GateOperation
    
    # 실제로는 이미 구성된 Qiskit 회로를 사용해야 하므로
    # 직접 실행하는 방식으로 변경
    batch_results = []

    import qiskit_aer as Aer
    
    results = Aer.AerSimulator().run(swap_circuits, shots=num_shots).result()
    # 각 회로를 개별적으로 실행 (배치 처리는 나중에 최적화)
    for i, swap_circuit in enumerate(swap_circuits):
        # 임시로 시뮬레이터 사용 (실제 IBM 하드웨어 연결 없이 테스트)
        counts = results.get_counts(i)
        
        # ancilla=0 확률 계산 (첫 번째 비트가 ancilla)
        total_shots = sum(counts.values())
        ancilla_0_count = sum(count for bitstring, count in counts.items() if bitstring[0] == '0')
        zero_probability = ancilla_0_count / total_shots
        
        # purity 계산: 2 * P(ancilla=0) - 1
        purity = 2 * zero_probability - 1
        
        # purity는 0과 1 사이여야 함
        purity = max(0, min(1, purity))
        batch_results.append(purity)
    
    return batch_results


def _batch_qubit_purities(circuit: CircuitSpec, num_shots: int) -> List[float]:
    """
    한 회로의 모든 큐빗 purity를 배치로 계산
    
    Args:
        circuit: 양자 회로 사양
        num_shots: 측정 횟수
        
    Returns:
        각 큐빗의 purity 리스트
    """
    n_qubits = circuit.num_qubits
    
    if n_qubits == 1:
        return [1.0]
    
    # 모든 큐빗에 대한 SWAP test 회로를 한 번에 구성
    purities = []
    
    # 각 큐빗에 대해 SWAP test 수행 (여전히 개별적이지만 최적화된 방식)
    for target_qubit in range(n_qubits):
        zero_probability, purity = _swap_test(circuit, target_qubit, num_shots)
        purities.append(max(0.0, min(1.0, purity)))
    
    return purities


def _single_qubit_purity(circuit: CircuitSpec, target_qubit: int, num_shots: int) -> float:
    """
    SWAP test를 사용한 단일 큐빗 purity 계산
    
    Args:
        circuit: 양자 회로 사양
        target_qubit: 대상 큐빗
        num_shots: 측정 횟수
    """
    n_qubits = circuit.num_qubits
    
    if n_qubits == 1:
        return 1.0
    
    # SWAP test 수행
    zero_probability, purity = _swap_test(circuit, target_qubit, num_shots)
    
    # purity는 0과 1 사이로 제한
    return max(0.0, min(1.0, purity))


def _swap_test(circuit: CircuitSpec, target_qubit: int, num_shots: int) -> tuple:
    """
    SWAP test 수행
    
    Args:
        circuit: 양자 회로 사양
        target_qubit: 대상 큐빗
        num_shots: 측정 횟수
        
    Returns:
        (zero_probability, purity)
    """
    n_qubits = circuit.num_qubits
    
    # SWAP test 회로 구성
    total_qubits = 2 * n_qubits + 1  # 두 복사본 + ancilla
    ancilla_idx = total_qubits - 1
    
    # 빈 회로 생성
    empty_spec = CircuitSpec(
        num_qubits=total_qubits,
        gates=[],
        circuit_id="swap_test_circuit"
    )
    base_qc_wrapper = QiskitQuantumCircuit(empty_spec)
    
    # 첫 번째 복사본에 원본 회로 적용
    _apply_circuit_to_qubits(base_qc_wrapper, circuit, 0)
    
    # 두 번째 복사본에 원본 회로 적용
    _apply_circuit_to_qubits(base_qc_wrapper, circuit, n_qubits)
    
    # Qiskit 회로 가져오기
    swap_qc = base_qc_wrapper.build()._qiskit_circuit
    creg_ancilla = ClassicalRegister(1, 'c_ancilla')
    swap_qc.add_register(creg_ancilla)
    
    # SWAP test 프로토콜
    swap_qc.h(ancilla_idx)
    swap_qc.cswap(ancilla_idx, target_qubit, target_qubit + n_qubits)
    swap_qc.h(ancilla_idx)
    
    # ancilla 측정
    swap_qc.measure(ancilla_idx, creg_ancilla[0])
    
    # 시뮬레이터로 실행
    simulator = AerSimulator()
    job = simulator.run(swap_qc, shots=num_shots)
    counts = job.result().get_counts()
    
    # ancilla=0 확률 계산
    ancilla_0_count = counts.get('0', 0)
    zero_probability = ancilla_0_count / num_shots
    
    # purity 계산: 2 * P(ancilla=0) - 1
    purity = 2 * zero_probability - 1
    
    return zero_probability, purity

def _apply_circuit_to_qubits(target_qc_wrapper: QiskitQuantumCircuit, circuit: CircuitSpec, qubit_offset: int):
    """원본 회로를 지정된 큐빗 오프셋에 적용"""
    for gate_spec in circuit.gates:
        offset_qubits = [q + qubit_offset for q in gate_spec.qubits]
        offset_gate = GateOperation(
            name=gate_spec.name,
            qubits=offset_qubits,
            parameters=gate_spec.parameters
        )
        target_qc_wrapper.add_gate(offset_gate)
    return target_qc_wrapper
        

def test_swap_test():
    """SWAP test 테스트"""
    from core.grid_graph_encoder import create_simple_circuit_example
    
    print("=== SWAP Test 테스트 ===")
    
    circuit = create_simple_circuit_example()
    print(f"테스트 회로: {circuit.num_qubits}큐빗, {len(circuit.gates)}게이트")
    
    mw_entropy = meyer_wallace_entropy_swap_test(circuit, num_shots=4096)
    
    print(f"\nMeyer-Wallach entropy: {mw_entropy:.6f}")
    
    return mw_entropy


def compare_with_exact():
    """SWAP test와 정확한 값 비교"""
    from core.grid_graph_encoder import create_simple_circuit_example
    from core.entangle_simulator import meyer_wallace_entropy
    
    print("=== SWAP Test vs 정확한 값 비교 ===")
    
    circuit = create_simple_circuit_example()
    
    # 정확한 값 (statevector 기반)
    exact_value = meyer_wallace_entropy(circuit)
    print(f"정확한 값 (statevector): {exact_value:.6f}")
    
    # SWAP test 값
    swap_value = meyer_wallace_entropy_swap_test(circuit, num_shots=8192)
    print(f"SWAP test: {swap_value:.6f}")
    
    # 차이 분석
    diff = abs(exact_value - swap_value)
    print(f"\n차이: {diff:.6f}")
    print(f"상대 오차: {diff/exact_value*100:.2f}%")
    
    return {"exact": exact_value, "swap": swap_value, "difference": diff}


if __name__ == "__main__":
    # 기본 테스트
    test_swap_test()
    
    print("\n" + "="*60 + "\n")
    
    # 정확한 값과 비교
    compare_with_exact()
