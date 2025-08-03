from core.circuit_interface import CircuitBuilder, CircuitSpec
import numpy as np
from config import ExperimentConfig
from typing import List, Dict, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))
from gates import GateOperation, gate_registry, GateType


def generate_random_circuit(exp_config: ExperimentConfig) -> List[CircuitSpec]:
    """
    Generate random quantum circuit specifications based on experiment configuration.
    
    Args:
        exp_config: 실험 설정 (num_qubits can be int or list of ints)
        
    Returns: 
        List of random circuit specifications (one per num_qubits value)
    """
    # 여러 큐빗 수에 대한 회로 생성
    circuits = []
    
    # depth 값이 리스트가 아닌 경우 리스트로 변환
    depths = exp_config.depth if isinstance(exp_config.depth, list) else [exp_config.depth]
    
    # exp_config.num_qubits는 __post_init__에 의해 항상 리스트 형태임
    for num_qubits in exp_config.num_qubits:
        # 각 큐빗 수에 대해 지정된 회로 수만큼 생성
        for two_qubit_ratio in exp_config.two_qubit_ratio:
            for depth in depths:
                for i in range(exp_config.num_circuits):
                    builder = CircuitBuilder()
                    builder.set_qubits(num_qubits)  # 특정 큐빗 수 설정
                    circuit_id = f"{exp_config.exp_name or 'circuit'}_{num_qubits}q_d{depth}_r{two_qubit_ratio}_{i}"
                    builder.set_circuit_id(circuit_id)
                
                    # 레지스트리에서 게이트 타입별로 게이트 가져오기
                    single_gates = []
                    parametric_single_gates = []
                    two_qubit_gates = []
                    parametric_two_qubit_gates = []
                    
                    # 레지스트리에서 모든 게이트를 타입별로 분류
                    for gate_name, gate_def in gate_registry._gates.items():
                        if gate_def.gate_type == GateType.SINGLE_QUBIT:
                            single_gates.append(gate_name)
                        elif gate_def.gate_type == GateType.PARAMETRIC:
                            parametric_single_gates.append(gate_name)
                        elif gate_def.gate_type == GateType.TWO_QUBIT:
                            two_qubit_gates.append(gate_name)
                        elif gate_def.gate_type == GateType.TWO_QUBIT_PARAMETRIC:
                            parametric_two_qubit_gates.append(gate_name)
                    
                    # 1. 전체 게이트 수 계산 및 게이트 위치 준비
                    total_gates_per_layer = num_qubits + 1   # 각 레이어에서 큐빗 당 평균 1개 게이트
                    total_gates = depth * total_gates_per_layer
                    
                    # 2. 2큐빗 게이트 수 계산 (전체 게이트의 two_qubit_ratio 비율)
                    num_two_qubit_gates = int(total_gates * two_qubit_ratio)
                    num_single_qubit_gates = total_gates - num_two_qubit_gates
                    
                    # 3. 게이트 위치 정보 준비 (위치와 게이트 종류를 나중에 랜덤하게 선택하기 위함)
                    gate_positions = []
                    
                    # 레이어별로 게이트 위치 미리 계산산
                    for layer in range(depth):
                        for _ in range(total_gates_per_layer):
                            gate_positions.append((layer, "placeholder"))
                    
                    # 4. 랜덤하게 셔플하여 게이트 위치 결정
                    np.random.shuffle(gate_positions)

                    # 5. 처음 num_two_qubit_gates 개의 위치에 2큐빗 게이트 배정
                    for i in range(len(gate_positions)):
                        if gate_positions[i][0] == 0:
                            gate_positions[i] = (gate_positions[i][0], "single_qubit")
                        else:
                            if i < num_two_qubit_gates and num_qubits > 1:
                                gate_positions[i] = (gate_positions[i][0], "two_qubit")
                            else:
                                gate_positions[i] = (gate_positions[i][0], "single_qubit")
                
                    # 6. 레이어별로 정렬 (시간 순서대로 게이트 적용을 위해)
                    gate_positions.sort(key=lambda x: x[0])
                    
                    # 7. 게이트 적용
                    for layer, gate_type in gate_positions:
                        if gate_type == "two_qubit":
                            # 2큐빗 게이트 적용
                            # 파라메트릭/비파라메트릭 2큐빗 게이트 중 선택
                            use_parametric = np.random.random() < 0.8  # 80% 확률로 파라메트릭 2큐빗 게이트 선택
                            
                            if use_parametric and parametric_two_qubit_gates:
                                gate_name = np.random.choice(parametric_two_qubit_gates)
                                control = np.random.randint(0, num_qubits)
                                target = np.random.randint(0, num_qubits)
                                while target == control:  # 다른 큐빗 선택
                                    target = np.random.randint(0, num_qubits)
                                param = np.random.uniform(0, 2 * np.pi)
                                builder.add_gate(gate_name, [control, target], [param])
                            elif two_qubit_gates:  # 비파라메트릭 2큐빗 게이트
                                gate_name = np.random.choice(two_qubit_gates)
                                control = np.random.randint(0, num_qubits)
                                target = np.random.randint(0, num_qubits)
                                while target == control:
                                    target = np.random.randint(0, num_qubits)
                                builder.add_gate(gate_name, [control, target])
                        else:
                            # 단일 큐빗 게이트 적용
                            qubit = np.random.randint(0, num_qubits)
                            # 파라메트릭/비파라메트릭 게이트 중 선택
                            use_parametric = np.random.random() < 0.9  # 90% 확률로 파라메트릭 게이트 선택
                            
                            if use_parametric and parametric_single_gates:
                                gate_name = np.random.choice(parametric_single_gates)
                                param = np.random.uniform(0, 2 * np.pi)
                                builder.add_gate(gate_name, qubit, [param])
                            elif single_gates:  # 비파라메트릭 게이트
                                gate_name = np.random.choice(single_gates)
                                builder.add_gate(gate_name, qubit)
                
                    circuits.append(builder.build_spec())
    
    return circuits




def create_random_parameterized_samples(circuit_spec: CircuitSpec, n: int) -> List[CircuitSpec]:
    """
    주어진 회로 사양의 파라미터만 랜덤하게 변경하여 n개의 샘플을 생성합니다.
    
    Args:
        circuit_spec: 파라미터를 랜덤화할 원본 회로 사양
        n: 생성할 샘플 수
    
    Returns:
        List[CircuitSpec]: 파라미터가 랜덤화된 새 회로 사양 리스트
    """
    random_param_circuits = []

    # n개의 샘플 생성
    for i in range(n):
        new_gates = []  # 각 샘플마다 gates 리스트 초기화
        
        # 각 게이트의 파라미터를 랜덤하게 변경
        for gate in circuit_spec.gates:
            if gate.parameters and len(gate.parameters) > 0:
                # 파라미터가 있는 게이트라면 랜덤 값으로 교체
                new_params = [np.random.uniform(0, 2 * np.pi) for _ in range(len(gate.parameters))]
                new_gate = GateOperation(gate.name, gate.qubits, new_params)
                new_gates.append(new_gate)
            else:
                # 파라미터가 없는 게이트는 그대로 유지
                new_gates.append(gate)

        # 이 샘플에 대한 CircuitSpec 생성
        new_circuit_spec = CircuitSpec(
            num_qubits=circuit_spec.num_qubits,
            gates=new_gates,
            circuit_id=f"{circuit_spec.circuit_id}_sample_{i}",
        )
        random_param_circuits.append(new_circuit_spec)
    # 수정된 게이트로 새 회로 사양들 반환
    return random_param_circuits