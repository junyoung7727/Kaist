"""
그리드-그래프 하이브리드 양자 회로 인코더 (Grid-Graph Hybrid Quantum Circuit Encoder)

이 모듈은 양자 회로를 그리드 기반의 간단한 그래프 표현으로 변환합니다:

1. 그리드 위치 인코딩:
   - 각 게이트는 [parallel_order, qubit_index] 그리드 위치로 표현
   - parallel_order: 해당 큐빗에서의 순서 (시간축)
   - qubit_index: 큐빗 인덱스 (공간축)

2. 최소한의 노드 특성:
   - 게이트 타입, 파라미터, 기본 속성만 포함
   - 복잡한 중앙성이나 위치 계산 제거

3. 명시적 에지 연결:
   - 다중 큐빗 게이트(cx 등)는 에지로 명시적 연결
   - 에지 정보로 비지역적 연결 표현 (예: [1,4])

이 접근법은 그래프와 그리드의 장점을 결합하여 효율적이고 직관적인 표현을 제공합니다.
"""

from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from core.circuit_interface import CircuitSpec
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))
from gates import GateType, GateOperation


@dataclass
class GridPosition:
    """그리드 위치를 나타내는 데이터 클래스"""
    parallel_order: int  # 해당 큐빗에서의 순서 (시간축)
    qubit_index: int     # 큐빗 인덱스 (공간축)
    
    def to_list(self) -> List[int]:
        return [self.parallel_order, self.qubit_index]


class GridGraphEncoder:
    """그리드-그래프 하이브리드 양자 회로 인코더"""
    
    def __init__(self):
        self.gate_registry = QuantumGateRegistry()
    
    def encode(self, circuit: CircuitSpec) -> Dict[str, Any]:
        """
        회로를 그리드-그래프 하이브리드 구조로 인코딩
        
        Args:
            circuit: 양자 회로 (CircuitSpec 객체)
        
        Returns:
            Dict[str, Any]: 인코딩된 회로 데이터
                - nodes: 게이트 노드 리스트
                - edges: 다중 큐빗 연결 에지 리스트
                - grid_shape: [max_parallel_order, num_qubits]
                - circuit_info: 회로 메타데이터
        """
        nodes = []
        edges = []
        
        # 각 큐빗별 병렬 순서 추적
        qubit_parallel_order = [0 for _ in range(circuit.num_qubits)]
        
        # 각 게이트에 대해 노드 생성
        for i, gate in enumerate(circuit.gates):

            # 게이트 기본 정보
            gate_name = gate.name.lower()
            gate_type = self.gate_registry.get_gate_type(gate_name)
            num_qubits = len(gate.qubits)
            
            # 게이트 특성
            is_hermitian = _is_hermitian(gate)
            is_parameterized = bool(gate.parameters)
            
            # 파라미터 특성
            param_features = {}
            if is_parameterized:
                param_features['parameter_value'] = float(gate.parameters[0] or 0.0)
                param_features['has_parameter'] = 1.0
            else:
                param_features['parameter_value'] = 0.0
                param_features['has_parameter'] = 0.0
            
            # 단일 큐빗 게이트 처리
            if gate_type == GateType.SINGLE_QUBIT or gate_type == GateType.PARAMETRIC:
                qubit_idx = gate.qubits[0]
                parallel_order = qubit_parallel_order[qubit_idx]
                qubit_parallel_order[qubit_idx] += 1    
                
                gate_id = f'{gate_name}_q{qubit_idx}_{parallel_order}'
                
                gate_node = {
                    'id': gate_id,
                    'node_name': f'gate_{gate_name}',
                    'type': 'GATE',
                    'gate_name': gate_name,
                    'is_hermitian': is_hermitian,
                    'is_parameterized': is_parameterized,
                    'grid_position': [parallel_order, qubit_idx],
                    'qubits': [qubit_idx],
                    **param_features
                }
                nodes.append(gate_node)
            
            # 다중 큐빗 게이트 처리
            elif gate_type == GateType.TWO_QUBIT or gate_type == GateType.TWO_QUBIT_PARAMETRIC:
                # 컨트롤과 타겟 큐빗
                control_qubit = gate.qubits[0]  # 첫 번째는 컨트롤
                target_qubit = gate.qubits[1]   # 두 번째는 타겟

                # 2큐빗 게이트는 동시에 실행되므로 두 큐빗 중 더 늦은 시점에 배치
                control_order = qubit_parallel_order[control_qubit]
                target_order = qubit_parallel_order[target_qubit]
                sync_order = max(control_order, target_order)
                
                # 두 큐빗 모두 동일한 시점으로 동기화 후 증가
                qubit_parallel_order[control_qubit] = sync_order + 1
                qubit_parallel_order[target_qubit] = sync_order + 1

                # 실제 배치될 위치는 sync_order
                control_order = sync_order
                target_order = sync_order
                    
                # 컨트롤 노드
                control_id = f'{gate_name}_control_q{control_qubit}_{control_order}'
                control_node = {
                    'id': control_id,
                    'node_name': f'source_{gate_name}',
                    'type': 'GATE',
                    'gate_name': gate_name,
                    'role': 'control',
                    'is_hermitian': is_hermitian,
                    'is_parameterized': is_parameterized,
                    'grid_position': [control_order, control_qubit],
                    'qubits': [control_qubit],
                    **param_features
                }
                nodes.append(control_node)
                
                # 타겟 노드
                target_id = f'{gate_name}_target_q{target_qubit}_{target_order}'
                target_node = {
                    'id': target_id,
                    'type': 'GATE',
                    'gate_name': gate_name,
                    'role': 'target',
                    'is_hermitian': is_hermitian,
                    'is_parameterized': is_parameterized,
                    'grid_position': [target_order, target_qubit],
                    'qubits': [target_qubit],
                    **param_features
                }
                nodes.append(target_node)
                
                # 에지 연결 (컨트롤 -> 타겟)
                entangle_edge = {
                    'id': f'entangle_edge_{i}_{gate_name}',
                    'type': 'ENTANGLE_CONNECTION',
                    'source': [control_order, control_qubit],
                    'target': [target_order, target_qubit],
                }

                edges.append(entangle_edge)

        qubit_parallel_order = [_ - 1 for _ in qubit_parallel_order] # i++ 메커니즘 맨 마지막 빈 레이어 회로로 제거

        for qubit in range(circuit.num_qubits):
            for order in range((qubit_parallel_order[qubit]))[:-1]:
                # 레지스터 에지
                register_edge = {
                    'id': f'register_edge_{order}_{qubit}',
                    'type': 'REGISTER_CONNECTION',
                    'source': [order, qubit],  # 그리드 좌표 간 연결
                    'target': [order+1, qubit],  # 그리드 좌표 간 연결
                }

                edges.append(register_edge)      

        # 그리드 형태 계산
        max_parallel_order = max(qubit_parallel_order)+1 if qubit_parallel_order else 0
        grid_shape = [max_parallel_order, circuit.num_qubits]
        
        return {
            'nodes': nodes,
            'edges': edges,
            'grid_shape': grid_shape,
            'circuit_info': {
                'circuit_id': circuit.circuit_id,
                'num_qubits': circuit.num_qubits,
                'num_gates': len(circuit.gates),
                'total_nodes': len(nodes)
            }
        }
    
    def to_grid_matrix(self, encoded_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        인코딩된 데이터를 그리드 매트릭스 형태로 변환
        
        Args:
            encoded_data: encode() 메서드의 출력
            
        Returns:
            그리드 매트릭스와 연결 정보
        """
        grid_shape = encoded_data['grid_shape']
        nodes = encoded_data['nodes']
        edges = encoded_data['edges']
        
        # 그리드 매트릭스 초기화 (None으로 빈 셀 표시)
        grid_matrix = [[None for _ in range(grid_shape[1])] for _ in range(grid_shape[0])]
        
        # 노드를 그리드에 배치
        for node in nodes:
            pos = node['grid_position']
            parallel_order, qubit_idx = pos[0], pos[1]
            grid_matrix[parallel_order][qubit_idx] = node['id']
        
        # 연결 매트릭스 (에지 정보)
        connections = []
        for edge in edges:
            connections.append({
                'grid_connection': f'{edge["source"]} --> {edge["target"]}',
            })
        
        return {
            'grid_matrix': grid_matrix,
            'connections': connections,
            'grid_shape': grid_shape,
            'node_lookup': {node['id']: node for node in nodes}
        }
    
    def visualize_grid(self, encoded_data: Dict[str, Any]) -> str:
        """
        그리드 형태로 회로를 시각화 (텍스트)
        
        Args:
            encoded_data: encode() 메서드의 출력
            
        Returns:
            텍스트 기반 그리드 시각화
        """
        grid_data = self.to_grid_matrix(encoded_data)
        grid_matrix = grid_data['grid_matrix']
        connections = grid_data['connections']
        node_lookup = grid_data['node_lookup']
        
        # 그리드 시각화
        lines = []
        lines.append("Circuit Grid Visualization:")
        lines.append("=" * 50)
        
        # 큐빗 헤더
        header = "Time\\Qubit "
        for q in range(len(grid_matrix[0])):
            header += f"  Q{q:2d}  "
        lines.append(header)
        lines.append("-" * len(header))
        
        # 각 시간 단계
        for t, row in enumerate(grid_matrix):
            line = f"   {t:2d}     "
            for cell in row:
                if cell is None:
                    line += "  ---  "
                else:
                    # 게이트 이름 축약
                    node = node_lookup[cell]
                    gate_short = node['gate_name'][:3].upper()
                    line += f" {gate_short:>3s}  "
            lines.append(line)
        
        # 연결 정보
        if connections:
            lines.append("\nConnections:")
            lines.append("-" * 20)
            for conn in connections:
                lines.append(f"{conn['grid_connection']}")
        
        return "\n".join(lines)


# 사용 예시를 위한 헬퍼 함수
def create_simple_circuit_example():
    """간단한 테스트 회로 생성 (얽힘을 만들기 위한 부분 추가)"""
    from circuit_interface import CircuitSpec
    
    gates = [
        # 초기화: 모든 큐빗을 중첩 상태로 만들기
        GateOperation('h', [0]),  # 쿼트럼 중첩 상태 생성
        GateOperation('h', [1]),
        GateOperation('h', [2]),
        GateOperation('h', [3]),
        GateOperation('h', [4]),
        # 추가 복잡성
        GateOperation('rz', [0], [1.5]),
        GateOperation('rz', [1], [2.3]),
        GateOperation('rz', [2], [3.2]),
        GateOperation('rz', [3], [1.1]),
        GateOperation('rz', [4], [2.4]),
        
        # 다양한 얽힘을 생성하는 게이트
        GateOperation('cx', [0, 1]),  # Bell 상태 생성
        GateOperation('cx', [1, 2]),  # GHZ 상태로 확장
        GateOperation('cx', [2, 3]),
        GateOperation('cx', [3, 4]),
    ]
    
    return CircuitSpec(
        circuit_id="test_grid",
        num_qubits=5,
        gates=gates
    )


if __name__ == "__main__":
    # 테스트 실행
    encoder = GridGraphEncoder()
    circuit = create_simple_circuit_example()
    
    # 인코딩
    encoded = encoder.encode(circuit)
    print("Encoded Circuit:")
    print(f"Nodes: {len(encoded['nodes'])}")
    print(f"Edges: {len(encoded['edges'])}")
    print(f"Grid Shape: {encoded['grid_shape']}")
    
    # 시각화
    print("\n" + encoder.visualize_grid(encoded))
