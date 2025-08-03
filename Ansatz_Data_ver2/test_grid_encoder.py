#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.grid_graph_encoder import GridGraphEncoder, create_simple_circuit_example

def test_grid_encoder():
    """그리드 인코더 테스트"""
    print("=== Grid-Graph Hybrid Encoder Test ===\n")
    
    # 인코더 생성
    encoder = GridGraphEncoder()
    
    # 테스트 회로 생성
    circuit = create_simple_circuit_example()
    print(f"Test Circuit: {circuit.circuit_id}")
    print(f"Qubits: {circuit.num_qubits}")
    print(f"Gates: {len(circuit.gates)}")
    
    # 각 게이트 정보 출력
    print("\nGates in circuit:")
    for i, gate in enumerate(circuit.gates):
        print(f"  {i}: {gate.name} on qubits {gate.qubits} (params: {gate.parameters})")
    
    # 인코딩 수행
    print("\n=== Encoding ===")
    encoded = encoder.encode(circuit)
    
    print(f"Total Nodes: {len(encoded['nodes'])}")
    print(f"Total Edges: {len(encoded['edges'])}")
    print(f"Grid Shape: {encoded['grid_shape']}")
    
    # 노드 정보 출력
    print("\nNodes:")
    for node in encoded['nodes']:
        print(f"  {node['id']}: {node['node_name']} at {node['grid_position']} (qubits: {node['qubits']})")
    
    # 에지 정보 출력
    print("\nEdges:")
    for edge in encoded['edges']:
        edge

    # 그리드 시각화
    print("\n=== Grid Visualization ===")
    print(encoder.visualize_grid(encoded))
    
    # 그리드 매트릭스 출력
    print("\n=== Grid Matrix ===")
    grid_data = encoder.to_grid_matrix(encoded)
    print(f"Grid Matrix Shape: {grid_data['grid_shape']}")
    print("Grid Matrix:")
    for i, row in enumerate(grid_data['grid_matrix']):
        print(f"  Time {i}: {row}")
    
    print("\nConnections:")
    for conn in grid_data['connections']:
        print(conn)
if __name__ == "__main__":
    test_grid_encoder()
