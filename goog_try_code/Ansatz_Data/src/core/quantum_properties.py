#!/usr/bin/env python3
"""
양자 회로 특성 계산 모듈 - 양자 회로의 다양한 수학적/물리적 특성을 계산합니다.
"""

import sys
import os
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

# 현재 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def calculate_quantum_properties(circuit_info, qc_transpiled=None):
    """
    실제 회로 구조에서 나오는 수치적 특성 계산
    
    Args:
        circuit_info (dict): 회로 정보
        qc_transpiled (QuantumCircuit, optional): 트랜스파일된 Qiskit 회로
        
    Returns:
        dict: 계산된 회로 특성
    """
    properties = {}
    
    try:
        # 기본 회로 정보
        n_qubits = circuit_info.get("n_qubits", 0)
        depth = circuit_info.get("depth", 0)
        gates = circuit_info.get("gates", [])
        wires_list = circuit_info.get("wires_list", [])
        
        # 게이트 통계
        gate_counts = {}
        for gate in gates:
            if gate in gate_counts:
                gate_counts[gate] += 1
            else:
                gate_counts[gate] = 1
        
        # 2큐빗 게이트 비율 계산
        two_qubit_gates = ["CNOT", "CZ", "SWAP"]
        two_qubit_count = sum(gate_counts.get(g, 0) for g in two_qubit_gates)
        total_gates = len(gates)
        two_qubit_ratio = two_qubit_count / total_gates if total_gates > 0 else 0
        
        # 파라미터화된 게이트 비율
        parameterized_gates = ["RX", "RY", "RZ"]
        param_gate_count = sum(gate_counts.get(g, 0) for g in parameterized_gates)
        param_gate_ratio = param_gate_count / total_gates if total_gates > 0 else 0
        
        # 회로 연결성 분석
        connectivity_matrix = np.zeros((n_qubits, n_qubits))
        for gate, wires in zip(gates, wires_list):
            if gate in two_qubit_gates and len(wires) >= 2:
                q1, q2 = wires[0], wires[1]
                if q1 < n_qubits and q2 < n_qubits:
                    connectivity_matrix[q1][q2] += 1
                    connectivity_matrix[q2][q1] += 1
        
        # 연결 밀도 (0~1)
        max_connections = n_qubits * (n_qubits - 1) / 2  # 최대 가능 연결 수
        actual_connections = np.count_nonzero(connectivity_matrix) / 2  # 실제 연결 수 (대칭 행렬)
        connectivity_density = actual_connections / max_connections if max_connections > 0 else 0
        
        # 큐빗당 게이트 수 분포
        qubit_gate_counts = [0] * n_qubits
        for gate, wires in zip(gates, wires_list):
            for wire in wires:
                if wire < n_qubits:
                    qubit_gate_counts[wire] += 1
        
        qubit_gate_stats = {
            "min": min(qubit_gate_counts) if qubit_gate_counts else 0,
            "max": max(qubit_gate_counts) if qubit_gate_counts else 0,
            "mean": sum(qubit_gate_counts) / n_qubits if n_qubits > 0 else 0,
            "std": np.std(qubit_gate_counts) if qubit_gate_counts else 0
        }
        
        # 결과 구성
        properties = {
            "n_qubits": n_qubits,
            "depth": depth,
            "total_gates": total_gates,
            "gate_counts": gate_counts,
            "two_qubit_count": two_qubit_count,
            "two_qubit_ratio": two_qubit_ratio,
            "param_gate_count": param_gate_count,
            "param_gate_ratio": param_gate_ratio,
            "connectivity_density": connectivity_density,
            "qubit_gate_stats": qubit_gate_stats
        }
        
        # Qiskit 트랜스파일드 회로 특성 (제공된 경우)
        if qc_transpiled:
            try:
                transpiled_depth = qc_transpiled.depth()
                transpiled_size = qc_transpiled.size()
                transpiled_properties = {
                    "transpiled_depth": transpiled_depth,
                    "transpiled_size": transpiled_size,
                    "transpiled_gate_counts": qc_transpiled.count_ops()
                }
                properties.update(transpiled_properties)
                
                # 기대 실행 시간 추정 (백엔드에 따라 다름)
                # 간단한 추정: 깊이 × 평균 게이트 시간 (5-10μs)
                estimated_time_us = transpiled_depth * 10  # 마이크로초 단위
                properties["estimated_time_us"] = estimated_time_us
            except Exception as e:
                print(f"Qiskit 트랜스파일 회로 분석 실패: {str(e)}")
        
    except Exception as e:
        print(f"회로 특성 계산 중 오류: {str(e)}")
    
    return properties


def create_coupling_features(coupling_map, n_qubits):
    """
    커플링맵을 트랜스포머가 이해하기 쉬운 수치적 특성으로 변환
    
    Args:
        coupling_map (list): 커플링 맵 (연결된 큐빗 쌍의 목록)
        n_qubits (int): 큐빗 수
        
    Returns:
        dict: 커플링 맵 특성
    """
    # 인접 행렬 생성
    adjacency = np.zeros((n_qubits, n_qubits))
    
    for pair in coupling_map:
        i, j = pair
        if i < n_qubits and j < n_qubits:
            adjacency[i, j] = 1
            adjacency[j, i] = 1
    
    # 연결성 통계
    connections_per_qubit = np.sum(adjacency, axis=1)
    avg_connections = np.mean(connections_per_qubit)
    std_connections = np.std(connections_per_qubit)
    min_connections = np.min(connections_per_qubit)
    max_connections = np.max(connections_per_qubit)
    
    # 그래프 지름
    diameter = calculate_graph_diameter(adjacency, n_qubits)
    
    # 클러스터링 계수
    clustering = calculate_clustering_coefficient(adjacency)
    
    return {
        "avg_connections": float(avg_connections),
        "std_connections": float(std_connections),
        "min_connections": int(min_connections),
        "max_connections": int(max_connections),
        "diameter": int(diameter),
        "clustering_coefficient": float(clustering),
        "connectivity_density": float(np.sum(adjacency) / (n_qubits * (n_qubits - 1))) if n_qubits > 1 else 0.0
    }


def calculate_graph_diameter(adjacency, n_qubits):
    """
    그래프의 지름 계산 (BFS 기반)
    
    Args:
        adjacency (numpy.ndarray): 인접 행렬
        n_qubits (int): 큐빗 수
        
    Returns:
        int: 그래프 지름
    """
    max_distance = 0
    
    for start in range(n_qubits):
        # BFS로 모든 노드까지의 거리 계산
        visited = [False] * n_qubits
        distance = [float('inf')] * n_qubits
        
        visited[start] = True
        distance[start] = 0
        queue = [start]
        
        while queue:
            node = queue.pop(0)
            
            for neighbor in range(n_qubits):
                if adjacency[node, neighbor] > 0 and not visited[neighbor]:
                    visited[neighbor] = True
                    distance[neighbor] = distance[node] + 1
                    queue.append(neighbor)
        
        # 연결되지 않은 노드는 무시하고 최대 거리 계산
        finite_distances = [d for d in distance if d != float('inf')]
        if finite_distances:
            max_distance = max(max_distance, max(finite_distances))
    
    return max_distance if max_distance != float('inf') else -1  # -1은 연결되지 않은 그래프 의미


def calculate_clustering_coefficient(adjacency):
    """
    클러스터링 계수 계산
    
    Args:
        adjacency (numpy.ndarray): 인접 행렬
        
    Returns:
        float: 평균 클러스터링 계수
    """
    n = adjacency.shape[0]
    clustering_coefficients = []
    
    for i in range(n):
        neighbors = [j for j in range(n) if adjacency[i, j] > 0]
        k_i = len(neighbors)
        
        if k_i < 2:  # 이웃이 1개 이하면 삼각형이 불가능
            clustering_coefficients.append(0)
            continue
        
        # 이웃 간 연결 수 계산
        triangles = 0
        for j in range(len(neighbors)):
            for k in range(j + 1, len(neighbors)):
                if adjacency[neighbors[j], neighbors[k]] > 0:
                    triangles += 1
        
        # 클러스터링 계수 = 실제 삼각형 수 / 가능한 삼각형 수
        possible_triangles = k_i * (k_i - 1) / 2
        c_i = triangles / possible_triangles if possible_triangles > 0 else 0
        clustering_coefficients.append(c_i)
    
    # 평균 클러스터링 계수
    return np.mean(clustering_coefficients) if clustering_coefficients else 0.0
