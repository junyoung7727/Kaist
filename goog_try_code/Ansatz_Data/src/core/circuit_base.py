#!/usr/bin/env python3
"""
양자 회로 기본 모듈
회로 생성 및 관리 기능 제공
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json
import uuid
from datetime import datetime
import os
import random
import statistics
import pennylane as qml
from qiskit import transpile

# 설정 모듈 임포트
from src.config import config

# 이제 config는 직접 사용 가능

class QuantumCircuitBase:
    """양자 회로 기본 클래스 - 회로 생성 및 기본 작업 처리"""
    
    def __init__(self, output_dir="grid_circuits"):
        """
        기본 양자 회로 생성기
        
        Args:
            output_dir (str): 출력 디렉토리
        """
        self.output_dir = output_dir
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "coherence_data"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "ansatz_data"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)  # 이미지 디렉토리 추가
        
        # 확장된 하드웨어 호환 게이트 정의
        self.single_qubit_gates = ["H", "X", "Y", "Z", "S", "T", "RZ", "RX", "RY"]
        self.two_qubit_gates = ["CNOT", "CZ"]  # CZ 게이트 추가
        self.parametric_gates = ["RZ", "RX", "RY"]  # 파라미터 게이트 확장
        self.all_gates = self.single_qubit_gates + self.two_qubit_gates
        
        # 회로 생성 전략 정의
        self.strategies = {
            "hardware_efficient": self._generate_hardware_efficient,
            "random": self._generate_random
        }
    
    def generate_random_circuit(self, n_qubits=4, depth=3, strategy="random", seed=None, two_qubit_ratio=0.3):
        """
        랜덤 양자 회로 생성
        
        Args:
            n_qubits (int): 큐빗 수
            depth (int): 회로 깊이
            strategy (str): 회로 생성 전략 ("random" 또는 "hardware_efficient")
            seed (int): 랜덤 시드 (재현성)
            two_qubit_ratio (float): 2-큐빗 게이트 비율 (0-1 사이)
            
        Returns:
            dict: 회로 정보
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 회로 정보
        circuit_info = {
            "n_qubits": n_qubits,
            "depth": depth,
            "strategy": strategy,
            "gates": [],
            "wires_list": [],
            "params_idx": [],
            "params": [],
            "metadata": {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "seed": seed,
                "two_qubit_ratio": two_qubit_ratio
            }
        }
        
        # 선택된 전략으로 회로 생성
        if strategy in self.strategies:
            self.strategies[strategy](circuit_info, two_qubit_ratio)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Available strategies: {list(self.strategies.keys())}")
        
        return circuit_info
    
    def _generate_hardware_efficient(self, circuit_info, two_qubit_ratio=0.3):
        """
        Hardware-Efficient Ansatz 생성 전략
        
        Args:
            circuit_info (dict): 회로 정보 (수정됨)
            two_qubit_ratio (float): 2-큐빗 게이트 비율
        """
        n_qubits = circuit_info["n_qubits"]
        depth = circuit_info["depth"]
        
        # 각 깊이마다 게이트 추가
        for d in range(depth):
            # 단일 큐빗 회전 게이트 레이어
            for q in range(n_qubits):
                # 랜덤 파라미터 회전 게이트
                gate = np.random.choice(self.parametric_gates)
                param = np.random.uniform(0, 2*np.pi)
                circuit_info["gates"].append(gate)
                circuit_info["wires_list"].append([q])
                circuit_info["params_idx"].append(len(circuit_info["gates"]) - 1)
                circuit_info["params"].append(param)
            
            # 얽힘 레이어 (CZ 또는 CNOT)
            if n_qubits >= 2:  # 2개 이상의 큐빗이 있어야 얽힘 가능
                # 가능한 모든 인접 큐빗 쌍
                qubit_pairs = []
                for i in range(n_qubits):
                    for j in range(i+1, n_qubits):
                        # 1차원 체인 구조로 제한
                        if abs(i - j) == 1:
                            qubit_pairs.append((i, j))
                
                # 2-큐빗 게이트 수 결정 (비율에 기반)
                n_two_qubit_gates = max(1, int(len(qubit_pairs) * two_qubit_ratio))
                
                # 랜덤하게 쌍 선택
                selected_pairs = random.sample(qubit_pairs, min(n_two_qubit_gates, len(qubit_pairs)))
                
                for q1, q2 in selected_pairs:
                    gate = np.random.choice(self.two_qubit_gates)
                    circuit_info["gates"].append(gate)
                    circuit_info["wires_list"].append([q1, q2])
    
    def _generate_random(self, circuit_info, two_qubit_ratio=0.3):
        """
        완전히 랜덤한 회로 생성 전략
        
        Args:
            circuit_info (dict): 회로 정보 (수정됨)
            two_qubit_ratio (float): 2-큐빗 게이트 비율
        """
        n_qubits = circuit_info["n_qubits"]
        depth = circuit_info["depth"]
        total_gates = n_qubits * depth
        
        # 2-큐빗 게이트 수 결정
        n_two_qubit_gates = int(total_gates * two_qubit_ratio)
        n_one_qubit_gates = total_gates - n_two_qubit_gates
        
        # 게이트 타입 목록 생성
        gates_types = ["one"] * n_one_qubit_gates + ["two"] * n_two_qubit_gates
        random.shuffle(gates_types)
        
        # 게이트 추가
        for gate_type in gates_types:
            if gate_type == "one":
                # 단일 큐빗 게이트
                q = random.randint(0, n_qubits - 1)
                gate = random.choice(self.single_qubit_gates)
                circuit_info["gates"].append(gate)
                circuit_info["wires_list"].append([q])
                
                # 파라메트릭 게이트인 경우 파라미터 추가
                if gate in self.parametric_gates:
                    param = np.random.uniform(0, 2*np.pi)
                    circuit_info["params_idx"].append(len(circuit_info["gates"]) - 1)
                    circuit_info["params"].append(param)
            
            else:
                # 2-큐빗 게이트 (최소 2개의 큐빗 필요)
                if n_qubits >= 2:
                    q1 = random.randint(0, n_qubits - 1)
                    q2 = random.randint(0, n_qubits - 1)
                    while q2 == q1:  # 다른 큐빗이어야 함
                        q2 = random.randint(0, n_qubits - 1)
                    
                    gate = random.choice(self.two_qubit_gates)
                    circuit_info["gates"].append(gate)
                    circuit_info["wires_list"].append([q1, q2])

    def save_circuit(self, circuit_info, filename=None):
        """
        회로 정보를 JSON 파일로 저장
        
        Args:
            circuit_info (dict): 회로 정보
            filename (str): 파일 이름 (기본값: 자동 생성)
            
        Returns:
            str: 저장된 파일 경로
        """
        if filename is None:
            strategy = circuit_info.get("strategy", "unknown")
            n_qubits = circuit_info.get("n_qubits", 0)
            depth = circuit_info.get("depth", 0)
            circuit_id = circuit_info.get("metadata", {}).get("id", str(uuid.uuid4()))
            filename = f"{strategy}_q{n_qubits}_d{depth}_{circuit_id[:8]}.json"
        
        filepath = os.path.join(self.output_dir, "ansatz_data", filename)
        
        with open(filepath, 'w') as f:
            json.dump(circuit_info, f, indent=2)
        
        return filepath

# 회로 빌드 유틸리티 함수
def build_qiskit_circuit(circuit_info, max_qubits=None):
    """
    Qiskit 회로 객체 생성
    
    Args:
        circuit_info (dict): 회로 정보
        max_qubits (int): 최대 큐빗 수 제한 (선택사항)
        
    Returns:
        qiskit.QuantumCircuit: Qiskit 양자 회로 객체
    """
    from qiskit import QuantumCircuit
    
    n_qubits = circuit_info.get("n_qubits")
    gates = circuit_info.get("gates")
    wires_list = circuit_info.get("wires_list")
    params_idx = circuit_info.get("params_idx", [])
    params = circuit_info.get("params", [])
    
    # 큐빗 수 제한 (옵션)
    if max_qubits is not None and n_qubits > max_qubits:
        n_qubits = max_qubits
    
    # Qiskit 양자 회로 생성
    qc = QuantumCircuit(n_qubits)
    
    # 게이트 적용
    for i, (gate, wires) in enumerate(zip(gates, wires_list)):
        # 큐빗 범위 확인
        if any(w >= n_qubits for w in wires):
            continue
            
        if gate == "H":
            qc.h(wires[0])
        elif gate == "X":
            qc.x(wires[0])
        elif gate == "Y":
            qc.y(wires[0])
        elif gate == "Z":
            qc.z(wires[0])
        elif gate == "S":
            qc.s(wires[0])
        elif gate == "T":
            qc.t(wires[0])
        elif gate == "RZ":
            # 파라미터 찾기
            param_value = None
            for j, idx in enumerate(params_idx):
                if idx == i:
                    param_value = params[j]
                    break
            if param_value is not None:
                qc.rz(param_value, wires[0])
        elif gate == "RX":
            param_value = None
            for j, idx in enumerate(params_idx):
                if idx == i:
                    param_value = params[j]
                    break
            if param_value is not None:
                qc.rx(param_value, wires[0])
        elif gate == "RY":
            param_value = None
            for j, idx in enumerate(params_idx):
                if idx == i:
                    param_value = params[j]
                    break
            if param_value is not None:
                qc.ry(param_value, wires[0])
        elif gate == "CZ":
            if len(wires) >= 2:
                qc.cz(wires[0], wires[1])
        elif gate == "CNOT":
            if len(wires) >= 2:
                qc.cx(wires[0], wires[1])
    
    return qc
