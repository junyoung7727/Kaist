import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json
import uuid
from datetime import datetime
import os
from tqdm import tqdm
import random
import statistics
import pennylane as qml
from config import config, get_shadow_params, get_simulator_shots
from expressibility_calculator import ExpressibilityCalculator
from qiskit import transpile

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
        self.circuit_strategies = {
            "hardware_efficient": {
                "single_gate_prob": 0.6,
                "two_qubit_prob": 0.8,
                "param_gate_ratio": 0.4,
                "layer_pattern": "alternating"
            },
            "expressibility_focused": {
                "single_gate_prob": 0.7,
                "two_qubit_prob": 0.9,
                "param_gate_ratio": 0.6,
                "layer_pattern": "dense"
            },
            "noise_resilient": {
                "single_gate_prob": 0.4,
                "two_qubit_prob": 0.5,
                "param_gate_ratio": 0.3,
                "layer_pattern": "sparse"
            }
        }
        
        # 표현력 계산기 초기화
        self.expressibility_calculator = ExpressibilityCalculator()
        
    def calculate_entropy(self, measurement_data):
        """
        측정 결과의 엔트로피 계산 (IBM 백엔드 호환)
        
        이 메서드는 양자 회로의 측정 결과로부터 샤논 엔트로피를 계산합니다.
        출력 분포의 불확실성을 측정하며, 값이 클수록 출력이 균일하고,
        작을수록 특정 상태에 집중된 분포를 나타냅니다.
        
        Args:
            measurement_data: 측정 결과 (다양한 형식 지원)
                - dict: {'00': 100, '01': 50, ...} 형태의 측정 카운트
                - list: [{'state': '00', 'count': 100}, ...] 형태의 측정 리스트
                - qiskit.result.Result: Qiskit의 Result 객체
                
        Returns:
            float: 측정 결과의 샤논 엔트로피 (비트 단위)
        """
        from qiskit.result import Result
        from expressibility_calculator import calculate_measurement_entropy
        
        try:
            # Qiskit Result 객체인 경우 처리
            if isinstance(measurement_data, Result):
                # 첫 번째 결과의 첫 번째 실험 결과 사용
                if not measurement_data.results:
                    return 0.0
                counts = measurement_data.get_counts(0)  # 첫 번째 실험 결과의 카운트 가져오기
                return calculate_measurement_entropy(counts)
            
            # 그 외의 경우는 calculate_measurement_entropy에 위임
            return calculate_measurement_entropy(measurement_data)
            
        except Exception as e:
            print(f"엔트로피 계산 중 오류 발생: {str(e)}")
            return 0.0
    
    def generate_random_circuit(self, n_qubits, depth, coupling_map=None, strategy="hardware_efficient", seed=None, two_qubit_ratio=None):
        """
        고급 레이어 기반 랜덤 양자 회로 생성
        
        Args:
            n_qubits (int): 큐빗 수
            depth (int): 레이어 수 (깊이)
            coupling_map (list): 커플링 맵 (None이면 격자 구조 사용)
            strategy (str): 회로 생성 전략 ("hardware_efficient", "expressibility_focused", "noise_resilient")
            seed (int): 랜덤 시드 (재현 가능한 회로 생성용)
            two_qubit_ratio (float): 2큐빗 게이트 비율 (0.0~1.0, None이면 전략 기본값 사용)
            
        Returns:
            dict: 회로 정보
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 전략 설정
        config = self.circuit_strategies.get(strategy, self.circuit_strategies["hardware_efficient"]).copy()
        
        # 2큐빗 게이트 비율 오버라이드
        if two_qubit_ratio is not None:
            if not (0.0 <= two_qubit_ratio <= 1.0):
                raise ValueError("two_qubit_ratio는 0.0과 1.0 사이여야 합니다.")
            config["two_qubit_prob"] = two_qubit_ratio
            print(f"2큐빗 게이트 비율 설정: {two_qubit_ratio:.1%}")
        
        gates = []
        wires_list = []
        params = []
        params_idx = []
        
        # 효율적인 커플링 맵 생성 및 인덱싱
        coupling_map, adjacency_dict = self._create_optimized_coupling_map(n_qubits, coupling_map)
        
        # 레이어별 게이트 생성 (개선된 패턴)
        for d in range(depth):
            layer_gates, layer_wires, layer_params, layer_params_idx = self._generate_layer(
                d, n_qubits, adjacency_dict, config, len(gates)
            )
            
            # 레이어 결과 통합
            gates.extend(layer_gates)
            wires_list.extend(layer_wires)
            params.extend(layer_params)
            params_idx.extend(layer_params_idx)
        
        # 회로 최적화 (중복 게이트 제거, 순서 최적화)
        gates, wires_list, params, params_idx = self._optimize_circuit(
            gates, wires_list, params, params_idx
        )
        
        # 회로 정보 반환
        circuit_info = {
            "gates": gates,
            "wires_list": wires_list,
            "params": params,
            "params_idx": params_idx,
            "n_qubits": n_qubits,
            "depth": depth,
            "coupling_map": coupling_map,
            "strategy": strategy,
            "seed": seed,
            "two_qubit_ratio": config["two_qubit_prob"],  # 실제 사용된 비율 저장
            "circuit_stats": self._calculate_circuit_stats(gates, wires_list, params)
        }
        
        return circuit_info
    
    def _create_optimized_coupling_map(self, n_qubits, coupling_map=None):
        """최적화된 커플링 맵 생성 및 인덱싱"""
        if coupling_map is None:
            # 개선된 격자 구조 (더 많은 연결성)
            coupling_map = []
            grid_size = int(np.ceil(np.sqrt(n_qubits)))
            
            for i in range(n_qubits):
                row, col = i // grid_size, i % grid_size
                
                # 기본 격자 연결 (오른쪽, 아래)
                neighbors = []
                if col < grid_size - 1 and row * grid_size + (col + 1) < n_qubits:
                    neighbors.append(row * grid_size + (col + 1))
                if row < grid_size - 1 and (row + 1) * grid_size + col < n_qubits:
                    neighbors.append((row + 1) * grid_size + col)
                
                # 대각선 연결 추가 (더 풍부한 연결성)
                if row < grid_size - 1 and col < grid_size - 1:
                    diag = (row + 1) * grid_size + (col + 1)
                    if diag < n_qubits:
                        neighbors.append(diag)
                
                for neighbor in neighbors:
                    coupling_map.append([i, neighbor])
        
        # 인접 리스트로 변환 (O(1) 접근)
        adjacency_dict = {i: set() for i in range(n_qubits)}
        for edge in coupling_map:
            if len(edge) >= 2:
                a, b = edge[0], edge[1]
                if a < n_qubits and b < n_qubits:
                    adjacency_dict[a].add(b)
                    adjacency_dict[b].add(a)
        
        return coupling_map, adjacency_dict
    
    def _generate_layer(self, layer_idx, n_qubits, adjacency_dict, config, gate_offset):
        """개선된 레이어 생성"""
        layer_gates = []
        layer_wires = []
        layer_params = []
        layer_params_idx = []
        
        used_qubits = set()
        
        # 레이어 패턴에 따른 처리
        if config["layer_pattern"] == "alternating":
            # 홀수/짝수 레이어 교대
            if layer_idx % 2 == 0:
                self._add_single_qubit_layer(layer_gates, layer_wires, layer_params, 
                                           layer_params_idx, n_qubits, config, gate_offset)
            else:
                self._add_two_qubit_layer(layer_gates, layer_wires, n_qubits, 
                                        adjacency_dict, config, used_qubits)
        
        elif config["layer_pattern"] == "dense":
            # 밀집 패턴 (단일 + 이중 큐빗 게이트 혼합)
            self._add_mixed_layer(layer_gates, layer_wires, layer_params, 
                                layer_params_idx, n_qubits, adjacency_dict, 
                                config, gate_offset, used_qubits)
        
        elif config["layer_pattern"] == "sparse":
            # 희소 패턴 (적은 게이트)
            self._add_sparse_layer(layer_gates, layer_wires, layer_params, 
                                 layer_params_idx, n_qubits, adjacency_dict, 
                                 config, gate_offset, used_qubits)
        
        return layer_gates, layer_wires, layer_params, layer_params_idx
    
    def _add_single_qubit_layer(self, layer_gates, layer_wires, layer_params, 
                               layer_params_idx, n_qubits, config, gate_offset):
        """단일 큐빗 레이어 추가"""
        for q in range(n_qubits):
            if random.random() < config["single_gate_prob"]:
                # 파라미터 게이트 우선 선택
                if random.random() < config["param_gate_ratio"]:
                    gate = random.choice(self.parametric_gates)
                    param = self._generate_parameter(gate)
                    layer_params.append(param)
                    layer_params_idx.append(gate_offset + len(layer_gates))
                else:
                    # 비파라미터 게이트
                    non_param_gates = [g for g in self.single_qubit_gates if g not in self.parametric_gates]
                    gate = random.choice(non_param_gates)
                
                layer_gates.append(gate)
                layer_wires.append([q])
    
    def _add_two_qubit_layer(self, layer_gates, layer_wires, n_qubits, 
                           adjacency_dict, config, used_qubits):
        """이중 큐빗 레이어 추가"""
        for q in range(n_qubits):
            if q in used_qubits:
                continue
                
            if random.random() < config["two_qubit_prob"]:
                # 연결된 큐빗 중 사용되지 않은 것 선택
                available_targets = [t for t in adjacency_dict[q] if t not in used_qubits]
                
                if available_targets:
                    target = random.choice(available_targets)
                    gate = random.choice(self.two_qubit_gates)
                    
                    layer_gates.append(gate)
                    layer_wires.append([q, target])
                    used_qubits.add(q)
                    used_qubits.add(target)
    
    def _add_mixed_layer(self, layer_gates, layer_wires, layer_params, 
                        layer_params_idx, n_qubits, adjacency_dict, 
                        config, gate_offset, used_qubits):
        """혼합 레이어 추가 (단일 + 이중 큐빗)"""
        # 먼저 이중 큐빗 게이트 배치
        self._add_two_qubit_layer(layer_gates, layer_wires, n_qubits, 
                                adjacency_dict, config, used_qubits)
        
        # 남은 큐빗에 단일 큐빗 게이트 추가
        for q in range(n_qubits):
            if q not in used_qubits and random.random() < config["single_gate_prob"]:
                if random.random() < config["param_gate_ratio"]:
                    gate = random.choice(self.parametric_gates)
                    param = self._generate_parameter(gate)
                    layer_params.append(param)
                    layer_params_idx.append(gate_offset + len(layer_gates))
                else:
                    non_param_gates = [g for g in self.single_qubit_gates if g not in self.parametric_gates]
                    gate = random.choice(non_param_gates)
                
                layer_gates.append(gate)
                layer_wires.append([q])
    
    def _add_sparse_layer(self, layer_gates, layer_wires, layer_params, 
                         layer_params_idx, n_qubits, adjacency_dict, 
                         config, gate_offset, used_qubits):
        """희소 레이어 추가 (노이즈 저항성)"""
        # 적은 수의 게이트만 선택
        num_gates = max(1, int(n_qubits * 0.3))  # 30%만 선택
        selected_qubits = random.sample(range(n_qubits), num_gates)
        
        for q in selected_qubits:
            if random.random() < 0.7:  # 단일 큐빗 게이트 우선
                gate = random.choice(["H", "X", "RZ"])  # 기본 게이트만
                if gate == "RZ":
                    param = self._generate_parameter(gate)
                    layer_params.append(param)
                    layer_params_idx.append(gate_offset + len(layer_gates))
                
                layer_gates.append(gate)
                layer_wires.append([q])
    
    def _generate_parameter(self, gate_type):
        """개선된 파라미터 생성 (다양한 분포)"""
        if gate_type in ["RZ", "RX", "RY"]:
            # 다양한 분포 사용
            distribution = random.choice(["uniform", "normal", "discrete"])
            
            if distribution == "uniform":
                return random.uniform(0, 2*np.pi)
            elif distribution == "normal":
                # 정규분포 (π 중심)
                param = np.random.normal(np.pi, np.pi/3)
                return np.clip(param, 0, 2*np.pi)
            else:  # discrete
                # 이산 값 (π/4 단위)
                return random.choice([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
        
        return random.uniform(0, 2*np.pi)
    
    def _optimize_circuit(self, gates, wires_list, params, params_idx):
        """회로 최적화 (중복 제거, 순서 최적화)"""
        # 연속된 동일 게이트 제거 (X-X = I 등)
        optimized_gates = []
        optimized_wires = []
        optimized_params = []
        optimized_params_idx = []
        
        i = 0
        param_counter = 0
        
        while i < len(gates):
            gate = gates[i]
            wires = wires_list[i]
            
            # 다음 게이트와 비교
            if i + 1 < len(gates):
                next_gate = gates[i + 1]
                next_wires = wires_list[i + 1]
                
                # 같은 큐빗에 연속된 X, Y, Z 게이트 제거
                if (gate in ["X", "Y", "Z"] and gate == next_gate and 
                    wires == next_wires):
                    i += 2  # 두 게이트 모두 건너뛰기
                    continue
            
            # 게이트 추가
            optimized_gates.append(gate)
            optimized_wires.append(wires)
            
            # 파라미터 처리
            if i in params_idx:
                param_idx = params_idx.index(i)
                optimized_params.append(params[param_idx])
                optimized_params_idx.append(len(optimized_gates) - 1)
            
            i += 1
        
        return optimized_gates, optimized_wires, optimized_params, optimized_params_idx
    
    def _calculate_circuit_stats(self, gates, wires_list, params):
        """회로 통계 계산"""
        stats = {
            "total_gates": len(gates),
            "single_qubit_gates": sum(1 for g in gates if g in self.single_qubit_gates),
            "two_qubit_gates": sum(1 for g in gates if g in self.two_qubit_gates),
            "parametric_gates": len(params),
            "gate_distribution": {},
            "qubit_usage": {}
        }
        
        # 게이트 분포
        for gate in gates:
            stats["gate_distribution"][gate] = stats["gate_distribution"].get(gate, 0) + 1
        
        # 큐빗 사용량
        for wires in wires_list:
            for qubit in wires:
                stats["qubit_usage"][qubit] = stats["qubit_usage"].get(qubit, 0) + 1
        
        return stats
    
    def create_circuit_qnode(self, circuit_info):
        """
        회로 정보에서 QNode 생성
        
        Args:
            circuit_info (dict): 회로 정보
            
        Returns:
            function: QNode 회로 함수
        """
        # 회로 정보 추출
        gates = circuit_info["gates"]
        wires_list = circuit_info["wires_list"]
        params_idx = circuit_info["params_idx"]
        n_qubits = circuit_info["n_qubits"]
        
        # 시뮬레이터 메모리 한계 고려
        max_sim_qubits = 20
        
        # 환경 변수에서 시뮬레이션 큐빗 제한 확인
        if "MAX_SIMULATION_QUBITS" in os.environ:
            try:
                max_sim_qubits = int(os.environ["MAX_SIMULATION_QUBITS"])
                print(f"시뮬레이션 큐빗 수 제한 설정: {max_sim_qubits}")
            except:
                pass
                
        if n_qubits > max_sim_qubits:
            print(f"⚠️ 경고: 시뮬레이터에서 {n_qubits}개 큐빗은 한계를 초과합니다. {max_sim_qubits}로 제한합니다.")
            n_qubits = max_sim_qubits
        
        # 기본 기기 설정
        device = qml.device("default.qubit", wires=n_qubits)
        
        # 회로 함수 생성
        @qml.qnode(device)
        def circuit(params=None):
            # 회로 구축
            for i, (gate, wires) in enumerate(zip(gates, wires_list)):
                # 큐빗 범위 확인
                if any(w >= n_qubits for w in wires):
                    continue  # 범위를 벗어난 큐빗은 건너뛰기
                    
                if gate == "H":
                    qml.Hadamard(wires=wires[0])
                elif gate == "X":
                    qml.PauliX(wires=wires[0])
                elif gate == "Y":
                    qml.PauliY(wires=wires[0])
                elif gate == "Z":
                    qml.PauliZ(wires=wires[0])
                elif gate == "S":
                    qml.S(wires=wires[0])
                elif gate == "T":
                    qml.T(wires=wires[0])
                elif gate == "RZ":
                    param_idx = params_idx.index(i)
                    qml.RZ(params[param_idx], wires=wires[0])
                elif gate == "RX":
                    param_idx = params_idx.index(i)
                    qml.RX(params[param_idx], wires=wires[0])
                elif gate == "RY":
                    param_idx = params_idx.index(i)
                    qml.RY(params[param_idx], wires=wires[0])
                elif gate == "CNOT":
                    if len(wires) >= 2:
                        qml.CNOT(wires=[wires[0], wires[1]])
                elif gate == "CZ":
                    if len(wires) >= 2:
                        qml.CZ(wires=[wires[0], wires[1]])
            
            # 확률 측정
            return qml.probs(wires=range(n_qubits))
        
        return circuit
    
    def create_inverse_circuit_qnode(self, circuit_info):
        """
        피델리티 측정을 위한 역회로 QNode 생성
        
        Args:
            circuit_info (dict): 회로 정보
            
        Returns:
            function: 역회로가 적용된 QNode 함수
        """
        # 회로 정보 추출
        gates = circuit_info["gates"]
        wires_list = circuit_info["wires_list"]
        params_idx = circuit_info["params_idx"]
        n_qubits = circuit_info["n_qubits"]
        
        # 시뮬레이터 제한
        max_sim_qubits = 30
        
        # 환경 변수에서 시뮬레이션 큐빗 제한 확인
        if "MAX_SIMULATION_QUBITS" in os.environ:
            try:
                max_sim_qubits = int(os.environ["MAX_SIMULATION_QUBITS"])
                print(f"역회로 시뮬레이션 큐빗 수 제한 설정: {max_sim_qubits}")
            except:
                pass
                
        if n_qubits > max_sim_qubits:
            print(f"⚠️ 경고: 시뮬레이터에서 {n_qubits}개 큐빗은 제한을 초과합니다. {max_sim_qubits}로 제한합니다.")
            n_qubits = max_sim_qubits
        
        # 기본 장치 설정
        device = qml.device("default.qubit", wires=n_qubits, shots=1024)  # 1024 샷으로 설정
        
        # 여기서 역회로 함수 구현
        @qml.qnode(device)
        def circuit(params=None):
            # 원래 회로 구현
            for i, (gate, wires) in enumerate(zip(gates, wires_list)):
                if any(w >= n_qubits for w in wires):
                    continue
                
                if gate == "H":
                    qml.Hadamard(wires=wires[0])
                elif gate == "X":
                    qml.PauliX(wires=wires[0])
                elif gate == "Y":
                    qml.PauliY(wires=wires[0])
                elif gate == "Z":
                    qml.PauliZ(wires=wires[0])
                elif gate == "S":
                    qml.S(wires=wires[0])
                elif gate == "T":
                    qml.T(wires=wires[0])
                elif gate == "RZ":
                    param_idx = params_idx.index(i)
                    qml.RZ(params[param_idx], wires=wires[0])
                elif gate == "RX":
                    param_idx = params_idx.index(i)
                    qml.RX(params[param_idx], wires=wires[0])
                elif gate == "RY":
                    param_idx = params_idx.index(i)
                    qml.RY(params[param_idx], wires=wires[0])
                elif gate == "CNOT":
                    if len(wires) >= 2:
                        qml.CNOT(wires=[wires[0], wires[1]])
                elif gate == "CZ":
                    if len(wires) >= 2:
                        qml.CZ(wires=[wires[0], wires[1]])
            
            # 역회로 구현
            for i in range(len(gates)-1, -1, -1):
                gate = gates[i]
                wires = wires_list[i]
                
                if any(w >= n_qubits for w in wires):
                    continue
                
                if gate == "H":
                    # H는 자기 자신이 인버스지만 adjoint로 일관되게 처리
                    qml.adjoint(qml.Hadamard)(wires=wires[0])
                elif gate == "X":
                    # X는 자기 자신이 인버스지만 adjoint로 일관되게 처리
                    qml.adjoint(qml.PauliX)(wires=wires[0])
                elif gate == "Y":
                    # Y는 자기 자신이 인버스지만 adjoint로 일관되게 처리
                    qml.adjoint(qml.PauliY)(wires=wires[0])
                elif gate == "Z":
                    # Z는 자기 자신이 인버스지만 adjoint로 일관되게 처리
                    qml.adjoint(qml.PauliZ)(wires=wires[0])
                elif gate == "S":
                    # S†는 S의 역행렬
                    qml.adjoint(qml.S)(wires=wires[0])
                elif gate == "T":
                    # T†는 T의 역행렬
                    qml.adjoint(qml.T)(wires=wires[0])
                elif gate == "RZ":
                    param_idx = params_idx.index(i)
                    # RZ(θ)†는 RZ(-θ)와 같음
                    qml.RZ(-params[param_idx], wires=wires[0])
                elif gate == "RX":
                    param_idx = params_idx.index(i)
                    # RX(θ)†는 RX(-θ)와 같음
                    qml.RX(-params[param_idx], wires=wires[0])
                elif gate == "RY":
                    param_idx = params_idx.index(i)
                    # RY(θ)†는 RY(-θ)와 같음
                    qml.RY(-params[param_idx], wires=wires[0])
                elif gate == "CNOT":
                    if len(wires) >= 2:
                        # CNOT은 자기 자신이 인버스지만 adjoint로 일관되게 처리
                        qml.adjoint(qml.CNOT)(wires=[wires[0], wires[1]])
                elif gate == "CZ":
                    if len(wires) >= 2:
                        # CZ는 자기 자신이 인버스지만 adjoint로 일관되게 처리
                        qml.adjoint(qml.CZ)(wires=[wires[0], wires[1]])
            
            # 상태 벡터 계산 대신 샘플링 반환 (메모리 효율적)
            return qml.sample()
        
        return circuit
    
    def calculate_fidelity(self, circuit_info):
        """
        피델리티 계산 (순방향+역방향 회로 실행)
        
        Args:
            circuit_info (dict): 회로 정보
            
        Returns:
            float: 피델리티 값 (0~1 사이)
        """
        # 큐빗 수 제한
        n_qubits = circuit_info["n_qubits"]
        
        # 시뮬레이터에서는 30큐빗로 제한
        if n_qubits > 30:
            print(f"⚠️ 경고: 피델리티 계산에서 {n_qubits}개 큐빗은 시뮬레이터 제한을 초과합니다. 30으로 제한합니다.")
            # 큐빗 수 조정된 새 회로 정보 생성
            circuit_info = circuit_info.copy()
            circuit_info["n_qubits"] = 30
            n_qubits = 30
        
        print("\n===== 피델리티 계산 디버깅 =====")
        print(f"피델리티 계산에 사용된 큐빗 수: {n_qubits}")
        print(f"회로 게이트 수: {len(circuit_info['gates'])}")
        print(f"시뮬레이터 디바이스: default.qubit, 샷 수: 1024")
        
        # 역회로가 적용된 QNode 생성
        circuit = self.create_inverse_circuit_qnode(circuit_info)
        
        # 샘플링 실행 (1024 샷)
        print("샘플링 실행 중...")
        samples = circuit(circuit_info["params"])
        
        print(f"반환된 샘플 타입: {type(samples)}")
        print(f"샘플 형태: {samples.shape if hasattr(samples, 'shape') else '형태 정보 없음'}")
        print(f"샘플 첫 10개: {samples[:10] if hasattr(samples, '__getitem__') else '접근 불가'}")
        
        # 디버깅: 샘플링 결과 요약
        try:
            # 샘플 결과 분석
            sample_counts = {}
            for i, sample in enumerate(samples):
                # 샘플을 비트 문자열로 변환
                if hasattr(sample, '__iter__'):
                    # 반복 가능한 샘플 (비트 배열)
                    bit_str = ''.join(str(int(bit)) for bit in sample)
                else:
                    # 정수 샘플
                    bit_str = format(int(sample), f'0{n_qubits}b')
                
                # 샘플 카운트
                if bit_str in sample_counts:
                    sample_counts[bit_str] += 1
                else:
                    sample_counts[bit_str] = 1
                
            # 상위 샘플 출력
            print("\n샘플링 결과 요약:")
            print(f"고유 상태 수: {len(sample_counts)}")
            print("상위 10개 상태:")
            for i, (state, count) in enumerate(sorted(sample_counts.items(), key=lambda x: x[1], reverse=True)[:10]):
                print(f"  |{state}⟩: {count}회 ({count/len(samples)*100:.2f}%)")
            
            # |0...0> 상태의 비율 계산
            zero_state_str = '0' * n_qubits
            zero_count = sample_counts.get(zero_state_str, 0)
            total_samples = len(samples)
            
            # 피델리티는 |0...0> 상태의 빈도
            fidelity = zero_count / total_samples
            
            print(f"피델리티 샘플링 결과: {zero_count}/{total_samples} 샘플이 |0...0> 상태")
            print(f"피델리티: {fidelity:.6f}")
            
        except Exception as e:
            print(f"샘플 분석 오류: {str(e)}")
            
            # 대체 방법: 영벡터 상태 확인
            zero_state = np.zeros(n_qubits)
            zero_count = 0
            
            # 샘플링 결과에서 |0...0> 상태 카운트
            for sample in samples:
                if all(bit == 0 for bit in sample):
                    zero_count += 1
            
            # 피델리티는 |0...0> 상태의 빈도
            fidelity = zero_count / len(samples)
            
            print(f"대체 방법으로 계산된 피델리티: {fidelity:.6f}")
            print(f"피델리티 샘플링 결과: {zero_count}/{len(samples)} 샘플이 |0...0> 상태")
        
        print("============================")
        
        return float(fidelity)
    
    def visualize_circuit(self, circuit_info, filename=None, include_inverse=False):
        """
        양자 회로 시각화
        
        Args:
            circuit_info (dict): 회로 정보
            filename (str): 저장할 파일명
            include_inverse (bool): 인버스 회로 포함 여부
        """
        # 시각화를 위한 큐빗 수 제한
        n_qubits = circuit_info["n_qubits"]
        max_visualization_qubits = 16  # 시각화에 적절한 큐빗 수
        
        if n_qubits > max_visualization_qubits:
            print(f"⚠️ 경고: {n_qubits}개 큐빗은 시각화하기 어렵습니다. {max_visualization_qubits}로 제한하여 일부만 표시합니다.")
            n_qubits = max_visualization_qubits
            # 원본 회로 정보는 수정하지 않고 새 회로 정보 생성
            vis_circuit_info = circuit_info.copy()
            vis_circuit_info["n_qubits"] = n_qubits
            # 큐빗 번호가 큰 게이트 필터링
            vis_circuit_info["gates"] = []
            vis_circuit_info["wires_list"] = []
            vis_circuit_info["params_idx"] = []
            
            # 작은 큐빗 번호의 게이트만 선택
            params_count = 0
            new_params = []
            
            for i, (gate, wires) in enumerate(zip(circuit_info["gates"], circuit_info["wires_list"])):
                valid_gate = True
                for w in wires:
                    if w >= n_qubits:
                        valid_gate = False
                        break
                
                if valid_gate:
                    vis_circuit_info["gates"].append(gate)
                    vis_circuit_info["wires_list"].append(wires)
                    if i in circuit_info["params_idx"]:
                        vis_circuit_info["params_idx"].append(params_count)
                        param_idx = circuit_info["params_idx"].index(i)
                        new_params.append(circuit_info["params"][param_idx])
                        params_count += 1
            
            vis_circuit_info["params"] = new_params
        else:
            vis_circuit_info = circuit_info
        
        # 회로 생성 (백엔드 사용하지 않음)
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev)
        def circuit(params=None):
            # 회로 정보 추출
            gates = vis_circuit_info["gates"]
            wires_list = vis_circuit_info["wires_list"]
            params_idx = vis_circuit_info["params_idx"]
            
            # 회로 적용
            for i, (gate, wires) in enumerate(zip(gates, wires_list)):
                if gate == "H":
                    qml.Hadamard(wires=wires[0])
                elif gate == "X":
                    qml.PauliX(wires=wires[0])
                elif gate == "Y":
                    qml.PauliY(wires=wires[0])
                elif gate == "Z":
                    qml.PauliZ(wires=wires[0])
                elif gate == "S":
                    qml.S(wires=wires[0])
                elif gate == "T":
                    qml.T(wires=wires[0])
                elif gate == "RZ":
                    param_idx = params_idx.index(i)
                    qml.RZ(params[param_idx], wires=wires[0])
                elif gate == "RX":
                    param_idx = params_idx.index(i)
                    qml.RX(params[param_idx], wires=wires[0])
                elif gate == "RY":
                    param_idx = params_idx.index(i)
                    qml.RY(params[param_idx], wires=wires[0])
                elif gate == "CNOT":
                    if len(wires) >= 2:
                        qml.CNOT(wires=[wires[0], wires[1]])
                elif gate == "CZ":
                    if len(wires) >= 2:
                        qml.CZ(wires=[wires[0], wires[1]])
            
            # 인버스 회로 추가 (요청 시)
            if include_inverse:
                # 구분선 추가 (시각적 구분을 위한 배리어)
                qml.Barrier(wires=range(n_qubits))
                
                # 역회로 구현 (역순으로 게이트 적용)
                for i in range(len(gates)-1, -1, -1):
                    gate = gates[i]
                    wires = wires_list[i]
                    
                    if gate == "H":
                        # H는 자기 자신이 인버스지만 adjoint로 일관되게 처리
                        qml.adjoint(qml.Hadamard)(wires=wires[0])
                    elif gate == "X":
                        # X는 자기 자신이 인버스지만 adjoint로 일관되게 처리
                        qml.adjoint(qml.PauliX)(wires=wires[0])
                    elif gate == "Y":
                        # Y는 자기 자신이 인버스지만 adjoint로 일관되게 처리
                        qml.adjoint(qml.PauliY)(wires=wires[0])
                    elif gate == "Z":
                        # Z는 자기 자신이 인버스지만 adjoint로 일관되게 처리
                        qml.adjoint(qml.PauliZ)(wires=wires[0])
                    elif gate == "S":
                        # S†는 S의 역행렬
                        qml.adjoint(qml.S)(wires=wires[0])
                    elif gate == "T":
                        # T†는 T의 역행렬
                        qml.adjoint(qml.T)(wires=wires[0])
                    elif gate == "RZ":
                        param_idx = params_idx.index(i)
                        # RZ(θ)†는 RZ(-θ)와 같음
                        qml.RZ(-params[param_idx], wires=wires[0])
                    elif gate == "RX":
                        param_idx = params_idx.index(i)
                        # RX(θ)†는 RX(-θ)와 같음
                        qml.RX(-params[param_idx], wires=wires[0])
                    elif gate == "RY":
                        param_idx = params_idx.index(i)
                        # RY(θ)†는 RY(-θ)와 같음
                        qml.RY(-params[param_idx], wires=wires[0])
                    elif gate == "CNOT":
                        if len(wires) >= 2:
                            # CNOT은 자기 자신이 인버스지만 adjoint로 일관되게 처리
                            qml.adjoint(qml.CNOT)(wires=[wires[0], wires[1]])
                    elif gate == "CZ":
                        if len(wires) >= 2:
                            # CZ는 자기 자신이 인버스지만 adjoint로 일관되게 처리
                            qml.adjoint(qml.CZ)(wires=[wires[0], wires[1]])
            
            return qml.state()
        
        # 회로 다이어그램 생성
        fig, ax = qml.draw_mpl(circuit)(vis_circuit_info["params"])
        
        # 제목에 인버스 정보 추가
        circuit_type = "Forward+Inverse" if include_inverse else "Forward only"
        plt.title(f"Grid Quantum Circuit ({n_qubits} qubits, {len(vis_circuit_info['gates'])} gates, {circuit_type})")
        
        if filename:
            # 파일명에 인버스 포함 여부 추가
            base, ext = os.path.splitext(filename)
            if include_inverse:
                filename = f"{base}_with_inverse{ext}"
            # 이미지 디렉토리에 저장
            image_path = os.path.join(self.output_dir, "images", os.path.basename(filename))
            plt.savefig(image_path)
            print(f"회로 다이어그램 저장됨: images/{os.path.basename(filename)}")
        
        plt.show()
    
    def create_heavy_hexagonal_layout(self, n_qubits):
        """
        Heavy-Hexagonal 레이아웃 생성 (IBM 양자 컴퓨터 구조와 유사)
        
        Args:
            n_qubits (int): 큐빗 수
            
        Returns:
            tuple: (pos, coupling_map) - 노드 위치와 연결 정보
        """
        # 기본 패러미터
        n_rows = int(np.ceil(np.sqrt(n_qubits) * 0.7))  # 행 수 (육각형 구조에 맞게 조정)
        n_cols = int(np.ceil(n_qubits / n_rows))  # 열 수
        
        # 위치 및 커플링 맵 초기화
        pos = {}
        coupling_map = []
        
        # 헥사고날 그리드에서의 상대적 오프셋
        hex_offsets = [
            (0, 0),      # 중앙
            (1, 0),      # 오른쪽
            (0.5, 0.866), # 오른쪽 위
            (-0.5, 0.866), # 왼쪽 위
            (-1, 0),     # 왼쪽
            (-0.5, -0.866), # 왼쪽 아래
            (0.5, -0.866)  # 오른쪽 아래
        ]
        
        # 큐빗 위치 계산 및 배치
        qubit_idx = 0
        for row in range(n_rows):
            row_offset = row * 1.5  # 행 간격
            for col in range(n_cols):
                if qubit_idx >= n_qubits:
                    break
                    
                # 행이 홀수/짝수인지에 따라 열 위치 오프셋 조정
                col_offset = col * 2 + (row % 2) * 1
                
                # 육각형 그리드에 큐빗 배치
                pos[qubit_idx] = (col_offset, -row_offset)
                
                # 다음 큐빗
                qubit_idx += 1
        
        # 연결 생성 (헥사고날 패턴)
        for i in range(n_qubits):
            x, y = pos[i]
            
            # 각 큐빗에 대해 가능한 연결 확인
            for j in range(n_qubits):
                if i == j:
                    continue
                    
                # 두 큐빗 간의 거리 계산
                x2, y2 = pos[j]
                dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                
                # 거리가 가까우면 연결 (헥사고날 그리드에서 이웃 노드 거리)
                # 1.1~1.8 사이의 거리는 헥사고날 그리드에서 인접한 큐빗으로 간주
                if 0.9 < dist < 2.2:
                    # 중복 연결 방지
                    if [i, j] not in coupling_map and [j, i] not in coupling_map:
                        coupling_map.append([i, j])
        
        return pos, coupling_map
    
    def visualize_grid(self, circuit_info, filename=None, use_heavy_hex=True):
        """
        격자 구조 시각화
        
        Args:
            circuit_info (dict): 회로 정보
            filename (str): 저장할 파일명
            use_heavy_hex (bool): Heavy-Hexagonal 구조 사용 여부
        """
        n_qubits = circuit_info["n_qubits"]
        coupling_map = circuit_info.get("coupling_map", [])
        
        # 그래프 생성
        G = nx.Graph()
        
        # Heavy-Hexagonal 레이아웃 사용
        if use_heavy_hex:
            pos, hex_coupling = self.create_heavy_hexagonal_layout(n_qubits)
            
            # 커플링 맵이 없으면 헥사고날 커플링 사용
            if not coupling_map:
                coupling_map = hex_coupling
        else:
            # 기존 격자 레이아웃 생성
            pos = {}
            grid_size = int(np.ceil(np.sqrt(n_qubits)))
            for i in range(n_qubits):
                row, col = i // grid_size, i % grid_size
                pos[i] = (col, -row)
        
        # 노드 추가
        for i in range(n_qubits):
            G.add_node(i)
        
        # 명시적으로 모든 노드에 pos 속성 설정
        nx.set_node_attributes(G, pos, 'pos')
        
        # 엣지 추가 (커플링)
        for edge in coupling_map:
            # 유효한 노드 인덱스인지 확인
            if edge[0] < n_qubits and edge[1] < n_qubits:
                G.add_edge(edge[0], edge[1])
        
        # 그래프 그리기
        plt.figure(figsize=(12, 10))
        
        # 노드 크기와 색상 (Heavy-Hex 구조일 때 다르게 표시)
        if use_heavy_hex:
            # 각 노드의 연결 수에 따라 다른 색상 지정
            node_colors = []
            for node in G.nodes():
                degree = G.degree(node)
                if degree <= 2:
                    node_colors.append('lightblue')  # 연결 2개 이하
                else:
                    node_colors.append('lightcoral')  # 연결 3개 이상
            
            # 그래프 그리기 (헥사고날 스타일)
            nx.draw(G, pos, with_labels=True, node_size=600, 
                    node_color=node_colors, font_weight='bold', 
                    width=2, edge_color='navy', font_size=10)
            
            layout_type = "Heavy-Hexagonal"
        else:
            # 기존 격자 스타일
            nx.draw(G, pos, with_labels=True, node_size=500, 
                    node_color='skyblue', font_weight='bold', 
                    width=2, edge_color='navy')
            
            layout_type = "Grid"
        
        plt.title(f"{layout_type} Quantum Circuit Structure ({n_qubits} qubits, {len(coupling_map)} connections)")
        
        if filename:
            # 이미지 디렉토리에 저장
            image_path = os.path.join(self.output_dir, "images", os.path.basename(filename))
            plt.savefig(image_path)
            print(f"격자 구조 다이어그램 저장됨: images/{os.path.basename(filename)}")
        plt.show()
    
    def save_results(self, ansatz_data, filename=None):
        """
        결과 저장
        
        Args:
            ansatz_data (dict): 회로 실행 결과 데이터
            filename (str): 저장할 파일명 (None이면 자동 생성)
            
        Returns:
            str: 저장된 파일 경로
        """
        if filename is None:
            filename = os.path.join(
                self.output_dir,
                "ansatz_data",
                f"ansatz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        
        with open(filename, 'w') as f:
            # 복잡한 객체는 문자열로 변환
            simplified_data = json.dumps(
                ansatz_data, 
                default=lambda o: str(o) if not isinstance(o, (dict, list, str, int, float, bool, type(None))) else o
            )
            json.dump(json.loads(simplified_data), f, indent=2)
        
        print(f"\n결과가 저장되었습니다: {filename}")
        return filename
    
    def calculate_expressibility(self, circuit_info, S=None, M=None, metric='classical_shadow', sigma=1.0):
        """
        표현력 계산 (ExpressibilityCalculator로 위임)
        
        Args:
            circuit_info (dict): 회로 정보
            S (int): 파라미터 샘플 수 (None이면 자동 설정)
            M (int): Shadow 크기 (None이면 자동 설정)
            metric (str): 'classical_shadow' (기본값)
            sigma (float): 사용하지 않음 (호환성 유지)
            
        Returns:
            dict: 표현력 측정 결과
        """
        return self.expressibility_calculator.calculate_expressibility(
            circuit_info, S, M, metric, sigma
        )

    def transpile_circuit(self, circuit_info, backend, optimization_level=0, verbose=True):
        """
        회로 트랜스파일
        
        Args:
            circuit_info (dict): 회로 정보
            backend (qiskit.backends.Backend): 트랜스파일할 백엔드
            optimization_level (int): 최적화 레벨 (0~4)
            verbose (bool): 트랜스파일 과정 출력 여부
            
        Returns:
            dict: 트랜스파일된 회로 정보
        """
        # 트랜스파일 전 회로 다이어그램 저장
        try:
            if verbose:
                qc_original = qml.QNode(lambda params: self.create_circuit_qnode(circuit_info)(params), self.device)(circuit_info["params"])
                fig = qc_original.draw(output='mpl', style={'name': 'bw'})
                fig.savefig(os.path.join(self.output_dir, "images", "before_transpile_circuit.png"))
                print("트랜스파일 전 회로 다이어그램 저장됨: images/before_transpile_circuit.png")
        except Exception as e:
            if verbose:
                print(f"회로 다이어그램 저장 오류: {str(e)}")
        
        # IBM 백엔드에 맞게 회로 트랜스파일
        if verbose:
            print("\n트랜스파일 수행 중...")
        # 최적화 레벨을 0으로 낮춰 최소한의 변환만 수행
        qc_transpiled = transpile(qc_original, backend=backend, optimization_level=optimization_level)
        
        # 트랜스파일 후 회로 정보 출력
        if verbose:
            print("\n===== 트랜스파일 후 회로 정보 =====")
            print(f"회로 깊이: {qc_transpiled.depth()}")
            print(f"게이트 수: {sum(len(qc_transpiled.data) for qc in [qc_transpiled])}")
            print("게이트 통계:")
            gate_counts = {}
            for gate in qc_transpiled.data:
                gate_name = gate[0].name
                if gate_name in gate_counts:
                    gate_counts[gate_name] += 1
                else:
                    gate_counts[gate_name] = 1
            for gate_name, count in gate_counts.items():
                print(f"  - {gate_name}: {count}개")
        
        # 트랜스파일 후 회로 다이어그램 저장
        try:
            if verbose:
                fig = qc_transpiled.draw(output='mpl', style={'name': 'bw'})
                fig.savefig(os.path.join(self.output_dir, "images", "after_transpile_circuit.png"))
                print("트랜스파일 후 회로 다이어그램 저장됨: images/after_transpile_circuit.png")
        except Exception as e:
            if verbose:
                print(f"회로 다이어그램 저장 오류: {str(e)}")
        
        # 트랜스파일된 회로 정보 반환
        transpiled_circuit_info = {
            "gates": qc_transpiled.data,
            "wires_list": [gate[1] for gate in qc_transpiled.data],
            "params": qc_transpiled.parameters,
            "params_idx": [i for i, gate in enumerate(qc_transpiled.data) if gate[0].name in self.parametric_gates],
            "n_qubits": qc_transpiled.num_qubits,
            "depth": qc_transpiled.depth(),
            "coupling_map": qc_transpiled.coupling_map,
            "strategy": "hardware_efficient",
            "seed": None,
            "two_qubit_ratio": 0.8,
            "circuit_stats": self._calculate_circuit_stats(
                [gate[0].name for gate in qc_transpiled.data],
                [wire for gate, wire in qc_transpiled.data],
                qc_transpiled.parameters
            )
        }
        
        return transpiled_circuit_info 

    def build_qiskit_circuit(self, circuit_info, max_qubits=None):
        from qiskit import QuantumCircuit
        # determine qubit count
        n_q = circuit_info.get("n_qubits", 0)
        if max_qubits is not None and n_q > max_qubits:
            n_q = max_qubits
        qc = QuantumCircuit(n_q, n_q)
        gates = circuit_info["gates"]
        wires_list = circuit_info["wires_list"]
        params_idx = circuit_info["params_idx"]
        params = circuit_info.get("params", [])
        # forward execution
        for j, (gate, wires) in enumerate(zip(gates, wires_list)):
            if any(w >= n_q for w in wires):
                continue
            if gate == "H": qc.h(wires[0])
            elif gate == "X": qc.x(wires[0])
            elif gate == "Y": qc.y(wires[0])
            elif gate == "Z": qc.z(wires[0])
            elif gate == "S": qc.s(wires[0])
            elif gate == "T": qc.t(wires[0])
            elif gate == "RZ":
                for k, idx in enumerate(params_idx):
                    if idx == j:
                        qc.rz(params[k], wires[0]); break
            elif gate == "RX":
                for k, idx in enumerate(params_idx):
                    if idx == j:
                        qc.rx(params[k], wires[0]); break
            elif gate == "RY":
                for k, idx in enumerate(params_idx):
                    if idx == j:
                        qc.ry(params[k], wires[0]); break
            elif gate == "CZ" and len(wires) >= 2:
                qc.cz(wires[0], wires[1])
            elif gate == "CNOT" and len(wires) >= 2:
                qc.cx(wires[0], wires[1])
        # reverse execution
        for j in range(len(gates)-1, -1, -1):
            gate = gates[j]; wires = wires_list[j]
            if any(w >= n_q for w in wires):
                continue
            if gate == "H": qc.h(wires[0])
            elif gate == "X": qc.x(wires[0])
            elif gate == "Y": qc.y(wires[0])
            elif gate == "Z": qc.z(wires[0])
            elif gate == "S": qc.sdg(wires[0])
            elif gate == "T": qc.tdg(wires[0])
            elif gate == "RZ":
                for k, idx in enumerate(params_idx):
                    if idx == j:
                        qc.rz(-params[k], wires[0]); break
            elif gate == "RX":
                for k, idx in enumerate(params_idx):
                    if idx == j:
                        qc.rx(-params[k], wires[0]); break
            elif gate == "RY":
                for k, idx in enumerate(params_idx):
                    if idx == j:
                        qc.ry(-params[k], wires[0]); break
            elif gate == "CZ" and len(wires) >= 2:
                qc.cz(wires[0], wires[1])
            elif gate == "CNOT" and len(wires) >= 2:
                qc.cx(wires[0], wires[1])
        # measurement
        qc.measure_all()
        return qc