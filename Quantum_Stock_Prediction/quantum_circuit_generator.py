#!/usr/bin/env python3
"""
사용자 요구사항에 맞는 양자회로 생성기
1단계: 예측 모델로 성능 평가
2단계: 강화학습/유전 알고리즘으로 회로 최적화
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from quantum_transformer_model import QuantumCircuitTransformer, MAX_SEQUENCE_LENGTH
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import copy

# 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################################
# 1. 사용자 요구사항 정의
#################################################

@dataclass
class UserRequirements:
    """사용자가 원하는 양자회로 성능 요구사항"""
    target_fidelity: float = 0.9        # 목표 피델리티
    target_expressibility: float = 0.8  # 목표 정규화된 표현력
    max_qubits: int = 30                # 최대 큐빗 수
    max_depth: int = 4                  # 최대 회로 깊이
    max_gates: int = 100                # 최대 게이트 수
    
    # 가중치 (중요도)
    fidelity_weight: float = 0.4
    expressibility_weight: float = 0.4
    efficiency_weight: float = 0.2      # 게이트 수 효율성
    
    def __post_init__(self):
        """가중치 정규화"""
        total = self.fidelity_weight + self.expressibility_weight + self.efficiency_weight
        self.fidelity_weight /= total
        self.expressibility_weight /= total
        self.efficiency_weight /= total

#################################################
# 2. 양자회로 표현 및 조작
#################################################

class QuantumCircuitRepresentation:
    """양자회로의 내부 표현"""
    
    def __init__(self, n_qubits: int, max_gates: int = 100):
        self.n_qubits = n_qubits
        self.max_gates = max_gates
        
        # 게이트 타입 정의 (우리 데이터와 일치)
        self.gate_types = {
            0: 'PADDING',  # 패딩
            1: 'H',        # Hadamard
            2: 'X',        # Pauli-X
            3: 'Y',        # Pauli-Y
            4: 'Z',        # Pauli-Z
            5: 'S',        # S gate
            6: 'T',        # T gate
            7: 'RZ',       # RZ rotation
            8: 'CNOT'      # CNOT
        }
        
        # 회로 시퀀스 초기화
        self.gate_sequence = [0] * MAX_SEQUENCE_LENGTH
        self.qubit_sequence = [-1] * MAX_SEQUENCE_LENGTH
        self.param_sequence = [0.0] * MAX_SEQUENCE_LENGTH
        self.gate_type_sequence = [0] * MAX_SEQUENCE_LENGTH
        
        self.actual_length = 0
    
    def add_gate(self, gate_type: int, qubits: List[int], param: float = 0.0):
        """게이트 추가"""
        if self.actual_length >= MAX_SEQUENCE_LENGTH:
            return False
        
        idx = self.actual_length
        self.gate_sequence[idx] = gate_type
        self.qubit_sequence[idx] = qubits[0] if qubits else -1
        self.param_sequence[idx] = param
        
        # 게이트 타입 분류 (단일/이중 큐빗)
        if gate_type in [1, 2, 3, 4, 5, 6, 7]:  # 단일 큐빗 게이트
            self.gate_type_sequence[idx] = 1
        elif gate_type == 8:  # CNOT (이중 큐빗)
            self.gate_type_sequence[idx] = 2
        else:
            self.gate_type_sequence[idx] = 0
        
        self.actual_length += 1
        return True
    
    def remove_gate(self, index: int):
        """특정 위치의 게이트 제거"""
        if 0 <= index < self.actual_length:
            # 뒤의 게이트들을 앞으로 이동
            for i in range(index, self.actual_length - 1):
                self.gate_sequence[i] = self.gate_sequence[i + 1]
                self.qubit_sequence[i] = self.qubit_sequence[i + 1]
                self.param_sequence[i] = self.param_sequence[i + 1]
                self.gate_type_sequence[i] = self.gate_type_sequence[i + 1]
            
            # 마지막 위치 초기화
            self.gate_sequence[self.actual_length - 1] = 0
            self.qubit_sequence[self.actual_length - 1] = -1
            self.param_sequence[self.actual_length - 1] = 0.0
            self.gate_type_sequence[self.actual_length - 1] = 0
            
            self.actual_length -= 1
    
    def mutate(self, mutation_rate: float = 0.1):
        """회로 변이 (유전 알고리즘용)"""
        circuit_copy = copy.deepcopy(self)
        
        for i in range(circuit_copy.actual_length):
            if random.random() < mutation_rate:
                # 게이트 타입 변경
                if random.random() < 0.5:
                    new_gate = random.choice([1, 2, 3, 4, 5, 6, 7, 8])
                    circuit_copy.gate_sequence[i] = new_gate
                    
                    # 게이트 타입에 따른 큐빗 조정
                    if new_gate in [1, 2, 3, 4, 5, 6, 7]:  # 단일 큐빗
                        circuit_copy.qubit_sequence[i] = random.randint(0, self.n_qubits - 1)
                        circuit_copy.gate_type_sequence[i] = 1
                    elif new_gate == 8:  # CNOT
                        circuit_copy.gate_type_sequence[i] = 2
                
                # 파라미터 변경 (RZ 게이트인 경우)
                if circuit_copy.gate_sequence[i] == 7:  # RZ
                    circuit_copy.param_sequence[i] = random.uniform(0, 2 * np.pi)
        
        return circuit_copy
    
    def crossover(self, other: 'QuantumCircuitRepresentation'):
        """두 회로의 교차 (유전 알고리즘용)"""
        child1 = copy.deepcopy(self)
        child2 = copy.deepcopy(other)
        
        # 단순 교차점 선택
        crossover_point = random.randint(1, min(self.actual_length, other.actual_length) - 1)
        
        # 교차 수행
        for i in range(crossover_point, MAX_SEQUENCE_LENGTH):
            child1.gate_sequence[i] = other.gate_sequence[i]
            child1.qubit_sequence[i] = other.qubit_sequence[i]
            child1.param_sequence[i] = other.param_sequence[i]
            child1.gate_type_sequence[i] = other.gate_type_sequence[i]
            
            child2.gate_sequence[i] = self.gate_sequence[i]
            child2.qubit_sequence[i] = self.qubit_sequence[i]
            child2.param_sequence[i] = self.param_sequence[i]
            child2.gate_type_sequence[i] = self.gate_type_sequence[i]
        
        # 실제 길이 조정
        child1.actual_length = max(self.actual_length, other.actual_length)
        child2.actual_length = max(self.actual_length, other.actual_length)
        
        return child1, child2
    
    def to_features(self):
        """트랜스포머 모델 입력용 특성 생성"""
        # 기본 구조적 특성
        gate_count = self.actual_length
        cnot_count = sum(1 for g in self.gate_sequence[:self.actual_length] if g == 8)
        single_qubit_gates = gate_count - cnot_count
        unique_gate_types = len(set(g for g in self.gate_sequence[:self.actual_length] if g != 0))
        
        # 파라미터 통계
        params = [p for i, p in enumerate(self.param_sequence[:self.actual_length]) 
                 if self.gate_sequence[i] == 7]  # RZ 게이트만
        param_count = len(params)
        param_mean = np.mean(params) if params else 0.0
        param_std = np.std(params) if params else 0.0
        param_min = np.min(params) if params else 0.0
        param_max = np.max(params) if params else 0.0
        
        # 시퀀스 특성
        sequence_length = self.actual_length
        unique_gates_in_sequence = unique_gate_types
        param_gate_ratio = param_count / gate_count if gate_count > 0 else 0.0
        two_qubit_gate_ratio = cnot_count / gate_count if gate_count > 0 else 0.0
        
        # 특성 벡터 구성 (quantum_transformer_model.py와 동일한 순서)
        features = torch.FloatTensor([
            # 구조적 특성 (7개)
            self.n_qubits / 127.0,
            4.0 / 10.0,  # 깊이 (임시로 4 설정)
            gate_count / 200.0,
            cnot_count / 100.0,
            single_qubit_gates / 150.0,
            unique_gate_types / 8.0,
            cnot_count / 50.0,  # cnot_connections (근사)
            
            # 커플링 특성 (6개) - 기본값 설정
            0.5,  # coupling_density
            2.0 / 10.0,  # max_degree
            1.5 / 5.0,   # avg_degree
            0.7,  # connectivity_ratio
            5.0 / 20.0,  # diameter
            0.3,  # clustering_coefficient
            
            # 파라미터 특성 (5개)
            param_count / 50.0,
            param_mean / (2 * np.pi),
            param_std / np.pi,
            param_min / (2 * np.pi),
            param_max / (2 * np.pi),
            
            # 측정 통계 특성 (7개) - 기본값 설정
            5.0 / 20.0,  # entropy
            0.5,  # zero_state_probability
            0.3,  # concentration
            128.0 / 1000.0,  # measured_states
            0.4,  # top_1_probability
            0.7,  # top_5_probability
            0.9,  # top_10_probability
            
            # 시퀀스 특성 (4개)
            sequence_length / MAX_SEQUENCE_LENGTH,
            unique_gates_in_sequence / 8.0,
            param_gate_ratio,
            two_qubit_gate_ratio,
            
            # 하드웨어 특성 (4개) - 기본값 설정
            1.2,  # gate_overhead
            1.1,  # depth_overhead
            gate_count * 1.5 / 50.0,  # transpiled_depth
            gate_count * 2.0 / 500.0,  # transpiled_gate_count
        ])
        
        return features

#################################################
# 3. 회로 평가기 (예측 모델 사용)
#################################################

class CircuitEvaluator:
    """훈련된 트랜스포머 모델을 사용한 회로 평가기"""
    
    def __init__(self, model_path: str):
        self.model = QuantumCircuitTransformer().to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
    
    def evaluate(self, circuit: QuantumCircuitRepresentation) -> Dict[str, float]:
        """회로 성능 예측"""
        with torch.no_grad():
            # 입력 데이터 준비
            gate_seq = torch.LongTensor(circuit.gate_sequence).unsqueeze(0).to(DEVICE)
            qubit_seq = torch.LongTensor(circuit.qubit_sequence).unsqueeze(0).to(DEVICE)
            param_seq = torch.FloatTensor(circuit.param_sequence).unsqueeze(0).to(DEVICE)
            gate_type_seq = torch.LongTensor(circuit.gate_type_sequence).unsqueeze(0).to(DEVICE)
            features = circuit.to_features().unsqueeze(0).to(DEVICE)
            
            # 예측 수행
            predictions = self.model(gate_seq, qubit_seq, param_seq, gate_type_seq, features)
            
            # 결과 반환
            return {
                'fidelity': predictions[0, 0].item(),
                'normalized_expressibility': predictions[0, 1].item(),
                'expressibility_distance': predictions[0, 2].item() * 1e-3,  # 역정규화
            }

#################################################
# 4. 유전 알고리즘 기반 회로 생성기
#################################################

class GeneticCircuitGenerator:
    """유전 알고리즘을 사용한 양자회로 생성기"""
    
    def __init__(self, evaluator: CircuitEvaluator, requirements: UserRequirements):
        self.evaluator = evaluator
        self.requirements = requirements
        self.population_size = 100
        self.generations = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.elite_size = 10
    
    def create_random_circuit(self) -> QuantumCircuitRepresentation:
        """랜덤 회로 생성"""
        circuit = QuantumCircuitRepresentation(self.requirements.max_qubits)
        
        # 랜덤 게이트 수 결정
        num_gates = random.randint(10, min(self.requirements.max_gates, MAX_SEQUENCE_LENGTH))
        
        for _ in range(num_gates):
            # 랜덤 게이트 타입 선택
            gate_type = random.choice([1, 2, 3, 4, 5, 6, 7, 8])
            
            # 큐빗 선택
            if gate_type in [1, 2, 3, 4, 5, 6, 7]:  # 단일 큐빗 게이트
                qubits = [random.randint(0, self.requirements.max_qubits - 1)]
            else:  # CNOT
                q1 = random.randint(0, self.requirements.max_qubits - 1)
                q2 = random.randint(0, self.requirements.max_qubits - 1)
                while q2 == q1:
                    q2 = random.randint(0, self.requirements.max_qubits - 1)
                qubits = [q1, q2]
            
            # 파라미터 (RZ 게이트인 경우)
            param = random.uniform(0, 2 * np.pi) if gate_type == 7 else 0.0
            
            circuit.add_gate(gate_type, qubits, param)
        
        return circuit
    
    def fitness(self, circuit: QuantumCircuitRepresentation) -> float:
        """적합도 함수"""
        try:
            metrics = self.evaluator.evaluate(circuit)
            
            # 목표와의 차이 계산
            fidelity_score = 1.0 - abs(metrics['fidelity'] - self.requirements.target_fidelity)
            expr_score = 1.0 - abs(metrics['normalized_expressibility'] - self.requirements.target_expressibility)
            
            # 효율성 점수 (게이트 수가 적을수록 좋음)
            efficiency_score = 1.0 - (circuit.actual_length / self.requirements.max_gates)
            
            # 가중 평균
            total_score = (
                self.requirements.fidelity_weight * max(0, fidelity_score) +
                self.requirements.expressibility_weight * max(0, expr_score) +
                self.requirements.efficiency_weight * max(0, efficiency_score)
            )
            
            return total_score
            
        except Exception as e:
            print(f"평가 오류: {e}")
            return 0.0
    
    def selection(self, population: List[QuantumCircuitRepresentation], 
                  fitness_scores: List[float]) -> List[QuantumCircuitRepresentation]:
        """토너먼트 선택"""
        selected = []
        
        # 엘리트 선택
        elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
        for idx in elite_indices:
            selected.append(population[idx])
        
        # 나머지 토너먼트 선택
        while len(selected) < self.population_size:
            tournament_size = 3
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        
        return selected
    
    def generate(self) -> Tuple[QuantumCircuitRepresentation, Dict[str, float]]:
        """유전 알고리즘으로 회로 생성"""
        print(f"🧬 유전 알고리즘 시작 (인구: {self.population_size}, 세대: {self.generations})")
        
        # 초기 인구 생성
        population = [self.create_random_circuit() for _ in range(self.population_size)]
        
        best_circuit = None
        best_fitness = -1
        best_metrics = None
        
        for generation in tqdm(range(self.generations), desc="세대 진화 중"):
            # 적합도 평가
            fitness_scores = [self.fitness(circuit) for circuit in population]
            
            # 최고 개체 추적
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_circuit = copy.deepcopy(population[max_fitness_idx])
                best_metrics = self.evaluator.evaluate(best_circuit)
            
            # 진행 상황 출력
            if generation % 10 == 0:
                avg_fitness = np.mean(fitness_scores)
                print(f"  세대 {generation}: 최고 적합도 = {best_fitness:.4f}, 평균 = {avg_fitness:.4f}")
                if best_metrics:
                    print(f"    최고 회로: 피델리티={best_metrics['fidelity']:.4f}, "
                          f"표현력={best_metrics['normalized_expressibility']:.4f}")
            
            # 선택
            selected = self.selection(population, fitness_scores)
            
            # 교차 및 변이
            new_population = selected[:self.elite_size]  # 엘리트 보존
            
            while len(new_population) < self.population_size:
                # 부모 선택
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)
                
                # 교차
                if random.random() < self.crossover_rate:
                    child1, child2 = parent1.crossover(parent2)
                else:
                    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                
                # 변이
                child1 = child1.mutate(self.mutation_rate)
                child2 = child2.mutate(self.mutation_rate)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        print(f"\n🎯 최종 결과:")
        print(f"  최고 적합도: {best_fitness:.4f}")
        if best_metrics:
            print(f"  피델리티: {best_metrics['fidelity']:.4f} (목표: {self.requirements.target_fidelity})")
            print(f"  표현력: {best_metrics['normalized_expressibility']:.4f} (목표: {self.requirements.target_expressibility})")
            print(f"  게이트 수: {best_circuit.actual_length}")
        
        return best_circuit, best_metrics

#################################################
# 5. 메인 실행 함수
#################################################

def main():
    """메인 실행 함수"""
    print("🚀 사용자 요구사항 기반 양자회로 생성기!")
    
    # 1. 훈련된 모델 로드
    model_path = "best_quantum_transformer.pth"
    if not os.path.exists(model_path):
        print(f"❌ 훈련된 모델을 찾을 수 없습니다: {model_path}")
        print("먼저 quantum_transformer_model.py를 실행하여 모델을 훈련하세요.")
        return
    
    print("📥 훈련된 예측 모델 로드 중...")
    evaluator = CircuitEvaluator(model_path)
    
    # 2. 사용자 요구사항 설정
    requirements = UserRequirements(
        target_fidelity=0.85,
        target_expressibility=0.75,
        max_qubits=20,
        max_depth=4,
        max_gates=50,
        fidelity_weight=0.5,
        expressibility_weight=0.3,
        efficiency_weight=0.2
    )
    
    print(f"\n🎯 사용자 요구사항:")
    print(f"  목표 피델리티: {requirements.target_fidelity}")
    print(f"  목표 표현력: {requirements.target_expressibility}")
    print(f"  최대 큐빗: {requirements.max_qubits}")
    print(f"  최대 게이트: {requirements.max_gates}")
    
    # 3. 유전 알고리즘으로 회로 생성
    generator = GeneticCircuitGenerator(evaluator, requirements)
    best_circuit, metrics = generator.generate()
    
    # 4. 결과 저장
    if best_circuit and metrics:
        result = {
            "requirements": {
                "target_fidelity": requirements.target_fidelity,
                "target_expressibility": requirements.target_expressibility,
                "max_qubits": requirements.max_qubits,
                "max_gates": requirements.max_gates
            },
            "generated_circuit": {
                "gate_sequence": best_circuit.gate_sequence[:best_circuit.actual_length],
                "qubit_sequence": best_circuit.qubit_sequence[:best_circuit.actual_length],
                "param_sequence": best_circuit.param_sequence[:best_circuit.actual_length],
                "gate_type_sequence": best_circuit.gate_type_sequence[:best_circuit.actual_length],
                "actual_length": best_circuit.actual_length
            },
            "predicted_metrics": metrics
        }
        
        with open("generated_circuit.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"\n💾 생성된 회로가 'generated_circuit.json'에 저장되었습니다.")
        
        # 5. 회로 정보 출력
        print(f"\n📊 생성된 회로 정보:")
        print(f"  게이트 수: {best_circuit.actual_length}")
        print(f"  사용된 큐빗: {requirements.max_qubits}")
        print(f"  예측 피델리티: {metrics['fidelity']:.4f}")
        print(f"  예측 표현력: {metrics['normalized_expressibility']:.4f}")
        
        # 게이트 분포
        gate_counts = {}
        for i in range(best_circuit.actual_length):
            gate_type = best_circuit.gate_sequence[i]
            gate_name = best_circuit.gate_types.get(gate_type, f"Unknown_{gate_type}")
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
        
        print(f"\n🔧 게이트 분포:")
        for gate, count in gate_counts.items():
            print(f"  {gate}: {count}개")
    
    print("\n🎉 회로 생성 완료!")

if __name__ == "__main__":
    main() 