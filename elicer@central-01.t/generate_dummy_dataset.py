#!/usr/bin/env python3
"""
더미 양자 회로 데이터셋 생성기
수백 개의 다양한 양자 회로와 결과를 포함한 JSON 파일 생성
"""

import json
import random
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import argparse

# 지원하는 게이트 타입들
SINGLE_QUBIT_GATES = [
    {"name": "h", "params": 0},
    {"name": "x", "params": 0},
    {"name": "y", "params": 0},
    {"name": "z", "params": 0},
    {"name": "s", "params": 0},
    {"name": "t", "params": 0},
    {"name": "rx", "params": 1},
    {"name": "ry", "params": 1},
    {"name": "rz", "params": 1},
    {"name": "p", "params": 1},
]

TWO_QUBIT_GATES = [
    {"name": "cx", "params": 0},
    {"name": "cy", "params": 0},
    {"name": "cz", "params": 0},
    {"name": "crx", "params": 1},
    {"name": "cry", "params": 1},
    {"name": "crz", "params": 1},
    {"name": "swap", "params": 0},
]

def generate_random_parameter() -> float:
    """랜덤 파라미터 생성 (0 ~ 2π)"""
    return random.uniform(0, 2 * np.pi)

def generate_gate(num_qubits: int) -> Dict[str, Any]:
    """랜덤 게이트 생성"""
    # 1-qubit vs 2-qubit 게이트 선택 (7:3 비율)
    if random.random() < 0.7 or num_qubits == 1:
        gate_info = random.choice(SINGLE_QUBIT_GATES)
        qubits = [random.randint(0, num_qubits - 1)]
    else:
        gate_info = random.choice(TWO_QUBIT_GATES)
        # 서로 다른 두 큐비트 선택
        qubit1 = random.randint(0, num_qubits - 1)
        qubit2 = random.randint(0, num_qubits - 1)
        while qubit2 == qubit1:
            qubit2 = random.randint(0, num_qubits - 1)
        qubits = [qubit1, qubit2]
    
    # 파라미터 생성
    parameters = []
    for _ in range(gate_info["params"]):
        parameters.append(generate_random_parameter())
    
    return {
        "name": gate_info["name"],
        "qubits": qubits,
        "parameters": parameters
    }

def generate_circuit(circuit_id: str, num_qubits: int, depth: int) -> Dict[str, Any]:
    """랜덤 양자 회로 생성"""
    gates = []
    for _ in range(depth):
        gate = generate_gate(num_qubits)
        gates.append(gate)
    
    return {
        "circuit_id": circuit_id,
        "num_qubits": num_qubits,
        "gates": gates,
        "qasm": None
    }

def generate_realistic_metrics(num_qubits: int, depth: int) -> Dict[str, Any]:
    """현실적인 메트릭 값들 생성"""
    # 큐비트 수와 깊이에 따라 현실적인 값 범위 설정
    base_fidelity = max(0.5, 1.0 - (depth * 0.01) - (num_qubits * 0.02))
    noise_factor = random.uniform(0.9, 1.1)
    
    fidelity = min(1.0, max(0.0, base_fidelity * noise_factor))
    robust_fidelity = fidelity * random.uniform(0.85, 0.98)
    
    # Expressibility (KL divergence): 0에 가까울수록 좋음
    kl_divergence = random.uniform(0.0, 0.5) * (1 + depth * 0.01)
    
    # Entanglement: 큐비트 수와 2-qubit 게이트 비율에 따라
    max_entanglement = 1.0 - (1.0 / (2 ** (num_qubits - 1)))
    entanglement = random.uniform(0.0, max_entanglement) * random.uniform(0.3, 1.0)
    
    return {
        "fidelity": round(fidelity, 6),
        "robust_fidelity": round(robust_fidelity, 6),
        "expressibility": {
            "kl_divergence": round(kl_divergence, 6)
        },
        "entanglement": round(entanglement, 6)
    }

def generate_experiment_dataset(
    experiment_name: str,
    num_circuits: int,
    qubit_range: tuple = (2, 5),
    depth_range: tuple = (5, 25),
    randomness_levels: List[float] = [0.1, 0.3, 0.5]
) -> Dict[str, Any]:
    """실험 데이터셋 생성"""
    
    circuits = {}
    results = {}
    timestamp = datetime.now().isoformat()
    
    for i in range(num_circuits):
        # 랜덤 파라미터들
        num_qubits = random.randint(*qubit_range)
        depth = random.randint(*depth_range)
        randomness = random.choice(randomness_levels)
        
        # 회로 ID 생성
        circuit_id = f"{experiment_name}_{num_qubits}q_d{depth}_r{randomness}_{i}"
        
        # 회로 생성
        circuit = generate_circuit(circuit_id, num_qubits, depth)
        circuits[circuit_id] = circuit
        
        # 결과 생성
        metrics = generate_realistic_metrics(num_qubits, depth)
        results[circuit_id] = {
            "circuit_id": circuit_id,
            "num_qubits": num_qubits,
            "depth": depth,
            "timestamp": timestamp,
            **metrics
        }
    
    return {
        "experiment_name": experiment_name,
        "circuits": circuits,
        "results": results,
        "summary": {
            "total_results": len(results),
            "total_circuits": len(circuits),
            "matched_pairs": len(circuits),
            "results_without_circuits": 0,
            "circuits_without_results": 0
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Generate dummy quantum circuit dataset")
    parser.add_argument("--num_circuits", type=int, default=500, help="Number of circuits to generate")
    parser.add_argument("--output", type=str, default="dummy_quantum_dataset.json", help="Output JSON file")
    parser.add_argument("--experiment_name", type=str, default="dummy_exp", help="Experiment name")
    parser.add_argument("--min_qubits", type=int, default=2, help="Minimum number of qubits")
    parser.add_argument("--max_qubits", type=int, default=6, help="Maximum number of qubits")
    parser.add_argument("--min_depth", type=int, default=5, help="Minimum circuit depth")
    parser.add_argument("--max_depth", type=int, default=30, help="Maximum circuit depth")
    
    args = parser.parse_args()
    
    print(f"🔧 Generating {args.num_circuits} quantum circuits...")
    print(f"   Qubits: {args.min_qubits}-{args.max_qubits}")
    print(f"   Depth: {args.min_depth}-{args.max_depth}")
    print(f"   Output: {args.output}")
    
    # 데이터셋 생성
    dataset = generate_experiment_dataset(
        experiment_name=args.experiment_name,
        num_circuits=args.num_circuits,
        qubit_range=(args.min_qubits, args.max_qubits),
        depth_range=(args.min_depth, args.max_depth),
        randomness_levels=[0.1, 0.2, 0.3, 0.4, 0.5]
    )
    
    # JSON 파일로 저장
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Successfully generated {len(dataset['circuits'])} circuits")
    print(f"   Saved to: {args.output}")
    print(f"   File size: {len(json.dumps(dataset)) / 1024 / 1024:.2f} MB")
    
    # 통계 출력
    qubit_counts = {}
    depth_counts = {}
    for circuit in dataset['circuits'].values():
        nq = circuit['num_qubits']
        depth = len(circuit['gates'])
        qubit_counts[nq] = qubit_counts.get(nq, 0) + 1
        depth_counts[depth] = depth_counts.get(depth, 0) + 1
    
    print("\n📊 Dataset Statistics:")
    print(f"   Qubit distribution: {dict(sorted(qubit_counts.items()))}")
    print(f"   Depth range: {min(depth_counts.keys())}-{max(depth_counts.keys())}")

if __name__ == "__main__":
    main()
