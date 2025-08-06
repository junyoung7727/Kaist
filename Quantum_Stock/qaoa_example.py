#!/usr/bin/env python3
"""
QAOA (Quantum Approximate Optimization Algorithm) 예제
Max-Cut 문제 해결
"""

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.quantum_info import SparsePauliOp
import matplotlib.pyplot as plt

def create_max_cut_problem(num_nodes=4):
    """Max-Cut 문제 생성"""
    # 간단한 그래프 생성
    G = nx.Graph()
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]  # 사각형 + 대각선
    G.add_edges_from(edges)
    
    print(f"📊 그래프: {num_nodes}개 노드, {len(edges)}개 엣지")
    print(f"🔗 엣지: {edges}")
    
    return G

def graph_to_hamiltonian(graph):
    """그래프를 Max-Cut 해밀토니안으로 변환"""
    num_nodes = len(graph.nodes())
    pauli_strings = []
    coefficients = []
    
    # 각 엣지에 대해 (I - ZZ)/2 항 추가
    for edge in graph.edges():
        i, j = edge
        
        # ZZ 항
        pauli_str = ['I'] * num_nodes
        pauli_str[i] = 'Z'
        pauli_str[j] = 'Z'
        pauli_strings.append(''.join(pauli_str))
        coefficients.append(-0.5)  # -1/2 * ZZ
        
        # 상수항 (I)
        pauli_strings.append('I' * num_nodes)
        coefficients.append(0.5)   # +1/2 * I
    
    return SparsePauliOp(pauli_strings, coefficients)

def create_qaoa_circuit(num_qubits, p_layers=1):
    """QAOA 회로 생성"""
    from qiskit.circuit import Parameter
    
    # 매개변수 정의
    beta = [Parameter(f'β_{i}') for i in range(p_layers)]
    gamma = [Parameter(f'γ_{i}') for i in range(p_layers)]
    
    qc = QuantumCircuit(num_qubits)
    
    # 초기 상태: |+⟩^⊗n (모든 큐비트에 H 게이트)
    qc.h(range(num_qubits))
    
    # QAOA 레이어들
    for layer in range(p_layers):
        # Problem Hamiltonian (예: Max-Cut)
        # 여기서는 간단한 예제로 구현
        for i in range(num_qubits-1):
            qc.rzz(2 * gamma[layer], i, i+1)
        
        # Mixer Hamiltonian (X 회전)
        for i in range(num_qubits):
            qc.rx(2 * beta[layer], i)
    
    return qc

def run_qaoa_example():
    """QAOA 실행 예제"""
    print("🧪 QAOA (Quantum Approximate Optimization Algorithm) 예제")
    print("=" * 60)
    
    # 1. Max-Cut 문제 생성
    graph = create_max_cut_problem(4)
    
    # 2. 해밀토니안 생성
    hamiltonian = graph_to_hamiltonian(graph)
    print(f"\n📊 해밀토니안 항 개수: {len(hamiltonian.paulis)}")
    
    # 3. QAOA 회로 생성
    num_qubits = len(graph.nodes())
    p_layers = 1
    ansatz = create_qaoa_circuit(num_qubits, p_layers)
    
    print(f"\n🔄 QAOA 회로:")
    print(f"   큐비트 수: {num_qubits}")
    print(f"   QAOA 레이어: {p_layers}")
    print(f"   매개변수 수: {ansatz.num_parameters}")
    print(f"   회로 깊이: {ansatz.depth()}")
    
    # 4. 옵티마이저 설정
    optimizer = COBYLA(maxiter=100)
    
    # 5. Sampler (측정 결과 샘플링)
    sampler = Sampler()
    
    try:
        # 6. QAOA 알고리즘 설정
        qaoa = QAOA(
            sampler=sampler,
            optimizer=optimizer,
            reps=p_layers,  # QAOA 레이어 수
            initial_point=np.random.random(ansatz.num_parameters)
        )
        
        # 7. QAOA 실행
        print("\n🚀 QAOA 최적화 시작...")
        result = qaoa.compute_minimum_eigenvalue(hamiltonian)
        
        print("\n✅ QAOA 결과:")
        print(f"🎯 최적값: {result.eigenvalue:.6f}")
        print(f"🎛️  최적 매개변수: {result.optimal_parameters}")
        print(f"🔄 함수 평가 횟수: {result.cost_function_evals}")
        
        # 8. 최적 해 분석
        optimal_circuit = ansatz.assign_parameters(result.optimal_parameters)
        job = sampler.run(optimal_circuit, shots=1000)
        counts = job.result().quasi_dists[0]
        
        print("\n📊 측정 결과 (상위 5개):")
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for bitstring, probability in sorted_counts[:5]:
            # 비트스트링을 정수로 변환
            binary_str = format(bitstring, f'0{num_qubits}b')
            cut_value = calculate_cut_value(graph, binary_str)
            print(f"   {binary_str}: {probability:.3f} (Cut value: {cut_value})")
        
    except Exception as e:
        print(f"❌ QAOA 실행 오류: {e}")
        print("💡 필요한 패키지:")
        print("   pip install qiskit-algorithms")
        print("   pip install qiskit-optimization")

def calculate_cut_value(graph, bitstring):
    """주어진 비트스트링에 대한 Cut 값 계산"""
    cut_value = 0
    for edge in graph.edges():
        i, j = edge
        if bitstring[i] != bitstring[j]:  # 다른 그룹에 속하면
            cut_value += 1
    return cut_value

def solve_classical_max_cut(graph):
    """고전적 Max-Cut 해법 (완전 탐색)"""
    num_nodes = len(graph.nodes())
    max_cut = 0
    best_partition = None
    
    # 모든 가능한 분할 시도
    for i in range(2**num_nodes):
        partition = format(i, f'0{num_nodes}b')
        cut_value = calculate_cut_value(graph, partition)
        if cut_value > max_cut:
            max_cut = cut_value
            best_partition = partition
    
    return max_cut, best_partition

def run_comparison():
    """QAOA vs 고전 알고리즘 비교"""
    print("\n" + "=" * 60)
    print("🆚 QAOA vs 고전 알고리즘 비교")
    print("=" * 60)
    
    graph = create_max_cut_problem(4)
    
    # 고전적 해법
    classical_max, classical_partition = solve_classical_max_cut(graph)
    print(f"🏆 고전적 최적해: {classical_max}")
    print(f"📝 최적 분할: {classical_partition}")
    
    print("\n💡 QAOA의 장점:")
    print("1. 큰 그래프에서 근사해를 빠르게 찾음")
    print("2. 양자 우위 가능성 (특정 문제에서)")
    print("3. 하이브리드 양자-고전 알고리즘")
    print("4. 노이즈가 있는 양자 컴퓨터에서도 작동")

if __name__ == "__main__":
    run_qaoa_example()
    run_comparison()
    
    print("\n" + "=" * 60)
    print("📋 QAOA 사용 팁:")
    print("1. p 레이어 수 조정 (깊이 vs 정확도 트레이드오프)")
    print("2. 다양한 옵티마이저 시도")
    print("3. 초기값 설정 최적화")
    print("4. 문제별 ansatz 커스터마이징")
    print("5. 노이즈 완화 기법 적용")
    print("6. qiskit-optimization으로 더 복잡한 문제 해결")
