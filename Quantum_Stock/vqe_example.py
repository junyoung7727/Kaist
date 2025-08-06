#!/usr/bin/env python3
"""
VQE (Variational Quantum Eigensolver) 예제
H2 분자의 기저상태 에너지 계산
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SPSA, SLSQP
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp

def create_h2_hamiltonian():
    """H2 분자 해밀토니안 생성 (간단한 예제)"""
    # H2 분자의 간단한 해밀토니안 (실제로는 qiskit-nature 사용 권장)
    pauli_strings = [
        "II", "IZ", "ZI", "ZZ", "XX"
    ]
    coefficients = [-1.052373245772859, 0.39793742484318045, -0.39793742484318045, 
                   -0.01128010425623538, 0.18093119978423156]
    
    return SparsePauliOp(pauli_strings, coefficients)

def create_ansatz(num_qubits=2, reps=1):
    """변분 회로 (ansatz) 생성"""
    # TwoLocal ansatz 사용
    ansatz = TwoLocal(
        num_qubits=num_qubits,
        rotation_blocks='ry',  # RY 회전 게이트
        entanglement_blocks='cz',  # CZ 얽힘 게이트
        entanglement='linear',  # 선형 얽힘
        reps=reps
    )
    return ansatz

def run_vqe_example():
    """VQE 실행 예제"""
    print("🧪 VQE (Variational Quantum Eigensolver) 예제")
    print("=" * 50)
    
    # 1. 해밀토니안 생성
    hamiltonian = create_h2_hamiltonian()
    print(f"📊 해밀토니안: {hamiltonian}")
    
    # 2. Ansatz 생성
    ansatz = create_ansatz(num_qubits=2, reps=1)
    print(f"🔄 Ansatz 깊이: {ansatz.depth()}")
    print(f"🎛️  매개변수 개수: {ansatz.num_parameters}")
    
    # 3. 옵티마이저 설정
    optimizer = SPSA(maxiter=100)
    
    # 4. Estimator (양자 기댓값 계산)
    estimator = Estimator()
    
    # 5. VQE 알고리즘 설정
    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        initial_point=np.random.random(ansatz.num_parameters)
    )
    
    # 6. VQE 실행
    print("\n🚀 VQE 최적화 시작...")
    try:
        result = vqe.compute_minimum_eigenvalue(hamiltonian)
        
        print("\n✅ VQE 결과:")
        print(f"🎯 최소 고유값 (기저상태 에너지): {result.eigenvalue:.6f}")
        print(f"🎛️  최적 매개변수: {result.optimal_parameters}")
        print(f"🔄 함수 평가 횟수: {result.cost_function_evals}")
        
        # 이론값과 비교 (H2 분자 기저상태 에너지)
        exact_energy = -1.857275030202
        error = abs(result.eigenvalue - exact_energy)
        print(f"📏 이론값: {exact_energy:.6f}")
        print(f"❌ 오차: {error:.6f}")
        
    except Exception as e:
        print(f"❌ VQE 실행 오류: {e}")
        print("💡 qiskit-algorithms 설치 필요: pip install qiskit-algorithms")

def create_custom_ansatz():
    """사용자 정의 ansatz 예제"""
    qc = QuantumCircuit(2)
    
    # 매개변수화된 회로
    from qiskit.circuit import Parameter
    theta1 = Parameter('θ₁')
    theta2 = Parameter('θ₂')
    theta3 = Parameter('θ₃')
    theta4 = Parameter('θ₄')
    
    # 레이어 1
    qc.ry(theta1, 0)
    qc.ry(theta2, 1)
    qc.cz(0, 1)
    
    # 레이어 2
    qc.ry(theta3, 0)
    qc.ry(theta4, 1)
    
    return qc

if __name__ == "__main__":
    run_vqe_example()
    
    print("\n" + "=" * 50)
    print("📋 VQE 사용 팁:")
    print("1. qiskit-nature로 실제 분자 해밀토니안 생성")
    print("2. 다양한 ansatz 시도 (Hardware Efficient, UCCSD 등)")
    print("3. 옵티마이저 튜닝 (SPSA, COBYLA, L-BFGS-B)")
    print("4. 노이즈 완화 기법 적용")
    print("5. 실제 양자 하드웨어에서 실행")
