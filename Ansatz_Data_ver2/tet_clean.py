from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt
import numpy as np

def create_5qubit_circuit_with_inverse():
    """
    5큐빗 간단하고 대칭적인 양자회로를 생성하고 역회로를 붙여서 시각화
    """
    # 5큐빗 양자회로 생성
    qc = QuantumCircuit(5)
    
    # Layer 1: 모든 큐빗에 Hadamard (완전 대칭)
    for i in range(5):
        qc.h(i)
    qc.barrier()  # 시각적 구분
    
    # Layer 2: 인접 큐빗 간 CNOT 체인
    for i in range(4):
        qc.cx(i, i+1)
    qc.barrier()
    
    # Layer 3: 회전 게이트 (간단한 패턴)
    qc.ry(np.pi/4, 2)  # 중앙 큐빗만
    qc.barrier()
    
    # Layer 4: 대칭 연결 (양 끝만)
    qc.cx(0, 4)  # 양 끝 연결
    qc.barrier()  # 최종 구분
    
    print("원본 회로 깊이:", qc.depth())
    print("원본 회로 게이트 수:", len(qc.data))
    print("회로 구조: 완전 대칭적 설계")
    
    # 역회로 생성
    inverse_qc = qc.inverse()
    
    # 원본 회로 + 역회로 합성
    combined_qc = qc.compose(inverse_qc)
    
    print("합성 회로 깊이:", combined_qc.depth())
    print("합성 회로 게이트 수:", len(combined_qc.data))
    
    return qc, inverse_qc, combined_qc

def visualize_circuits():
    """
    원본 + 역회로를 하나의 합성된 회로로 시각화
    """
    # 회로 생성
    original, inverse, combined = create_5qubit_circuit_with_inverse()
    
    # 시각화 설정 - 하나의 큰 그림으로
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    
    # 합성 회로만 시각화 (원본 + 역회로)
    combined_img = circuit_drawer(combined, output='mpl', style='iqp', ax=ax)
    ax.set_title('5큐빗 양자회로 + 역회로 (완전한 합성)', fontsize=16, fontweight='bold', pad=30)
    
    plt.tight_layout()
    plt.savefig('5qubit_combined_circuit.png', dpi=300, bbox_inches='tight')
    print("\n✅ 합성 회로 이미지가 '5qubit_combined_circuit.png'로 저장되었습니다.")
    plt.show()
    
    return original, inverse, combined

def analyze_circuit_properties(qc):
    """
    회로의 속성 분석
    """
    print(f"\n=== 회로 분석 ===")
    print(f"큐빗 수: {qc.num_qubits}")
    print(f"회로 깊이: {qc.depth()}")
    print(f"총 게이트 수: {len(qc.data)}")
    
    # 게이트 종류별 카운트
    gate_counts = {}
    for instruction in qc.data:
        gate_name = instruction[0].name
        gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
    
    print("\n게이트 종류별 개수:")
    for gate, count in sorted(gate_counts.items()):
        print(f"  {gate}: {count}개")
    
    return gate_counts

def main():
    """
    메인 실행 함수
    """
    print("🚀 5큐빗 양자회로 + 역회로 시각화")
    print("=" * 50)
    
    try:
        # 회로 생성 및 시각화
        original, inverse, combined = visualize_circuits()
        
        # 각 회로 분석
        print("\n📊 원본 회로 분석:")
        analyze_circuit_properties(original)
        
        print("\n📊 역회로 분석:")
        analyze_circuit_properties(inverse)
        
        print("\n📊 합성 회로 분석:")
        analyze_circuit_properties(combined)
        
        # 회로 정보 출력
        print("\n" + "="*50)
        print("💡 회로 정보:")
        print("- 원본 회로: 간단하고 대칭적인 구조 (H → CNOT체인 → 회전 → 교차연결)")
        print("- 게이트 수: 최소화된 효율적 설계")
        print("- 역회로: 원본의 모든 연산을 역순으로 수행")
        print("- 합성 회로: 원본 + 역회로 = 항등 연산 (이론적으로 |0⟩⊗5 상태로 복원)")
        print("- 시각적 특징: 하나의 연결된 회로로 전체 과정 표시")
        print("\n✨ 시각화 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
