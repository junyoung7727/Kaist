#!/usr/bin/env python3
"""
IBM Quantum 샷 수 계산기
1회 제출량 1천만 샷 제한 내에서 실험 설계 검증
"""

from config import Exp_Box

def calculate_total_shots(exp_config):
    """실험 설정의 총 샷 수 계산"""
    
    # 기본 파라미터
    num_qubits_list = exp_config.num_qubits
    depth_list = exp_config.depth if isinstance(exp_config.depth, list) else [exp_config.depth]
    two_qubit_ratios = exp_config.two_qubit_ratio
    num_circuits = exp_config.num_circuits
    
    # 각 메트릭별 샷 수 계산
    total_combinations = len(num_qubits_list) * len(depth_list) * len(two_qubit_ratios) * num_circuits
    
    print(f"🔍 실험 설정: {exp_config.exp_name}")
    print(f"  - 큐빗 수: {num_qubits_list} ({len(num_qubits_list)}개)")
    print(f"  - 깊이: {depth_list} ({len(depth_list)}개)")
    print(f"  - Two-qubit 비율: {two_qubit_ratios} ({len(two_qubit_ratios)}개)")
    print(f"  - 회로 수: {num_circuits}개")
    print(f"  - 총 조합: {total_combinations}개")
    
    # 1. 피델리티 샷 수
    fidelity_shots = exp_config.fidelity_shots * total_combinations
    print(f"\n📊 피델리티 측정:")
    print(f"  - 회로당 샷: {exp_config.fidelity_shots}")
    print(f"  - 총 샷: {fidelity_shots:,}")
    
    # 2. 표현력 샷 수 (SWAP test 페어)
    num_samples = exp_config.num_samples
    pairs_per_circuit = num_samples * (num_samples - 1) // 2  # C(n,2)
    expressibility_shots = exp_config.shots * pairs_per_circuit * total_combinations
    print(f"\n📊 표현력 측정:")
    print(f"  - 샘플 수: {num_samples}개")
    print(f"  - 페어 수: {pairs_per_circuit}개")
    print(f"  - 페어당 샷: {exp_config.shots}")
    print(f"  - 총 샷: {expressibility_shots:,}")
    
    # 3. 얽힘도 샷 수 (큐빗별 SWAP test)
    entanglement_shots_total = 0
    for num_qubits in num_qubits_list:
        shots_for_this_qubit = exp_config.entangle_shots * num_qubits * len(depth_list) * len(two_qubit_ratios) * num_circuits
        entanglement_shots_total += shots_for_this_qubit
    
    print(f"\n📊 얽힘도 측정:")
    print(f"  - 큐빗당 샷: {exp_config.entangle_shots}")
    print(f"  - 총 샷: {entanglement_shots_total:,}")
    
    # 총합 계산
    total_shots = fidelity_shots + expressibility_shots + entanglement_shots_total
    
    print(f"\n🎯 총 샷 수 요약:")
    print(f"  - 피델리티: {fidelity_shots:,} ({fidelity_shots/total_shots*100:.1f}%)")
    print(f"  - 표현력: {expressibility_shots:,} ({expressibility_shots/total_shots*100:.1f}%)")
    print(f"  - 얽힘도: {entanglement_shots_total:,} ({entanglement_shots_total/total_shots*100:.1f}%)")
    print(f"  - 총합: {total_shots:,}")
    
    # IBM 제한 확인
    ibm_limit = 10_000_000  # 1천만 샷
    print(f"\n🚨 IBM 제한 확인:")
    print(f"  - IBM 1회 제출 제한: {ibm_limit:,} 샷")
    print(f"  - 현재 설계: {total_shots:,} 샷")
    print(f"  - 사용률: {total_shots/ibm_limit*100:.1f}%")
    
    if total_shots <= ibm_limit:
        print(f"  - ✅ 제한 내 (여유: {ibm_limit-total_shots:,} 샷)")
    else:
        print(f"  - ❌ 제한 초과 (초과: {total_shots-ibm_limit:,} 샷)")
        print(f"  - 권장: 샷 수를 {total_shots/ibm_limit:.1f}배 줄이세요")
    
    return total_shots

if __name__ == "__main__":
    print("=" * 60)
    print("IBM Quantum 샷 수 계산기")
    print("=" * 60)
    
    # 기존 exp1 설정 테스트
    print("\n" + "=" * 40)
    calculate_total_shots(Exp_Box.exp1)
    
    # 새로운 scalability_test 설정 테스트
    print("\n" + "=" * 40)
    calculate_total_shots(Exp_Box.scalability_test)
