#!/usr/bin/env python3
"""
데이터셋의 게이트 다양성 검증 스크립트
"""

import json
import sys
from pathlib import Path
from collections import Counter

# 경로 추가
sys.path.append(str(Path(__file__).parent.parent / "quantumcommon"))
from gates import QuantumGateRegistry

def verify_data_diversity(data_path: str):
    """데이터셋의 게이트 다양성 검증"""
    
    print(f"🔍 데이터셋 검증: {data_path}")
    
    # 게이트 레지스트리 초기화
    gate_registry = QuantumGateRegistry()
    gate_vocab = gate_registry.get_gate_vocab()
    
    print(f"📋 사용 가능한 게이트 타입: {len(gate_vocab)}")
    print(f"게이트 vocab: {gate_vocab}")
    
    # 데이터 로드
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {data_path}")
        return
    
    # 회로 분석 (다양한 키 시도)
    circuits = data.get('merged_circuits', data.get('circuits', {}))
    print(f"📊 총 회로 수: {len(circuits)}")
    
    # 데이터 구조 확인
    if len(circuits) == 0:
        print("⚠️  다른 키들 확인 중...")
        for key in data.keys():
            if isinstance(data[key], dict) and len(data[key]) > 0:
                print(f"   발견된 키: '{key}' ({len(data[key])}개 항목)")
                # 첫 번째 항목이 회로 데이터인지 확인
                first_item = next(iter(data[key].values()))
                if isinstance(first_item, dict) and 'gates' in first_item:
                    circuits = data[key]
                    print(f"   ✅ '{key}'를 회로 데이터로 사용")
                    break
    
    # 게이트 이름 수집
    all_gate_names = []
    gate_counter = Counter()
    
    for circuit_id, circuit_data in circuits.items():
        gates = circuit_data.get('gates', [])
        for gate in gates:
            gate_name = gate.get('name', '').lower()
            all_gate_names.append(gate_name)
            gate_counter[gate_name] += 1
    
    print(f"\n🎯 발견된 게이트 타입:")
    for gate_name, count in gate_counter.most_common():
        gate_index = gate_vocab.get(gate_name, -1)
        print(f"  {gate_name}: {count}개 (인덱스: {gate_index})")
    
    print(f"\n📈 게이트 다양성 통계:")
    print(f"  고유 게이트 타입 수: {len(gate_counter)}")
    print(f"  총 게이트 수: {len(all_gate_names)}")
    if len(circuits) > 0:
        print(f"  평균 게이트/회로: {len(all_gate_names) / len(circuits):.1f}")
    else:
        print(f"  평균 게이트/회로: N/A (회로 없음)")
    
    # 문제 진단
    if len(gate_counter) == 1:
        print(f"\n❌ 심각한 문제: 단일 게이트 타입만 발견!")
        single_gate = list(gate_counter.keys())[0]
        print(f"   모든 게이트가 '{single_gate}'입니다.")
    elif len(gate_counter) < 5:
        print(f"\n⚠️  경고: 게이트 다양성 부족 ({len(gate_counter)}개 타입)")
    else:
        print(f"\n✅ 양호: {len(gate_counter)}개의 다양한 게이트 타입 발견")
    
    return gate_counter

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="데이터셋 게이트 다양성 검증")
    parser.add_argument('--data_path', type=str, 
                       default=r'C:\Users\jungh\Documents\GitHub\Kaist\OAT_Model\raw_data\merged_data.json',
                       help='검증할 데이터셋 경로')
    
    args = parser.parse_args()
    
    verify_data_diversity(args.data_path)
