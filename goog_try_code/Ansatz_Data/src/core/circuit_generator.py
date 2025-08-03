#!/usr/bin/env python3
"""
회로 생성 모듈 - 양자 회로 생성 로직을 처리합니다.
"""

import sys
import os
from typing import Dict, List, Any, Optional, Union, Tuple

# 현재 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import config
from src.core.circuit_base import QuantumCircuitBase  # QuantumCircuitBase 클래스 임포트


def generate_all_circuits() -> List[Dict[str, Any]]:
    """
    config 설정을 활용한 2큐빗 게이트 비율 테스트용 회로 생성
    
    Returns:
        List[Dict[str, Any]]: 생성된 회로 정보 목록
    """
    # 지연 임포트 패턴 유지
    from src.core.circuit_base import QuantumCircuitBase
    
    # 중앙화된 구성에서 쿠빗 설정 가져오기
    # 속성 접근 방식 사용
    n_qubits_list = config.data_generation.qubit_presets  # ConfigBox를 통한 속성 접근
    
    # 회로 깊이 리스트 생성
    depth_list = config.data_generation.depth_presets

    # 회로 생성 파라미터 가져오기
    two_qubit_ratios = config.circuit_generation_params.two_qubit_ratios
    circuits_per_config = config.circuit_generation_params.circuits_per_config
    generation_strategy = config.circuit_generation_params.generation_strategy

    # 총 회로 수 계산
    total_circuits = len(n_qubits_list) * len(depth_list) * len(two_qubit_ratios) * circuits_per_config
    print(f"🔧 테스트용 2큐빗 게이트 비율 테스트 {total_circuits}개 회로 생성 중...")
    print(f"   큐빗 수: {n_qubits_list}")
    print(f"   회로 깊이: {depth_list}")
    print(f"   2큐빗 게이트 비율: {[f'{r:.1%}' for r in two_qubit_ratios]}")
    print(f"   각 설정당 회로 수: {circuits_per_config}")
    print(f"   생성 전략: {generation_strategy}")
    
    base_circuit = QuantumCircuitBase()
    all_circuits = []
    
    circuit_id = 0
    for n_qubits in n_qubits_list:
        for depth in depth_list:
            for two_qubit_ratio in two_qubit_ratios:
                print(f"  생성 중: {n_qubits}큐빗, 깊이{depth}, 2큐빗비율{two_qubit_ratio:.1%} - {circuits_per_config}개 회로")
                
                for i in range(circuits_per_config):
                    # 회로 생성 (2큐빗 게이트 비율 지정)
                    circuit_info = base_circuit.generate_random_circuit(
                        n_qubits=n_qubits,
                        depth=depth,
                        strategy=generation_strategy,  # 설정에서 가져온 전략 사용
                        seed=circuit_id + i,  # 재현 가능한 시드
                        two_qubit_ratio=two_qubit_ratio  # 2큐빗 게이트 비율 설정
                    )
                    
                    # 회로 ID 및 메타데이터 추가
                    circuit_info["circuit_id"] = circuit_id
                    circuit_info["config_group"] = f"q{n_qubits}_d{depth}_r{int(two_qubit_ratio*100)}"
                    circuit_info["two_qubit_ratio_target"] = two_qubit_ratio
                    
                    all_circuits.append(circuit_info)
                    circuit_id += 1
                
                # 진행 상황 출력
                progress = (circuit_id / total_circuits) * 100
                print(f"    진행률: {progress:.1f}% ({circuit_id}/{total_circuits})")
    
    print(f"✅ 총 {len(all_circuits)}개 회로 생성 완료!")
    
    # 설정별 회로 수 요약
    print("\n📊 설정별 회로 수 요약:")
    config_counts = {}
    for circuit in all_circuits:
        config_group = circuit["config_group"]
        if config_group in config_counts:
            config_counts[config_group] += 1
        else:
            config_counts[config_group] = 1
    
    for config_group, count in sorted(config_counts.items()):
        print(f"  {config_group}: {count}개")
    
    return all_circuits
