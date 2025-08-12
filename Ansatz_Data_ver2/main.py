#!/usr/bin/env python3
"""
Quantum Circuit Backend - Main Entry Point

IBM Quantum 통합 배치 처리 시스템 (최종 버전)
기존 3회 백엔드 연결을 1회로 최적화한 메인 실행 파일입니다.

주요 기능:
- 피델리티, 표현력, 얽힘도 통합 배치 처리
- 단일 백엔드 연결로 모든 측정 수행
- JSON 결과 저장 및 성능 분석
- 하위 호환성 보장
"""

# 경로 설정
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import time
import json
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# 핵심 모듈 임포트
from config import default_config, Exp_Box
from execution.executor import QuantumExecutorFactory
from core.random_circuit_generator import generate_random_circuit
from utils.result_handler import ResultHandler

# 통합 배치 처리 시스템
from core.batch_manager import QuantumCircuitBatchManager, ResultDistributor

# 태스크 모듈들
from core.error_fidelity import run_error_fidelity
from expressibility.fidelity_divergence import Divergence_Expressibility
from core.entangle_hardware import meyer_wallace_entropy_swap_test
from core.entangle_simulator import meyer_wallace_entropy


def print_summary(results: Dict[str, Any]):
    """
    통합 배치 처리 결과 요약 출력
    
    Args:
        results: 통합 배치 처리 결과 딕셔너리
    """
    print("\n" + "="*50)
    print("IBM QUANTUM 통합 배치 처리 결과")
    print("="*50)
    
    # 기본 정보
    print(f"🔌 백엔드: {results.get('backend_type', 'Unknown').upper()}")
    print(f"📅 실험 시간: {results.get('timestamp', 'N/A')}")
    
    # 성능 지표
    performance = results.get('performance', {})
    print(f"\n🚀 성능 지표:")
    print(f"   - 총 실행 시간: {performance.get('total_time', 0):.2f}초")
    print(f"   - 배치 실행 시간: {performance.get('batch_time', 0):.2f}초")
    print(f"   - 백엔드 연결 수: {performance.get('backend_connections', 1)}회")
    print(f"   - 최적화 비율: {performance.get('optimization_ratio', 0)*100:.0f}% 단축")
    
    # 회로 정보
    circuit_results = results.get('circuit_results', [])
    total_circuits = results.get('total_circuits', len(circuit_results))
    print(f"\n📊 회로 정보:")
    print(f"   - 총 회로 수: {total_circuits}개")
    print(f"   - 처리된 회로: {len(circuit_results)}개")
    print(f"   - 성공률: {results.get('success_rate', 0)*100:.1f}%")
    
    # 결과 요약
    if circuit_results:
        fidelities = [r.get('fidelity') for r in circuit_results if r.get('fidelity') is not None]
        expressibilities = [r.get('expressibility', {}).get('kl_divergence') for r in circuit_results if r.get('expressibility') is not None]
        entanglements = [r.get('entanglement') for r in circuit_results if r.get('entanglement') is not None]
        
        print(f"\n📈 측정 결과:")
        if fidelities:
            avg_fidelity = sum(fidelities) / len(fidelities)
            print(f"   - 피델리티: {len(fidelities)}개 회로, 평균 {avg_fidelity:.4f}")
        
        if expressibilities:
            avg_expr = sum(e for e in expressibilities if e is not None) / len([e for e in expressibilities if e is not None])
            print(f"   - 표현력: KL divergence {avg_expr:.4f}")
        
        if entanglements:
            avg_entangle = sum(entanglements) / len(entanglements)
            print(f"   - 얽힘도: {len(entanglements)}개 회로, 평균 MW entropy {avg_entangle:.4f}")
    
    # 오류 정보
    errors = results.get('errors', [])
    if errors:
        print(f"\n⚠️ 오류 정보: {len(errors)}건")
        for error in errors[:3]:  # 최대 3개만 표시
            print(f"   - {error}")
    
    print("="*50)


def run_unified_batch_experiment(backend_type: str, exp_config) -> Dict[str, Any]:
    """
    통합 배치 처리를 사용한 완전한 실험 실행
    
    Args:
        backend_type: 백엔드 유형 ("ibm" 또는 "simulator")
        exp_config: 실험 설정
        
    Returns:
        완전한 실험 결과 딕셔너리
    """
    print(f"\n🚀 {backend_type.upper()} 백엔드에서 통합 배치 처리 실행...")
    
    experiment_start_time = time.time()
    
    # 실험 회로 생성
    print(f"🔧 실험 회로 생성 중...")
    
    # generate_random_circuit는 리스트를 반환하므로 직접 사용
    exp_circuits = generate_random_circuit(exp_config)
    
    # 회로 ID가 없는 경우 설정
    for i, circuit in enumerate(exp_circuits):
        if not hasattr(circuit, 'circuit_id') or not circuit.circuit_id:
            circuit.circuit_id = f"circuit_{i}"
    
    print(f"✅ {len(exp_circuits)}개 회로 생성 완료")
    
    # 배치 처리 실행
    batch_start_time = time.time()
    
    if backend_type == "ibm":
        circuit_results, errors = run_ibm_unified_batch_processing(exp_circuits, exp_config)
    else:
        circuit_results, errors = run_simulator_unified_batch_processing(exp_circuits, exp_config)
    
    batch_time = time.time() - batch_start_time
    total_time = time.time() - experiment_start_time
    
    # 결과 종합
    results = {
        "timestamp": datetime.now().isoformat(),
        "backend_type": backend_type,
        "experiment_config": {
            "num_circuits": exp_config.num_circuits,
            "num_qubits": exp_config.num_qubits,
            "depth": exp_config.depth,
            "shots": getattr(exp_config, 'shots', 1024)
        },
        "total_circuits": len(exp_circuits),
        "circuit_results": circuit_results,
        "circuit_specs": exp_circuits,  # 회로 정보 추가
        "performance": {
            "total_time": total_time,
            "batch_time": batch_time,
            "circuits_per_second": len(exp_circuits) / batch_time if batch_time > 0 else 0,
            "backend_connections": 1,
            "optimization_ratio": 0.67  # 3회 → 1회 연결
        },
        "errors": errors,
        "success_rate": (len(circuit_results) - len(errors)) / len(circuit_results) if circuit_results else 0
    }
    
    return results


def run_ibm_unified_batch_processing(exp_circuits: List, exp_config) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    IBM 하드웨어에서 통합 배치 처리 실행
    
    Args:
        exp_circuits: 실험 회로 리스트
        exp_config: 실험 설정
        
    Returns:
        (circuit_results, errors): 회로별 결과와 오류 리스트
    """
    print("\n🚀 IBM 하드웨어 통합 배치 처리 시작...")
    
    errors = []
    
    # 실행자 생성
    executor = QuantumExecutorFactory.create_executor("ibm")
    exp_config.executor = executor
    
    # 배치 매니저 초기화
    batch_manager = QuantumCircuitBatchManager(exp_config)
    
    # 1. 피델리티 태스크 수집
    print("🎯 피델리티 태스크 수집...")
    fidelity_indices = run_error_fidelity(exp_circuits, exp_config, batch_manager)
    
    # 2. 표현력 태스크 수집 (모든 회로별로)
    print("📈 표현력 태스크 수집...")
    expr_indices = Divergence_Expressibility.batch_circuit(
            exp_circuits, exp_config, num_samples=exp_config.num_samples, batch_manager=batch_manager
        )
    
    # 3. 얽힘도 태스크 수집
    print("🔗 얽힘도 태스크 수집...")
    entangle_indices = meyer_wallace_entropy_swap_test(exp_circuits, exp_config, batch_manager)
    
    # 4. 통합 배치 실행
    print("🚀 통합 배치 실행 시작...")
    task_results = batch_manager.execute_unified_batch()
    
    if not task_results:
        raise Exception("배치 실행 실패")
    
    # 5. 결과 분배 및 조합
    print("📊 결과 분배 및 조합...")
    
    # 피델리티 결과
    fidelity_batch_results = batch_manager.get_task_results("fidelity", fidelity_indices)
    fidelity_results = ResultDistributor.distribute_fidelity_results(
        fidelity_batch_results, exp_circuits, exp_config
    )
    
    # 표현력 결과
    expr_results = []
    for circuit_idx, circuit in enumerate(exp_circuits):
        expr_circuit_results = batch_manager.get_task_results("expressibility", expr_indices[circuit_idx])
        expr_result = ResultDistributor.distribute_expressibility_results(
            expr_circuit_results, {"circuit_spec": circuit}
        )
        expr_results.append(expr_result)

    # 얽힘도 결과
    entangle_batch_results = batch_manager.get_task_results("entanglement", entangle_indices)
    circuit_qubit_mapping = []
    for circuit_idx, circuit in enumerate(exp_circuits):
        for target_qubit in range(circuit.num_qubits):
            circuit_qubit_mapping.append((circuit_idx, target_qubit, circuit.num_qubits))
    
    entangle_results = ResultDistributor.distribute_entanglement_results(
        entangle_batch_results, circuit_qubit_mapping
    )
    
    # 최종 결과 조합
    circuit_results = combine_all_results(exp_circuits, fidelity_results, expr_results, entangle_results)
    
    print(f"✅ IBM 배치 처리 완료: {len(circuit_results)}개 회로 결과")
    return circuit_results, errors
    

def run_simulator_unified_batch_processing(exp_circuits: List, exp_config) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    시뮬레이터에서 통합 배치 처리 실행
    
    Args:
        exp_circuits: 실험 회로 리스트
        exp_config: 실험 설정
        
    Returns:
        (circuit_results, errors): 회로별 결과와 오류 리스트
    """
    print("\n💻 시뮬레이터 통합 배치 처리 시작...")
    
    errors = []
    
    # 실행자 생성
    executor = QuantumExecutorFactory.create_executor("simulator")
    exp_config.executor = executor
    
    # 1. 피델리티 태스크 수집
    print("🎯 피델리티 태스크 수집...")
    fidelity_results, robust_fidelity_results = run_error_fidelity(exp_circuits, exp_config)
    
    # 피델리티 결과를 딕셔너리 형태로 변환
    combined_fidelity_results = []
    for i in range(len(fidelity_results)):
        combined_fidelity_results.append({
            'standard': fidelity_results[i],
            'robust': robust_fidelity_results[i]
        })
    
    # 2. 표현력 태스크 수집
    print("📈 표현력 태스크 수집...")
    from expressibility.fidelity_divergence import Divergence_Expressibility
    expr_results = Divergence_Expressibility.calculate_from_circuit_specs_divergence_list(
        exp_circuits, num_samples=getattr(exp_config, 'num_samples', 50)
    )
    
    # 3. 얽힘도 태스크 수집
    print("🔗 얽힘도 태스크 수집...")
    from core.entangle_simulator import meyer_wallace_entropy_list
    entangle_results = meyer_wallace_entropy_list(exp_circuits)
    
    # 4. 결과 분배 및 조합
    print("📊 결과 분배 및 조합...")

    # 최종 결과 조합
    circuit_results = combine_all_results(exp_circuits, combined_fidelity_results, expr_results, entangle_results)
    
    print(f"✅ 시뮬레이터 배치 처리 완료: {len(circuit_results)}개 회로 결과")
    return circuit_results, errors
    

def combine_all_results(exp_circuits: List, fidelity_results: List, expr_results: List, entangle_results: List) -> List[Dict[str, Any]]:
    """
    모든 측정 결과를 조합하여 최종 실험 결과 생성
    
    Args:
        exp_circuits: 실험 회로 리스트
        fidelity_results: 피델리티 결과 리스트
        expr_results: 표현력 결과 리스트 (단일 값)
        entangle_results: 얽힘도 결과 리스트
        
    Returns:
        회로별 종합 결과 리스트
    """
    combined_results = []
    
    print(f"🔍 결과 조합 디버깅:")
    print(f"   - 전체 회로 수: {len(exp_circuits)}")
    print(f"   - 피델리티 결과 수: {len(fidelity_results)}")
    print(f"   - 표현력 결과 수: {len(expr_results) if expr_results else 0}")
    print(f"   - 얽힘도 결과 수: {len(entangle_results)}")
    
    for i, circuit in enumerate(exp_circuits):
        result = {
            "circuit_id": getattr(circuit, 'circuit_id', f"circuit_{i}"),
            "num_qubits": circuit.num_qubits,
            "depth": len(circuit.gates) if hasattr(circuit, 'gates') else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        # 피델리티 결과 추가 (standard 및 robust 포함)
        if i < len(fidelity_results):
            fidelity_data = fidelity_results[i]
            if isinstance(fidelity_data, dict):
                # 새로운 형식: standard와 robust 피델리티 모두 포함
                result["fidelity"] = fidelity_data.get('standard', 0.0)
                result["robust_fidelity"] = fidelity_data.get('robust', 0.0)
            else:
                # 기존 형식: 단일 값
                result["fidelity"] = fidelity_data
                result["robust_fidelity"] = None
        else:
            result["fidelity"] = None  # 결과 없음 명시
            result["robust_fidelity"] = None
        
        # 표현력 결과 추가
        if expr_results and len(expr_results) > 0:
            if isinstance(expr_results[i], dict):
                result["expressibility"] = expr_results[i]
            else:
                result["expressibility"] = {"kl_divergence": expr_results[i]}
        else:
            result["expressibility"] = None
        
        # 얽힘도 결과 추가
        if i < len(entangle_results):
            result["entanglement"] = entangle_results[i]
        else:
            result["entanglement"] = None  # 결과 없음 명시
        
        combined_results.append(result)
    
    print(f"✅ 최종 조합 결과: {len(combined_results)}개")
    return combined_results


def main():
    """
    메인 실행 함수 - IBM Quantum 통합 배치 처리 시스템
    """
    print("="*60)
    print("🚀 IBM Quantum 통합 배치 처리 시스템")
    print("   기존 3회 연결 → 1회 연결로 최적화!")
    print("="*60)
    
    try:
        # 설정 로드
        exp_box = Exp_Box()
        exp_config = exp_box.get_setting("simulator_data_set")
        
        print(f"\n📋 실험 설정:")
        print(f"   - 큐빗 수: {exp_config.num_qubits}")
        print(f"   - 회로 깊이: {exp_config.depth}")
        print(f"   - 회로 개수: {exp_config.num_circuits}")
        print(f"   - 샷 수: {getattr(exp_config, 'shots', 1024)}")
        
        # 백엔드 선택
        backend_type = "simulator"  # 테스트용, 실제로는 "ibm" 사용
        
        print(f"\n🔌 백엔드: {backend_type.upper()}")
        
        # 통합 배치 실험 실행
        results = run_unified_batch_experiment(backend_type, exp_config)
        
        # 성능 요약 출력
        print_summary(results)
        
        # JSON 결과 저장 (ResultHandler 사용)
        from utils.result_handler import ResultHandler
        
        # 기존 결과만 저장
        output_file = ResultHandler.save_experiment_results(
            experiment_results=results.get('circuit_results', []),
            exp_config=exp_config,
            output_dir="results",
            filename="unified_batch_experiment_results.json"
        )
        
        # 회로 정보와 결과를 함께 저장
        output_file_with_circuits = ResultHandler.save_experiment_results_with_circuits(
            experiment_results=results.get('circuit_results', []),
            circuit_specs=results.get('circuit_specs', []),
            exp_config=exp_config,
            output_dir="results",
            filename="unified_batch_experiment_results_with_circuits.json"
        )
        
        print(f"\n🎉 실험 완료!")
        print(f"   - 총 회로 수: {results.get('total_circuits', 0)}개")
        print(f"   - 성공률: {results.get('success_rate', 0)*100:.1f}%")
        print(f"   - 결과 파일: {output_file}")
        
        # 오류 요약
        errors = results.get('errors', [])
        if errors:
            print(f"\n⚠️ 오류 {len(errors)}건 발생:")
            for error in errors[:3]:  # 최대 3개만 표시
                print(f"   - {error}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 메인 실행 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
