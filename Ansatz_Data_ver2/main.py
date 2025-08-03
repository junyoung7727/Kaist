#!/usr/bin/env python3
"""
Quantum Circuit Backend - Main Entry Point

This is the main entry point for the quantum circuit backend system.
It demonstrates the use of the abstract interfaces and implementations.
"""

# quantum_common 패키지를 찾기 위한 간단한 경로 설정
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import time
import json
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
from utils.result_handler import ResultHandler
from core.entangle_simulator import meyer_wallace_entropy
from core.entangle_hardware import meyer_wallace_entropy_swap_test

from config import default_config, Exp_Box
from expressibility.fidelity_divergence import Divergence_Expressibility
from execution.executor import QuantumExecutorFactory
from core.error_fidelity import run_error_fidelity
from core.random_circuit_generator import generate_random_circuit
import numpy as np
import json


def print_summary(results: List[Dict[str, Any]]):
    """
    Print experiment summary.
    
    Args:
        results: Experiment results
    """
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    
    if 'error' in results:
        print(f"❌ Experiment failed: {results['error']}")
        return
    
    # Backend info
    backend_info = results.get('backend_info', {})
    print(f"Backend: {backend_info.get('backend_name', 'Unknown')}")
    print(f"Backend Type: {backend_info.get('backend_type', 'Unknown')}")
    
    # Circuit info
    circuits = results.get('circuits', [])
    print(f"\nCircuits: {len(circuits)}")
    
    # Expressibility
    expressibility = results.get('expressibility')
    if expressibility and not expressibility.get('error'):
        print(f"\nExpressibility: {expressibility.get('expressibility', 'N/A'):.4f}")
        print(f"KS Statistic: {expressibility.get('ks_statistic', 'N/A'):.4f}")
        print(f"Valid Samples: {expressibility.get('valid_samples', 'N/A')}")
    elif expressibility:
        print(f"\nExpressibility: ❌ {expressibility.get('error', 'Unknown error')}")
    
    print("="*50)


def main():
    
    # 구성 로드
    config = default_config
    exp_box = Exp_Box()
    exp_setting = "exp1"
    exp_config = exp_box.get_setting(exp_setting)
    fidelity_shots = exp_config.fidelity_shots
    shots = exp_config.shots
    
    
    # 사용 가능한 백엔드 표시
    print("사용 가능한 백엔드:")
    available_backends = QuantumExecutorFactory.list_available_backends()
    for i, backend in enumerate(available_backends):
        print(f"  {i+1}. {backend}")
    

    # 백엔드 선택
    choice = input(f"\n백엔드 선택 (1-{len(available_backends)}) [기본값: 1]: ").strip()
    backend_type = available_backends[0]  # 기본값: 첫 번째 백엔드
    if choice:
        backend_idx = int(choice) - 1
        if 0 <= backend_idx < len(available_backends):
            backend_type = available_backends[backend_idx]
    
    print(f"선택된 백엔드: {backend_type}")
    
    # 실행 컨텍스트
    print(f"\n실험 1 실행 중: {exp_config.num_qubits} 큐빗, {exp_config.depth} 깊이...")
    
    # 첫 번째 실험 실행 - 회로 생성
    exp_circuits = generate_random_circuit(exp_config)
    print(f"생성된 회로 수: {len(exp_circuits)}개 ({[q for q in exp_config.num_qubits]} 큐빗 각각 {exp_config.num_circuits}개)")
    
    # 실험 결과 분석 및 저장
    experiment_results = []

    executor = QuantumExecutorFactory.create_executor(backend_type)
    exp_config.executor = executor
    
    print(f"\n🚀 {backend_type} 백엔드 - 배치 모드 (연결 3번만!)")
    print(f"생성된 회로 수: {len(exp_circuits)}개")
    
    # 배치 처리로 연결 최소화
    if backend_type == "ibm":
        print("📊 1/3: 피델리티 배치 측정...")
        fidelity_result = run_error_fidelity(exp_circuits, exp_config)
        
        print("📊 2/3: 표현력 배치 측정...")
        expr_result = Divergence_Expressibility.calculate_from_circuit_specs_divergence_hardware(
            exp_circuits, exp_config, num_samples=10
        )
        
        print("📊 3/3: 얽힘도 배치 측정...")
        from core.entangle_hardware import meyer_wallace_entropy_swap_test
        entangle_results = meyer_wallace_entropy_swap_test(exp_circuits, exp_config)
        
    else:  # simulator
        print("📊 1/3: 피델리티 배치 측정...")
        fidelity_results = [run_error_fidelity(circuit, exp_config) for circuit in exp_circuits]
        
        print("📊 2/3: 표현력 배치 측정...")
        expr_results = []
        for circuit in exp_circuits:
            result = Divergence_Expressibility.calculate_from_circuit_specs_divergence_simulator(
                circuit, num_samples=50
            )
            expr_results.append(result)
        
        print("📊 3/3: 얽힘도 배치 측정...")
        from core.entangle_simulator import meyer_wallace_entropy
        entangle_results = [meyer_wallace_entropy(circuit) for circuit in exp_circuits]
    
    # 결과 조합
    for i, circuit in enumerate(exp_circuits):
        circuit_info = {
            "circuit_id": circuit.circuit_id,
            "num_qubits": circuit.num_qubits,
            "gate_count": len(circuit.gates),
            "two_qubit_ratio": sum(1 for g in circuit.gates if len(g.qubits) > 1) / len(circuit.gates) if circuit.gates else 0
        }
        
        if backend_type == "ibm":
            circuit_info["error_fidelity"] = fidelity_result if isinstance(fidelity_result, (int, float)) else 0.0
            circuit_info["expressibility_divergence"] = expr_result if isinstance(expr_result, (int, float)) else 0.0
            circuit_info["entanglement_ability"] = entangle_results[i] if i < len(entangle_results) else 0.0
        else:
            circuit_info["error_fidelity"] = fidelity_results[i] if i < len(fidelity_results) else 0.0
            circuit_info["expressibility_divergence"] = expr_results[i] if i < len(expr_results) else 0.0
            circuit_info["entanglement_ability"] = entangle_results[i] if i < len(entangle_results) else 0.0
        
        experiment_results.append(circuit_info)
        print(f"회로 {i+1}/{len(exp_circuits)} 분석 완료")
    
    print(f"\n✅ 배치 처리 완료: 연결 3번으로 {len(exp_circuits)}개 회로 분석!")
    
    # 결과 저장 - 새 ResultHandler 사용
    output_path = ResultHandler.save_experiment_results(
        experiment_results=experiment_results,
        exp_config=exp_config,
        output_dir="output",
        filename="experiment_results.json"
    )
    
    # CircuitSpec 객체 저장 - 써킷 스펙 리스트 저장
    circuit_specs_path = ResultHandler.save_circuit_specs(
        circuit_specs=exp_circuits,  # CircuitSpec 객체 리스트
        exp_config=exp_config,
        output_dir="output",
        filename="circuit_specs.json"
    )
    
    # 파일 생성 확인 및 경로 검증
    print(f"\n실험 결과 파일 경로: {os.path.abspath(output_path)}")  
    print(f"써킷 스펙 파일 경로: {os.path.abspath(circuit_specs_path)}")
    
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"파일 생성 성공: {output_path} (크기: {file_size:,} 바이트)")
        
        # JSON 파일 유효성 검증
        try:
            with open(output_path, 'r') as f:
                json_data = json.load(f)
            print(f"JSON 파일 유효성 검증 성공: {len(json_data.get('results', [])):,}개 결과 포함")
            
            # 결과 요약 정보 표시
            if 'summary' in json_data and json_data['summary']:
                print("\n요약 정보:")
                for key, value in json_data['summary'].items():
                    print(f"  - {key}: {value}")
        except json.JSONDecodeError as e:
            print(f"경고: JSON 파일 형식 오류: {e}")
        except Exception as e:
            print(f"경고: 파일 내용 검증 중 오류 발생: {e}")
    else:
        print(f"경고: 파일이 생성되지 않았습니다: {output_path}")
    
    # 결과 요약 출력
    print(f"\n결과 요약:")
    ResultHandler.print_result_summary(experiment_results)

    # 결과 처리가 이미 위에서 완료되었으므로 여기서는 중복 처리 없이 종료

if __name__ == "__main__":
    main()
