#!/usr/bin/env python3
"""
Quantum Circuit Backend - Main Entry Point

This is the main entry point for the quantum circuit backend system.
It demonstrates the clean separation between simulator and IBM backends
using a unified interface.
"""

import json
import os
from typing import List, Dict, Any
import numpy as np

from config import default_config, Config, Exp_Box
from core.circuit_interface import CircuitBuilder, CircuitSpec
from core.inverse import create_fidelity_circuit_spec
from core.error_fidelity import calculate_error_fidelity_from_result
from expressibility.fidelity_divergence import Divergence_Expressibility
from expressibility.classical_shadow import calculate_shadow_expressibility_all
from execution.executor import QuantumExecutorFactory
from execution.simulator_executor import QiskitQuantumCircuit
from core.random_circuit_generator import generate_random_circuit
import numpy as np
import json
import os

def run_fidelity_experiment(config: Config) -> Dict[str, Any]:
    """
    Run fidelity experiment using the specified backend.
    
    This function demonstrates the clean separation:
    - No backend-specific code here
    - Backend selection happens only in ExecutorFactory
    - All circuits use the same abstract interface
    
    Args:
        config: Application configuration
        
    Returns:
        Experiment results
    """
    print(f"Starting fidelity experiment with {config.backend_type} backend...")
    
    # Create executor (this is the ONLY place where backend type matters)
    executor = ExecutorFactory.create_executor(config.backend_type)
    
    results = {
        'config': config.to_dict(),
        'circuits': [],
        'fidelities': [],
        'expressibility': None,
        'backend_info': None
    }
    
    try:
        with executor:
            # Get backend info
            results['backend_info'] = executor.get_backend_info()
            print(f"Using backend: {results['backend_info']['backend_name']}")
            
            # Generate random circuits
            circuit_specs = []
            for i in range(config.num_circuits):
                spec = generate_random_circuit_spec(
                    config.num_qubits, 
                    config.circuit_depth, 
                    f"circuit_{i}"
                )
                circuit_specs.append(spec)
            
            print(f"Generated {len(circuit_specs)} random circuits")
            
            # Create fidelity measurement circuits
            fidelity_circuits = []
            for spec in circuit_specs:
                # Create fidelity circuit (original + inverse)
                fidelity_spec = create_fidelity_circuit_spec(spec)
                fidelity_circuit = QiskitQuantumCircuit(fidelity_spec)
                fidelity_circuits.append(fidelity_circuit)
            
            print(f"Created {len(fidelity_circuits)} fidelity measurement circuits")
            
            # Execute circuits
            print("Executing circuits...")
            execution_results = executor.execute_circuits(fidelity_circuits)
            
            # Calculate fidelities
            fidelities = []
            for i, (spec, exec_result) in enumerate(zip(circuit_specs, execution_results)):
                if exec_result.success:
                    fidelity = calculate_fidelity_from_result(exec_result, config.num_qubits)
                    fidelities.append(fidelity)
                    
                    # Save circuit info
                    circuit_info = {
                        'name': spec.name,
                        'num_qubits': spec.num_qubits,
                        'num_gates': len(spec.gates),
                        'fidelity': fidelity,
                        'execution_time': exec_result.execution_time,
                        'shots': exec_result.shots
                    }
                    results['circuits'].append(circuit_info)
                    
                    if i % 10 == 0:
                        print(f"Processed {i+1}/{len(circuit_specs)} circuits, fidelity: {fidelity:.4f}")
                else:
                    print(f"Circuit {i} failed: {exec_result.error_message}")
            
            results['fidelities'] = fidelities
            print(f"Calculated {len(fidelities)} fidelities")
            
            # Calculate expressibility
            if len(fidelities) >= config.min_fidelity_samples:
                print("Calculating expressibility...")
                expressibility_result = calculate_expressibility_from_results(
                    execution_results, config.num_qubits
                )
                results['expressibility'] = expressibility_result
                print(f"Expressibility: {expressibility_result.get('expressibility', 'N/A')}")
            else:
                print(f"Insufficient samples for expressibility: {len(fidelities)} < {config.min_fidelity_samples}")
                results['expressibility'] = {
                    'error': f'Insufficient samples: {len(fidelities)} < {config.min_fidelity_samples}'
                }
    
    except Exception as e:
        print(f"Experiment failed: {e}")
        results['error'] = str(e)
    
    return results


def save_results(results, config: Config):
    """
    Save experiment results to files.
    
    Args:
        results: Experiment results
        config: Application configuration
    """
    if not config.save_results:
        return
    
    # Save main results
    results_file = os.path.join(config.output_dir, 'experiment_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {results_file}")
    
    # Save fidelities separately for analysis
    if results.get('fidelities'):
        fidelities_file = os.path.join(config.output_dir, 'fidelities.json')
        with open(fidelities_file, 'w') as f:
            json.dump(results['fidelities'], f, indent=2)
        
        print(f"Fidelities saved to {fidelities_file}")


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
    
    # Fidelity statistics
    fidelities = results.get('fidelities', [])
    if fidelities:
        print(f"\nFidelity Statistics:")
        print(f"  Count: {len(fidelities)}")
        print(f"  Mean: {np.mean(fidelities):.4f}")
        print(f"  Std: {np.std(fidelities):.4f}")
        print(f"  Min: {np.min(fidelities):.4f}")
        print(f"  Max: {np.max(fidelities):.4f}")
    
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
    """
    Main entry point.
    
    이 함수는 새로운 API를 활용한 간단하고 직관적인 실행 방식을 보여줍니다:
    1. 실행자 직접 생성: ExecutorFactory("simulator") 또는 ExecutorFactory("ibm")
    2. 실험 설정 직접 접근: config.exp1, config.exp2 등
    3. 간단한 실행: executor.run(config.exp1)
    4. 명확한 코드 분리와 가독성
    """
    print("🚀 Quantum Circuit Backend - 간소화된 API 데모")
    print("")
    
    # 구성 로드
    config = default_config
    exp_box = exp_box = Exp_Box()
    
    # 사용 가능한 백엔드 표시
    print("사용 가능한 백엔드:")
    available_backends = QuantumExecutorFactory.list_available_backends()
    for i, backend in enumerate(available_backends):
        print(f"  {i+1}. {backend}")
    
    try:
        # 백엔드 선택
        choice = input(f"\n백엔드 선택 (1-{len(available_backends)}) [기본값: 1]: ").strip()
        backend_type = available_backends[0]  # 기본값: 첫 번째 백엔드
        if choice:
            backend_idx = int(choice) - 1
            if 0 <= backend_idx < len(available_backends):
                backend_type = available_backends[backend_idx]
        
        print(f"선택된 백엔드: {backend_type}")
        
        # 실행자 직접 생성 - 새로운 API 사용
        hardware_executor = QuantumExecutorFactory.create_executor(backend_type)
        
        # 실행 컨텍스트
        print(f"\n실험 1 실행 중: {exp_box.exp1.num_qubits} 큐빗, {exp_box.exp1.depth} 깊이...")
        
        # 첫 번째 실험 실행 - 회로 생성
        exp1_circuits = generate_random_circuit(exp_box.exp1)
        print(f"생성된 회로 수: {len(exp1_circuits)}개 ({[q for q in exp_box.exp1.num_qubits]} 큐빗 각각 {exp_box.exp1.num_circuits}개)")
        


        # 회로 실행
        with hardware_executor:
            results1 = hardware_executor.run(exp1_circuits, exp_box.exp1)
        print(f"실험 1 완료: {len(results1)} 회로 실행됨")
        
        # 실험 결과 분석 및 저장
        experiment_results = []
        
        # 회로별 분석 (각 회로마다 별도로 피델리티/표현력 계산), 여기서 서킷은 스펙객체임임
        for i, circuit in enumerate(exp1_circuits):
            circuit_results = [result for result in results1 if result.circuit_id == circuit.circuit_id]
            if not circuit_results:
                print(f"회로 {i+1}의 결과가 없습니다.")
                continue
                
            # 기본 정보 수집
            circuit_info = {
                "circuit_id": circuit.circuit_id,
                "num_qubits": circuit.num_qubits,
                "gate_count": len(circuit.gates),
                "two_qubit_ratio": sum(1 for g in circuit.gates if len(g.qubits) > 1) / len(circuit.gates) if circuit.gates else 0
            }
            
            # 피델리티 계산
            try:
                # 각 결과의 피델리티 계산
                fidelities = []
                for result in circuit_results:
                    if result.success and result.counts:
                        fidelity = calculate_error_fidelity_from_result(result, circuit.num_qubits, exp_box.exp1)
                        fidelities.append(fidelity)
                
                # 통계 계산
                if fidelities:
                    circuit_info["fidelity"] = {
                        "mean": float(np.mean(fidelities)),
                        "std": float(np.std(fidelities)),
                        "min": float(np.min(fidelities)),
                        "max": float(np.max(fidelities)),
                        "values": [float(f) for f in fidelities],
                        "valid_samples": len(fidelities)
                    }
                else:
                    circuit_info["fidelity"] = {"error": "No valid fidelity samples"}

                 # 표현력 계산 (시뮬레이터 - 피델리티 다이버전스)
                expr_result = None
                try:
                    expr_result = Divergence_Expressibility.calculate_from_circuit_specs_divergence(circuit)
                    print("표현력" + "="*50)
                    print(expr_result)
                    circuit_info["expressibility_divergence"] = expr_result
                except Exception as e:
                    circuit_info["expressibility_divergence"] = {"error": str(e)}
                    
            except Exception as e:
                circuit_info["fidelity"] = {"error": str(e)}
            
           
            # 클래식 쉐도우 표현력 계산 (IBM)
            # shadow_result = None
            # try:
            #     shadow_result = calculate_shadow_expressibility_all(circuit_results, circuit.num_qubits)
            #     circuit_info["expressibility_shadow"] = shadow_result
            # except Exception as e:
            #     circuit_info["expressibility_shadow"] = {"error": str(e)}
            
            # 결과 저장
            experiment_results.append(circuit_info)
            
            # 진행 상황 출력
            print(f"회로 {i+1}/{len(exp1_circuits)} 분석 완료")
            print(f"  - 피델리티: {circuit_info['fidelity'].get('mean', 'N/A')}")
            #print(f"  - 표현력(다이버전스): {circuit_info['expressibility_divergence'].get('expressibility', 'N/A')}")
            # 클래식 쉐도우 기능이 비활성화되어 있으므로 이 부분 주석 처리
            # print(f"  - 표현력(쉐도우): {circuit_info.get('expressibility_shadow', {}).get('summary', {}).get('local2_expressibility', 'N/A')}")
        
        # 결과 저장
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "experiment_results.json")
        with open(output_path, 'w') as f:
            json.dump({
                "experiment_name": exp_box.exp1.exp_name,
                "experiment_config": {
                    "num_qubits": [int(q) for q in exp_box.exp1.num_qubits],
                    "depth": exp_box.exp1.depth if isinstance(exp_box.exp1.depth, int) else [int(d) for d in exp_box.exp1.depth],
                    "shots": exp_box.exp1.shots,
                    "num_circuits": exp_box.exp1.num_circuits,
                    "optimization_level": exp_box.exp1.optimization_level,
                    "two_qubit_ratio": [float(r) for r in exp_box.exp1.two_qubit_ratio]
                },
                "results": experiment_results,
                "summary": {
                    "total_circuits": len(exp1_circuits),
                    "successful_circuits": len([r for r in experiment_results if "fidelity" in r and "error" not in r["fidelity"]]),
                    "average_fidelity": float(np.mean([r["fidelity"]["mean"] for r in experiment_results 
                                                if "fidelity" in r and "mean" in r["fidelity"]])) 
                                                if any("fidelity" in r and "mean" in r["fidelity"] for r in experiment_results) else None,
                    "average_expressibility_div": float(np.mean([r["expressibility_divergence"]["expressibility"] for r in experiment_results 
                                                      if "expressibility_divergence" in r and "expressibility" in r["expressibility_divergence"]])) 
                                                      if any("expressibility_divergence" in r and "expressibility" in r["expressibility_divergence"] for r in experiment_results) else None,
                    "average_expressibility_shadow": float(np.mean([r["expressibility_shadow"]["summary"]["local2_expressibility"] for r in experiment_results 
                                                         if "expressibility_shadow" in r and "summary" in r["expressibility_shadow"] and "local2_expressibility" in r["expressibility_shadow"]["summary"]])) 
                                                         if any("expressibility_shadow" in r and "summary" in r["expressibility_shadow"] and "local2_expressibility" in r["expressibility_shadow"]["summary"] for r in experiment_results) else None
                }
            }, f, indent=2)
        
        print(f"\n결과 저장 완료: {output_path}")
        print("=== 실험 요약 ===")
        print(f"총 회로 수: {len(exp1_circuits)}")
        print(f"성공한 회로 수: {len([r for r in experiment_results if 'fidelity' in r and 'error' not in r['fidelity']])}")

        print(results1)


        for result in results1:
            # 결과 저장 및 표시
            save_results(result, config)
            
            print("\n=== 실험 1 요약 ===")
        #print_summary(result)

    except KeyboardInterrupt:
        print("\n❌ 사용자에 의해 실험이 취소되었습니다")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
