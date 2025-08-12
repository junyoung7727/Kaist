#!/usr/bin/env python3
"""
Quantum Circuit Generation Script

학습된 Decision Transformer 모델을 사용하여 양자 회로를 생성하는 메인 스크립트
"""

import argparse
import torch
from pathlib import Path
import sys

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent / "src"))

from src.inference.quantum_circuit_generator import QuantumCircuitGenerator, GenerationConfig


def main():
    parser = argparse.ArgumentParser(description='Generate quantum circuits using trained Decision Transformer')
    
    # 모델 설정
    parser.add_argument('--model_path', type=str,default=r"C:\Users\jungh\Documents\GitHub\Kaist\checkpoints\best_model.pt",
                       help='')
    
    # 생성 설정
    parser.add_argument('--num_circuits', type=int, default=5,
                       help='Number of circuits to generate')
    parser.add_argument('--max_length', type=int, default=20,
                       help='Maximum circuit length (number of gates)')
    parser.add_argument('--num_qubits', type=int, default=4,
                       help='Number of qubits in generated circuits')
    
    # 샘플링 설정
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Top-p (nucleus) sampling parameter')
    parser.add_argument('--greedy', action='store_true',
                       help='Use greedy decoding instead of sampling')
    
    # 목표 메트릭 (선택적)
    parser.add_argument('--target_fidelity', type=float, default=None,
                       help='Target fidelity (0.0-1.0)')
    parser.add_argument('--target_entanglement', type=float, default=None,
                       help='Target entanglement (0.0-1.0)')
    parser.add_argument('--target_expressibility', type=float, default=None,
                       help='Target expressibility (0.0-1.0)')
    
    # 출력 설정
    parser.add_argument('--output', type=str, default='generated_circuits.json',
                       help='Output file path for generated circuits')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # 모델 파일 존재 확인
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        print("Please train a model first using train_decision_transformer.py")
        return
    
    # 생성 설정
    config = GenerationConfig(
        max_circuit_length=args.max_length,
        target_num_qubits=args.num_qubits,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=not args.greedy
    )
    
    # 목표 메트릭 설정
    target_metrics = {}
    if args.target_fidelity is not None:
        target_metrics['target_fidelity'] = args.target_fidelity
    if args.target_entanglement is not None:
        target_metrics['target_entanglement'] = args.target_entanglement
    if args.target_expressibility is not None:
        target_metrics['target_expressibility'] = args.target_expressibility
    
    print("🚀 Quantum Circuit Generation with Decision Transformer")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Number of circuits: {args.num_circuits}")
    print(f"Max circuit length: {args.max_length}")
    print(f"Number of qubits: {args.num_qubits}")
    print(f"Temperature: {args.temperature}")
    print(f"Sampling: {'Greedy' if args.greedy else f'Top-k={args.top_k}, Top-p={args.top_p}'}")
    if target_metrics:
        print(f"Target metrics: {target_metrics}")
    print("=" * 60)
    
    try:
        # 생성기 초기화
        generator = QuantumCircuitGenerator(args.model_path, config)
        
        # 회로 생성
        print(f"\nGenerating {args.num_circuits} quantum circuits...")
        circuits = generator.generate_multiple_circuits(
            num_circuits=args.num_circuits,
            target_metrics=target_metrics if target_metrics else None
        )
        
        # 결과 저장
        generator.save_circuits(circuits, args.output)
        
        # 결과 요약
        print("\n" + "=" * 60)
        print("📊 Generation Summary")
        print("=" * 60)
        
        total_gates = sum(len(circuit.gates) for circuit in circuits)
        avg_gates = total_gates / len(circuits) if circuits else 0
        
        print(f"Generated circuits: {len(circuits)}")
        print(f"Total gates: {total_gates}")
        print(f"Average gates per circuit: {avg_gates:.1f}")
        print(f"Output saved to: {args.output}")
        
        # 각 회로 상세 정보
        if args.verbose:
            print("\n📋 Circuit Details:")
            for i, circuit in enumerate(circuits):
                print(f"\nCircuit {i+1} ({circuit.circuit_id}):")
                print(f"  Qubits: {circuit.num_qubits}")
                print(f"  Gates: {len(circuit.gates)}")
                
                # 게이트 타입 분포
                gate_counts = {}
                for gate in circuit.gates:
                    gate_counts[gate.name] = gate_counts.get(gate.name, 0) + 1
                
                print("  Gate distribution:")
                for gate_name, count in sorted(gate_counts.items()):
                    print(f"    {gate_name}: {count}")
                
                # 처음 몇 개 게이트 출력
                print("  First few gates:")
                for j, gate in enumerate(circuit.gates[:5]):
                    params_str = f", params={gate.parameters}" if gate.parameters else ""
                    print(f"    {j+1}. {gate.name}(qubits={gate.qubits}{params_str})")
                if len(circuit.gates) > 5:
                    print(f"    ... and {len(circuit.gates) - 5} more gates")
        
        print("\n✅ Circuit generation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during circuit generation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
