#!/usr/bin/env python3
"""
Quantum Circuit Generation Script

학습된 Decision Transformer 모델과 Property Prediction 모델을 사용하여
타겟 메트릭을 기반으로 양자 회로를 생성하는 메인 스크립트
"""

import argparse
import torch
import json
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Optional

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent / "src"))

from src.inference.quantum_circuit_generator import QuantumCircuitGenerator, GenerationConfig
from src.property_prediction_transformer import create_property_prediction_model, PropertyPredictionConfig
from src.training.property_prediction_trainer import PropertyPredictionTrainer


def load_property_prediction_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Load the property prediction model from checkpoint"""
    try:
        print(f"Loading property prediction model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get config if available in checkpoint or use default
        config = checkpoint.get('config', PropertyPredictionConfig())
        model = create_property_prediction_model(config)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        print("✅ Property prediction model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Failed to load property prediction model: {e}")
        return None


def predict_circuit_metrics(model, circuit_spec) -> Dict[str, float]:
    """Predict metrics for a given circuit using the property prediction model"""
    with torch.no_grad():
        predictions = model(circuit_spec)
        # Extract predicted values
        metrics = {
            'entanglement': float(predictions['entanglement'].item()),
            'fidelity': float(predictions['fidelity'].item()),
            'expressibility': float(predictions['expressibility'].item()),
            'robust_fidelity': float(predictions['robust_fidelity'].item())
        }
        # Convert from model output range (-1, 1) to actual metrics range (0, 1)
        for key in metrics:
            metrics[key] = (metrics[key] + 1) / 2.0
        return metrics


def main():
    parser = argparse.ArgumentParser(description='Generate quantum circuits using trained Decision Transformer with property prediction')
    
    # 모델 설정
    parser.add_argument('--model_path', type=str, default=r"C:\Users\jungh\Documents\GitHub\Kaist\OAT_Model\checkpoints\best_model.pt",
                       help='Path to the trained Decision Transformer model')
    parser.add_argument('--property_model_path', type=str, default=r"C:\Users\jungh\Documents\GitHub\Kaist\OAT_Model\property_prediction_checkpoints\best_model.pt",
                       help='Path to the trained Property Prediction model')
    
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
    
    # 목표 메트릭 설정
    target_metrics = {}
    if args.target_fidelity is not None:
        target_metrics['target_fidelity'] = args.target_fidelity
    if args.target_entanglement is not None:
        target_metrics['target_entanglement'] = args.target_entanglement
    if args.target_expressibility is not None:
        target_metrics['target_expressibility'] = args.target_expressibility
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 생성 설정
    config = GenerationConfig(
        max_circuit_length=args.max_length,
        target_num_qubits=args.num_qubits,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=not args.greedy,
        use_target_metrics=bool(target_metrics),  # 목표 메트릭이 있으면 활성화
        verbose=args.verbose  # 상세 출력 설정
    )
    
    # 프로퍼티 예측 모델 로드
    property_model = None
    if args.property_model_path and Path(args.property_model_path).exists():
        property_model = load_property_prediction_model(args.property_model_path, device)
    
    print("🚀 Quantum Circuit Generation with Decision Transformer & Property Prediction")
    print("=" * 60)
    print(f"Decision Transformer Model: {args.model_path}")
    print(f"Property Prediction Model: {args.property_model_path if property_model else 'Not loaded'}")
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
        
        # 생성된 회로 메트릭 평가 (프로퍼티 예측 모델이 있는 경우)
        if property_model:
            print("\n📊 Evaluating circuit properties with Property Prediction Model...")
            for circuit in circuits:
                circuit_spec = circuit.to_dict()
                predicted_metrics = predict_circuit_metrics(property_model, circuit_spec)
                
                # 메트릭 추가
                circuit.metrics = predicted_metrics
                
                if args.verbose:
                    print(f"\nCircuit {circuit.circuit_id} predicted metrics:")
                    print(f"  Entanglement: {predicted_metrics['entanglement']:.4f}")
                    print(f"  Fidelity: {predicted_metrics['fidelity']:.4f}")
                    print(f"  Expressibility: {predicted_metrics['expressibility']:.4f}")
                    print(f"  Robust Fidelity: {predicted_metrics['robust_fidelity']:.4f}")
                    
                    # 타겟 메트릭과 비교
                    if target_metrics:
                        print("  Comparison with targets:")
                        if 'target_entanglement' in target_metrics:
                            diff = abs(predicted_metrics['entanglement'] - target_metrics['target_entanglement'])
                            print(f"    Entanglement diff: {diff:.4f}")
                        if 'target_fidelity' in target_metrics:
                            diff = abs(predicted_metrics['fidelity'] - target_metrics['target_fidelity'])
                            print(f"    Fidelity diff: {diff:.4f}")
                        if 'target_expressibility' in target_metrics:
                            diff = abs(predicted_metrics['expressibility'] - target_metrics['target_expressibility'])
                            print(f"    Expressibility diff: {diff:.4f}")
        
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
        
        # 메트릭 평균 (프로퍼티 예측 모델이 있는 경우)
        if property_model and circuits:
            avg_metrics = {
                'entanglement': np.mean([circuit.metrics.get('entanglement', 0) for circuit in circuits]),
                'fidelity': np.mean([circuit.metrics.get('fidelity', 0) for circuit in circuits]),
                'expressibility': np.mean([circuit.metrics.get('expressibility', 0) for circuit in circuits]),
                'robust_fidelity': np.mean([circuit.metrics.get('robust_fidelity', 0) for circuit in circuits])
            }
            
            print("\nAverage predicted metrics:")
            print(f"  Entanglement: {avg_metrics['entanglement']:.4f}")
            print(f"  Fidelity: {avg_metrics['fidelity']:.4f}")
            print(f"  Expressibility: {avg_metrics['expressibility']:.4f}")
            print(f"  Robust Fidelity: {avg_metrics['robust_fidelity']:.4f}")
            
            # 타겟 메트릭과 비교
            if target_metrics:
                avg_diffs = {}
                if 'target_entanglement' in target_metrics:
                    avg_diffs['entanglement'] = abs(avg_metrics['entanglement'] - target_metrics['target_entanglement'])
                if 'target_fidelity' in target_metrics:
                    avg_diffs['fidelity'] = abs(avg_metrics['fidelity'] - target_metrics['target_fidelity'])
                if 'target_expressibility' in target_metrics:
                    avg_diffs['expressibility'] = abs(avg_metrics['expressibility'] - target_metrics['target_expressibility'])
                
                if avg_diffs:
                    print("\nAverage difference from targets:")
                    for metric, diff in avg_diffs.items():
                        print(f"  {metric.capitalize()}: {diff:.4f}")
        
        print(f"\nOutput saved to: {args.output}")
        
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
                
                # 예측된 메트릭 출력 (프로퍼티 예측 모델이 있는 경우)
                if property_model and hasattr(circuit, 'metrics'):
                    print("  Predicted metrics:")
                    metrics = circuit.metrics
                    print(f"    Entanglement: {metrics.get('entanglement', 0):.4f}")
                    print(f"    Fidelity: {metrics.get('fidelity', 0):.4f}")
                    print(f"    Expressibility: {metrics.get('expressibility', 0):.4f}")
                    print(f"    Robust Fidelity: {metrics.get('robust_fidelity', 0):.4f}")
        
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
