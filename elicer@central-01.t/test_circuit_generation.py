#!/usr/bin/env python3
"""
Test script for quantum circuit generation with target metrics conditioning
"""

import sys
import torch
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.inference.quantum_circuit_generator import QuantumCircuitGenerator, GenerationConfig
from src.models.property_prediction_transformer import PropertyPredictionTransformer, load_property_prediction_model
from src.evaluation.circuit_evaluator import evaluate_circuit_properties


def evaluate_circuit_with_property_model(circuit, property_model):
    """Evaluate a circuit using the property prediction model"""
    if property_model is None:
        print("⚠️ Property prediction model not available. Skipping evaluation.")
        return None
    
    try:
        # Get property predictions from the model
        with torch.no_grad():
            predictions = property_model.predict_properties(circuit)
            
            # Convert from model output range (-1, 1) to actual metrics range (0, 1)
            # This assumes the model outputs are in (-1, 1) range and need to be rescaled
            for key in predictions:
                predictions[key] = (predictions[key] + 1) / 2.0
        
        return predictions
    except Exception as e:
        print(f"Error evaluating circuit with property model: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_metrics_comparison(test_cases, results, output_dir):
    """Visualize the comparison between target and predicted metrics"""
    try:
        # Create figure for all test cases
        fig, axes = plt.subplots(len(test_cases), 1, figsize=(12, 5 * len(test_cases)))
        if len(test_cases) == 1:
            axes = [axes]  # Make it iterable if only one test case
        
        # Map of metric names to friendly display names
        metric_display_names = {
            'fidelity': 'Fidelity',
            'target_fidelity': 'Fidelity',
            'entanglement': 'Entanglement',
            'target_entanglement': 'Entanglement',
            'expressibility': 'Expressibility',
            'target_expressibility': 'Expressibility',
            'robust_fidelity': 'Robust Fidelity'
        }
        
        # Metric colors
        metric_colors = {
            'fidelity': 'blue',
            'entanglement': 'green',
            'expressibility': 'red',
            'robust_fidelity': 'purple'
        }
        
        for i, (test_case, result) in enumerate(zip(test_cases, results)):
            ax = axes[i]
            
            # Get target and predicted metrics
            target_metrics = {}
            for key, value in test_case['metrics'].items():
                # Extract the metric name without 'target_' prefix
                metric_name = key[7:] if key.startswith('target_') else key
                target_metrics[metric_name] = value
            
            predicted_metrics = result.get('predicted_metrics', {})
            if not predicted_metrics:
                ax.set_title(f"Test Case: {test_case['name']} - No predicted metrics available")
                continue
            
            # Metrics to display
            all_metrics = list(set(list(target_metrics.keys()) + list(predicted_metrics.keys())))
            
            # Bar positions
            x = np.arange(len(all_metrics))  
            width = 0.35
            
            # Plot bars
            target_values = [target_metrics.get(m, 0) for m in all_metrics]
            predicted_values = [predicted_metrics.get(m, 0) for m in all_metrics]
            
            ax.bar(x - width/2, target_values, width, label='Target', color='skyblue', alpha=0.7)
            ax.bar(x + width/2, predicted_values, width, label='Predicted', color='orange', alpha=0.7)
            
            # Add labels and legend
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Value')
            ax.set_title(f"Test Case: {test_case['name']} - Target vs Predicted Metrics")
            ax.set_xticks(x)
            ax.set_xticklabels([metric_display_names.get(m, m) for m in all_metrics])
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add values above bars
            for j, v in enumerate(target_values):
                ax.text(j - width/2, v + 0.02, f"{v:.2f}", ha='center')
                
            for j, v in enumerate(predicted_values):
                ax.text(j + width/2, v + 0.02, f"{v:.2f}", ha='center')
            
            # Set y-axis limits
            ax.set_ylim(0, 1.1)
            
        plt.tight_layout()
        plt.savefig(output_dir / "metrics_comparison.png", dpi=300)
        plt.close()
        
        print(f"📊 Metrics comparison visualization saved to {output_dir / 'metrics_comparison.png'}")
        
    except Exception as e:
        print(f"Error generating metrics visualization: {e}")
        import traceback
        traceback.print_exc()


def test_metrics_conditioning(path: str = None):
    """Test circuit generation with different target metrics"""
    
    # Model path - update if needed
    if path is None:
        model_path = r"C:\Users\jungh\Documents\GitHub\Kaist\OAT_Model\checkpoints\best_model.pt"
        property_model_path = r"C:\Users\jungh\Documents\GitHub\Kaist\OAT_Model\property_prediction_checkpoints\best_model.pt"
    else:
        model_path = path
        # Assume property model is in the same directory structure
        property_model_path = str(Path(path).parent.parent / "property_prediction_checkpoints" / "best_model.pt")
        
    # Load property prediction model
    print(f"\nLoading property prediction model from {property_model_path}...")
    try:
        property_model = load_property_prediction_model(property_model_path)
        print("✅ Property prediction model loaded successfully")
    except Exception as e:
        print(f"⚠️ Failed to load property prediction model: {e}")
        property_model = None
    
    # Test configurations
    test_cases = [
        {
            "name": "high_fidelity",
            "metrics": {"target_fidelity": 0.9, "target_entanglement": 0.3},
            "config": {
                "max_circuit_length": 15,
                "target_num_qubits": 4,
                "temperature": 0.8,
                "top_k": 5,
                "top_p": 0.9
            }
        },
        {
            "name": "high_entanglement",
            "metrics": {"target_fidelity": 0.6, "target_entanglement": 0.8},
            "config": {
                "max_circuit_length": 15,
                "target_num_qubits": 4, 
                "temperature": 0.8,
                "top_k": 5,
                "top_p": 0.9
            }
        },
        {
            "name": "high_expressibility",
            "metrics": {"target_expressibility": 0.85},
            "config": {
                "max_circuit_length": 15,
                "target_num_qubits": 4,
                "temperature": 0.8,
                "top_k": 5,
                "top_p": 0.9
            }
        },
        {
            "name": "balanced_metrics",
            "metrics": {
                "target_fidelity": 0.7,
                "target_entanglement": 0.7,
                "target_expressibility": 0.7
            },
            "config": {
                "max_circuit_length": 15,
                "target_num_qubits": 4,
                "temperature": 0.8,
                "top_k": 5,
                "top_p": 0.9
            }
        }
    ]
    
    results = []
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent / "test_results"
    output_dir.mkdir(exist_ok=True)
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test Case {i+1}: {test_case['name']}")
        print(f"Target metrics: {test_case['metrics']}")
        print(f"{'='*60}")
        
        # Configure generator
        config = GenerationConfig(
            max_circuit_length=test_case["config"]["max_circuit_length"],
            target_num_qubits=test_case["config"]["target_num_qubits"],
            temperature=test_case["config"]["temperature"],
            top_k=test_case["config"]["top_k"],
            top_p=test_case["config"]["top_p"],
            use_target_metrics=True,
            verbose=True
        )
        
        try:
            # Initialize generator
            generator = QuantumCircuitGenerator(model_path, config)
            
            # Generate circuit
            start_time = time.time()
            circuit = generator.generate_circuit(target_metrics=test_case["metrics"])
            generation_time = time.time() - start_time
            
            # Save circuit
            output_file = output_dir / f"circuit_{test_case['name']}.json"
            generator.save_circuits([circuit], str(output_file))
            
            # Evaluate circuit with property prediction model
            print("\nEvaluating circuit with property prediction model...")
            predicted_metrics = evaluate_circuit_with_property_model(circuit, property_model)
            
            # Record results
            test_result = {
                "test_name": test_case["name"],
                "target_metrics": test_case["metrics"],
                "circuit_id": circuit.circuit_id,
                "num_qubits": circuit.num_qubits,
                "num_gates": len(circuit.gates),
                "gate_types": {},
                "predicted_metrics": predicted_metrics if predicted_metrics else {},
                "generation_time_seconds": round(generation_time, 2)
            }
            
            # Count gate types
            for gate in circuit.gates:
                test_result["gate_types"][gate.name] = test_result["gate_types"].get(gate.name, 0) + 1
            
            # Add metrics comparison
            if predicted_metrics:
                metric_diff = {}
                for metric, value in predicted_metrics.items():
                    target_key = f"target_{metric}"
                    if target_key in test_case["metrics"]:
                        target_value = test_case["metrics"][target_key]
                        diff = abs(value - target_value)
                        metric_diff[metric] = {
                            "target": target_value,
                            "predicted": value,
                            "absolute_diff": diff,
                            "relative_diff": diff / target_value if target_value > 0 else 0
                        }
                test_result["metric_comparisons"] = metric_diff
            
            results.append(test_result)
            
            # Display results
            print(f"Circuit generated: {circuit.circuit_id}")
            print(f"Number of gates: {len(circuit.gates)}")
            print(f"Generation time: {generation_time:.2f} seconds")
            print(f"Output saved to: {output_file}")
            
            # Print gate distribution
            print("\nGate distribution:")
            gate_counts = {}
            for gate in circuit.gates:
                gate_counts[gate.name] = gate_counts.get(gate.name, 0) + 1
            
            for gate_name, count in sorted(gate_counts.items()):
                print(f"  {gate_name}: {count}")
                
            # Display first few gates
            print("\nFirst few gates:")
            for j, gate in enumerate(circuit.gates[:5]):
                params_str = f", params={gate.parameters}" if gate.parameters else ""
                print(f"  {j+1}. {gate.name}(qubits={gate.qubits}{params_str})")
            
        except Exception as e:
            print(f"Error in test case {test_case['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Display metrics comparison for each test case
    print("\n" + "="*80)
    print("METRICS COMPARISON SUMMARY")
    print("="*80)
    
    for result in results:
        if "metric_comparisons" in result and result["metric_comparisons"]:
            print(f"\nTest Case: {result['test_name']}")
            print("-"*60)
            
            for metric, data in result["metric_comparisons"].items():
                target = data["target"]
                predicted = data["predicted"]
                abs_diff = data["absolute_diff"]
                rel_diff = data["relative_diff"] * 100  # Convert to percentage
                
                # Use color coding for difference (green for small diff, yellow for medium, red for large)
                if abs_diff < 0.1:
                    diff_indicator = "✓"
                elif abs_diff < 0.2:
                    diff_indicator = "△"
                else:
                    diff_indicator = "✗"
                    
                print(f"{metric:15s} | Target: {target:.4f} | Predicted: {predicted:.4f} | Diff: {abs_diff:.4f} ({rel_diff:.1f}%) {diff_indicator}")
            print()
    
    # Generate visualization if we have results with predictions
    if any("predicted_metrics" in result and result["predicted_metrics"] for result in results):
        print("Generating metrics visualization...")
        visualize_metrics_comparison(test_cases, results, output_dir)
    
    # Save all test results
    with open(output_dir / "test_results_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nAll tests completed. Summary saved to test_results/test_results_summary.json")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test circuit generation with target metrics conditioning")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=r"C:\Users\jungh\Documents\GitHub\Kaist\OAT_Model\checkpoints\best_model.pt", 
        help="Path to the trained model checkpoint"
    )
    
    args = parser.parse_args()
    test_metrics_conditioning(args.model_path)
