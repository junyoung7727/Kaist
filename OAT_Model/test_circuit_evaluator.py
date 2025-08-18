#!/usr/bin/env python3
"""
Test script for the QuantumCircuitEvaluator integration
"""

import sys
import torch
import json
from pathlib import Path
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "quantumcommon"))

from evaluation.circuit_evaluater import QuantumCircuitEvaluator
from models.property_prediction_transformer import create_property_prediction_model
from circuit_interface import CircuitSpec, GateOperation


def load_test_circuits(file_path):
    """Load test circuits from a JSON file"""
    print(f"Loading circuits from {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert dictionary circuits to CircuitSpec objects
    circuits = []
    
    # Handle different JSON formats
    if isinstance(data, dict) and 'merged_results' in data:
        circuit_list = data.get('merged_results', [])
        print(f"Using merged_results format with {len(circuit_list)} entries")
    elif isinstance(data, dict) and 'results' in data:
        circuit_list = data.get('results', [])
        print(f"Using results format with {len(circuit_list)} entries")
    elif isinstance(data, dict) and 'circuits' in data:
        circuit_list = data.get('circuits', [])
        print(f"Using circuits format with {len(circuit_list)} entries")
    elif isinstance(data, list):
        circuit_list = data
        print(f"Using direct list format with {len(circuit_list)} entries")
    else:
        # Try to find circuits elsewhere in the structure
        circuit_list = []
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                # Check if first item looks like circuit data
                first_item = value[0]
                if isinstance(first_item, dict) and ('circuit_id' in first_item or 'num_qubits' in first_item):
                    circuit_list = value
                    print(f"Found circuits in '{key}' with {len(circuit_list)} entries")
                    break
    
    print(f"Found {len(circuit_list)} circuit entries in JSON")
    
    # Process each circuit
    for circuit_data in circuit_list:
        if not isinstance(circuit_data, dict):
            print(f"Warning: Skipping non-dict circuit data: {type(circuit_data)}")
            continue
            
        circuit_id = circuit_data.get('circuit_id', f'circuit_{len(circuits)}')
        num_qubits = circuit_data.get('num_qubits', 0)
        
        # Handle merged_data.json format which might not have gates directly
        # but has evaluation results
        if 'gates' not in circuit_data and 'gate_operations' not in circuit_data:
            if all(key in circuit_data for key in ['fidelity', 'expressibility', 'entanglement']):
                print(f"Creating dummy gates for results-only circuit: {circuit_id}")
                # Create dummy gates for a circuit that only has results
                gates = []
                for i in range(min(num_qubits, 5)):
                    gates.append(GateOperation('h', [i], []))
                    if i < num_qubits - 1:
                        gates.append(GateOperation('cx', [i, i+1], []))
                
                # Create CircuitSpec with dummy gates
                circuit_depth = circuit_data.get('depth', len(gates))
                try:
                    circuit = CircuitSpec(circuit_id, num_qubits, gates, circuit_depth)
                    # Add evaluation results if available
                    if hasattr(circuit, 'set_fidelity') and 'fidelity' in circuit_data:
                        circuit.set_fidelity(circuit_data['fidelity'])
                    if hasattr(circuit, 'set_expressibility') and 'expressibility' in circuit_data:
                        kl_div = circuit_data['expressibility'].get('kl_divergence', 0)
                        circuit.set_expressibility(kl_div)
                    if hasattr(circuit, 'set_entanglement') and 'entanglement' in circuit_data:
                        circuit.set_entanglement(circuit_data['entanglement'])
                    circuits.append(circuit)
                    continue
                except Exception as e:
                    print(f"Error creating dummy CircuitSpec: {e}")
                    continue
        
        # Get gates data from various possible keys
        gates_data = circuit_data.get('gates', [])
        if not gates_data and 'gate_operations' in circuit_data:
            gates_data = circuit_data.get('gate_operations', [])
            
        # Create gate operations
        gates = []
            
        for gate_dict in gates_data:
            if not isinstance(gate_dict, dict):
                continue
                
            # Handle different key formats
            gate_name = gate_dict.get('name', gate_dict.get('type', ''))
            qubits = gate_dict.get('qubits', [])
            params = gate_dict.get('parameters', gate_dict.get('params', []))
            gates.append(GateOperation(gate_name, qubits, params))
        
        # Make sure we have actual gate objects
        if not gates:
            print(f"Warning: No gates found for circuit {circuit_id}")
            continue
        
        # Create CircuitSpec
        try:
            circuit_depth = circuit_data.get('depth', len(gates))
            circuit = CircuitSpec(circuit_id, num_qubits, gates, circuit_depth)
            # Add evaluation results if available
            if hasattr(circuit, 'set_fidelity') and 'fidelity' in circuit_data:
                circuit.set_fidelity(circuit_data['fidelity'])
            if hasattr(circuit, 'set_expressibility') and 'expressibility' in circuit_data:
                kl_div = circuit_data['expressibility'].get('kl_divergence', 0)
                circuit.set_expressibility(kl_div)
            if hasattr(circuit, 'set_entanglement') and 'entanglement' in circuit_data:
                circuit.set_entanglement(circuit_data['entanglement'])
            circuits.append(circuit)
        except Exception as e:
            print(f"Error creating CircuitSpec: {e}")
            continue
    
    return circuits


def main(args):
    print("=== Testing QuantumCircuitEvaluator Integration ===")
    
    # Load property prediction model if available
    property_model = None
    if args.property_model_path and Path(args.property_model_path).exists():
        try:
            print(f"Loading property prediction model from {args.property_model_path}")
            # We'll skip loading the actual model for now - the evaluator doesn't use it directly
            print("Using default model initialization for testing")
        except Exception as e:
            print(f"Error loading property prediction model: {e}")
    
    # Load test circuits
    test_circuits = load_test_circuits(args.test_data)
    print(f"Loaded {len(test_circuits)} test circuits")
    
    # Create evaluator - no property_model parameter
    evaluator = QuantumCircuitEvaluator()
    
    # Test evaluate_circuits
    print("\nTesting evaluate_circuits...")
    circuit_evaluations = evaluator.evaluate_circuits(test_circuits)
    print(f"Evaluated {len(circuit_evaluations.get('circuit_evaluations', []))} circuits")
    
    # Test evaluate_quantum_properties
    print("\nTesting evaluate_quantum_properties...")
    properties = evaluator.evaluate_quantum_properties(test_circuits)
    print("Quantum properties:")
    for key, value in properties.items():
        print(f"  {key}: {value}")
    
    # If we have baseline circuits, test comparison
    if args.baseline_data and Path(args.baseline_data).exists():
        print("\nTesting compare_with_baseline...")
        baseline_circuits = load_test_circuits(args.baseline_data)
        print(f"Loaded {len(baseline_circuits)} baseline circuits")
        
        comparison = evaluator.compare_with_baseline(test_circuits, baseline_circuits)
        print("Comparison metrics:")
        for key, value in comparison.items():
            print(f"  {key}: {value}")
    
    # Test comprehensive evaluation if output directory is provided
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\nRunning comprehensive evaluation, saving to {output_dir}...")
        # Convert to format expected by the original method
        test_data_path = output_dir / "test_circuits.json"
        
        # Save test circuits to a file
        test_circuits_json = {'circuits': []}
        
        # Process circuits safely
        for i, circ in enumerate(test_circuits[:5]):
            # Skip circuits with invalid gates
            if not hasattr(circ, 'gates') or not isinstance(circ.gates, (list, tuple)):
                print(f"Skipping circuit {i} with invalid gates type: {type(getattr(circ, 'gates', None))}")
                continue
                
            # Create circuit entry
            circuit_entry = {
                'circuit_id': getattr(circ, 'circuit_id', f'test_{i}'),
                'num_qubits': circ.num_qubits,
                'gates': []
            }
            
            # Process gates
            valid_gates = []
            for gate in circ.gates:
                if not hasattr(gate, 'qubits'):
                    print(f"Skipping invalid gate in circuit {i}: {gate}")
                    continue
                    
                gate_entry = {
                    'type': gate.gate_type if hasattr(gate, 'gate_type') else 
                           (gate.name if hasattr(gate, 'name') else 'unknown'),
                    'qubits': gate.qubits,
                    'params': gate.parameters if hasattr(gate, 'parameters') else []
                }
                circuit_entry['gates'].append(gate_entry)
                valid_gates.append(gate)
                
            # Only include if we have valid gates
            if valid_gates:
                test_circuits_json['circuits'].append(circuit_entry)
        
        with open(test_data_path, 'w') as f:
            json.dump(test_circuits_json, f, indent=2)
        
        # Call the original method
        try:
            evaluator.run_comprehensive_evaluation(
                test_data_path=str(test_data_path),
                num_generated=5,  # Small number for testing
                save_results=True,
                output_dir=str(output_dir)
            )
        except TypeError as e:
            # Handle missing parameter or other type errors
            print(f"Type error in run_comprehensive_evaluation: {e}")
            print("Trying with more limited arguments...")
            evaluator.run_comprehensive_evaluation(
                test_data_path=str(test_data_path),
                num_generated=5  # Small number for testing
            )
        print("Comprehensive evaluation completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the QuantumCircuitEvaluator integration")
    parser.add_argument("--test_data", type=str, default="output/circuit_specs.json",
                        help="Path to test circuit data JSON file")
    parser.add_argument("--baseline_data", type=str, default="dummy_experiment_results.json", 
                        help="Path to baseline circuit data JSON file")
    parser.add_argument("--property_model_path", type=str, default="property_prediction_checkpoints/best_model.pt",
                        help="Path to property prediction model checkpoint")
    parser.add_argument("--output_dir", type=str, default="evaluation_test_results",
                        help="Directory to save evaluation results")
    
    args = parser.parse_args()
    main(args)
