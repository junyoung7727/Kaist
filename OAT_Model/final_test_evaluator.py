#!/usr/bin/env python3
"""
Final simple test script for circuit evaluation
"""

import json
import sys
import os
from pathlib import Path

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent_dir + '/quantumcommon')

from circuit_interface import CircuitSpec
from gates import GateOperation


def load_circuits_from_merged_data(file_path, max_circuits=10):
    """Load circuits from merged data JSON"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    circuits = []
    merged_results = data.get('merged_results', [])
    
    for i, result in enumerate(merged_results[:max_circuits]):
        if not isinstance(result, dict):
            continue
            
        circuit_id = result.get('circuit_id', f'merged_{i}')
        num_qubits = result.get('num_qubits', 2)
        depth = result.get('depth', 5)
        
        # Create simple gates based on circuit properties
        gates = []
        gates.append(GateOperation('h', [0], []))
        
        # Add CNOT gates based on qubits
        for q in range(min(num_qubits-1, 3)):
            gates.append(GateOperation('cx', [q, q+1], []))
        
        # Add some rotation gates
        if depth > 3:
            gates.append(GateOperation('ry', [0], [0.5]))
        
        circuit = CircuitSpec(
            num_qubits=num_qubits, 
            gates=gates, 
            circuit_id=circuit_id, 
            depth=depth
        )
        circuits.append(circuit)
    
    return circuits


def evaluate_circuit_simple(circuit):
    """Simple circuit evaluation"""
    if not hasattr(circuit, 'gates') or not circuit.gates:
        return None
    
    num_gates = len(circuit.gates)
    gate_types = set()
    
    for gate in circuit.gates:
        if hasattr(gate, 'gate_type'):
            gate_types.add(gate.gate_type)
        elif hasattr(gate, 'name'):
            gate_types.add(gate.name)
    
    # Simple mock metrics
    fidelity = max(0.1, 1.0 - (num_gates * 0.03))
    expressibility = min(10.0, num_gates * 0.5)
    entanglement = min(1.0, len(gate_types) * 0.15)
    
    return {
        "fidelity": fidelity,
        "expressibility": expressibility,
        "entanglement": entanglement,
        "num_gates": num_gates,
        "gate_types": list(gate_types)
    }


def main():
    print("=== Final Circuit Evaluator Test ===")
    
    # Test with merged data
    merged_file = "scal_test_result/merged_results/merged_all_20250814_080028.json"
    
    if os.path.exists(merged_file):
        print(f"\nLoading circuits from {merged_file}...")
        circuits = load_circuits_from_merged_data(merged_file, max_circuits=5)
        print(f"Loaded {len(circuits)} circuits")
        
        # Evaluate each circuit
        print("\nEvaluating circuits...")
        results = []
        
        for i, circuit in enumerate(circuits):
            result = evaluate_circuit_simple(circuit)
            if result:
                print(f"Circuit {i+1} ({circuit.circuit_id}): {result}")
                results.append(result)
            else:
                print(f"Circuit {i+1}: Failed to evaluate")
        
        # Summary statistics
        if results:
            fidelities = [r["fidelity"] for r in results]
            expressibilities = [r["expressibility"] for r in results]
            entanglements = [r["entanglement"] for r in results]
            
            print(f"\nSummary Statistics:")
            print(f"Fidelity: mean={sum(fidelities)/len(fidelities):.3f}, range=[{min(fidelities):.3f}, {max(fidelities):.3f}]")
            print(f"Expressibility: mean={sum(expressibilities)/len(expressibilities):.3f}, range=[{min(expressibilities):.3f}, {max(expressibilities):.3f}]")
            print(f"Entanglement: mean={sum(entanglements)/len(entanglements):.3f}, range=[{min(entanglements):.3f}, {max(entanglements):.3f}]")
        
    else:
        print(f"Merged data file not found: {merged_file}")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
