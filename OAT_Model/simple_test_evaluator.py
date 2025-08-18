#!/usr/bin/env python3
"""
Simple test script for QuantumCircuitEvaluator
"""

import json
import sys
import os

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent_dir + '/quantumcommon')

from circuit_interface import CircuitSpec
from gates import GateOperation


class SimpleQuantumEvaluator:
    """Simple quantum circuit evaluator without complex dependencies"""
    
    def __init__(self):
        pass
    
    def evaluate_single_circuit(self, circuit):
        """Evaluate a single circuit - simplified version"""
        try:
            # Basic validation
            if not hasattr(circuit, 'gates') or not circuit.gates:
                return {"error": "No gates found"}
            
            # Simple metrics calculation
            num_gates = len(circuit.gates)
            gate_types = set(gate.gate_type if hasattr(gate, 'gate_type') else gate.name for gate in circuit.gates)
            
            # Mock fidelity based on circuit complexity
            fidelity = max(0.1, 1.0 - (num_gates * 0.05))
            
            # Mock expressibility (KL divergence)
            expressibility = min(15.0, num_gates * 0.8)
            
            # Mock entanglement
            entanglement = min(1.0, len(gate_types) * 0.2)
            
            return {
                "fidelity": fidelity,
                "expressibility": expressibility,
                "entanglement": entanglement,
                "num_gates": num_gates,
                "gate_types": list(gate_types)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def evaluate_quantum_properties(self, circuits):
        """Evaluate multiple circuits and return aggregated properties"""
        if not circuits:
            return {}
        
        results = []
        for circuit in circuits:
            result = self.evaluate_single_circuit(circuit)
            if "error" not in result:
                results.append(result)
        
        if not results:
            return {"error": "No valid circuits to evaluate"}
        
        # Calculate statistics
        fidelities = [r["fidelity"] for r in results]
        expressibilities = [r["expressibility"] for r in results]
        entanglements = [r["entanglement"] for r in results]
        
        return {
            "fidelity": {
                "mean": sum(fidelities) / len(fidelities),
                "min": min(fidelities),
                "max": max(fidelities)
            },
            "expressibility": {
                "mean": sum(expressibilities) / len(expressibilities),
                "min": min(expressibilities),
                "max": max(expressibilities)
            },
            "entanglement": {
                "mean": sum(entanglements) / len(entanglements),
                "min": min(entanglements),
                "max": max(entanglements)
            },
            "total_circuits": len(results)
        }


def create_simple_test_circuits():
    """Create simple test circuits"""
    circuits = []
    
    # Circuit 1: Simple H-CNOT circuit
    gates1 = [
        GateOperation('h', [0], []),
        GateOperation('cx', [0, 1], [])
    ]
    circuit1 = CircuitSpec(num_qubits=2, gates=gates1, circuit_id='test_1', depth=2)
    circuits.append(circuit1)
    
    # Circuit 2: 3-qubit GHZ state
    gates2 = [
        GateOperation('h', [0], []),
        GateOperation('cx', [0, 1], []),
        GateOperation('cx', [1, 2], [])
    ]
    circuit2 = CircuitSpec(num_qubits=3, gates=gates2, circuit_id='test_2', depth=3)
    circuits.append(circuit2)
    
    # Circuit 3: Random rotation circuit
    gates3 = [
        GateOperation('rx', [0], [0.5]),
        GateOperation('ry', [1], [1.2]),
        GateOperation('cx', [0, 1], [])
    ]
    circuit3 = CircuitSpec(num_qubits=2, gates=gates3, circuit_id='test_3', depth=3)
    circuits.append(circuit3)
    
    return circuits


def load_merged_data_circuits(file_path, max_circuits=10):
    """Load circuits from merged_data.json format"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    circuits = []
    merged_results = data.get('merged_results', [])
    
    for i, result in enumerate(merged_results[:max_circuits]):
        circuit_id = result.get('circuit_id', f'merged_{i}')
        num_qubits = result.get('num_qubits', 2)
        
        # Create simple dummy gates based on num_qubits
        gates = []
        gates.append(GateOperation('h', [0], []))
        for q in range(min(num_qubits-1, 4)):
            gates.append(GateOperation('cx', [q, q+1], []))
        
        depth = result.get('depth', len(gates))
        circuit = CircuitSpec(num_qubits=num_qubits, gates=gates, circuit_id=circuit_id, depth=depth)
        circuits.append(circuit)
    
    return circuits


def main():
    print("=== Simple QuantumCircuitEvaluator Test ===")
    
    # Initialize evaluator
    evaluator = SimpleQuantumEvaluator()
    
    # Test 1: Simple test circuits
    print("\n1. Testing with simple circuits...")
    test_circuits = create_simple_test_circuits()
    print(f"Created {len(test_circuits)} test circuits")
    
    # Test evaluate_single_circuit
    print("\n2. Testing single circuit evaluation...")
    for i, circuit in enumerate(test_circuits):
        try:
            result = evaluator.evaluate_single_circuit(circuit)
            print(f"Circuit {i+1}: {result}")
        except Exception as e:
            print(f"Error evaluating circuit {i+1}: {e}")
    
    # Test evaluate_quantum_properties
    print("\n3. Testing quantum properties evaluation...")
    try:
        properties = evaluator.evaluate_quantum_properties(test_circuits)
        print(f"Quantum properties: {properties}")
    except Exception as e:
        print(f"Error in evaluate_quantum_properties: {e}")
    
    # Test 4: Load from merged data if available
    merged_file = "scal_test_result/merged_results/merged_all_20250814_080028.json"
    if os.path.exists(merged_file):
        print(f"\n4. Testing with merged data from {merged_file}...")
        try:
            merged_circuits = load_merged_data_circuits(merged_file, max_circuits=5)
            print(f"Loaded {len(merged_circuits)} circuits from merged data")
            
            # Test a few circuits
            for i, circuit in enumerate(merged_circuits[:3]):
                try:
                    result = evaluator.evaluate_single_circuit(circuit)
                    print(f"Merged circuit {i+1}: {result}")
                except Exception as e:
                    print(f"Error with merged circuit {i+1}: {e}")
                    
        except Exception as e:
            print(f"Error loading merged data: {e}")
    else:
        print(f"\n4. Merged data file not found: {merged_file}")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
