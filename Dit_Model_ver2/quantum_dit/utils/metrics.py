"""
Quantum Circuit Metrics for Evaluation
Comprehensive metrics for quantum circuit quality assessment
"""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter
import math


class QuantumCircuitMetrics:
    """
    Comprehensive metrics for evaluating quantum circuit generation quality
    """
    
    def __init__(self):
        self.gate_types = {
            'single_qubit': ['I', 'X', 'Y', 'Z', 'H', 'S', 'T', 'RX', 'RY', 'RZ'],
            'two_qubit': ['CNOT', 'CZ', 'SWAP', 'CRX', 'CRY', 'CRZ'],
            'multi_qubit': ['CCX', 'CSWAP', 'Fredkin', 'Toffoli']
        }
    
    def compute_circuit_depth(self, circuit: Dict[str, Any]) -> int:
        """
        Compute the depth of a quantum circuit
        
        Args:
            circuit: Circuit representation with gates and qubits
            
        Returns:
            Circuit depth
        """
        if not circuit.get('gates') or circuit.get('qubits', 0) == 0:
            return 0
        
        # Simple depth calculation based on gate dependencies
        # In practice, this would require more sophisticated analysis
        num_gates = len(circuit['gates'])
        num_qubits = circuit['qubits']
        
        # Estimate depth assuming some parallelization
        estimated_depth = max(1, num_gates // max(1, num_qubits // 2))
        return min(estimated_depth, num_gates)
    
    def compute_gate_diversity(self, circuit: Dict[str, Any]) -> float:
        """
        Compute gate diversity (Shannon entropy of gate distribution)
        
        Args:
            circuit: Circuit representation
            
        Returns:
            Gate diversity score (0 to log(num_gate_types))
        """
        gates = circuit.get('gates', [])
        if not gates:
            return 0.0
        
        # Count gate occurrences
        gate_counts = Counter(gates)
        total_gates = len(gates)
        
        # Compute Shannon entropy
        entropy = 0.0
        for count in gate_counts.values():
            prob = count / total_gates
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def compute_entanglement_capability(self, circuit: Dict[str, Any]) -> float:
        """
        Estimate entanglement generation capability
        
        Args:
            circuit: Circuit representation
            
        Returns:
            Entanglement capability score (0 to 1)
        """
        gates = circuit.get('gates', [])
        if not gates:
            return 0.0
        
        # Count two-qubit and multi-qubit gates
        entangling_gates = 0
        for gate_idx in gates:
            # Simple heuristic: assume higher indices are entangling gates
            if gate_idx > len(self.gate_types['single_qubit']):
                entangling_gates += 1
        
        # Normalize by total gates
        entanglement_ratio = entangling_gates / len(gates)
        
        # Apply sigmoid to get score between 0 and 1
        return 2 / (1 + math.exp(-5 * entanglement_ratio)) - 1
    
    def compute_circuit_efficiency(self, circuit: Dict[str, Any]) -> float:
        """
        Compute circuit efficiency (gates per qubit ratio)
        
        Args:
            circuit: Circuit representation
            
        Returns:
            Efficiency score
        """
        gates = circuit.get('gates', [])
        qubits = circuit.get('qubits', 1)
        
        if not gates or qubits == 0:
            return 0.0
        
        # Compute gates per qubit
        gates_per_qubit = len(gates) / qubits
        
        # Normalize to reasonable range (assuming 1-10 gates per qubit is reasonable)
        normalized_efficiency = min(1.0, gates_per_qubit / 10.0)
        
        return normalized_efficiency
    
    def compute_circuit_validity(self, circuit: Dict[str, Any]) -> float:
        """
        Check circuit validity based on various criteria
        
        Args:
            circuit: Circuit representation
            
        Returns:
            Validity score (0 to 1)
        """
        validity_score = 1.0
        
        # Check basic structure
        if not circuit.get('gates'):
            return 0.0
        
        if circuit.get('qubits', 0) <= 0:
            validity_score *= 0.5
        
        # Check for reasonable gate distribution
        gates = circuit['gates']
        gate_counts = Counter(gates)
        
        # Penalize circuits with excessive repetition
        max_repetition = max(gate_counts.values()) / len(gates)
        if max_repetition > 0.8:
            validity_score *= 0.6
        
        # Check for reasonable circuit length
        if len(gates) < 2:
            validity_score *= 0.7
        elif len(gates) > 1000:  # Too long
            validity_score *= 0.8
        
        # Check qubit count reasonableness
        qubits = circuit.get('qubits', 0)
        if qubits > 50:  # Too many qubits
            validity_score *= 0.9
        
        return validity_score
    
    def compute_quantum_volume(self, circuit: Dict[str, Any]) -> float:
        """
        Estimate quantum volume-like metric
        
        Args:
            circuit: Circuit representation
            
        Returns:
            Quantum volume estimate
        """
        depth = self.compute_circuit_depth(circuit)
        qubits = circuit.get('qubits', 0)
        
        if depth == 0 or qubits == 0:
            return 0.0
        
        # Quantum volume is typically min(depth, qubits)^2
        effective_size = min(depth, qubits)
        quantum_volume = effective_size ** 2
        
        # Normalize to reasonable range
        return min(1.0, quantum_volume / 100.0)
    
    def compute_expressivity(self, circuits: List[Dict[str, Any]]) -> float:
        """
        Compute expressivity of a set of circuits
        
        Args:
            circuits: List of circuit representations
            
        Returns:
            Expressivity score
        """
        if not circuits:
            return 0.0
        
        # Compute diversity metrics across all circuits
        all_depths = [self.compute_circuit_depth(c) for c in circuits]
        all_diversities = [self.compute_gate_diversity(c) for c in circuits]
        all_entanglements = [self.compute_entanglement_capability(c) for c in circuits]
        
        # Compute variance in each metric (higher variance = more expressive)
        depth_variance = np.var(all_depths) if len(all_depths) > 1 else 0
        diversity_variance = np.var(all_diversities) if len(all_diversities) > 1 else 0
        entanglement_variance = np.var(all_entanglements) if len(all_entanglements) > 1 else 0
        
        # Combine variances (normalized)
        expressivity = (
            min(1.0, depth_variance / 10.0) +
            min(1.0, diversity_variance / 2.0) +
            min(1.0, entanglement_variance / 0.25)
        ) / 3.0
        
        return expressivity
    
    def compute_fidelity_estimate(self, 
                                generated_circuit: Dict[str, Any],
                                reference_circuit: Dict[str, Any]) -> float:
        """
        Estimate fidelity between generated and reference circuits
        
        Args:
            generated_circuit: Generated circuit
            reference_circuit: Reference circuit
            
        Returns:
            Fidelity estimate (0 to 1)
        """
        # Simple structural similarity
        gen_gates = generated_circuit.get('gates', [])
        ref_gates = reference_circuit.get('gates', [])
        
        if not gen_gates or not ref_gates:
            return 0.0
        
        # Compute gate overlap
        gen_counter = Counter(gen_gates)
        ref_counter = Counter(ref_gates)
        
        # Jaccard similarity
        intersection = sum((gen_counter & ref_counter).values())
        union = sum((gen_counter | ref_counter).values())
        
        jaccard = intersection / union if union > 0 else 0.0
        
        # Consider circuit structure similarity
        gen_depth = self.compute_circuit_depth(generated_circuit)
        ref_depth = self.compute_circuit_depth(reference_circuit)
        
        depth_similarity = 1.0 - abs(gen_depth - ref_depth) / max(gen_depth, ref_depth, 1)
        
        # Combine similarities
        fidelity = (jaccard + depth_similarity) / 2.0
        
        return fidelity
    
    def evaluate_circuit_set(self, circuits: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Comprehensive evaluation of a set of circuits
        
        Args:
            circuits: List of circuit representations
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not circuits:
            return {}
        
        # Individual circuit metrics
        depths = [self.compute_circuit_depth(c) for c in circuits]
        diversities = [self.compute_gate_diversity(c) for c in circuits]
        entanglements = [self.compute_entanglement_capability(c) for c in circuits]
        efficiencies = [self.compute_circuit_efficiency(c) for c in circuits]
        validities = [self.compute_circuit_validity(c) for c in circuits]
        quantum_volumes = [self.compute_quantum_volume(c) for c in circuits]
        
        # Aggregate metrics
        results = {
            # Mean values
            'mean_depth': np.mean(depths),
            'mean_diversity': np.mean(diversities),
            'mean_entanglement': np.mean(entanglements),
            'mean_efficiency': np.mean(efficiencies),
            'mean_validity': np.mean(validities),
            'mean_quantum_volume': np.mean(quantum_volumes),
            
            # Standard deviations
            'std_depth': np.std(depths),
            'std_diversity': np.std(diversities),
            'std_entanglement': np.std(entanglements),
            'std_efficiency': np.std(efficiencies),
            'std_validity': np.std(validities),
            'std_quantum_volume': np.std(quantum_volumes),
            
            # Set-level metrics
            'expressivity': self.compute_expressivity(circuits),
            'valid_circuit_ratio': np.mean([1.0 if v > 0.8 else 0.0 for v in validities]),
            
            # Distribution metrics
            'min_depth': np.min(depths),
            'max_depth': np.max(depths),
            'median_depth': np.median(depths),
            
            # Quality metrics
            'high_quality_ratio': np.mean([
                1.0 if (d > 0.5 and e > 0.3 and v > 0.8) else 0.0 
                for d, e, v in zip(diversities, entanglements, validities)
            ])
        }
        
        return results
    
    def compare_circuit_sets(self, 
                           generated_circuits: List[Dict[str, Any]],
                           reference_circuits: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compare generated circuits with reference circuits
        
        Args:
            generated_circuits: Generated circuits
            reference_circuits: Reference circuits
            
        Returns:
            Comparison metrics
        """
        gen_metrics = self.evaluate_circuit_set(generated_circuits)
        ref_metrics = self.evaluate_circuit_set(reference_circuits)
        
        comparison = {}
        
        # Compute relative differences
        for key in gen_metrics:
            if key.startswith('mean_') or key.startswith('std_'):
                gen_val = gen_metrics[key]
                ref_val = ref_metrics.get(key, 0)
                
                if ref_val != 0:
                    rel_diff = (gen_val - ref_val) / abs(ref_val)
                    comparison[f'{key}_relative_diff'] = rel_diff
                
                comparison[f'{key}_generated'] = gen_val
                comparison[f'{key}_reference'] = ref_val
        
        # Compute average fidelity
        if reference_circuits:
            fidelities = []
            for gen_circuit in generated_circuits[:100]:  # Sample for efficiency
                best_fidelity = 0.0
                for ref_circuit in reference_circuits[:100]:
                    fidelity = self.compute_fidelity_estimate(gen_circuit, ref_circuit)
                    best_fidelity = max(best_fidelity, fidelity)
                fidelities.append(best_fidelity)
            
            comparison['average_fidelity'] = np.mean(fidelities)
            comparison['fidelity_std'] = np.std(fidelities)
        
        return comparison
