"""
Property Guided Quantum Circuit Generator

This module implements a quantum circuit generator that leverages the PropertyPredictionTransformer
to generate quantum circuits with desired target properties.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import sys
import json
from dataclasses import dataclass
import random
import copy

# Project module imports
sys.path.append(str(Path(__file__).parent.parent))
from models.property_prediction_transformer import PropertyPredictionTransformer, PropertyPredictionConfig
from models.decision_transformer import DecisionTransformer
from data.embedding_pipeline import EmbeddingPipeline, EmbeddingConfig
from data.quantum_circuit_dataset import CircuitSpec

# quantumcommon module imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))
from gates import QuantumGateRegistry, GateOperation
from circuit_interface import QuantumCircuit


@dataclass
class PropertyGuidedConfig:
    """Configuration for property-guided circuit generation"""
    # Circuit parameters
    max_circuit_length: int = 30
    min_circuit_length: int = 5
    target_num_qubits: int = 4
    
    # Generation parameters
    temperature: float = 1.0
    top_k: int = 10
    top_p: float = 0.95
    do_sample: bool = True
    
    # Target property objectives
    target_entanglement: Optional[float] = None
    target_fidelity: Optional[float] = None
    target_expressibility: Optional[float] = None
    target_robust_fidelity: Optional[float] = None
    
    # Weights for property importance
    entanglement_weight: float = 1.0
    fidelity_weight: float = 1.0
    expressibility_weight: float = 1.0
    robust_fidelity_weight: float = 1.0
    
    # Generation strategy
    max_iterations: int = 100
    property_tolerance: float = 0.1
    optimization_steps: int = 10
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PropertyGuidedGenerator:
    """
    Property-guided quantum circuit generator using PropertyPredictionTransformer
    for guiding the circuit generation process to achieve target property objectives.
    """
    
    def __init__(self, 
                 property_model_path: str,
                 decision_model_path: Optional[str] = None,
                 config: Optional[PropertyGuidedConfig] = None):
        """
        Initialize the property-guided generator.
        
        Args:
            property_model_path: Path to the PropertyPredictionTransformer model
            decision_model_path: Optional path to the DecisionTransformer model for direct generation
            config: Configuration for the generator
        """
        self.config = config or PropertyGuidedConfig()
        self.device = torch.device(self.config.device)
        
        # Initialize gate registry
        self.gate_registry = QuantumGateRegistry()
        self.gate_vocab = self.gate_registry.get_gate_vocab()
        self.idx_to_gate = {idx: gate for gate, idx in self.gate_vocab.items()}
        
        # Load property prediction model
        self.property_model = self._load_property_model(property_model_path)
        
        # Load decision transformer model if provided
        self.decision_model = None
        if decision_model_path:
            self.decision_model = self._load_decision_model(decision_model_path)
        
        # Initialize embedding pipeline
        self.embedding_pipeline = self._create_embedding_pipeline()
        
        print(f"PropertyGuidedGenerator initialized on {self.device}")
        print(f"Gate vocabulary size: {len(self.gate_vocab)} gates")
        print(f"Target objectives: {self._get_target_objectives()}")
    
    def _load_property_model(self, model_path: str) -> PropertyPredictionTransformer:
        """Load the property prediction transformer model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model config from checkpoint
        model_config = None
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
        elif 'config' in checkpoint:
            model_config = checkpoint['config']
        
        # Create model configuration
        if model_config is not None:
            config = PropertyPredictionConfig(
                d_model=model_config.get('d_model', 512),
                n_layers=model_config.get('n_layers', 6),
                n_heads=model_config.get('n_heads', 8),
                dropout=model_config.get('dropout', 0.1),
            )
        else:
            # Default config if not found in checkpoint
            config = PropertyPredictionConfig()
        
        # Create and load model
        model = PropertyPredictionTransformer(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Property prediction model loaded from {model_path}")
        return model
    
    def _load_decision_model(self, model_path: str) -> Optional[DecisionTransformer]:
        """Load the decision transformer model for direct generation"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model config
            ckpt_cfg = None
            if 'model_config' in checkpoint:
                ckpt_cfg = checkpoint['model_config']
            elif 'config' in checkpoint:
                ckpt_cfg = checkpoint['config']

            if ckpt_cfg is not None:
                model_config = {
                    'd_model': ckpt_cfg.get('d_model', 512),
                    'n_layers': ckpt_cfg.get('n_layers', 6),
                    'n_heads': ckpt_cfg.get('n_heads', 8),
                    'n_gate_types': ckpt_cfg.get('n_gate_types', len(self.gate_vocab)),
                    'dropout': ckpt_cfg.get('dropout', 0.1),
                }
            else:
                # Default settings
                model_config = {
                    'd_model': 512,
                    'n_layers': 6,
                    'n_heads': 8,
                    'n_gate_types': len(self.gate_vocab),
                    'dropout': 0.1
                }
            
            # Create and load model
            model = DecisionTransformer(**model_config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            print(f"Decision transformer model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Warning: Failed to load decision transformer model: {e}")
            return None
    
    def _create_embedding_pipeline(self) -> EmbeddingPipeline:
        """Create embedding pipeline for circuit processing"""
        embedding_config = EmbeddingConfig(
            d_model=512,
            n_gate_types=len(self.gate_vocab),
            n_qubits=self.config.target_num_qubits,
            max_seq_len=self.config.max_circuit_length * 3  # For S-A-R pattern if needed
        )
        return EmbeddingPipeline(embedding_config)
    
    def _get_target_objectives(self) -> Dict[str, float]:
        """Extract target objectives from configuration"""
        objectives = {}
        
        if self.config.target_entanglement is not None:
            objectives['entanglement'] = self.config.target_entanglement
            
        if self.config.target_fidelity is not None:
            objectives['fidelity'] = self.config.target_fidelity
            
        if self.config.target_expressibility is not None:
            objectives['expressibility'] = self.config.target_expressibility
            
        if self.config.target_robust_fidelity is not None:
            objectives['robust_fidelity'] = self.config.target_robust_fidelity
            
        return objectives
    
    def generate_circuit(self, 
                       target_objectives: Optional[Dict[str, float]] = None, 
                       initial_circuit: Optional[CircuitSpec] = None) -> CircuitSpec:
        """
        Generate a circuit with properties matching target objectives
        
        Args:
            target_objectives: Dictionary of target property values
            initial_circuit: Optional starting circuit
            
        Returns:
            CircuitSpec with generated circuit
        """
        # Update target objectives if provided
        if target_objectives:
            self._update_target_objectives(target_objectives)
        
        # Create initial circuit if not provided
        if initial_circuit is None:
            initial_circuit = self._create_initial_circuit()
        
        print(f"Starting property-guided circuit generation:")
        print(f"Target objectives: {self._get_target_objectives()}")
        
        # Use property-guided optimization to generate circuit
        best_circuit = self._property_guided_optimization(initial_circuit)
        
        print(f"Circuit generation completed: {len(best_circuit.gates)} gates")
        print(f"Estimated properties: {self._predict_circuit_properties(best_circuit)}")
        
        return best_circuit
    
    def _update_target_objectives(self, target_objectives: Dict[str, float]) -> None:
        """Update target objectives in the configuration"""
        if 'entanglement' in target_objectives:
            self.config.target_entanglement = target_objectives['entanglement']
            
        if 'fidelity' in target_objectives:
            self.config.target_fidelity = target_objectives['fidelity']
            
        if 'expressibility' in target_objectives:
            self.config.target_expressibility = target_objectives['expressibility']
            
        if 'robust_fidelity' in target_objectives:
            self.config.target_robust_fidelity = target_objectives['robust_fidelity']
    
    def _create_initial_circuit(self) -> CircuitSpec:
        """Create an initial empty circuit"""
        return CircuitSpec(
            circuit_id="initial_circuit",
            num_qubits=self.config.target_num_qubits,
            gates=[]
        )
    
    def _predict_circuit_properties(self, circuit: CircuitSpec) -> Dict[str, float]:
        """
        Predict properties for a given circuit using the property prediction model
        
        Args:
            circuit: Input circuit
            
        Returns:
            Dictionary of predicted properties
        """
        # Process circuit through embedding pipeline
        embedded_data = self.embedding_pipeline.process_single_circuit(circuit)
        
        # Get input tensors
        input_sequence = embedded_data['input_sequence'].to(self.device)
        attention_mask = embedded_data.get('attention_mask')
        if attention_mask is None:
            # Create causal mask if not provided
            seq_len = input_sequence.shape[1]
            attention_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).unsqueeze(0).to(self.device)
        
        # Create grid structure and edges (if needed by model)
        grid_structure = embedded_data.get('grid_structure', {})
        edges = embedded_data.get('edges', [])
        
        # Forward pass through model
        with torch.no_grad():
            predictions = self.property_model.forward_single(
                input_sequence, 
                attention_mask=attention_mask,
                grid_structure=grid_structure,
                edges=edges
            )
        
        # Extract property predictions
        properties = {}
        for key in ['entanglement', 'fidelity', 'expressibility', 'robust_fidelity']:
            if key in predictions:
                properties[key] = predictions[key].item()
        
        return properties
    
    def _property_guided_optimization(self, initial_circuit: CircuitSpec) -> CircuitSpec:
        """
        Use property-guided optimization to generate a circuit meeting target objectives
        
        Args:
            initial_circuit: Starting circuit
            
        Returns:
            Optimized circuit
        """
        best_circuit = initial_circuit
        best_score = float('inf')
        
        current_circuit = copy.deepcopy(initial_circuit)
        
        for i in range(self.config.max_iterations):
            # Predict current properties
            current_properties = self._predict_circuit_properties(current_circuit)
            
            # Calculate objective score
            current_score = self._calculate_objective_score(current_properties)
            
            # Update best circuit if better
            if current_score < best_score:
                best_score = current_score
                best_circuit = copy.deepcopy(current_circuit)
                
                print(f"Iteration {i}: Found better circuit (score: {best_score:.4f})")
                print(f"  Properties: {current_properties}")
            
            # Check if we're close enough to target
            if best_score < self.config.property_tolerance:
                print(f"Reached target objectives within tolerance (score: {best_score:.4f})")
                break
            
            # Apply optimization step
            if len(current_circuit.gates) >= self.config.max_circuit_length:
                # If max length reached, modify existing gates
                current_circuit = self._modify_existing_circuit(current_circuit)
            else:
                # Otherwise, add new gates
                current_circuit = self._extend_circuit(current_circuit, current_properties)
        
        return best_circuit
    
    def _calculate_objective_score(self, properties: Dict[str, float]) -> float:
        """
        Calculate how well the circuit properties match the target objectives
        
        Args:
            properties: Dictionary of circuit properties
            
        Returns:
            Score (lower is better)
        """
        score = 0.0
        
        # Calculate weighted MSE for each target objective
        if self.config.target_entanglement is not None and 'entanglement' in properties:
            score += self.config.entanglement_weight * (properties['entanglement'] - self.config.target_entanglement) ** 2
            
        if self.config.target_fidelity is not None and 'fidelity' in properties:
            score += self.config.fidelity_weight * (properties['fidelity'] - self.config.target_fidelity) ** 2
            
        if self.config.target_expressibility is not None and 'expressibility' in properties:
            score += self.config.expressibility_weight * (properties['expressibility'] - self.config.target_expressibility) ** 2
            
        if self.config.target_robust_fidelity is not None and 'robust_fidelity' in properties:
            score += self.config.robust_fidelity_weight * (properties['robust_fidelity'] - self.config.target_robust_fidelity) ** 2
        
        return score
    
    def _modify_existing_circuit(self, circuit: CircuitSpec) -> CircuitSpec:
        """
        Modify an existing circuit by changing gates
        
        Args:
            circuit: Input circuit
            
        Returns:
            Modified circuit
        """
        # Create a copy of the circuit to modify
        modified_circuit = copy.deepcopy(circuit)
        
        if not modified_circuit.gates:
            return modified_circuit
            
        # Select a random gate to modify
        gate_idx = random.randint(0, len(modified_circuit.gates) - 1)
        
        # Choose modification type
        mod_type = random.choice(['replace', 'remove', 'parameter'])
        
        if mod_type == 'replace':
            # Replace gate with a new one
            modified_circuit.gates[gate_idx] = self._generate_random_gate()
            
        elif mod_type == 'remove':
            # Remove the gate
            modified_circuit.gates.pop(gate_idx)
            
        elif mod_type == 'parameter':
            # Only modify parameters if the gate has any
            if modified_circuit.gates[gate_idx].parameters:
                # Perturb parameters
                for i in range(len(modified_circuit.gates[gate_idx].parameters)):
                    # Add small random change to parameter
                    delta = random.uniform(-0.3, 0.3)
                    modified_circuit.gates[gate_idx].parameters[i] += delta
        
        return modified_circuit
    
    def _extend_circuit(self, circuit: CircuitSpec, 
                      current_properties: Dict[str, float]) -> CircuitSpec:
        """
        Extend a circuit by adding a new gate
        
        Args:
            circuit: Input circuit
            current_properties: Current circuit properties
            
        Returns:
            Extended circuit
        """
        # Create a copy to extend
        extended_circuit = copy.deepcopy(circuit)
        
        # Generate a new gate
        new_gate = self._generate_guided_gate(current_properties)
        
        # Add the new gate
        extended_circuit.gates.append(new_gate)
        
        return extended_circuit
    
    def _generate_random_gate(self) -> GateOperation:
        """Generate a random quantum gate"""
        # Select a random gate from vocabulary (excluding special tokens)
        valid_gates = [gate for gate in self.gate_vocab.keys() 
                       if gate not in ['[PAD]', '[EOS]', '[EMPTY]']]
        gate_name = random.choice(valid_gates)
        
        # Get gate definition
        gate_def = self.gate_registry.get_gate(gate_name)
        
        # Select qubits
        num_qubits = gate_def.num_qubits
        qubits = random.sample(range(self.config.target_num_qubits), min(num_qubits, self.config.target_num_qubits))
        
        # Generate parameters
        num_params = gate_def.num_parameters
        parameters = [random.uniform(0, 2 * np.pi) for _ in range(num_params)]
        
        # Create gate operation
        return GateOperation(name=gate_name, qubits=qubits, parameters=parameters)
    
    def _generate_guided_gate(self, current_properties: Dict[str, float]) -> GateOperation:
        """
        Generate a gate guided by current properties and target objectives
        
        Args:
            current_properties: Current circuit properties
            
        Returns:
            New gate operation
        """
        # If we have a decision model, use it for guided generation
        if self.decision_model:
            return self._generate_gate_with_decision_model(current_properties)
        
        # Default to random gate generation
        return self._generate_random_gate()
    
    def _generate_gate_with_decision_model(self, current_properties: Dict[str, float]) -> GateOperation:
        """Generate a gate using the decision transformer model"""
        # Implement this if decision model is available for guided generation
        # This is just a placeholder for now
        return self._generate_random_gate()
    
    def generate_multiple_circuits(self, 
                                num_circuits: int = 5,
                                target_objectives: Optional[Dict[str, float]] = None) -> List[CircuitSpec]:
        """
        Generate multiple circuits with target properties
        
        Args:
            num_circuits: Number of circuits to generate
            target_objectives: Dictionary of target property values
            
        Returns:
            List of generated circuits
        """
        circuits = []
        
        for i in range(num_circuits):
            print(f"\nGenerating circuit {i+1}/{num_circuits}")
            circuit = self.generate_circuit(target_objectives=target_objectives)
            circuits.append(circuit)
        
        return circuits
    
    def save_circuits(self, circuits: List[CircuitSpec], output_path: str):
        """
        Save generated circuits to a JSON file
        
        Args:
            circuits: List of circuits to save
            output_path: Output file path
        """
        circuits_data = []
        
        for i, circuit in enumerate(circuits):
            # Get predicted properties for this circuit
            properties = self._predict_circuit_properties(circuit)
            
            circuit_data = {
                'circuit_id': circuit.circuit_id or f"generated_circuit_{i}",
                'num_qubits': circuit.num_qubits,
                'gates': [
                    {
                        'name': gate.name,
                        'qubits': [int(q) for q in gate.qubits],
                        'parameters': [float(p) for p in gate.parameters]
                    }
                    for gate in circuit.gates
                ],
                'predicted_properties': properties
            }
            circuits_data.append(circuit_data)
        
        with open(output_path, 'w') as f:
            json.dump(circuits_data, f, indent=2)
        
        print(f"Saved {len(circuits)} circuits to {output_path}")


def main():
    """Usage example"""
    # Configuration
    config = PropertyGuidedConfig(
        max_circuit_length=20,
        target_num_qubits=4,
        temperature=0.8,
        
        # Target properties
        target_entanglement=0.7,
        target_fidelity=0.85,
        target_expressibility=0.5,
        
        # Weights
        entanglement_weight=1.0,
        fidelity_weight=2.0,
        expressibility_weight=1.0,
        
        # Generation settings
        max_iterations=50,
        property_tolerance=0.05
    )
    
    # Model paths
    property_model_path = "checkpoints/best_model.pt"  # PropertyPredictionTransformer checkpoint
    decision_model_path = None  # Optional DecisionTransformer checkpoint
    
    # Initialize generator
    generator = PropertyGuidedGenerator(
        property_model_path=property_model_path,
        decision_model_path=decision_model_path,
        config=config
    )
    
    # Generate multiple circuits
    circuits = generator.generate_multiple_circuits(
        num_circuits=3,
        target_objectives={
            'entanglement': 0.7,
            'fidelity': 0.85,
            'expressibility': 0.5
        }
    )
    
    # Save results
    generator.save_circuits(circuits, "generated_property_circuits.json")
    
    # Print results summary
    for i, circuit in enumerate(circuits):
        properties = generator._predict_circuit_properties(circuit)
        print(f"\nCircuit {i+1}:")
        print(f"  Qubits: {circuit.num_qubits}")
        print(f"  Gates: {len(circuit.gates)}")
        print(f"  Properties: {properties}")
        
        # Show first few gates
        for j, gate in enumerate(circuit.gates[:5]):
            print(f"    {j+1}. {gate.name} on qubits {gate.qubits}")
        if len(circuit.gates) > 5:
            print(f"    ... and {len(circuit.gates) - 5} more gates")


if __name__ == "__main__":
    main()
