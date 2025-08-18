"""
Inference Script for Quantum Gate Prediction Transformer
Generate quantum circuits based on user requirements using trained transformer model
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

# Import model
from models.quantum_transformer import QuantumTransformer, QuantumTransformerConfig

# Add quantumcommon to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "quantumcommon"))

from gates import QuantumGateRegistry


class QuantumCircuitGenerator:
    """Generate quantum circuits using trained transformer model"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize circuit generator
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.device = self._setup_device(device)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        
        # Create model
        self.model = QuantumTransformer(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize gate registry
        self.gate_registry = QuantumGateRegistry()
        self.gate_vocab = self.gate_registry.get_gate_vocab()
        self.gate_to_idx = {gate: idx for idx, gate in enumerate(self.gate_vocab)}
        self.idx_to_gate = {idx: gate for gate, idx in self.gate_to_idx.items()}
        
        print(f"Loaded model with {self.model.get_num_params():,} parameters")
        print(f"Gate vocabulary: {self.gate_vocab}")
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(device)
    
    def _create_requirements_tensor(self, requirements: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Convert requirements dictionary to tensor format"""
        req_tensors = {}
        
        # Target properties
        req_tensors['target_fidelity'] = torch.tensor([[requirements.get('target_fidelity', 0.8)]], 
                                                     dtype=torch.float, device=self.device)
        req_tensors['target_expressibility'] = torch.tensor([[requirements.get('target_expressibility', 0.5)]], 
                                                           dtype=torch.float, device=self.device)
        req_tensors['target_entanglement'] = torch.tensor([[requirements.get('target_entanglement', 0.3)]], 
                                                         dtype=torch.float, device=self.device)
        req_tensors['target_depth'] = torch.tensor([[requirements.get('target_depth', 20.0)]], 
                                                   dtype=torch.float, device=self.device)
        
        # Circuit constraints
        req_tensors['num_qubits'] = torch.tensor([requirements.get('num_qubits', 4)], 
                                                dtype=torch.long, device=self.device)
        req_tensors['max_depth'] = torch.tensor([[requirements.get('max_depth', 50.0)]], 
                                               dtype=torch.float, device=self.device)
        req_tensors['two_qubit_ratio'] = torch.tensor([[requirements.get('two_qubit_ratio', 0.3)]], 
                                                     dtype=torch.float, device=self.device)
        
        return req_tensors
    
    def _encode_gate_sequence(self, gates: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode gate sequence into tensor format"""
        seq_len = len(gates)
        
        # Initialize tensors
        gate_indices = torch.full((1, seq_len), self.model.config.gate_vocab_size, 
                                 dtype=torch.long, device=self.device)
        qubit_indices = torch.zeros((1, seq_len, self.config.max_qubits), 
                                   dtype=torch.long, device=self.device)
        parameters = torch.zeros((1, seq_len, self.config.max_parameters), 
                                dtype=torch.float, device=self.device)
        
        for i, gate in enumerate(gates):
            # Gate type
            gate_name = gate['name']
            if gate_name in self.gate_to_idx:
                gate_indices[0, i] = self.gate_to_idx[gate_name]
            
            # Qubits
            gate_qubits = gate['qubits']
            for j, qubit in enumerate(gate_qubits):
                if j < self.config.max_qubits:
                    qubit_indices[0, i, j] = qubit
            
            # Parameters
            gate_params = gate.get('parameters', [])
            for j, param in enumerate(gate_params):
                if j < self.config.max_parameters:
                    parameters[0, i, j] = param
        
        return gate_indices, qubit_indices, parameters
    
    def generate_circuit(self, 
                        requirements: Dict[str, float],
                        initial_gates: List[Dict] = None,
                        max_length: int = 50,
                        temperature: float = 1.0,
                        top_k: int = None,
                        top_p: float = None,
                        stop_gates: List[str] = None) -> List[Dict]:
        """
        Generate quantum circuit based on requirements
        
        Args:
            requirements: Dictionary with target properties and constraints
            initial_gates: Optional initial gate sequence to continue from
            max_length: Maximum circuit length to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (only consider top k gates)
            top_p: Top-p (nucleus) sampling
            stop_gates: List of gate names that should stop generation
            
        Returns:
            List of gate dictionaries representing the generated circuit
        """
        # Initialize with empty circuit or provided initial gates
        if initial_gates is None:
            generated_gates = []
        else:
            generated_gates = initial_gates.copy()
        
        # Convert requirements to tensor format
        req_tensors = self._create_requirements_tensor(requirements)
        
        # Default stop gates
        if stop_gates is None:
            stop_gates = ['measure', 'barrier']
        
        num_qubits = requirements.get('num_qubits', 4)
        
        with torch.no_grad():
            for step in range(max_length):
                # Encode current sequence
                if len(generated_gates) == 0:
                    # Start with identity gates on all qubits
                    current_gates = [{'name': 'id', 'qubits': [i], 'parameters': []} 
                                   for i in range(num_qubits)]
                else:
                    current_gates = generated_gates
                
                gate_indices, qubit_indices, parameters = self._encode_gate_sequence(current_gates)
                
                # Forward pass through model
                outputs = self.model(
                    gates=gate_indices,
                    qubits=qubit_indices,
                    parameters=parameters,
                    requirements=req_tensors
                )
                
                # Get logits for next gate
                gate_logits = outputs['gate_logits'][0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = gate_logits < torch.topk(gate_logits, top_k)[0][..., -1, None]
                    gate_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(gate_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    gate_logits[indices_to_remove] = float('-inf')
                
                # Sample next gate
                gate_probs = F.softmax(gate_logits, dim=-1)
                gate_idx = torch.multinomial(gate_probs, 1).item()
                
                # Convert to gate name
                if gate_idx >= len(self.idx_to_gate):
                    break  # End of vocabulary
                
                gate_name = self.idx_to_gate[gate_idx]
                
                # Check for stop condition
                if gate_name in stop_gates:
                    break
                
                # Generate qubits and parameters for the gate
                next_gate = self._generate_gate_details(
                    gate_name, num_qubits, outputs, requirements
                )
                
                generated_gates.append(next_gate)
                
                # Check target depth
                if len(generated_gates) >= requirements.get('target_depth', 20):
                    break
        
        return generated_gates
    
    def _generate_gate_details(self, 
                              gate_name: str, 
                              num_qubits: int, 
                              model_outputs: Dict[str, torch.Tensor],
                              requirements: Dict[str, float]) -> Dict:
        """Generate qubit assignments and parameters for a gate"""
        gate_info = {'name': gate_name, 'qubits': [], 'parameters': []}
        
        # Get gate properties from registry
        gate_properties = self.gate_registry.get_gate_info(gate_name)
        
        if gate_properties:
            # Determine number of qubits needed
            if gate_properties.get('type') == 'single':
                num_gate_qubits = 1
            elif gate_properties.get('type') == 'two':
                num_gate_qubits = 2
            else:
                num_gate_qubits = 1  # Default
            
            # Sample qubits
            if self.config.predict_qubits and 'qubit_logits' in model_outputs:
                qubit_logits = model_outputs['qubit_logits'][0, -1, :num_qubits]
                qubit_probs = F.softmax(qubit_logits, dim=-1)
                
                # Sample qubits without replacement
                sampled_qubits = []
                remaining_qubits = list(range(num_qubits))
                
                for _ in range(min(num_gate_qubits, num_qubits)):
                    if len(remaining_qubits) == 0:
                        break
                    
                    # Create probability distribution over remaining qubits
                    remaining_probs = qubit_probs[remaining_qubits]
                    remaining_probs = remaining_probs / remaining_probs.sum()
                    
                    # Sample qubit
                    idx = torch.multinomial(remaining_probs, 1).item()
                    selected_qubit = remaining_qubits[idx]
                    
                    sampled_qubits.append(selected_qubit)
                    remaining_qubits.remove(selected_qubit)
                
                gate_info['qubits'] = sampled_qubits
            else:
                # Random qubit assignment
                gate_info['qubits'] = np.random.choice(
                    num_qubits, size=num_gate_qubits, replace=False
                ).tolist()
            
            # Generate parameters
            if self.config.predict_parameters and 'parameter_logits' in model_outputs:
                param_values = model_outputs['parameter_logits'][0, -1, :].cpu().numpy()
                
                # Use predicted parameters based on gate requirements
                num_params = gate_properties.get('num_parameters', 0)
                gate_info['parameters'] = param_values[:num_params].tolist()
            else:
                # Random parameters for parameterized gates
                num_params = gate_properties.get('num_parameters', 0)
                if num_params > 0:
                    gate_info['parameters'] = (np.random.uniform(0, 2*np.pi, num_params)).tolist()
        
        return gate_info
    
    def evaluate_circuit(self, gates: List[Dict], requirements: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate how well a generated circuit meets the requirements
        (This is a simplified evaluation - in practice you'd use quantum simulators)
        """
        num_qubits = requirements.get('num_qubits', 4)
        
        # Basic metrics
        circuit_depth = len(gates)
        two_qubit_gates = sum(1 for gate in gates if len(gate['qubits']) >= 2)
        two_qubit_ratio = two_qubit_gates / len(gates) if gates else 0
        
        # Simplified scoring (in practice, use quantum metrics)
        depth_score = 1.0 - abs(circuit_depth - requirements.get('target_depth', 20)) / 20
        ratio_score = 1.0 - abs(two_qubit_ratio - requirements.get('two_qubit_ratio', 0.3))
        
        return {
            'circuit_depth': circuit_depth,
            'two_qubit_ratio': two_qubit_ratio,
            'depth_score': max(0, depth_score),
            'ratio_score': max(0, ratio_score),
            'overall_score': (depth_score + ratio_score) / 2
        }


def main():
    parser = argparse.ArgumentParser(description="Generate quantum circuits using trained transformer")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--requirements', type=str, required=True,
                        help='JSON string or file path with circuit requirements')
    parser.add_argument('--output_path', type=str, default='generated_circuit.json',
                        help='Output path for generated circuit')
    parser.add_argument('--max_length', type=int, default=50,
                        help='Maximum circuit length')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=None,
                        help='Top-p sampling')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of circuits to generate')
    
    args = parser.parse_args()
    
    # Load requirements
    if os.path.exists(args.requirements):
        with open(args.requirements, 'r') as f:
            requirements = json.load(f)
    else:
        requirements = json.loads(args.requirements)
    
    # Create generator
    generator = QuantumCircuitGenerator(args.model_path)
    
    # Generate circuits
    generated_circuits = []
    
    for i in range(args.num_samples):
        print(f"Generating circuit {i+1}/{args.num_samples}...")
        
        circuit = generator.generate_circuit(
            requirements=requirements,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        
        # Evaluate circuit
        evaluation = generator.evaluate_circuit(circuit, requirements)
        
        circuit_data = {
            'circuit_id': f'generated_{i}',
            'requirements': requirements,
            'gates': circuit,
            'evaluation': evaluation
        }
        
        generated_circuits.append(circuit_data)
        
        print(f"Generated circuit with {len(circuit)} gates")
        print(f"Evaluation: {evaluation}")
        print()
    
    # Save results
    output_data = {
        'requirements': requirements,
        'generation_config': {
            'max_length': args.max_length,
            'temperature': args.temperature,
            'top_k': args.top_k,
            'top_p': args.top_p
        },
        'circuits': generated_circuits
    }
    
    with open(args.output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Generated {len(generated_circuits)} circuits saved to {args.output_path}")


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) == 1:
        print("Example usage:")
        print("python inference_gate_prediction.py \\")
        print("  --model_path outputs/gate_prediction/best_model.pt \\")
        print("  --requirements '{\"target_fidelity\": 0.8, \"target_expressibility\": 0.5, \"target_entanglement\": 0.3, \"num_qubits\": 4, \"target_depth\": 20, \"two_qubit_ratio\": 0.3}' \\")
        print("  --output_path generated_circuit.json \\")
        print("  --temperature 0.8 \\")
        print("  --top_p 0.9 \\")
        print("  --num_samples 3")
    else:
        main()
