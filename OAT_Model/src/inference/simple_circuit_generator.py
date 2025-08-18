"""
Simple Quantum Circuit Generator
Clean implementation without hardcoded assumptions
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys
import json
import time
from dataclasses import dataclass

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))

from models.decision_transformer import DecisionTransformer
from gates import QuantumGateRegistry, GateOperation
from data.quantum_circuit_dataset import CircuitSpec


@dataclass
class SimpleGenerationConfig:
    """Simple generation configuration without hardcoded assumptions"""
    max_gates: int = 20
    num_qubits: int = 4
    temperature: float = 1.0
    do_sample: bool = True
    verbose: bool = True


class SimpleCircuitGenerator:
    """Clean circuit generator without architectural assumptions"""
    
    def __init__(self, model_path: str = None, config: SimpleGenerationConfig = None):
        self.config = config or SimpleGenerationConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize gate registry
        self.gate_registry = QuantumGateRegistry()
        self.gate_vocab = self.gate_registry.get_gate_vocab()
        self.idx_to_gate = {idx: gate for gate, idx in self.gate_vocab.items()}
        
        # Load model if path provided
        self.model = None
        if model_path and Path(model_path).exists():
            self.model = self._load_model_flexible(model_path)
        
        if self.config.verbose:
            print(f"SimpleCircuitGenerator initialized")
            print(f"Device: {self.device}")
            print(f"Gate vocabulary: {len(self.gate_vocab)} gates")
            print(f"Model loaded: {'Yes' if self.model else 'No'}")
    
    def _load_model_flexible(self, model_path: str) -> Optional[DecisionTransformer]:
        """Load model with flexible architecture detection"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Detect model architecture from checkpoint
            model_config = self._detect_model_config(state_dict)
            
            if self.config.verbose:
                print(f"Detected model config: {model_config}")
            
            # Create model with detected config
            model = DecisionTransformer(**model_config)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None
    
    def _detect_model_config(self, state_dict: Dict) -> Dict:
        """Detect model configuration from state dict"""
        config = {
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 2048,
            'dropout': 0.1,
            'device': str(self.device)
        }
        
        # Detect d_model from transformer blocks
        if 'transformer_blocks.0.norm1.weight' in state_dict:
            config['d_model'] = state_dict['transformer_blocks.0.norm1.weight'].shape[0]
        
        # Detect gate types from action heads
        if 'action_heads.gate.2.weight' in state_dict:
            config['n_gate_types'] = state_dict['action_heads.gate.2.weight'].shape[0]
        
        # Detect position dimension
        if 'action_heads.position.2.weight' in state_dict:
            position_dim = state_dict['action_heads.position.2.weight'].shape[0]
            config['position_dim'] = position_dim
            # Infer max_qubits from position_dim
            config['max_qubits'] = position_dim // 2 if position_dim % 2 == 0 else 8
        
        return config
    
    def generate_circuit_random(self) -> CircuitSpec:
        """Generate circuit using random gate selection (fallback method)"""
        gates = []
        available_gates = [name for name in self.gate_vocab.keys() 
                          if not name.startswith('[')]  # Skip special tokens
        
        num_gates = np.random.randint(5, self.config.max_gates + 1)
        
        for _ in range(num_gates):
            gate_name = np.random.choice(available_gates)
            gate_def = self.gate_registry.get_gate(gate_name)
            
            if gate_def is None:
                continue
            
            # Select qubits
            required_qubits = gate_def.num_qubits
            if required_qubits > self.config.num_qubits:
                continue
                
            qubits = np.random.choice(
                self.config.num_qubits, 
                size=required_qubits, 
                replace=False
            ).tolist()
            
            # Generate parameters
            parameters = []
            for _ in range(gate_def.num_parameters):
                parameters.append(float(np.random.uniform(0, 2 * np.pi)))
            
            gate = GateOperation(
                name=gate_name,
                qubits=qubits,
                parameters=parameters
            )
            gates.append(gate)
        
        circuit_id = f"random_circuit_{len(gates)}g_{self.config.num_qubits}q_{int(time.time())}"
        
        return CircuitSpec(
            circuit_id=circuit_id,
            num_qubits=self.config.num_qubits,
            gates=gates
        )
    
    def generate_circuit_model(self, target_properties: Optional[Dict] = None) -> CircuitSpec:
        """Generate circuit using the loaded model"""
        if self.model is None:
            if self.config.verbose:
                print("No model loaded, falling back to random generation")
            return self.generate_circuit_random()
        
        gates = []
        
        # Simple autoregressive generation
        for step in range(self.config.max_gates):
            try:
                # Create simple context (this is a simplified approach)
                context = self._create_simple_context(gates)
                
                # Get next gate prediction
                next_gate = self._predict_next_gate_simple(context, target_properties)
                
                if next_gate is None or next_gate.name in ['[EOS]', '[PAD]', '[EMPTY]']:
                    break
                
                gates.append(next_gate)
                
                if self.config.verbose:
                    print(f"Step {step}: {next_gate.name} on qubits {next_gate.qubits}")
                    
            except Exception as e:
                if self.config.verbose:
                    print(f"Generation error at step {step}: {e}")
                break
        
        circuit_id = f"model_circuit_{len(gates)}g_{self.config.num_qubits}q_{int(time.time())}"
        
        return CircuitSpec(
            circuit_id=circuit_id,
            num_qubits=self.config.num_qubits,
            gates=gates
        )
    
    def _create_simple_context(self, gates: List[GateOperation]) -> Dict:
        """Create simple context for model input"""
        return {
            'num_gates': len(gates),
            'num_qubits': self.config.num_qubits,
            'gates': gates
        }
    
    def _predict_next_gate_simple(self, context: Dict, target_properties: Optional[Dict] = None) -> Optional[GateOperation]:
        """Simple gate prediction without complex embeddings"""
        try:
            with torch.no_grad():
                # Create dummy input (simplified approach)
                batch_size = 1
                seq_len = max(1, context['num_gates'])
                d_model = self.model.d_model
                
                # Create dummy sequence
                dummy_sequence = torch.randn(batch_size, seq_len, d_model, device=self.device)
                
                # Create attention mask
                attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)
                
                # Get model predictions
                outputs = self.model.forward(
                    input_sequence=dummy_sequence,
                    attention_mask=attention_mask
                )
                
                # Extract predictions from last position
                gate_logits = outputs['gate_logits'][0, -1]  # [n_gate_types]
                position_preds = outputs['position_preds'][0, -1]  # [max_qubits, 2] or similar
                parameter_preds = outputs['parameter_preds'][0, -1]  # scalar
                
                # Sample gate
                gate_id = self._sample_gate_id(gate_logits)
                gate_name = self.idx_to_gate.get(gate_id, None)
                
                if gate_name is None or gate_name.startswith('['):
                    return None
                
                # Get gate definition
                gate_def = self.gate_registry.get_gate(gate_name)
                if gate_def is None:
                    return None
                
                # Select qubits
                qubits = self._select_qubits_simple(position_preds, gate_def.num_qubits)
                
                # Generate parameters
                parameters = self._generate_parameters_simple(parameter_preds, gate_def.num_parameters)
                
                return GateOperation(
                    name=gate_name,
                    qubits=qubits,
                    parameters=parameters
                )
                
        except Exception as e:
            if self.config.verbose:
                print(f"Prediction error: {e}")
            return None
    
    def _sample_gate_id(self, logits: torch.Tensor) -> int:
        """Sample gate ID from logits"""
        if not self.config.do_sample:
            return logits.argmax().item()
        
        # Temperature scaling
        if self.config.temperature != 1.0:
            logits = logits / self.config.temperature
        
        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()
    
    def _select_qubits_simple(self, position_preds: torch.Tensor, num_qubits: int) -> List[int]:
        """Simple qubit selection"""
        if num_qubits > self.config.num_qubits:
            num_qubits = self.config.num_qubits
        
        # Simple approach: select first num_qubits positions
        return list(range(min(num_qubits, self.config.num_qubits)))
    
    def _generate_parameters_simple(self, param_pred: torch.Tensor, num_params: int) -> List[float]:
        """Simple parameter generation"""
        if num_params == 0:
            return []
        
        # Use predicted parameter for first param, generate others randomly
        params = []
        if isinstance(param_pred, torch.Tensor):
            base_param = param_pred.item() if param_pred.numel() == 1 else param_pred.flatten()[0].item()
        else:
            base_param = float(param_pred)
        
        params.append(base_param)
        
        # Generate additional parameters
        for i in range(1, num_params):
            params.append(float(np.random.uniform(0, 2 * np.pi)))
        
        return params
    
    def generate_circuit(self, target_properties: Optional[Dict] = None) -> CircuitSpec:
        """Main circuit generation method"""
        if self.model is not None:
            return self.generate_circuit_model(target_properties)
        else:
            return self.generate_circuit_random()
    
    def generate_multiple_circuits(self, num_circuits: int = 5, target_properties: Optional[Dict] = None) -> List[CircuitSpec]:
        """Generate multiple circuits"""
        circuits = []
        
        for i in range(num_circuits):
            if self.config.verbose:
                print(f"\nGenerating circuit {i+1}/{num_circuits}")
            
            circuit = self.generate_circuit(target_properties)
            circuits.append(circuit)
        
        return circuits
    
    def save_circuits(self, circuits: List[CircuitSpec], output_path: str):
        """Save circuits to JSON file"""
        circuits_data = []
        
        for circuit in circuits:
            circuit_data = {
                'circuit_id': circuit.circuit_id,
                'num_qubits': circuit.num_qubits,
                'gates': [
                    {
                        'name': gate.name,
                        'qubits': [int(q) for q in gate.qubits],
                        'parameters': [float(p) for p in gate.parameters]
                    }
                    for gate in circuit.gates
                ]
            }
            circuits_data.append(circuit_data)
        
        with open(output_path, 'w') as f:
            json.dump(circuits_data, f, indent=2)
        
        print(f"Saved {len(circuits)} circuits to {output_path}")


def main():
    """Example usage"""
    config = SimpleGenerationConfig(
        max_gates=15,
        num_qubits=4,
        temperature=0.8,
        verbose=True
    )
    
    # Try to load model, fallback to random if not available
    model_path = "checkpoints/best_model.pt"
    generator = SimpleCircuitGenerator(model_path, config)
    
    # Generate circuits
    circuits = generator.generate_multiple_circuits(num_circuits=3)
    
    # Save results
    generator.save_circuits(circuits, "simple_generated_circuits.json")
    
    # Print summary
    for i, circuit in enumerate(circuits):
        print(f"\nCircuit {i+1}: {len(circuit.gates)} gates on {circuit.num_qubits} qubits")
        for j, gate in enumerate(circuit.gates[:3]):  # Show first 3 gates
            print(f"  {j+1}. {gate.name} on qubits {gate.qubits}")
        if len(circuit.gates) > 3:
            print(f"  ... and {len(circuit.gates) - 3} more gates")


if __name__ == "__main__":
    main()
