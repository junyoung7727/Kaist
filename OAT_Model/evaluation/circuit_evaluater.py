"""
Evaluation script for Quantum Circuit Generation and Property Evaluation
Comprehensive evaluation with quantum-specific metrics
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Import from src directory
from src.models.decision_transformer import DecisionTransformer
from src.models.property_prediction_transformer import PropertyPredictionTransformer, create_property_prediction_model
from src.data.quantum_circuit_dataset import CircuitSpec
from src.inference.quantum_circuit_generator import QuantumCircuitGenerator
from src.data.embedding_pipeline import EmbeddingConfig, EmbeddingPipeline

# Import quantumcommon
sys.path.append(str(Path(__file__).parent.parent.parent / "quantumcommon"))
from gates import QuantumGateRegistry, GateOperation


class QuantumCircuitEvaluator:
    """Comprehensive evaluator for quantum circuit generation"""
    
    def __init__(self, model_path: str = None, config_path: str = None, property_model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize gate registry
        self.gate_registry = QuantumGateRegistry()
        
        # Load model and config if provided
        self.model = None
        self.property_model = None
        self.config = {}
        
        if model_path:
            self.load_model(model_path, config_path)
            
        if property_model_path:
            self.load_property_model(property_model_path)
        
        print(f"Evaluator initialized on {self.device}")
    
    def load_model(self, model_path: str, config_path: str = None):
        """Load trained Decision Transformer model and configuration"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load config
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            self.config = config_dict
        else:
            # Try to get config from checkpoint
            if 'model_config' in checkpoint:
                self.config = checkpoint['model_config']
            elif 'config' in checkpoint:
                self.config = checkpoint['config']
            else:
                # Use default config values
                self.config = {
                    'd_model': 512,
                    'n_layers': 6,
                    'n_heads': 8,
                    'n_gate_types': len(self.gate_registry.get_gate_vocab()),
                    'dropout': 0.1
                }
                print("Warning: Using default config values")
        
        # Create model
        self.model = DecisionTransformer(
            d_model=self.config.get('d_model', 512),
            n_layers=self.config.get('n_layers', 6),
            n_heads=self.config.get('n_heads', 8),
            n_gate_types=self.config.get('n_gate_types', len(self.gate_registry.get_gate_vocab())),
            dropout=self.config.get('dropout', 0.1)
        )
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Warning: model_state_dict not found in checkpoint")
            
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Decision Transformer model loaded from: {model_path}")
        if 'epoch' in checkpoint:
            print(f"Trained for {checkpoint['epoch']} epochs")
        if 'best_val_loss' in checkpoint:
            print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
            
    def load_property_model(self, model_path: str):
        """Load property prediction model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model
            self.property_model = create_property_prediction_model()
            
            # Load weights
            if 'model_state_dict' in checkpoint:
                self.property_model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.property_model.load_state_dict(checkpoint['state_dict'])
            else:
                print("Warning: model_state_dict not found in property model checkpoint")
                
            self.property_model.to(self.device)
            self.property_model.eval()
            
            print(f"Property prediction model loaded from: {model_path}")
        except Exception as e:
            print(f"Failed to load property model: {e}")
            self.property_model = None
    
    def generate_circuits(self, 
                         num_samples: int = 10,
                         target_metrics: Optional[Dict[str, float]] = None,
                         max_length: int = 20,
                         target_num_qubits: int = 4) -> List[CircuitSpec]:
        """Generate quantum circuits using Decision Transformer
        
        Args:
            num_samples: Number of circuits to generate
            target_metrics: Target metrics for circuit generation (fidelity, entanglement, expressibility)
            max_length: Maximum circuit length
            target_num_qubits: Number of qubits in the generated circuits
            
        Returns:
            List of generated circuit specifications
        """
        print("[INFO] Generating circuits using Decision Transformer")
        
        if self.model is None:
            print("[WARNING] No model loaded. Cannot generate circuits.")
            return []
            
        # Initialize circuit generator
        from src.inference.quantum_circuit_generator import GenerationConfig
        
        config = GenerationConfig(
            max_circuit_length=max_length,
            target_num_qubits=target_num_qubits,
            temperature=0.8,
            top_k=5,
            top_p=0.9
        )
        
        # If target metrics provided, add them to config
        if target_metrics:
            if 'fidelity' in target_metrics:
                config.target_fidelity = target_metrics['fidelity']
            if 'entanglement' in target_metrics:
                config.target_entanglement = target_metrics['entanglement']
            if 'expressibility' in target_metrics:
                config.target_expressibility = target_metrics['expressibility']
        
        # Create generator using our loaded model
        generator = QuantumCircuitGenerator(
            model_path=None,  # We'll provide our loaded model
            config=config
        )
        
        # Replace the generator's model with our loaded model
        generator.model = self.model
        
        generated_circuits = []
        
        for i in tqdm(range(num_samples), desc="Generating circuits"):
            try:
                circuit = generator.generate_circuit(target_metrics=target_metrics)
                generated_circuits.append(circuit)
            except Exception as e:
                print(f"Error generating circuit {i}: {e}")
        
        print(f"Generated {len(generated_circuits)} circuits successfully")
        return generated_circuits
    
    def circuit_spec_to_dict(self, circuit: CircuitSpec) -> Dict[str, Any]:
        """Convert CircuitSpec to a serializable dictionary"""
        gate_operations = []
        
        for gate in circuit.gates:
            gate_dict = {
                'gate_type': gate.gate_type,
                'qubits': gate.qubits,
                'parameters': gate.parameters if hasattr(gate, 'parameters') else []
            }
            gate_operations.append(gate_dict)
            
        return {
            'circuit_id': circuit.circuit_id if hasattr(circuit, 'circuit_id') else 'unknown',
            'num_qubits': circuit.num_qubits,
            'gate_operations': gate_operations,
            'gate_count': len(gate_operations),
            'depth': getattr(circuit, 'depth', None)
        }
        
    def evaluate_circuit_properties(self, circuit: Union[CircuitSpec, Dict], property_model=None) -> Dict[str, float]:
        """Evaluate circuit properties using a property prediction model
        
        Args:
            circuit: Circuit specification as CircuitSpec or dictionary
            property_model: Property prediction model (optional, uses self.property_model if None)
            
        Returns:
            Dictionary of property predictions
        """
        if property_model is None:
            property_model = self.property_model
            
        if property_model is None:
            print("[WARNING] No property prediction model available")
            return {}
        
        try:
            # Convert dictionary to CircuitSpec if needed
            if isinstance(circuit, dict):
                gate_operations = []
                for gate in circuit.get('gate_operations', []):
                    gate_op = GateOperation(
                        gate_type=gate.get('gate_type'),
                        qubits=gate.get('qubits', []),
                        parameters=gate.get('parameters', [])
                    )
                    gate_operations.append(gate_op)
                
                circuit_spec = CircuitSpec(
                    circuit_id=circuit.get('circuit_id', 'unknown'),
                    num_qubits=circuit.get('num_qubits', 4),
                    gate_operations=gate_operations
                )
            else:
                circuit_spec = circuit
            
            # Forward pass through property model
            with torch.no_grad():
                property_model.eval()
                predictions = property_model(circuit_spec)
            
            # Convert tensor outputs to Python floats
            result = {}
            for key, value in predictions.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.item() if value.numel() == 1 else value.tolist()
                else:
                    result[key] = value
            
            return result
        
        except Exception as e:
            print(f"[WARNING] Error evaluating circuit properties: {e}")
            return {}
    
    def calculate_circuit_metrics(self, circuit_spec: CircuitSpec) -> Dict[str, float]:
        """Calculate basic circuit metrics without using property prediction model
        
        Args:
            circuit_spec: CircuitSpec object
            
        Returns:
            Dictionary of circuit metrics
        """
        # Basic metrics that don't require property prediction
        metrics = {
            'num_qubits': circuit_spec.num_qubits,
            'gate_count': len(circuit_spec.gates),
        }
        
        # Calculate circuit depth
        if not hasattr(circuit_spec, 'depth') or circuit_spec.depth is None:
            # Simple depth calculation (can be improved)
            depth = 0
            qubit_last_layer = [-1] * circuit_spec.num_qubits
            
            for gate in circuit_spec.gates:
                max_prev_layer = -1
                for q in gate.qubits:
                    if q < len(qubit_last_layer):
                        max_prev_layer = max(max_prev_layer, qubit_last_layer[q])
                
                current_layer = max_prev_layer + 1
                for q in gate.qubits:
                    if q < len(qubit_last_layer):
                        qubit_last_layer[q] = current_layer
                        
                depth = max(depth, current_layer + 1)
                
            metrics['depth'] = depth
        else:
            metrics['depth'] = circuit_spec.depth
            
        # Calculate gate type diversity
        gate_types = set(gate.gate_type for gate in circuit_spec.gates)
        metrics['gate_diversity'] = len(gate_types)
        
        # Calculate 2-qubit gate ratio
        two_qubit_gates = sum(1 for gate in circuit_spec.gates if len(gate.qubits) > 1)
        if metrics['gate_count'] > 0:
            metrics['two_qubit_ratio'] = two_qubit_gates / metrics['gate_count']
        else:
            metrics['two_qubit_ratio'] = 0.0
            
        return metrics
    
    def infer_qubits(self, gate_indices: torch.Tensor) -> int:
        """Infer number of qubits from gate sequence"""
        # Simple heuristic - can be improved based on gate registry
        return min(8, max(2, len(gate_indices) // 4))
    
    def evaluate_circuits(self, circuits: List[CircuitSpec]) -> Dict[str, List[Dict[str, float]]]:
        """Evaluate a list of circuits using both basic metrics and property prediction
        
        Args:
            circuits: List of CircuitSpec objects
            
        Returns:
            Dictionary with circuit evaluations
        """
        results = []
        
        for i, circuit in enumerate(tqdm(circuits, desc="Evaluating circuits")):
            try:
                # Calculate basic circuit metrics
                metrics = self.calculate_circuit_metrics(circuit)
                
                # Add property predictions if model available
                if self.property_model is not None:
                    property_predictions = self.evaluate_circuit_properties(circuit)
                    metrics.update(property_predictions)
                
                # Add circuit identifier
                metrics['circuit_id'] = getattr(circuit, 'circuit_id', f'circuit_{i}')
                results.append(metrics)
            except Exception as e:
                print(f"[WARNING] Error evaluating circuit {i}: {e}")
                
        return {
            'circuit_evaluations': results
        }
    
    def check_circuit_validity(self, circuit: CircuitSpec) -> float:
        """Check if circuit is valid (simplified)
        
        Args:
            circuit: CircuitSpec object to check
            
        Returns:
            Validity score between 0.0 and 1.0
        """
        # Basic validity checks
        if len(circuit.gates) == 0:
            return 0.0
        
        if circuit.num_qubits < 1:
            return 0.0
        
        # Check for reasonable gate distribution
        gate_counts = {}
        for gate in circuit.gates:
            gate_counts[gate.gate_type] = gate_counts.get(gate.gate_type, 0) + 1
        
        # Penalize circuits with too much repetition of the same gate
        total_gates = len(circuit.gates)
        if total_gates > 0:
            max_repetition = max(gate_counts.values()) / total_gates
            if max_repetition > 0.8:
                return 0.5
        
        # Check if qubits are used reasonably (no unused qubits)
        used_qubits = set()
        for gate in circuit.gates:
            for qubit in gate.qubits:
                used_qubits.add(qubit)
        
        qubit_usage_ratio = len(used_qubits) / circuit.num_qubits if circuit.num_qubits > 0 else 0
        if qubit_usage_ratio < 0.5:  # Less than half of the qubits are used
            return 0.7
        
        return 1.0
    
    def evaluate_quantum_properties(self, circuits: List[CircuitSpec]) -> Dict[str, float]:
        """Evaluate quantum properties for a list of circuits
        
        Args:
            circuits: List of CircuitSpec objects
            
        Returns:
            Dictionary with aggregate statistics (mean, std, min, max) for all metrics
        """
        # Evaluate all circuits
        evaluations_result = self.evaluate_circuits(circuits)
        circuit_evaluations = evaluations_result.get('circuit_evaluations', [])
        
        if not circuit_evaluations:
            return {}
            
        # Extract all metrics
        all_metrics = {}
        for eval_dict in circuit_evaluations:
            for key, value in eval_dict.items():
                if isinstance(value, (int, float)):
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
        
        # Calculate statistics for each metric
        stats = {}
        for key, values in all_metrics.items():
            if len(values) > 0:
                stats[f"{key}_mean"] = float(np.mean(values))
                stats[f"{key}_std"] = float(np.std(values))
                stats[f"{key}_min"] = float(np.min(values))
                stats[f"{key}_max"] = float(np.max(values))
        
        return stats
    
    def compare_with_baseline(self, 
                            generated_circuits: List[CircuitSpec],
                            baseline_circuits: List[CircuitSpec]) -> Dict[str, float]:
        """Compare generated circuits with baseline/reference circuits
        
        Args:
            generated_circuits: List of generated CircuitSpec objects
            baseline_circuits: List of baseline CircuitSpec objects
            
        Returns:
            Dictionary with comparison metrics
        """
        gen_metrics = self.evaluate_quantum_properties(generated_circuits)
        baseline_metrics = self.evaluate_quantum_properties(baseline_circuits)
        
        comparison = {}
        
        # Compare all available metrics
        for key in gen_metrics:
            if key in baseline_metrics:
                gen_val = gen_metrics[key]
                baseline_val = baseline_metrics[key]
                
                # Store absolute values
                comparison[f'{key}_generated'] = gen_val
                comparison[f'{key}_baseline'] = baseline_val
                
                # Compute relative difference if applicable
                if isinstance(baseline_val, (int, float)) and baseline_val != 0:
                    rel_diff = (gen_val - baseline_val) / abs(baseline_val)
                    comparison[f'{key}_relative_diff'] = rel_diff
        
        return comparison
    
    def run_comprehensive_evaluation(self, 
                                   test_data_path: str,
                                   num_generated: int = 1000,
                                   save_results: bool = True,
                                   output_dir: str = "./output") -> Dict[str, Any]:
        """Run comprehensive evaluation using Decision Transformer model
        
        Args:
            test_data_path: Path to test data file
            num_generated: Number of circuits to generate
            save_results: Whether to save results to file
            output_dir: Directory to save results in
            
        Returns:
            Dictionary with evaluation results
        """
        print("Starting comprehensive evaluation...")
        
        # Load test data
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        # Convert test data to CircuitSpec objects if needed
        test_circuits = []
        for circuit_data in test_data['circuits'][:100]:  # Limit to 100 test circuits
            try:
                if isinstance(circuit_data, dict) and not isinstance(circuit_data, CircuitSpec):
                    # Convert dictionary to CircuitSpec
                    circuit_spec = CircuitSpec(
                        num_qubits=circuit_data.get('qubits', 0),
                        gate_operations=[]
                    )
                    
                    # Add gates if available in the old format
                    if 'gates' in circuit_data:
                        for gate_info in circuit_data['gates']:
                            if isinstance(gate_info, dict):
                                # Add gate from dictionary format
                                gate_op = GateOperation(
                                    gate_type=gate_info.get('type', 0),
                                    qubits=gate_info.get('qubits', []),
                                    parameters=gate_info.get('params', [])
                                )
                                circuit_spec.gates.append(gate_op)
                    
                    test_circuits.append(circuit_spec)
                elif isinstance(circuit_data, CircuitSpec):
                    test_circuits.append(circuit_data)
            except Exception as e:
                print(f"Warning: Error converting circuit: {e}")
        
        # Generate new circuits
        print("Generating new circuits...")
        generated_circuits = self.generate_circuits(
            num_samples=num_generated,
            target_metrics={
                'fidelity': 0.8,
                'expressibility': 0.5,
                'entanglement': 0.6
            }
        )
        
        # Evaluate quantum properties
        print("Evaluating quantum properties...")
        gen_quantum_props = self.evaluate_quantum_properties(generated_circuits)
        baseline_quantum_props = self.evaluate_quantum_properties(test_circuits)
        
        # Compare with baseline
        comparison = self.compare_with_baseline(generated_circuits, test_circuits)
        
        # Compile results
        results = {
            'generated_properties': gen_quantum_props,
            'baseline_properties': baseline_quantum_props,
            'comparison': comparison,
            'num_circuits': {
                'generated': len(generated_circuits),
                'baseline': len(test_circuits)
            }
        }
        
        # Save results if requested
        if save_results:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
            circuits_file = os.path.join(output_dir, f"generated_circuits_{timestamp}.json")
            
            # Save evaluation results
            with open(results_file, 'w') as f:
                # Don't include actual circuits in the main results file to keep it small
                save_results = results.copy()
                json.dump(save_results, f, indent=2)
            print(f"Results saved to {results_file}")
                
            # Save generated circuits separately (need to convert to serializable format)
            try:
                serialized_circuits = []
                for i, circuit in enumerate(generated_circuits):
                    # Convert CircuitSpec to serializable dictionary
                    circuit_dict = {
                        'circuit_id': getattr(circuit, 'circuit_id', f'circuit_{i}'),
                        'num_qubits': circuit.num_qubits,
                        'gates': [
                            {
                                'type': gate.gate_type,
                                'qubits': gate.qubits,
                                'params': gate.parameters if hasattr(gate, 'parameters') else []
                            } for gate in circuit.gates
                        ]
                    }
                    serialized_circuits.append(circuit_dict)
                    
                with open(circuits_file, 'w') as f:
                    json.dump({
                        'generated_circuits': serialized_circuits
                    }, f, indent=2)
                print(f"Generated circuits saved to {circuits_file}")
                
            except Exception as e:
                print(f"Error saving generated circuits: {e}")
        
        print("Evaluation complete!")
        return results
    
    def save_evaluation_results(self, 
                              results: Dict[str, Any], 
                              generated_circuits: List[CircuitSpec],
                              output_dir: str = "evaluation_results"):
        """Save evaluation results and generated circuits
        
        Args:
            results: Dictionary with evaluation results
            generated_circuits: List of generated CircuitSpec objects
            output_dir: Directory to save results in
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save generated circuits (need to convert to serializable format)
        serialized_circuits = []
        for i, circuit in enumerate(generated_circuits):
            # Convert CircuitSpec to serializable dictionary
            circuit_dict = {
                'circuit_id': getattr(circuit, 'circuit_id', f'circuit_{i}'),
                'num_qubits': circuit.num_qubits,
                'gates': [
                    {
                        'type': gate.gate_type,
                        'qubits': gate.qubits,
                        'params': gate.parameters if hasattr(gate, 'parameters') else []
                    } for gate in circuit.gates
                ]
            }
            serialized_circuits.append(circuit_dict)
            
        # Save the serialized circuits
        with open(os.path.join(output_dir, "generated_circuits.json"), 'w') as f:
            json.dump({
                'circuits': serialized_circuits
            }, f, indent=2)  # Save first 100
        
        print("Evaluation results saved to evaluation_results/")


def visualize_training_history(loss_history_path: str, save_path: str = None):
    """Visualize training and validation loss curves"""
    with open(loss_history_path, 'r') as f:
        history = json.load(f)
    
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot training loss by epoch
    train_epochs = [x['epoch'] for x in train_losses]
    train_loss_values = [x['loss'] for x in train_losses]
    
    ax1.plot(train_epochs, train_loss_values, 'b-', label='Training Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Epochs')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot validation loss by step
    val_steps = [x['step'] for x in val_losses]
    val_loss_values = [x['loss'] for x in val_losses]
    
    ax2.plot(val_steps, val_loss_values, 'r-', label='Validation Loss', linewidth=2)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss Over Training Steps')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate DiT Quantum Circuit Model")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to trained model checkpoint")
    parser.add_argument("--test_data", type=str, required=True,
                       help="Path to test dataset")
    parser.add_argument("--config_path", type=str, 
                       help="Path to model config file")
    parser.add_argument("--num_generated", type=int, default=1000,
                       help="Number of circuits to generate")
    parser.add_argument("--visualize_training", type=str,
                       help="Path to loss history JSON for visualization")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = QuantumCircuitEvaluator(args.model_path, args.config_path)
    results = evaluator.run_comprehensive_evaluation(
        args.test_data, 
        args.num_generated
    )
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    for category, metrics in results.items():
        print(f"\n{category.upper()}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    # Visualize training history if provided
    if args.visualize_training:
        visualize_training_history(
            args.visualize_training,
            "evaluation_results/training_curves.png"
        )


if __name__ == "__main__":
    main()
