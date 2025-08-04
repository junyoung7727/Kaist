"""
Evaluation script for DiT Quantum Circuit Generation Model
Comprehensive evaluation with quantum-specific metrics
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from quantum_dit.models.dit_model import QuantumDiT, DiTConfig, create_dit_model
from quantum_dit.data.quantum_dataset import QuantumCircuitDataset
from quantum_dit.models.diffusion import DiffusionScheduler
from quantum_dit.utils.metrics import QuantumCircuitMetrics
from quantum_dit.utils.visualization import plot_training_curves, plot_quantum_metrics


class QuantumCircuitEvaluator:
    """Comprehensive evaluator for quantum circuit generation"""
    
    def __init__(self, model_path: str, config_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and config
        self.load_model(model_path, config_path)
        
        # Setup diffusion scheduler
        self.diffusion = DiffusionScheduler(
            timesteps=self.config.timesteps,
            noise_schedule=self.config.noise_schedule
        )
        
        # Setup metrics
        self.metrics = QuantumCircuitMetrics()
        
        print(f"Evaluator initialized on {self.device}")
        print(f"Model parameters: {self.model.get_num_params():,}")
    
    def load_model(self, model_path: str, config_path: str = None):
        """Load trained model and configuration"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Load config
        if config_path:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            # Extract model_config from the JSON structure
            model_config = config_dict.get('model_config', config_dict)
            # Keep diffusion mode disabled for existing checkpoints
            model_config['diffusion_mode'] = False
            self.config = DiTConfig(**model_config)
        else:
            # Try to get config from checkpoint
            if 'config' in checkpoint:
                model_config = checkpoint['config']['model_config']
                # Keep diffusion mode disabled for existing checkpoints
                model_config['diffusion_mode'] = False
                self.config = DiTConfig(**model_config)
            else:
                # Use default config
                self.config = DiTConfig(diffusion_mode=False)
                print("Warning: Using default config")
        
        # Create and load model with correct target count
        # Infer target count from checkpoint
        if 'model_state_dict' in checkpoint:
            # Check the final layer size to determine number of targets
            final_layer_key = 'property_predictor.3.weight'
            if final_layer_key in checkpoint['model_state_dict']:
                num_targets = checkpoint['model_state_dict'][final_layer_key].shape[0]
            else:
                num_targets = 2  # Default
        else:
            num_targets = 2  # Default
        
        print(f"Creating model with {num_targets} targets")
        self.model = create_dit_model(
            self.config,
            num_targets=num_targets,
            target_names=['expressibility', 'two_qubit_ratio'][:num_targets]
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from: {model_path}")
        if 'epoch' in checkpoint:
            print(f"Trained for {checkpoint['epoch']} epochs")
        if 'best_val_loss' in checkpoint:
            print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    def generate_circuits(self, 
                         num_samples: int = 100,
                         max_length: int = None,
                         guidance_scale: float = 1.0) -> List[Dict[str, Any]]:
        """Simulate circuit generation using property prediction model
        
        Note: This is NOT true diffusion-based generation since the model
        was trained for property prediction. For real generation, train
        a diffusion model using train_diffusion_dit.py
        """
        print("⚠️  WARNING: Using property prediction model for 'generation'")
        print("   This is NOT true diffusion-based generation!")
        print("   For real generation, train a diffusion model first.")
        
        if max_length is None:
            max_length = self.config.max_circuit_length
        
        generated_circuits = []
        
        with torch.no_grad():
            for i in tqdm(range(num_samples), desc="Simulating circuits"):
                # Generate random circuit parameters
                num_qubits = torch.randint(4, 16, (1,), device=self.device)
                gate_count = torch.randint(10, 50, (1,), device=self.device)
                depth = torch.randint(5, 20, (1,), device=self.device)
                two_qubit_ratio = torch.rand(1, device=self.device)
                
                # Generate random gate sequence
                gates = torch.randint(0, self.model.vocab_size, (1, max_length), device=self.device)
                
                # Predict properties using the model
                properties = self.model(
                    gates=gates,
                    num_qubits=num_qubits,
                    gate_count=gate_count,
                    depth=depth,
                    two_qubit_ratio=two_qubit_ratio
                )
                
                # Create circuit representation
                circuit = {
                    'circuit_id': f'simulated_{i}',
                    'gates': gates.cpu().numpy().flatten().tolist(),
                    'num_qubits': num_qubits.item(),
                    'gate_count': gate_count.item(),
                    'depth': depth.item(),
                    'two_qubit_ratio': two_qubit_ratio.item(),
                    'predicted_expressibility': properties[0, 0].item() if properties.shape[1] > 0 else 0.0,
                    'predicted_two_qubit_ratio': properties[0, 1].item() if properties.shape[1] > 1 else 0.0,
                    'generation_method': 'property_prediction_simulation',
                    # Add fields expected by evaluation code
                    'length': len(gates.cpu().numpy().flatten().tolist()),
                    'qubits': num_qubits.item()
                }
                
                generated_circuits.append(circuit)
        
        return generated_circuits
    
    def postprocess_circuit(self, logits: torch.Tensor) -> Dict[str, Any]:
        """Convert model output to discrete circuit representation"""
        # Apply softmax and sample
        probs = F.softmax(logits, dim=-1)
        gate_indices = torch.multinomial(probs, 1).squeeze(-1)
        
        # Convert to circuit format
        circuit = {
            'gates': gate_indices.cpu().numpy().tolist(),
            'length': len(gate_indices),
            'qubits': self.infer_qubits(gate_indices)
        }
        
        return circuit
    
    def infer_qubits(self, gate_indices: torch.Tensor) -> int:
        """Infer number of qubits from gate sequence"""
        # Simple heuristic - can be improved based on gate registry
        return min(8, max(2, len(gate_indices) // 4))
    
    def evaluate_reconstruction(self, dataset_path: str) -> Dict[str, float]:
        """Evaluate reconstruction quality on test dataset"""
        dataset = QuantumCircuitDataset(dataset_path)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=False
        )
        
        reconstruction_losses = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating reconstruction"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Add noise and reconstruct
                timesteps = torch.randint(0, self.config.timesteps, 
                                        (len(batch['circuit']),), device=self.device)
                noise = torch.randn_like(batch['circuit'])
                noisy_circuits = self.diffusion.add_noise(batch['circuit'], noise, timesteps)
                
                # Predict noise
                predicted_noise = self.model(noisy_circuits, timesteps)
                
                # Compute loss
                loss = F.mse_loss(predicted_noise, noise)
                reconstruction_losses.append(loss.item())
        
        return {
            'reconstruction_loss': np.mean(reconstruction_losses),
            'reconstruction_std': np.std(reconstruction_losses)
        }
    
    def evaluate_quantum_properties(self, circuits: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate quantum-specific properties of generated circuits"""
        metrics = {
            'avg_depth': [],
            'avg_gates': [],
            'gate_diversity': [],
            'entanglement_capability': [],
            'circuit_validity': []
        }
        
        for circuit in circuits:
            # Circuit depth (simplified)
            depth = self.estimate_circuit_depth(circuit)
            metrics['avg_depth'].append(depth)
            
            # Number of gates
            metrics['avg_gates'].append(circuit['length'])
            
            # Gate diversity (unique gates / total gates)
            unique_gates = len(set(circuit['gates']))
            diversity = unique_gates / max(1, circuit['length'])
            metrics['gate_diversity'].append(diversity)
            
            # Entanglement capability (heuristic)
            entanglement = self.estimate_entanglement_capability(circuit)
            metrics['entanglement_capability'].append(entanglement)
            
            # Circuit validity
            validity = self.check_circuit_validity(circuit)
            metrics['circuit_validity'].append(validity)
        
        # Compute statistics
        results = {}
        for key, values in metrics.items():
            results[f'{key}_mean'] = np.mean(values)
            results[f'{key}_std'] = np.std(values)
        
        return results
    
    def estimate_circuit_depth(self, circuit: Dict[str, Any]) -> int:
        """Estimate circuit depth (simplified)"""
        # This is a simplified estimation
        # In practice, you'd need to consider gate dependencies
        return max(1, circuit['length'] // circuit['qubits'])
    
    def estimate_entanglement_capability(self, circuit: Dict[str, Any]) -> float:
        """Estimate entanglement capability (heuristic)"""
        # Count two-qubit gates as proxy for entanglement
        two_qubit_gates = 0
        for gate_idx in circuit['gates']:
            # Assume gates > threshold are two-qubit gates
            if gate_idx > 10:  # Simplified heuristic
                two_qubit_gates += 1
        
        return two_qubit_gates / max(1, circuit['length'])
    
    def check_circuit_validity(self, circuit: Dict[str, Any]) -> float:
        """Check if circuit is valid (simplified)"""
        # Basic validity checks
        if circuit['length'] == 0:
            return 0.0
        
        if circuit['qubits'] < 1:
            return 0.0
        
        # Check for reasonable gate distribution
        gate_counts = {}
        for gate in circuit['gates']:
            gate_counts[gate] = gate_counts.get(gate, 0) + 1
        
        # Penalize circuits with too much repetition
        max_repetition = max(gate_counts.values()) / circuit['length']
        if max_repetition > 0.8:
            return 0.5
        
        return 1.0
    
    def compare_with_baseline(self, 
                            generated_circuits: List[Dict[str, Any]],
                            baseline_circuits: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compare generated circuits with baseline/reference circuits"""
        gen_metrics = self.evaluate_quantum_properties(generated_circuits)
        baseline_metrics = self.evaluate_quantum_properties(baseline_circuits)
        
        comparison = {}
        for key in gen_metrics:
            if key.endswith('_mean'):
                base_key = key
                gen_val = gen_metrics[key]
                baseline_val = baseline_metrics[key]
                
                # Compute relative difference
                if baseline_val != 0:
                    rel_diff = (gen_val - baseline_val) / abs(baseline_val)
                    comparison[f'{base_key}_relative_diff'] = rel_diff
                
                comparison[f'{base_key}_generated'] = gen_val
                comparison[f'{base_key}_baseline'] = baseline_val
        
        return comparison
    
    def run_comprehensive_evaluation(self, 
                                   test_data_path: str,
                                   num_generated: int = 1000,
                                   save_results: bool = True) -> Dict[str, Any]:
        """Run comprehensive evaluation"""
        print("Starting comprehensive evaluation...")
        
        results = {}
        
        # 1. Reconstruction evaluation
        print("Evaluating reconstruction quality...")
        reconstruction_metrics = self.evaluate_reconstruction(test_data_path)
        results['reconstruction'] = reconstruction_metrics
        
        # 2. Generation evaluation
        print(f"Generating {num_generated} circuits...")
        generated_circuits = self.generate_circuits(num_generated)
        
        # 3. Quantum properties evaluation
        print("Evaluating quantum properties...")
        quantum_metrics = self.evaluate_quantum_properties(generated_circuits)
        results['quantum_properties'] = quantum_metrics
        
        # 4. Load baseline circuits for comparison
        try:
            baseline_dataset = QuantumCircuitDataset(test_data_path)
            baseline_circuits = []
            for i in range(min(num_generated, len(baseline_dataset))):
                baseline_circuits.append(baseline_dataset[i]['circuit'])
            
            print("Comparing with baseline circuits...")
            comparison_metrics = self.compare_with_baseline(generated_circuits, baseline_circuits)
            results['comparison'] = comparison_metrics
        except Exception as e:
            print(f"Could not load baseline circuits: {e}")
        
        # 5. Save results
        if save_results:
            self.save_evaluation_results(results, generated_circuits)
        
        return results
    
    def save_evaluation_results(self, 
                              results: Dict[str, Any], 
                              generated_circuits: List[Dict[str, Any]]):
        """Save evaluation results and generated circuits"""
        os.makedirs("evaluation_results", exist_ok=True)
        
        # Save metrics
        with open("evaluation_results/metrics.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save generated circuits
        with open("evaluation_results/generated_circuits.json", 'w') as f:
            json.dump(generated_circuits[:100], f, indent=2)  # Save first 100
        
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
