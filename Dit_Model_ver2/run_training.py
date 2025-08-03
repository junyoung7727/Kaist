#!/usr/bin/env python3
"""
Main execution script for DiT Quantum Circuit Training
Easy-to-use interface for training and evaluation
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from quantum_dit.models.dit_model import DiTConfig
from quantum_dit.training.train_dit import TrainingConfig, AdvancedTrainer
from evaluation.evaluate_dit import QuantumCircuitEvaluator


def create_default_configs():
    """Create default configurations for different model sizes"""
    
    configs = {
        'small': DiTConfig(
            d_model=256,
            n_layers=6,
            n_heads=4,
            d_ff=1024,
            max_circuit_length=128,
            max_qubits=16,
            dropout=0.1,
            use_flash_attention=True,
            use_rotary_pe=True,
            use_swiglu=True
        ),
        
        'medium': DiTConfig(
            d_model=512,
            n_layers=12,
            n_heads=8,
            d_ff=2048,
            max_circuit_length=256,
            max_qubits=32,
            dropout=0.1,
            use_flash_attention=True,
            use_rotary_pe=True,
            use_swiglu=True
        ),
        
        'large': DiTConfig(
            d_model=768,
            n_layers=18,
            n_heads=12,
            d_ff=3072,
            max_circuit_length=512,
            max_qubits=64,
            dropout=0.1,
            use_flash_attention=True,
            use_rotary_pe=True,
            use_swiglu=True,
            gradient_checkpointing=True
        )
    }
    
    return configs


def train_model(args):
    """Train DiT model"""
    print("🚀 Starting DiT Quantum Circuit Training")
    print("="*50)
    
    # Get model configuration
    model_configs = create_default_configs()
    model_config = model_configs[args.model_size]
    
    # Create training configuration
    training_config = TrainingConfig(
        model_config=model_config,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        circuit_spec_path="data/raw/circuit_specs.json",  # Add circuit specs path
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        run_name=args.run_name,
        use_amp=args.use_amp,
        compile_model=False,
        use_ema=args.use_ema
    )
    
    # Create trainer and start training
    trainer = AdvancedTrainer(training_config)
    trainer.train()
    
    print("✅ Training completed successfully!")


def evaluate_model(args):
    """Evaluate trained model"""
    print("🔍 Starting DiT Model Evaluation")
    print("="*50)
    
    # Create evaluator
    evaluator = QuantumCircuitEvaluator(args.model_path, args.config_path)
    
    # Run evaluation
    results = evaluator.run_comprehensive_evaluation(
        args.test_data,
        args.num_generated,
        save_results=True
    )
    
    print("✅ Evaluation completed successfully!")
    return results


def generate_circuits(args):
    """Generate quantum circuits using trained model"""
    print("⚡ Generating Quantum Circuits")
    print("="*50)
    
    # Create evaluator
    evaluator = QuantumCircuitEvaluator(args.model_path, args.config_path)
    
    # Generate circuits
    circuits = evaluator.generate_circuits(
        num_samples=args.num_samples,
        max_length=args.max_length,
        guidance_scale=args.guidance_scale
    )
    
    # Save generated circuits
    import json
    output_path = args.output_path or "generated_circuits.json"
    with open(output_path, 'w') as f:
        json.dump(circuits, f, indent=2)
    
    print(f"✅ Generated {len(circuits)} circuits saved to: {output_path}")


def visualize_training(args):
    """Visualize training results"""
    print("📊 Creating Training Visualizations")
    print("="*50)
    
    from quantum_dit.utils.visualization import plot_training_curves, create_training_dashboard
    
    if args.dashboard:
        create_training_dashboard(args.log_dir, args.output_path)
    else:
        loss_history_path = os.path.join(args.log_dir, "loss_history.json")
        plot_training_curves(loss_history_path, args.output_path)
    
    print("✅ Visualizations created successfully!")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="DiT Quantum Circuit Generation")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train DiT model')
    train_parser.add_argument('--model_size', choices=['small', 'medium', 'large'], 
                             default='medium', help='Model size')
    train_parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--train_data', type=str, default='data/raw/experiment_results.json', 
                             help='Training data path')
    train_parser.add_argument('--val_data', type=str, default='data/raw/experiment_results.json', 
                             help='Validation data path')
    train_parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    train_parser.add_argument('--save_dir', type=str, default='checkpoints', 
                             help='Checkpoint directory')
    train_parser.add_argument('--run_name', type=str, help='Run name for logging')
    train_parser.add_argument('--use_amp', action='store_true', default=True, 
                             help='Use automatic mixed precision')
    train_parser.add_argument('--compile_model', action='store_true', default=True,
                             help='Use torch.compile')
    train_parser.add_argument('--use_ema', action='store_true', default=True,
                             help='Use exponential moving average')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model_path', type=str, required=True, 
                            help='Path to trained model')
    eval_parser.add_argument('--config_path', type=str, help='Path to model config')
    eval_parser.add_argument('--test_data', type=str, required=True, 
                            help='Test data path')
    eval_parser.add_argument('--num_generated', type=int, default=1000,
                            help='Number of circuits to generate')
    
    # Generation command
    gen_parser = subparsers.add_parser('generate', help='Generate quantum circuits')
    gen_parser.add_argument('--model_path', type=str, required=True,
                           help='Path to trained model')
    gen_parser.add_argument('--config_path', type=str, help='Path to model config')
    gen_parser.add_argument('--num_samples', type=int, default=100,
                           help='Number of circuits to generate')
    gen_parser.add_argument('--max_length', type=int, help='Maximum circuit length')
    gen_parser.add_argument('--guidance_scale', type=float, default=1.0,
                           help='Guidance scale for generation')
    gen_parser.add_argument('--output_path', type=str, help='Output file path')
    
    # Visualization command
    viz_parser = subparsers.add_parser('visualize', help='Create visualizations')
    viz_parser.add_argument('--log_dir', type=str, required=True, help='Log directory')
    viz_parser.add_argument('--output_path', type=str, help='Output image path')
    viz_parser.add_argument('--dashboard', action='store_true', 
                           help='Create full dashboard instead of simple curves')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'generate':
        generate_circuits(args)
    elif args.command == 'visualize':
        visualize_training(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
