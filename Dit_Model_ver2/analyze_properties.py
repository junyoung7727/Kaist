#!/usr/bin/env python3
"""
Simple Quantum Circuit Property Analysis
Analyze circuit properties using the trained Property Prediction model
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Add quantumcommon to path
quantumcommon_path = Path(__file__).parent.parent / "quantumcommon"
if quantumcommon_path.exists():
    sys.path.append(str(quantumcommon_path))

from quantum_dit.models.dit_model import create_dit_model, DiTConfig
from quantum_dit.data.experiment_dataset import ExperimentResultsDataset


def analyze_circuits(checkpoint_path: str, data_path: str, circuit_spec_path: str = None):
    """Analyze quantum circuit properties"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Load checkpoint
    print(f"📁 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model config
    config = DiTConfig()
    
    # Determine number of targets
    final_layer_key = 'property_predictor.3.weight'
    if final_layer_key in checkpoint['model_state_dict']:
        num_targets = checkpoint['model_state_dict'][final_layer_key].shape[0]
    else:
        num_targets = 2
    
    # Create and load model
    model = create_dit_model(
        config,
        num_targets=num_targets,
        target_names=['expressibility', 'two_qubit_ratio'][:num_targets]
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded with {num_targets} target properties")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load dataset
    print(f"📂 Loading dataset: {data_path}")
    dataset = ExperimentResultsDataset(
        data_path=data_path,
        circuit_spec_path=circuit_spec_path,
        target_properties=['expressibility', 'two_qubit_ratio'][:num_targets],
        train_mode=False
    )
    
    if len(dataset) == 0:
        print("❌ No circuits found in dataset!")
        return
    
    print(f"📋 Found {len(dataset)} circuits to analyze")
    
    # Analyze circuits
    predictions = []
    ground_truth = []
    circuit_ids = []
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="🔍 Analyzing circuits"):
            sample = dataset[i]
            
            # Prepare inputs
            gates = sample['gates'].unsqueeze(0).to(device)
            num_qubits = sample['num_qubits'].unsqueeze(0).to(device)
            gate_count = sample['gate_count'].unsqueeze(0).to(device)
            depth = sample['depth'].unsqueeze(0).to(device)
            two_qubit_ratio = sample['two_qubit_ratio'].unsqueeze(0).to(device)
            targets = sample['targets'].cpu().numpy()
            
            # Predict
            pred = model(
                gates=gates,
                num_qubits=num_qubits,
                gate_count=gate_count,
                depth=depth,
                two_qubit_ratio=two_qubit_ratio
            )
            
            predictions.append(pred.cpu().numpy().flatten())
            ground_truth.append(targets)
            circuit_ids.append(sample['circuit_id'])
    
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Calculate metrics
    property_names = ['expressibility', 'two_qubit_ratio'][:num_targets]
    
    print(f"\n🎯 Analysis Results:")
    print("=" * 50)
    
    for i, prop in enumerate(property_names):
        pred_vals = predictions[:, i]
        true_vals = ground_truth[:, i]
        
        mse = np.mean((pred_vals - true_vals) ** 2)
        mae = np.mean(np.abs(pred_vals - true_vals))
        rmse = np.sqrt(mse)
        
        print(f"\n📊 {prop.upper()}:")
        print(f"  MSE:  {mse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  True range: [{true_vals.min():.3f}, {true_vals.max():.3f}]")
        print(f"  Pred range: [{pred_vals.min():.3f}, {pred_vals.max():.3f}]")
    
    # Show some examples
    print(f"\n🔍 Sample Predictions:")
    print("-" * 50)
    for i in range(min(5, len(circuit_ids))):
        print(f"\nCircuit {circuit_ids[i]}:")
        for j, prop in enumerate(property_names):
            true_val = ground_truth[i, j]
            pred_val = predictions[i, j]
            error = abs(pred_val - true_val)
            print(f"  {prop}: {true_val:.3f} → {pred_val:.3f} (error: {error:.3f})")
    
    print(f"\n✅ Analysis complete! Analyzed {len(dataset)} circuits.")


def main():
    parser = argparse.ArgumentParser(description="Analyze quantum circuit properties")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='data/raw/experiment_results.json',
                       help='Path to experiment results')
    parser.add_argument('--circuit_specs', type=str, default='data/raw/circuit_specs.json',
                       help='Path to circuit specifications')
    
    args = parser.parse_args()
    
    analyze_circuits(args.checkpoint, args.data, args.circuit_specs)


if __name__ == "__main__":
    main()
