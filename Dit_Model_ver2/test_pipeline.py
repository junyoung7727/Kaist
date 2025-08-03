#!/usr/bin/env python3
"""
Simple test script to verify the complete pipeline
"""

import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from quantum_dit.data.experiment_dataset import ExperimentResultsDataset
from quantum_dit.models.dit_model import DiTConfig, create_dit_model


def test_dataset():
    """Test dataset loading"""
    print("🔍 Testing Dataset Loading...")
    
    # Paths to data files
    experiment_results_path = r"Dit_Model_ver2\data\raw\experiment_results.json"
    circuit_spec_path = r"Dit_Model_ver2\data\raw\circuit_specs.json"
    
    # Check if files exist
    if not os.path.exists(experiment_results_path):
        print(f"❌ File not found: {experiment_results_path}")
        return None
    
    if not os.path.exists(circuit_spec_path):
        print(f"⚠️  File not found: {circuit_spec_path} (will use synthetic gates)")
        circuit_spec_path = None
    
    # Create dataset
    dataset = ExperimentResultsDataset(
        data_path=experiment_results_path,
        circuit_spec_path=circuit_spec_path,
        target_properties=['expressibility', 'two_qubit_ratio', 'simulator_error_fidelity'],
        train_mode=False  # No augmentation for testing
    )
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   - Number of circuits: {len(dataset)}")
    
    # Show dataset statistics
    stats = dataset.get_statistics()
    print(f"   - Target properties: {dataset.get_target_names()}")
    print(f"   - Vocab size: {dataset.get_vocab_size()}")
    
    # Test a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n📊 Sample Data:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"   - {key}: {value.shape} | {value.dtype}")
            else:
                print(f"   - {key}: {value}")
    
    return dataset


def test_model(dataset):
    """Test model creation and forward pass"""
    print("\n🤖 Testing Model...")
    
    # Create model configuration
    config = DiTConfig(
        d_model=256,  # Smaller for testing
        n_layers=6,
        n_heads=4,
        max_circuit_length=64,
        use_flash_attention=True,
        use_rotary_pe=True,
        use_swiglu=True
    )
    
    # Create model
    model = create_dit_model(
        config, 
        num_targets=3,
        target_names=['expressibility', 'two_qubit_ratio', 'simulator_error_fidelity']
    )
    
    print(f"✅ Model created successfully!")
    
    # Test forward pass with a sample
    if dataset and len(dataset) > 0:
        sample = dataset[0]
        
        # Add batch dimension
        gates = sample['gates'].unsqueeze(0)
        num_qubits = sample['num_qubits'].unsqueeze(0)
        gate_count = sample['gate_count'].unsqueeze(0)
        depth = sample['depth'].unsqueeze(0)
        two_qubit_ratio = sample['two_qubit_ratio'].unsqueeze(0)
        targets = sample['targets'].unsqueeze(0)
        
        print(f"\n🔄 Testing Forward Pass...")
        print(f"   - Input shapes:")
        print(f"     • gates: {gates.shape}")
        print(f"     • num_qubits: {num_qubits.shape}")
        print(f"     • gate_count: {gate_count.shape}")
        print(f"     • depth: {depth.shape}")
        print(f"     • two_qubit_ratio: {two_qubit_ratio.shape}")
        
        # Forward pass
        with torch.no_grad():
            predictions = model(
                gates=gates,
                num_qubits=num_qubits,
                gate_count=gate_count,
                depth=depth,
                two_qubit_ratio=two_qubit_ratio
            )
        
        print(f"   - Output shape: {predictions.shape}")
        print(f"   - Predictions: {predictions.squeeze().numpy()}")
        print(f"   - Targets: {targets.squeeze().numpy()}")
        
        # Calculate loss
        loss = F.mse_loss(predictions, targets)
        print(f"   - MSE Loss: {loss.item():.6f}")
        
        print(f"✅ Forward pass successful!")
        
        return model
    
    return model


def test_training_step(dataset, model):
    """Test a single training step"""
    print("\n🏋️ Testing Training Step...")
    
    # Create a simple dataloader
    from torch.utils.data import DataLoader
    from quantum_dit.data.experiment_dataset import ExperimentResultsCollator
    
    collator = ExperimentResultsCollator(pad_token_id=dataset.get_vocab_size() - 1)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collator)
    
    # Get a batch
    batch = next(iter(dataloader))
    
    print(f"   - Batch size: {batch['gates'].shape[0]}")
    
    # Forward pass
    predictions = model(
        gates=batch['gates'],
        num_qubits=batch['num_qubits'],
        gate_count=batch['gate_count'],
        depth=batch['depth'],
        two_qubit_ratio=batch['two_qubit_ratio']
    )
    
    # Calculate loss
    targets = batch['targets']
    loss = F.mse_loss(predictions, targets)
    
    print(f"   - Batch predictions shape: {predictions.shape}")
    print(f"   - Batch targets shape: {targets.shape}")
    print(f"   - Batch loss: {loss.item():.6f}")
    
    # Test backward pass
    loss.backward()
    print(f"✅ Backward pass successful!")
    
    return loss.item()


def main():
    """Main test function"""
    print("🚀 Testing DiT Quantum Circuit Property Prediction Pipeline")
    print("=" * 60)
    
    try:
        # Test dataset
        dataset = test_dataset()
        if dataset is None:
            print("❌ Dataset test failed!")
            return
        
        # Test model
        model = test_model(dataset)
        
        # Test training step
        loss = test_training_step(dataset, model)
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed successfully!")
        print(f"   - Dataset: ✅ {len(dataset)} circuits loaded")
        print(f"   - Model: ✅ {model.get_num_params():,} parameters")
        print(f"   - Training: ✅ Loss = {loss:.6f}")
        print("\n💡 Ready to start full training!")
        print("   Run: python run_training.py train --model_size small")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
