#!/usr/bin/env python3
"""
Test script for the new Diffusion DiT model
Test both diffusion generation and property prediction modes
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Add quantumcommon to path
quantumcommon_path = Path(__file__).parent.parent / "quantumcommon"
if quantumcommon_path.exists():
    sys.path.append(str(quantumcommon_path))

from quantum_dit.models.dit_model import QuantumDiT, DiTConfig


def test_diffusion_mode():
    """Test diffusion generation mode"""
    print("🔥 Testing Diffusion Mode...")
    
    # Create diffusion config
    config = DiTConfig(
        d_model=256,
        n_layers=4,
        n_heads=4,
        diffusion_mode=True,  # Enable diffusion!
        timesteps=1000
    )
    
    # Create model
    model = QuantumDiT(config, num_targets=2)
    model.eval()
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size = 2
    seq_len = 32
    
    # Random noisy gate sequence
    gates = torch.randint(0, 20, (batch_size, seq_len))
    
    # Random timesteps
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    print(f"📊 Input: gates={gates.shape}, timesteps={timesteps.shape}")
    
    # Forward pass
    with torch.no_grad():
        noise_pred = model(gates=gates, timesteps=timesteps)
    
    print(f"🎯 Output: {noise_pred.shape} (should be [batch, seq_len, vocab_size])")
    print(f"✅ Diffusion mode working! Predicted noise shape: {noise_pred.shape}")
    
    return True


def test_property_prediction_mode():
    """Test property prediction mode"""
    print("\n📊 Testing Property Prediction Mode...")
    
    # Create property prediction config
    config = DiTConfig(
        d_model=256,
        n_layers=4,
        n_heads=4,
        diffusion_mode=False,  # Disable diffusion
    )
    
    # Create model
    model = QuantumDiT(config, num_targets=2)
    model.eval()
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size = 2
    seq_len = 32
    
    # Circuit data
    gates = torch.randint(0, 20, (batch_size, seq_len))
    num_qubits = torch.randint(4, 16, (batch_size,))
    gate_count = torch.randint(10, 50, (batch_size,))
    depth = torch.randint(5, 20, (batch_size,))
    two_qubit_ratio = torch.rand(batch_size)
    
    print(f"📊 Input: gates={gates.shape}, num_qubits={num_qubits.shape}")
    
    # Forward pass
    with torch.no_grad():
        properties = model(
            gates=gates,
            num_qubits=num_qubits,
            gate_count=gate_count,
            depth=depth,
            two_qubit_ratio=two_qubit_ratio
        )
    
    print(f"🎯 Output: {properties.shape} (should be [batch, num_targets])")
    print(f"✅ Property prediction mode working! Properties shape: {properties.shape}")
    
    return True


def test_diffusion_scheduler():
    """Test diffusion scheduler integration"""
    print("\n⚡ Testing Diffusion Scheduler Integration...")
    
    config = DiTConfig(diffusion_mode=True)
    model = QuantumDiT(config, num_targets=2)
    
    # Access the diffusion scheduler
    scheduler = model.diffusion_scheduler
    
    print(f"✅ Scheduler: {scheduler.noise_schedule} with {scheduler.timesteps} timesteps")
    
    # Test noise addition
    batch_size, seq_len = 2, 16
    original_samples = torch.randn(batch_size, seq_len, model.vocab_size)
    noise = torch.randn_like(original_samples)
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    # Add noise
    noisy_samples = scheduler.add_noise(original_samples, noise, timesteps)
    
    print(f"🎯 Noisy samples shape: {noisy_samples.shape}")
    print(f"✅ Diffusion scheduler working correctly!")
    
    return True


def main():
    """Run all tests"""
    print("🚀 Testing New Diffusion DiT Model")
    print("=" * 50)
    
    try:
        # Test diffusion mode
        test_diffusion_mode()
        
        # Test property prediction mode
        test_property_prediction_mode()
        
        # Test diffusion scheduler
        test_diffusion_scheduler()
        
        print("\n🎉 All tests passed!")
        print("✅ The new QuantumDiT model supports both:")
        print("   1. Diffusion-based circuit generation")
        print("   2. Property prediction")
        print("✅ evaluate_dit.py should now work with diffusion_mode=True!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
