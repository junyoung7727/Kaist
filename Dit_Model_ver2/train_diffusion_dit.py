#!/usr/bin/env python3
"""
Train Diffusion DiT Model for Quantum Circuit Generation
Train a true diffusion-based generative model
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Add quantumcommon to path
quantumcommon_path = Path(__file__).parent.parent / "quantumcommon"
if quantumcommon_path.exists():
    sys.path.append(str(quantumcommon_path))

from quantum_dit.models.dit_model import QuantumDiT, DiTConfig
from quantum_dit.data.experiment_dataset import ExperimentResultsDataset, ExperimentResultsCollator


def create_diffusion_config():
    """Create configuration for diffusion training"""
    return DiTConfig(
        d_model=256,
        n_layers=6,
        n_heads=4,
        d_ff=1024,
        max_circuit_length=128,
        max_qubits=16,
        dropout=0.1,
        timesteps=1000,
        noise_schedule="cosine",
        diffusion_mode=True,  # Enable diffusion!
        use_flash_attention=True,
        use_rotary_pe=True,
        use_swiglu=True,
        gradient_checkpointing=False
    )


def train_diffusion_model(data_path: str, circuit_spec_path: str, output_dir: str = "diffusion_checkpoints"):
    """Train diffusion model for circuit generation"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Create diffusion config
    config = create_diffusion_config()
    print(f"📋 Diffusion Config: {config}")
    
    # Create model
    model = QuantumDiT(config, num_targets=2)  # Not used in diffusion mode
    model.to(device)
    
    print(f"🎯 Created Diffusion DiT with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dataset (for gate sequences)
    dataset = ExperimentResultsDataset(
        data_path=data_path,
        circuit_spec_path=circuit_spec_path,
        target_properties=['expressibility', 'two_qubit_ratio'],
        train_mode=True
    )
    
    collator = ExperimentResultsCollator()
    dataloader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=True, 
        collate_fn=collator,
        num_workers=0
    )
    
    print(f"📊 Dataset: {len(dataset)} circuits")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Training loop
    model.train()
    num_epochs = 10
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Move to device (only tensors)
            batch_device = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch_device[key] = value.to(device)
                else:
                    batch_device[key] = value
            
            gates = batch_device['gates']  # [batch_size, seq_len]
            batch_size, seq_len = gates.shape
            
            # Sample random timesteps
            timesteps = torch.randint(0, config.timesteps, (batch_size,), device=device)
            
            # Add noise to gates (convert to one-hot first)
            gates_onehot = torch.zeros(batch_size, seq_len, model.vocab_size, device=device)
            gates_onehot.scatter_(2, gates.unsqueeze(-1), 1)
            
            # Sample noise
            noise = torch.randn_like(gates_onehot)
            
            # Add noise using diffusion scheduler
            noisy_gates = model.diffusion_scheduler.add_noise(gates_onehot, noise, timesteps)
            
            # Convert back to discrete for model input
            noisy_gates_discrete = noisy_gates.argmax(dim=-1)
            
            # Forward pass: predict noise
            optimizer.zero_grad()
            predicted_noise = model(gates=noisy_gates_discrete, timesteps=timesteps)
            
            # Loss: MSE between predicted and actual noise
            loss = nn.MSELoss()(predicted_noise, noise)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': loss.item():.4f})
        
        avg_loss = epoch_loss / num_batches
        print(f"🎯 Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config.__dict__,
            'loss': avg_loss
        }
        
        checkpoint_path = os.path.join(output_dir, f"diffusion_dit_epoch_{epoch+1}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    print(f"🎉 Training complete! Checkpoints saved in {output_dir}")
    return os.path.join(output_dir, f"diffusion_dit_epoch_{num_epochs}.pt")


def main():
    parser = argparse.ArgumentParser(description="Train Diffusion DiT Model")
    parser.add_argument('--data', type=str, default='data/raw/experiment_results.json',
                       help='Path to experiment results')
    parser.add_argument('--circuit_specs', type=str, default='data/raw/circuit_specs.json',
                       help='Path to circuit specifications')
    parser.add_argument('--output_dir', type=str, default='diffusion_checkpoints',
                       help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    print("🚀 Training Diffusion DiT Model for Quantum Circuit Generation")
    print("=" * 60)
    
    checkpoint_path = train_diffusion_model(args.data, args.circuit_specs, args.output_dir)
    
    print(f"\n✅ Training completed!")
    print(f"🎯 Final checkpoint: {checkpoint_path}")
    print(f"🔥 Now you can use this checkpoint with evaluate_dit.py for real circuit generation!")


if __name__ == "__main__":
    main()
