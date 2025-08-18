"""
Create a test checkpoint file to replace the corrupted one
"""
import torch
import sys
from pathlib import Path
import os

# Add paths
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))
sys.path.append(str(Path(__file__).parent))

from src.models.decision_transformer import DecisionTransformer

def create_test_checkpoint():
    """Create a valid test checkpoint file"""
    print("🔧 Creating test checkpoint...")
    
    # Create model with same config as training
    model = DecisionTransformer(
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        n_gate_types=20,
        max_qubits=32,
        dropout=0.1
    )
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create checkpoint structure
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': {},  # Empty for test
        'scheduler_state_dict': {},  # Empty for test
        'global_step': 0,
        'best_val_loss': float('inf'),
        'config': {
            'd_model': 512,
            'n_layers': 6,
            'n_heads': 8,
            'n_gate_types': 20,
            'dropout': 0.1,
            'learning_rate': 3e-4,
            'batch_size': 32,
            'num_epochs': 20
        }
    }
    
    # Ensure checkpoint directory exists
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save with atomic write
    checkpoint_path = checkpoint_dir / "best_model.pt"
    temp_path = checkpoint_dir / "best_model.pt.tmp"
    
    try:
        print(f"💾 Saving checkpoint to {checkpoint_path}...")
        torch.save(checkpoint, temp_path)
        
        # Verify the saved file can be loaded
        print("🔍 Verifying saved checkpoint...")
        test_checkpoint = torch.load(temp_path, map_location='cpu')
        print("✅ Checkpoint verification successful!")
        
        # Atomic move
        if checkpoint_path.exists():
            backup_path = checkpoint_dir / "best_model_old_backup.pt"
            checkpoint_path.rename(backup_path)
            print(f"📦 Old checkpoint backed up to {backup_path}")
        
        temp_path.rename(checkpoint_path)
        print(f"✅ New checkpoint saved successfully: {checkpoint_path}")
        
        # Final verification
        final_test = torch.load(checkpoint_path, map_location='cpu')
        print("✅ Final verification passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create checkpoint: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False

if __name__ == "__main__":
    success = create_test_checkpoint()
    if success:
        print("\n🎉 Test checkpoint created successfully!")
        print("💡 You can now test the circuit generation pipeline.")
    else:
        print("\n❌ Failed to create test checkpoint.")
