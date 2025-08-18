#!/usr/bin/env python3
"""
Test script to diagnose model checkpoint loading issues
"""

import torch
import sys
import os
from pathlib import Path

# Add project paths
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "src"))
sys.path.append(str(current_dir.parent / "quantumcommon"))

def test_checkpoint_file(checkpoint_path):
    """Test if checkpoint file can be loaded"""
    print(f"Testing checkpoint: {checkpoint_path}")
    
    # Check if file exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ File does not exist: {checkpoint_path}")
        return False
    
    # Check file size
    file_size = os.path.getsize(checkpoint_path)
    print(f"📁 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    if file_size == 0:
        print("❌ File is empty")
        return False
    
    # Try to load with torch.load
    try:
        print("🔄 Attempting to load checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("✅ Checkpoint loaded successfully!")
        
        # Analyze checkpoint structure
        if isinstance(checkpoint, dict):
            print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
            
            # Check for common keys
            if 'model_state_dict' in checkpoint:
                print(f"🎯 Model state dict found with {len(checkpoint['model_state_dict'])} parameters")
            if 'optimizer_state_dict' in checkpoint:
                print("🎯 Optimizer state dict found")
            if 'epoch' in checkpoint:
                print(f"🎯 Epoch: {checkpoint['epoch']}")
            if 'loss' in checkpoint:
                print(f"🎯 Loss: {checkpoint['loss']}")
            if 'config' in checkpoint:
                print(f"🎯 Config found: {checkpoint['config']}")
                
        else:
            print(f"📋 Checkpoint type: {type(checkpoint)}")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        
        # Try alternative loading methods
        print("🔄 Trying alternative loading methods...")
        
        # Try loading with weights_only=True (for newer PyTorch versions)
        try:
            print("   Trying weights_only=True...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            print("✅ Loaded with weights_only=True!")
            return True
        except:
            pass
        
        # Try loading as pickle
        try:
            print("   Trying pickle loading...")
            import pickle
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            print("✅ Loaded as pickle!")
            return True
        except:
            pass
        
        # Check if file is truncated
        try:
            print("   Checking file integrity...")
            with open(checkpoint_path, 'rb') as f:
                # Try to read the entire file
                data = f.read()
                print(f"   Read {len(data):,} bytes successfully")
                
                # Check if it looks like a ZIP file (PyTorch format)
                if data[:2] == b'PK':
                    print("   ✅ File appears to be a ZIP archive")
                else:
                    print("   ❌ File does not appear to be a ZIP archive")
                    print(f"   First 20 bytes: {data[:20]}")
                    
        except Exception as read_error:
            print(f"   ❌ Cannot read file: {read_error}")
        
        return False

def test_model_creation():
    """Test if we can create a model without loading weights"""
    print("\n" + "="*60)
    print("Testing model creation...")
    
    try:
        from models.decision_transformer import DecisionTransformer
        
        # Try to create model with default config
        print("🔄 Creating DecisionTransformer...")
        model = DecisionTransformer(
            d_model=256,
            n_layers=6,
            n_heads=8,
            d_ff=1024,
            n_gate_types=20,
            max_qubits=8,
            dropout=0.1
        )
        print("✅ Model created successfully!")
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model
        
    except Exception as e:
        print(f" Failed to create model: {e}")
        return None

def test_model_loading_with_weights(model_from_creation, checkpoint_path):
    """Test loading weights into model"""
    print("\n" + "="*60)
    print("Testing weight loading...")
    
    if model_from_creation is None:
        print(" Model creation failed, cannot test weight loading")
        return False
    
    try:
        # First load checkpoint to get configuration
        print(" Loading checkpoint to extract configuration...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract config from checkpoint
        if 'config' in checkpoint:
            ckpt_config = checkpoint['config']
            print(f" Found config in checkpoint: {ckpt_config}")
            
            # Create new model with matching dimensions
            from models.decision_transformer import DecisionTransformer
            
            print(" Creating model with checkpoint dimensions...")
            model = DecisionTransformer(
                d_model=ckpt_config.get('d_model', 512),
                n_layers=ckpt_config.get('n_layers', 6),
                n_heads=ckpt_config.get('n_heads', 8),
                d_ff=ckpt_config.get('d_model', 512) * 4,  # Typically 4x d_model
                n_gate_types=ckpt_config.get('n_gate_types', 20),
                max_qubits=32,  # Safe default
                dropout=ckpt_config.get('dropout', 0.1)
            )
            print(f" Model created with matching dimensions: d_model={ckpt_config.get('d_model')}")
            print(f" Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        else:
            print(" No config found in checkpoint, using originally created model")
            model = model_from_creation
        
        # Try to load weights
        print(" Loading state dict into model...")
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(" Weights loaded successfully!")
            return True
        else:
            print(" No 'model_state_dict' found in checkpoint")
            return False
    
    except Exception as e:
        print(f" Failed to load weights: {e}")
        return False

def main():
    print(" Model Loading Diagnostic Test")
    print("🧪 Model Loading Diagnostic Test")
    print("="*60)
    
    # Test different possible checkpoint paths
    checkpoint_paths = [
        r"C:\Users\jungh\Documents\GitHub\Kaist\OAT_Model\checkpoints\best_model.pt",
    ]
    
    successful_path = None
    
    for path in checkpoint_paths:
        full_path = Path(path).resolve()
        print(f"\n📍 Testing path: {full_path}")
        
        if test_checkpoint_file(str(full_path)):
            successful_path = str(full_path)
            break
        print("-" * 40)
    
    if successful_path:
        print(f"\n✅ Found working checkpoint: {successful_path}")
        
        # Test model creation
        model = test_model_creation()
        
        # Test weight loading
        if model:
            test_model_loading_with_weights(model, successful_path)
    else:
        print("\n❌ No working checkpoint found")
        print("💡 Suggestions:")
        print("   1. Check if training completed successfully")
        print("   2. Verify checkpoint save path in training script")
        print("   3. Re-run training to generate new checkpoint")
        
        # Still test model creation
        test_model_creation()

if __name__ == "__main__":
    main()
