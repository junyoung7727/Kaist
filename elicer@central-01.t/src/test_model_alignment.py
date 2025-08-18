"""
Test script to validate tensor dimension alignment in Decision Transformer
"""

import torch
import sys
from pathlib import Path
import argparse

# Add project paths
src_path = Path(__file__).parent
sys.path.append(str(src_path))
sys.path.append(str(src_path.parent))

from src.training.trainer import QuantumCircuitCollator
from src.data.quantum_circuit_dataset import DatasetManager
from src.data.embedding_pipeline import create_embedding_pipeline, EmbeddingConfig
from src.models.decision_transformer import create_decision_transformer
from utils.debug_utils import debug_tensor_info


def test_model_alignment(data_path, debug=False):
    """Test tensor alignment between model output and masks"""
    print("\n=== Testing Model Tensor Alignment ===")
    
    # 1. Create dataset with a very small subset
    print("\n🔍 Loading dataset...")
    manager = DatasetManager(unified_data_path=data_path)
    circuit_data = manager.merge_data(enable_filtering=True)
    train_dataset, val_dataset, _ = manager.split_dataset(
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )
    
    # Take only a small sample
    small_dataset = train_dataset[:5] if len(train_dataset) > 5 else train_dataset
    print(f"Using {len(small_dataset)} samples for testing")
    
    # 2. Create embedding pipeline
    print("\n🔍 Setting up embedding pipeline...")
    embed_config = EmbeddingConfig(
        d_model=256,  # Use smaller model for quick testing
        n_gate_types=20,
        max_seq_len=2000
    )
    embedding_pipeline = create_embedding_pipeline(embed_config)
    
    # 3. Create collator and process batch
    print("\n🔍 Processing batch with collator...")
    collator = QuantumCircuitCollator(embedding_pipeline)
    batch = collator(small_dataset)
    
    # 4. Check shapes before model processing
    print("\n🔍 Initial tensor shapes:")
    print(f"  • Input sequence: {batch['input_sequence'].shape}")
    print(f"  • Attention mask: {batch['attention_mask'].shape}")
    print(f"  • Action prediction mask: {batch['action_prediction_mask'].shape}")
    if debug:
        print(f"  • Action mask values: {batch['action_prediction_mask'].sum(dim=2)}")
    
    # 5. Create model
    print("\n🔍 Creating model...")
    model = create_decision_transformer(
        d_model=256,  # Small for testing
        n_layers=4,   # Small for testing
        n_heads=4,
        n_gate_types=20,
        dropout=0.1
    )
    
    # 6. Process through model
    print("\n🔍 Running forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Move batch to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # Set model to eval mode
    model.eval()
    
    with torch.no_grad():
        model_kwargs = {
            'input_sequence': batch['input_sequence'],
            'attention_mask': batch['attention_mask'],
            'action_prediction_mask': batch['action_prediction_mask']
        }
        
        # Get model outputs
        outputs = model(**model_kwargs)
        
        # Prepare targets for loss computation (simplified)
        squeezed_action_mask = batch['action_prediction_mask'].squeeze(1) if batch['action_prediction_mask'].dim() == 3 else batch['action_prediction_mask']
        squeezed_target_actions = batch['target_actions'].squeeze(1) if batch['target_actions'].dim() == 3 else batch['target_actions']
        
        targets = {
            'gate_targets': squeezed_target_actions,
            'position_targets': batch.get('target_qubits', []),
            'parameter_targets': batch.get('target_params', [])
        }
    
    # 7. Check shapes after model processing
    print("\n🔍 Model output tensor shapes:")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  • {k}: {v.shape}")
    
    # 8. Check dimension alignment between mask and outputs
    print("\n🔍 Verifying dimension alignment...")
    
    # Get key shapes for comparison
    input_seq_shape = batch['input_sequence'].shape
    action_mask_shape = squeezed_action_mask.shape
    gate_logits_shape = outputs['gate_logits'].shape
    
    print(f"  • Input sequence shape: {input_seq_shape}")
    print(f"  • Action mask shape: {action_mask_shape}")
    print(f"  • Gate logits shape: {gate_logits_shape}")
    
    # Verify alignment
    if gate_logits_shape[0] == action_mask_shape[0] and gate_logits_shape[1] == action_mask_shape[1]:
        print("\n✅ PASS: Gate logits and action mask have compatible dimensions!")
    else:
        print("\n❌ FAIL: Dimension mismatch between gate logits and action mask!")
        print(f"  • Gate logits: {gate_logits_shape}")
        print(f"  • Action mask: {action_mask_shape}")

    # 9. Try to compute loss to verify loss calculation works
    print("\n🔍 Testing loss computation...")
    try:
        # Add missing batch metadata for loss computation
        num_qubits = [circuit_data.num_qubits for circuit_data in small_dataset]
        num_gates = [len(circuit_data.gates) for circuit_data in small_dataset]
        
        loss_outputs = model.compute_loss(
            outputs, 
            targets, 
            squeezed_action_mask,
            num_qubits=num_qubits,
            num_gates=num_gates
        )
        print(f"✅ PASS: Loss computation successful!")
        print(f"  • Total loss: {loss_outputs['loss'].item():.4f}")
        
        if 'gate_loss' in loss_outputs:
            print(f"  • Gate loss: {loss_outputs['gate_loss'].item():.4f}")
        if 'position_loss' in loss_outputs:
            print(f"  • Position loss: {loss_outputs['position_loss'].item():.4f}")
        if 'parameter_loss' in loss_outputs:
            print(f"  • Parameter loss: {loss_outputs['parameter_loss'].item():.4f}")
            
    except Exception as e:
        print(f"❌ FAIL: Loss computation failed with error:")
        print(f"  • {str(e)}")
        
        # Additional debugging information
        if debug:
            print("\nDetailed tensor shapes for debugging:")
            debug_tensor_info(outputs, "Model outputs")
            debug_tensor_info(targets, "Targets")
            debug_tensor_info({"action_mask": squeezed_action_mask}, "Action mask")
    
    print("\n=== Test Completed ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Decision Transformer tensor alignment")
    parser.add_argument("--data_path", type=str, default=r"C:\Users\jungh\Documents\GitHub\Kaist\OAT_Model\raw_data\merged_data.json",
                      help="Path to data file")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debugging output")
    
    args = parser.parse_args()
    test_model_alignment(args.data_path, args.debug)
