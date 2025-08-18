#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import os
from pathlib import Path
from datetime import datetime

def save_model_weights(model, save_dir, model_name, epoch=None, val_loss=None):
    """Simple model weight saving"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if epoch is not None:
        filename = f"{model_name}_epoch_{epoch}_{timestamp}.pt"
    else:
        filename = f"{model_name}_{timestamp}.pt"
    
    save_path = save_dir / filename
    
    # Save model state dict
    torch.save(model.state_dict(), save_path)
    
    # Also save as "latest" for easy loading
    latest_path = save_dir / f"{model_name}_latest.pt"
    torch.save(model.state_dict(), latest_path)
    
    # Save as "best" if val_loss provided
    if val_loss is not None:
        best_path = save_dir / f"{model_name}_best.pt"
        # Check if this is better than previous best
        best_loss_file = save_dir / f"{model_name}_best_loss.txt"
        
        should_save_best = True
        if best_loss_file.exists():
            with open(best_loss_file, 'r') as f:
                prev_best_loss = float(f.read().strip())
                should_save_best = val_loss < prev_best_loss
        
        if should_save_best:
            torch.save(model.state_dict(), best_path)
            with open(best_loss_file, 'w') as f:
                f.write(str(val_loss))
            print(f"💾 New best model saved! Val loss: {val_loss:.4f}")
    
    print(f"📁 Model weights saved: {save_path}")
    return str(save_path)

def load_model_weights(model, checkpoint_path):
    """Simple model weight loading"""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    print(f"📂 Model weights loaded: {checkpoint_path}")
    return model
