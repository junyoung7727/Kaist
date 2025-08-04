"""
Training Script for Quantum Gate Prediction Transformer
Train transformer model to predict next quantum gate based on user requirements
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Import model and dataset
from models.quantum_transformer import QuantumTransformer, QuantumTransformerConfig, create_quantum_transformer
from data.gate_prediction_dataset import create_gate_prediction_dataloaders

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GatePredictionTrainer:
    """Trainer for quantum gate prediction transformer"""
    
    def __init__(self, 
                 model: QuantumTransformer,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 config: Optional[Dict] = None):
        """
        Initialize trainer
        
        Args:
            model: Quantum transformer model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        
        # Training configuration
        self.learning_rate = self.config.get('learning_rate', 1e-4)
        self.weight_decay = self.config.get('weight_decay', 0.01)
        self.num_epochs = self.config.get('num_epochs', 100)
        self.warmup_steps = self.config.get('warmup_steps', 1000)
        self.gradient_clip = self.config.get('gradient_clip', 1.0)
        self.save_every = self.config.get('save_every', 10)
        self.eval_every = self.config.get('eval_every', 5)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95)
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.warmup_steps,
            T_mult=2,
            eta_min=self.learning_rate * 0.1
        )
        
        # Loss functions
        self.gate_criterion = nn.CrossEntropyLoss(ignore_index=model.config.gate_vocab_size)
        self.qubit_criterion = nn.BCEWithLogitsLoss()
        self.parameter_criterion = nn.MSELoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Output directory
        self.output_dir = Path(self.config.get('output_dir', 'outputs/gate_prediction'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Tensorboard
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {self.model.get_num_params():,}")
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        gate_loss_total = 0.0
        qubit_loss_total = 0.0
        param_loss_total = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Move requirements to device
            requirements = {}
            for key, value in batch['requirements'].items():
                requirements[key] = value.to(self.device)
            
            # Forward pass
            outputs = self.model(
                gates=batch['input_gates'],
                qubits=batch['input_qubits'],
                parameters=batch['input_parameters'],
                requirements=requirements,
                mask=batch.get('attention_mask')
            )
            
            # Calculate losses
            gate_loss = self.gate_criterion(
                outputs['gate_logits'].view(-1, outputs['gate_logits'].size(-1)),
                batch['target_gates'].view(-1)
            )
            
            total_batch_loss = gate_loss
            gate_loss_total += gate_loss.item()
            
            # Qubit prediction loss (if enabled)
            if self.model.config.predict_qubits and 'qubit_logits' in outputs:
                # Convert target qubits to binary format
                target_qubits_binary = torch.zeros_like(outputs['qubit_logits'][:, -1, :])
                for i, qubits in enumerate(batch['target_qubits']):
                    for qubit in qubits:
                        if qubit < target_qubits_binary.size(-1):
                            target_qubits_binary[i, qubit] = 1.0
                
                qubit_loss = self.qubit_criterion(
                    outputs['qubit_logits'][:, -1, :],
                    target_qubits_binary
                )
                total_batch_loss += 0.5 * qubit_loss
                qubit_loss_total += qubit_loss.item()
            
            # Parameter prediction loss (if enabled)
            if self.model.config.predict_parameters and 'parameter_logits' in outputs:
                param_loss = self.parameter_criterion(
                    outputs['parameter_logits'][:, -1, :],
                    batch['target_parameters']
                )
                total_batch_loss += 0.3 * param_loss
                param_loss_total += param_loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += total_batch_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_batch_loss.item():.4f}",
                'gate_loss': f"{gate_loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log to tensorboard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('train/total_loss', total_batch_loss.item(), global_step)
            self.writer.add_scalar('train/gate_loss', gate_loss.item(), global_step)
            self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], global_step)
            
            if self.model.config.predict_qubits and 'qubit_logits' in outputs:
                self.writer.add_scalar('train/qubit_loss', qubit_loss.item(), global_step)
            if self.model.config.predict_parameters and 'parameter_logits' in outputs:
                self.writer.add_scalar('train/param_loss', param_loss.item(), global_step)
        
        avg_loss = total_loss / num_batches
        avg_gate_loss = gate_loss_total / num_batches
        avg_qubit_loss = qubit_loss_total / num_batches if num_batches > 0 else 0.0
        avg_param_loss = param_loss_total / num_batches if num_batches > 0 else 0.0
        
        return {
            'total_loss': avg_loss,
            'gate_loss': avg_gate_loss,
            'qubit_loss': avg_qubit_loss,
            'param_loss': avg_param_loss
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        gate_loss_total = 0.0
        qubit_loss_total = 0.0
        param_loss_total = 0.0
        num_batches = 0
        
        gate_correct = 0
        gate_total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Move requirements to device
                requirements = {}
                for key, value in batch['requirements'].items():
                    requirements[key] = value.to(self.device)
                
                # Forward pass
                outputs = self.model(
                    gates=batch['input_gates'],
                    qubits=batch['input_qubits'],
                    parameters=batch['input_parameters'],
                    requirements=requirements,
                    mask=batch.get('attention_mask')
                )
                
                # Calculate losses
                gate_loss = self.gate_criterion(
                    outputs['gate_logits'].view(-1, outputs['gate_logits'].size(-1)),
                    batch['target_gates'].view(-1)
                )
                
                total_batch_loss = gate_loss
                gate_loss_total += gate_loss.item()
                
                # Gate accuracy
                gate_preds = outputs['gate_logits'][:, -1, :].argmax(dim=-1)
                gate_correct += (gate_preds == batch['target_gates']).sum().item()
                gate_total += batch['target_gates'].size(0)
                
                # Additional losses
                if self.model.config.predict_qubits and 'qubit_logits' in outputs:
                    target_qubits_binary = torch.zeros_like(outputs['qubit_logits'][:, -1, :])
                    for i, qubits in enumerate(batch['target_qubits']):
                        for qubit in qubits:
                            if qubit < target_qubits_binary.size(-1):
                                target_qubits_binary[i, qubit] = 1.0
                    
                    qubit_loss = self.qubit_criterion(
                        outputs['qubit_logits'][:, -1, :],
                        target_qubits_binary
                    )
                    total_batch_loss += 0.5 * qubit_loss
                    qubit_loss_total += qubit_loss.item()
                
                if self.model.config.predict_parameters and 'parameter_logits' in outputs:
                    param_loss = self.parameter_criterion(
                        outputs['parameter_logits'][:, -1, :],
                        batch['target_parameters']
                    )
                    total_batch_loss += 0.3 * param_loss
                    param_loss_total += param_loss.item()
                
                total_loss += total_batch_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_gate_loss = gate_loss_total / num_batches
        avg_qubit_loss = qubit_loss_total / num_batches if num_batches > 0 else 0.0
        avg_param_loss = param_loss_total / num_batches if num_batches > 0 else 0.0
        gate_accuracy = gate_correct / gate_total if gate_total > 0 else 0.0
        
        # Log to tensorboard
        self.writer.add_scalar('val/total_loss', avg_loss, epoch)
        self.writer.add_scalar('val/gate_loss', avg_gate_loss, epoch)
        self.writer.add_scalar('val/gate_accuracy', gate_accuracy, epoch)
        
        if self.model.config.predict_qubits:
            self.writer.add_scalar('val/qubit_loss', avg_qubit_loss, epoch)
        if self.model.config.predict_parameters:
            self.writer.add_scalar('val/param_loss', avg_param_loss, epoch)
        
        return {
            'total_loss': avg_loss,
            'gate_loss': avg_gate_loss,
            'qubit_loss': avg_qubit_loss,
            'param_loss': avg_param_loss,
            'gate_accuracy': gate_accuracy
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.model.config,
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.output_dir / 'latest_checkpoint.pt')
        
        # Save best model
        if metrics.get('total_loss', float('inf')) < self.best_val_loss:
            self.best_val_loss = metrics['total_loss']
            torch.save(checkpoint, self.output_dir / 'best_model.pt')
            logger.info(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
        
        # Save periodic checkpoint
        if epoch % self.save_every == 0:
            torch.save(checkpoint, self.output_dir / f'checkpoint_epoch_{epoch}.pt')
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        for epoch in range(1, self.num_epochs + 1):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_metrics['total_loss'])
            
            # Validate
            val_metrics = {}
            if epoch % self.eval_every == 0:
                val_metrics = self.validate(epoch)
                if val_metrics:
                    self.val_losses.append(val_metrics['total_loss'])
            
            epoch_time = time.time() - start_time
            
            # Log metrics
            logger.info(f"Epoch {epoch}/{self.num_epochs} ({epoch_time:.2f}s)")
            logger.info(f"  Train Loss: {train_metrics['total_loss']:.4f} "
                       f"(Gate: {train_metrics['gate_loss']:.4f})")
            
            if val_metrics:
                logger.info(f"  Val Loss: {val_metrics['total_loss']:.4f} "
                           f"(Gate: {val_metrics['gate_loss']:.4f}, "
                           f"Accuracy: {val_metrics['gate_accuracy']:.4f})")
            
            # Save checkpoint
            if epoch % self.save_every == 0 or val_metrics:
                metrics_to_save = val_metrics if val_metrics else train_metrics
                self.save_checkpoint(epoch, metrics_to_save)
        
        logger.info("Training completed!")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train Quantum Gate Prediction Transformer")
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training data JSON file')
    parser.add_argument('--val_data_path', type=str, default=None,
                        help='Path to validation data JSON file')
    parser.add_argument('--output_dir', type=str, default='outputs/gate_prediction',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to model configuration JSON file')
    
    # Model configuration
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--max_circuit_length', type=int, default=256)
    parser.add_argument('--max_qubits', type=int, default=32)
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    
    # Data configuration
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--augment_data', action='store_true')
    
    args = parser.parse_args()
    
    # Load model configuration
    if args.config_path and os.path.exists(args.config_path):
        with open(args.config_path, 'r') as f:
            model_config_dict = json.load(f)
        model_config = QuantumTransformerConfig(**model_config_dict)
    else:
        model_config = QuantumTransformerConfig(
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            max_circuit_length=args.max_circuit_length,
            max_qubits=args.max_qubits,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        )
    
    # Create model
    model = create_quantum_transformer(model_config)
    
    # Create data loaders
    train_loader, val_loader = create_gate_prediction_dataloaders(
        train_path=args.data_path,
        val_path=args.val_data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_circuit_length=args.max_circuit_length,
        max_qubits=args.max_qubits,
        augment_data=args.augment_data
    )
    
    # Training configuration
    train_config = {
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'num_epochs': args.num_epochs,
        'warmup_steps': args.warmup_steps,
        'gradient_clip': args.gradient_clip,
        'output_dir': args.output_dir,
        'save_every': 10,
        'eval_every': 5
    }
    
    # Create trainer and start training
    trainer = GatePredictionTrainer(model, train_loader, val_loader, train_config)
    trainer.train()


if __name__ == "__main__":
    main()
