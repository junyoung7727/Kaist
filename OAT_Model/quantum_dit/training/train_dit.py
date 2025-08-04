"""
Training script for DiT Quantum Circuit Generation Model
State-of-the-art training pipeline with advanced optimization techniques
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.dit_model import QuantumDiT, DiTConfig, create_dit_model
from quantum_dit.data.experiment_dataset import ExperimentResultsDataset, ExperimentResultsCollator, create_experiment_dataloaders
from utils.diffusion import DiffusionScheduler
from utils.metrics import QuantumCircuitMetrics

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model
    model_config: DiTConfig
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    num_epochs: int = 100
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Validation
    val_every_n_steps: int = 500
    save_every_n_steps: int = 1000
    
    # Optimization
    optimizer: str = "adamw"  # "adamw", "lion", "sophia"
    scheduler: str = "cosine"  # "cosine", "linear", "polynomial"
    use_amp: bool = True  # Automatic Mixed Precision
    compile_model: bool = True  # torch.compile
    
    # Data
    train_data_path: str = "data/train"
    val_data_path: str = "data/val"
    circuit_spec_path: Optional[str] = None  # Path to circuit specifications
    num_workers: int = 4
    
    # Logging
    log_dir: str = "logs"
    save_dir: str = "checkpoints"
    project_name: str = "quantum_dit"
    run_name: Optional[str] = None
    
    # Advanced
    gradient_accumulation_steps: int = 1
    use_ema: bool = True  # Exponential Moving Average
    ema_decay: float = 0.9999


class ExponentialMovingAverage:
    """Exponential Moving Average for model parameters"""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Apply shadow parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])


class AdvancedTrainer:
    """Advanced trainer with state-of-the-art techniques"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup logging
        self.setup_logging()
        
        # Create model
        self.model = create_dit_model(
            config.model_config,
            num_targets=2,
            target_names=['expressibility', 'two_qubit_ratio']
        )
        self.model.to(self.device)
        
        # Compile model for better performance (PyTorch 2.0+)
        if config.compile_model and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
            self.logger.info("Model compiled with torch.compile")
        
        # Setup diffusion scheduler
        self.diffusion = DiffusionScheduler(
            timesteps=config.model_config.timesteps,
            noise_schedule=config.model_config.noise_schedule
        )
        
        # Setup optimizer
        self.optimizer = self.create_optimizer()
        
        # Setup scheduler
        self.scheduler = self.create_scheduler()
        
        # Setup EMA
        if config.use_ema:
            self.ema = ExponentialMovingAverage(self.model, config.ema_decay)
        
        # Setup AMP
        if config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Setup metrics
        self.metrics = QuantumCircuitMetrics()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Loss tracking
        self.train_losses = []
        self.val_losses = []
        
        # Setup tensorboard
        self.writer = SummaryWriter(
            log_dir=os.path.join(config.log_dir, config.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        )
    
    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with advanced techniques"""
        if self.config.optimizer.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.95),  # Better for transformers
                eps=1e-8
            )
        elif self.config.optimizer.lower() == "lion":
            try:
                from lion_pytorch import Lion
                return Lion(
                    self.model.parameters(),
                    lr=self.config.learning_rate * 0.1,  # Lion needs lower LR
                    weight_decay=self.config.weight_decay
                )
            except ImportError:
                self.logger.warning("Lion optimizer not available, falling back to AdamW")
                return self.create_adamw_optimizer()
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.num_epochs // 4,
                T_mult=2,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler == "linear":
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config.warmup_steps
            )
        else:
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.num_epochs // 3,
                gamma=0.1
            )
    
    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load training and validation datasets"""
        # Create datasets
        train_dataset = ExperimentResultsDataset(
            data_path=self.config.train_data_path,
            circuit_spec_path=getattr(self.config, 'circuit_spec_path', None),
            target_properties=['expressibility', 'two_qubit_ratio'],
            train_mode=True
        )
        
        val_dataset = ExperimentResultsDataset(
            data_path=self.config.val_data_path,
            circuit_spec_path=getattr(self.config, 'circuit_spec_path', None),
            target_properties=['expressibility', 'two_qubit_ratio'],
            train_mode=False
        )
        
        # Create collator
        collator = ExperimentResultsCollator(pad_token_id=train_dataset.get_vocab_size() - 1)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collator
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=collator
        )
        
        self.logger.info(f"Loaded {len(train_dataset)} training samples")
        self.logger.info(f"Loaded {len(val_dataset)} validation samples")
        
        return train_loader, val_loader
    
    def compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute property prediction loss"""
        # Extract inputs and targets
        gates = batch['gates'].to(self.device)
        num_qubits = batch['num_qubits'].to(self.device)
        gate_count = batch['gate_count'].to(self.device)
        depth = batch['depth'].to(self.device)
        two_qubit_ratio = batch['two_qubit_ratio'].to(self.device)
        targets = batch['targets'].to(self.device)
        
        # Predict properties
        if self.config.use_amp:
            with torch.cuda.amp.autocast():
                predictions = self.model(
                    gates=gates,
                    num_qubits=num_qubits,
                    gate_count=gate_count,
                    depth=depth,
                    two_qubit_ratio=two_qubit_ratio
                )
                loss = nn.functional.mse_loss(predictions, targets)
        else:
            predictions = self.model(
                gates=gates,
                num_qubits=num_qubits,
                gate_count=gate_count,
                depth=depth,
                two_qubit_ratio=two_qubit_ratio
            )
            loss = nn.functional.mse_loss(predictions, targets)
        
        return loss
    
    def train_step(self, batch: Dict[str, Any]) -> float:
        """Single training step"""
        self.model.train()
        
        # Compute loss
        loss = self.compute_loss(batch)
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.config.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update parameters
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            # Update EMA
            if self.config.use_ema:
                self.ema.update()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation loop"""
        self.model.eval()
        val_losses = []
        
        # Apply EMA if available
        if self.config.use_ema:
            self.ema.apply_shadow()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                loss = self.compute_loss(batch)
                val_losses.append(loss.item())
        
        # Restore original parameters
        if self.config.use_ema:
            self.ema.restore()
        
        avg_val_loss = np.mean(val_losses)
        
        return {
            'val_loss': avg_val_loss,
            'val_perplexity': np.exp(avg_val_loss)
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': asdict(self.config),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        if self.config.use_ema:
            checkpoint['ema_state_dict'] = self.ema.shadow
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.save_dir, 
            f"checkpoint_epoch_{self.epoch}_step_{self.global_step}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config.save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with val_loss: {self.best_val_loss:.4f}")
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {self.model.get_num_params():,}")
        
        # Load data
        train_loader, val_loader = self.load_data()
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_losses = []
            
            # Training
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device (only tensors)
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                # Training step
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                self.global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # Log to tensorboard
                if self.global_step % 10 == 0:
                    self.writer.add_scalar('train/loss', loss, self.global_step)
                    self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                
                # Validation
                if self.global_step % self.config.val_every_n_steps == 0:
                    val_metrics = self.validate(val_loader)
                    
                    # Log validation metrics
                    for key, value in val_metrics.items():
                        self.writer.add_scalar(f'val/{key}', value, self.global_step)
                    
                    # Check if best model
                    is_best = val_metrics['val_loss'] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_metrics['val_loss']
                    
                    # Save losses for visualization
                    self.val_losses.append({
                        'step': self.global_step,
                        'epoch': epoch,
                        'loss': val_metrics['val_loss']
                    })
                    
                    self.logger.info(
                        f"Step {self.global_step} - Val Loss: {val_metrics['val_loss']:.4f} "
                        f"(Best: {self.best_val_loss:.4f})"
                    )
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_every_n_steps == 0:
                        self.save_checkpoint(is_best)
            
            # End of epoch
            avg_train_loss = np.mean(epoch_losses)
            self.train_losses.append({
                'epoch': epoch,
                'loss': avg_train_loss
            })
            
            # Update scheduler
            self.scheduler.step()
            
            self.logger.info(
                f"Epoch {epoch+1} completed - Avg Train Loss: {avg_train_loss:.4f}"
            )
        
        # Final validation and save
        final_val_metrics = self.validate(val_loader)
        self.save_checkpoint(final_val_metrics['val_loss'] < self.best_val_loss)
        
        # Save loss history
        self.save_loss_history()
        
        self.logger.info("Training completed!")
    
    def save_loss_history(self):
        """Save training and validation loss history for visualization"""
        loss_history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': asdict(self.config)
        }
        
        history_path = os.path.join(self.config.log_dir, 'loss_history.json')
        with open(history_path, 'w') as f:
            json.dump(loss_history, f, indent=2)
        
        self.logger.info(f"Loss history saved: {history_path}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train DiT Quantum Circuit Model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--run_name", type=str, help="Run name for logging")
    
    args = parser.parse_args()
    
    # Create model config
    model_config = DiTConfig(
        d_model=512,
        n_layers=12,
        n_heads=8,
        d_ff=2048,
        max_circuit_length=256,
        dropout=0.1,
        use_flash_attention=True,
        use_rotary_pe=True,
        use_swiglu=True
    )
    
    # Create training config
    training_config = TrainingConfig(
        model_config=model_config,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        run_name=args.run_name,
        use_amp=True,
        compile_model=True,
        use_ema=True
    )
    
    # Create trainer and start training
    trainer = AdvancedTrainer(training_config)
    trainer.train()


if __name__ == "__main__":
    main()
