"""
Unified Training Script
Uses the unified configuration system for consistent hyperparameter management
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import time

# Add project paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "quantumcommon"))

# Import unified config
from config.unified_training_config import (
    UnifiedTrainingConfig, 
    ConfigManager,
    get_config_by_name
)

# Import models
from models.decision_transformer import DecisionTransformer
from models.property_prediction_transformer import PropertyPredictionTransformer, PropertyPredictionConfig

# Import data
from data.quantum_circuit_dataset import DatasetManager, create_dataloaders

# Import gates
from gates import QuantumGateRegistry


class UnifiedTrainer:
    """Unified trainer for both Decision Transformer and Property Predictor"""
    
    def __init__(self, config: UnifiedTrainingConfig, model_type: str = "decision_transformer"):
        self.config = config
        self.model_type = model_type
        self.device = torch.device(config.model.get_device())
        
        print(f"🚀 Initializing {model_type} trainer")
        print(f"Device: {self.device}")
        print(f"Experiment: {config.experiment.experiment_name}")
        
        # Setup directories and seed
        config.setup_directories()
        config.set_seed()
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.criterion = self._create_loss_function()
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _create_model(self):
        """Create model based on type"""
        if self.model_type == "decision_transformer":
            model_config = self.config.get_model_config_for_decision_transformer()
            return DecisionTransformer(config=model_config)
        
        elif self.model_type == "property_predictor":
            # Create PropertyPredictionConfig from unified config
            prop_config = PropertyPredictionConfig(
                d_model=self.config.model.d_model,
                n_heads=self.config.model.n_heads,
                n_layers=self.config.model.n_layers,
                d_ff=self.config.model.d_ff,
                dropout=self.config.model.dropout,
                max_qubits=self.config.model.max_qubits
            )
            return PropertyPredictionTransformer(prop_config)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _create_optimizer(self):
        """Create optimizer"""
        return AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(self.config.training.beta1, self.config.training.beta2),
            eps=self.config.training.eps
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.training.scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.training.learning_rate * 0.01
            )
        else:
            return None
    
    def _create_loss_function(self):
        """Create loss function based on model type"""
        if self.model_type == "decision_transformer":
            return nn.CrossEntropyLoss()  # Simplified for now
        elif self.model_type == "property_predictor":
            from models.property_prediction_transformer import PropertyPredictionLoss
            return PropertyPredictionLoss(
                entanglement_weight=self.config.property_predictor.entanglement_weight,
                fidelity_weight=self.config.property_predictor.fidelity_weight,
                expressibility_weight=self.config.property_predictor.expressibility_weight
            )
        else:
            return nn.MSELoss()
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Forward pass
                if self.model_type == "decision_transformer":
                    loss = self._train_step_decision_transformer(batch)
                elif self.model_type == "property_predictor":
                    loss = self._train_step_property_predictor(batch)
                else:
                    continue
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.gradient_clip_norm
                    )
                
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                self.current_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
                })
                
                # Log periodically
                if self.current_step % self.config.training.log_every_n_steps == 0:
                    print(f"Step {self.current_step}: loss = {loss.item():.4f}")
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def _train_step_decision_transformer(self, batch):
        """Training step for Decision Transformer"""
        # Simplified training step - implement based on your data format
        # This is a placeholder that needs to be adapted to your actual data
        return torch.tensor(0.1, requires_grad=True, device=self.device)
    
    def _train_step_property_predictor(self, batch):
        """Training step for Property Predictor"""
        # Simplified training step - implement based on your data format
        # This is a placeholder that needs to be adapted to your actual data
        return torch.tensor(0.1, requires_grad=True, device=self.device)
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                try:
                    if self.model_type == "decision_transformer":
                        loss = self._train_step_decision_transformer(batch)
                    elif self.model_type == "property_predictor":
                        loss = self._train_step_property_predictor(batch)
                    else:
                        continue
                    
                    total_loss += loss.item()
                    
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def save_checkpoint(self, filepath):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__,
            'model_type': self.model_type
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['current_epoch']
        self.current_step = checkpoint['current_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded from {filepath}")
    
    def train(self, train_loader, val_loader=None):
        """Main training loop"""
        print(f"🎯 Starting training for {self.config.training.num_epochs} epochs")
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_model_path = Path(self.config.experiment.checkpoint_dir) / "best_model.pt"
                    self.save_checkpoint(best_model_path)
                    print(f"New best model saved with val_loss={val_loss:.4f}")
            else:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}")
            
            # Step scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = Path(self.config.experiment.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
                self.save_checkpoint(checkpoint_path)
        
        print("🎉 Training completed!")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Unified Training Script")
    parser.add_argument("--config", type=str, default="medium", 
                       help="Configuration name (small, medium, large) or path to config file")
    parser.add_argument("--model", type=str, choices=["decision_transformer", "property_predictor"],
                       default="decision_transformer", help="Model type to train")
    parser.add_argument("--experiment-name", type=str, help="Custom experiment name")
    parser.add_argument("--data-path", type=str, help="Path to training data")
    
    args = parser.parse_args()
    
    # Load or create configuration
    config_manager = ConfigManager()
    
    if Path(args.config).exists():
        # Load from file
        config = UnifiedTrainingConfig.load(args.config)
    else:
        # Load predefined config
        config = get_config_by_name(args.config)
    
    # Apply command line overrides
    if args.experiment_name:
        config.experiment.experiment_name = args.experiment_name
    
    if args.data_path:
        config.data.data_path = args.data_path
    
    # Save the configuration
    config_path = Path(config.experiment.output_dir) / f"{config.experiment.experiment_name}_config.json"
    config.save(config_path)
    
    print(f"📋 Configuration saved to {config_path}")
    print(f"🎯 Training {args.model} with config: {args.config}")
    print(f"📊 Experiment: {config.experiment.experiment_name}")
    
    # Create trainer
    trainer = UnifiedTrainer(config, args.model)
    
    print(f"Loading data from {config.data.data_path}")
    
    # Create actual dataloaders
    try:
        # First display model configuration (for verification)
        print("✅ Unified training system initialized successfully!")
        print(f"Model architecture:")
        print(f"  - d_model: {config.model.d_model}")
        print(f"  - n_layers: {config.model.n_layers}")
        print(f"  - n_heads: {config.model.n_heads}")
        print(f"  - dropout: {config.model.dropout}")
        print(f"Training settings:")
        print(f"  - learning_rate: {config.training.learning_rate}")
        print(f"  - batch_size: {config.training.train_batch_size}")
        print(f"  - num_epochs: {config.training.num_epochs}")
        
        # Import the necessary dataset classes based on model type
        if args.model == "property_predictor":
            from data.quantum_circuit_dataset import DatasetManager, create_dataloaders
            from training.property_prediction_trainer import PropertyPredictionDataset, collate_fn
            
            # Create dataset manager and load data
            print(f"\n🔄 Loading quantum circuit data from {config.data.data_path}")
            dataset_manager = DatasetManager(unified_data_path=config.data.data_path)
            
            # Split dataset into train/val/test
            train_quantum_dataset, val_quantum_dataset, test_quantum_dataset = dataset_manager.split_dataset(
                train_ratio=config.data.train_split,
                val_ratio=config.data.val_split,
                test_ratio=config.data.test_split
            )
            
            # Create train/val datasets
            train_dataset = PropertyPredictionDataset(train_quantum_dataset)
            val_dataset = PropertyPredictionDataset(val_quantum_dataset)
            
            print(f"📊 Data loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
            
            # Create data loaders
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config.training.train_batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0
            )
            
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=config.training.val_batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0
            )
            
            print(f"\n🚀 Starting model training with {len(train_loader)} batches per epoch")
            
            # Use the specialized PropertyPredictionTrainer for actual training
            from training.property_prediction_trainer import PropertyPredictionTrainer
            from models.property_prediction_transformer import PropertyPredictionConfig
            
            # Create PropertyPredictionConfig from unified config
            prop_config = PropertyPredictionConfig(
                d_model=config.model.d_model,
                n_heads=config.model.n_heads,
                n_layers=config.model.n_layers,
                d_ff=config.model.d_ff,
                dropout=config.model.dropout,
                attention_mode=config.model.attention_mode,  # 통합 어텐션 모드
                use_rotary_pe=config.model.use_rotary_pe,    # 통합 위치 인코딩 설정
                max_qubits=config.model.max_qubits,
                train_batch_size=config.training.train_batch_size,
                val_batch_size=config.training.val_batch_size,
                learning_rate=config.training.learning_rate
            )
            
            # 실험별 저장 디렉토리 설정
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 모델 설정 기반 저장 이름 생성
            model_info = f"{config.model.attention_mode}_{config.model.d_model}d_{config.model.n_layers}l_{config.model.n_heads}h"
            save_dir = f"{config.experiment.checkpoint_dir}/property_{model_info}_{timestamp}"
            
            # Create PropertyPredictionTrainer
            prop_trainer = PropertyPredictionTrainer(
                config=prop_config,
                model=trainer.model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                save_dir=save_dir
            )
            
            print(f"💾 모델 저장 경로: {save_dir}")
            
            # Start the actual training with the specialized trainer
            prop_trainer.train(num_epochs=config.training.num_epochs)
        
        elif args.model == "decision_transformer":
            # TODO: Implement decision transformer data loading and training
            print("Decision transformer training not yet implemented")
        
        print("\n✅ Training complete!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
