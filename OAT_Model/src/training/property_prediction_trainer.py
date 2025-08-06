"""
Property Prediction Transformer Training Pipeline

CircuitSpec으로부터 얽힘도, fidelity, robust fidelity를 예측하는 
트랜스포머 모델의 학습 파이프라인
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Tuple
import wandb
import json
import numpy as np
from dataclasses import asdict
import time
import os

# Import model components
from ..models.property_prediction_transformer import (
    PropertyPredictionTransformer,
    PropertyPredictionConfig,
    PropertyPredictionLoss,
    create_property_prediction_model
)

# Import circuit interface
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))
from circuit_interface import CircuitSpec
from gates import GateOperation


class PropertyPredictionDataset(Dataset):
    """양자 회로 특성 예측 데이터셋"""
    
    def __init__(self, data_path: str, split: str = 'train'):
        """
        Args:
            data_path: JSON 데이터 파일 경로
            split: 'train', 'val', 'test'
        """
        self.data_path = Path(data_path)
        self.split = split
        
        # Load data
        self.data = self._load_data()
        
        print(f"✅ {split} 데이터셋 로드 완료: {len(self.data)} 샘플")
    
    def _load_data(self) -> List[Dict]:
        """JSON 데이터 로드 및 전처리 (더미 데이터셋 구조 대응)"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        processed_data = []
        
        # 더미 데이터셋 구조: {"circuits": {circuit_id: circuit_data, ...}}
        if 'circuits' in raw_data:
            circuits_data = raw_data['circuits']
        else:
            # 기존 구조 (리스트) 지원
            circuits_data = raw_data if isinstance(raw_data, list) else [raw_data]
        
        # 딕셔너리인 경우 값들을 리스트로 변환
        if isinstance(circuits_data, dict):
            circuit_items = list(circuits_data.values())
        else:
            circuit_items = circuits_data
        
        for circuit_data in circuit_items:
            try:
                # CircuitSpec 생성
                gates = []
                for gate_data in circuit_data['gates']:
                    gate = GateOperation(
                        name=gate_data['name'],
                        qubits=gate_data['qubits'],
                        parameters=gate_data.get('parameters', [])
                    )
                    gates.append(gate)
                
                circuit_spec = CircuitSpec(
                    circuit_id=circuit_data['circuit_id'],
                    num_qubits=circuit_data['num_qubits'],
                    gates=gates
                )
                
                # 더미 타겟 값 생성 (실제 계산된 값이 없으므로)
                # 회로 복잡도 기반으로 대략적인 값 추정
                num_gates = len(gates)
                num_qubits = circuit_data['num_qubits']
                
                # 간단한 휴리스틱으로 타겟 값 생성
                entanglement = min(0.9, (num_gates * 0.1) / num_qubits)  # 0-0.9 범위
                fidelity = max(0.1, 1.0 - (num_gates * 0.02))  # 게이트 많을수록 fidelity 감소
                robust_fidelity = fidelity * 0.8  # robust는 일반 fidelity보다 낮음
                
                targets = {
                    'entanglement': float(entanglement),
                    'fidelity': float(fidelity),
                    'robust_fidelity': float(robust_fidelity)
                }
                
                # Combined target
                targets['combined'] = torch.tensor([
                    targets['entanglement'],
                    targets['fidelity'], 
                    targets['robust_fidelity']
                ], dtype=torch.float32)
                
                processed_data.append({
                    'circuit_spec': circuit_spec,
                    'targets': targets,
                    'metadata': {
                        'num_qubits': circuit_spec.num_qubits,
                        'num_gates': len(circuit_spec.gates),
                        'circuit_id': circuit_spec.circuit_id
                    }
                })
                
            except Exception as e:
                print(f"⚠️ 데이터 처리 오류 (건너뜀): {e}")
                continue
        
        return processed_data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]


def collate_fn(batch: List[Dict]) -> Dict:
    """배치 데이터 collation"""
    circuit_specs = [item['circuit_spec'] for item in batch]
    
    # 타겟 값들을 텐서로 변환
    targets = {}
    for key in ['entanglement', 'fidelity', 'robust_fidelity']:
        targets[key] = torch.tensor([item['targets'][key] for item in batch], dtype=torch.float32)
    
    targets['combined'] = torch.stack([item['targets']['combined'] for item in batch])
    
    # 메타데이터
    metadata = [item['metadata'] for item in batch]
    
    return {
        'circuit_specs': circuit_specs,
        'targets': targets,
        'metadata': metadata
    }


class PropertyPredictionTrainer:
    """Property Prediction Transformer 학습기"""
    
    def __init__(
        self,
        model: PropertyPredictionTransformer,
        config: PropertyPredictionConfig,
        train_dataset: PropertyPredictionDataset,
        val_dataset: PropertyPredictionDataset,
        save_dir: str = "property_prediction_checkpoints"
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"🎯 학습 디바이스: {self.device}")
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.95)
        )
        
        # Loss function
        self.criterion = PropertyPredictionLoss(
            entanglement_weight=1.0,
            fidelity_weight=1.0,
            robust_fidelity_weight=1.0,
            combined_weight=0.5
        )
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=16,  # 작은 배치 크기 (회로 처리 복잡도 고려)
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0  # 멀티프로세싱 비활성화 (안정성)
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Learning rate scheduler
        total_steps = len(self.train_loader) * 100  # 100 epochs 가정
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []
    
    def train_epoch(self) -> Dict[str, float]:
        """한 에폭 학습"""
        self.model.train()
        
        total_losses = {
            'total': 0.0,
            'entanglement': 0.0,
            'fidelity': 0.0,
            'robust_fidelity': 0.0,
            'combined': 0.0
        }
        
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Forward pass
                circuit_specs = batch['circuit_specs']
                targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
                
                # Model prediction
                predictions = self.model(circuit_specs)
                
                # Move predictions to device
                for key in predictions:
                    predictions[key] = predictions[key].to(self.device)
                
                # Calculate loss
                losses = self.criterion(predictions, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                losses['total'].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                
                # Accumulate losses
                for key, loss in losses.items():
                    total_losses[key] += loss.item()
                
                num_batches += 1
                
                # Update progress bar
                current_lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}",
                    'lr': f"{current_lr:.2e}"
                })
                
            except Exception as e:
                print(f"⚠️ 배치 {batch_idx} 학습 오류: {e}")
                continue
        
        # Average losses
        avg_losses = {key: total_loss / max(num_batches, 1) for key, total_loss in total_losses.items()}
        
        return avg_losses
    
    def validate(self) -> Dict[str, float]:
        """검증"""
        self.model.eval()
        
        total_losses = {
            'total': 0.0,
            'entanglement': 0.0,
            'fidelity': 0.0,
            'robust_fidelity': 0.0,
            'combined': 0.0
        }
        
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                try:
                    circuit_specs = batch['circuit_specs']
                    targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
                    
                    # Model prediction
                    predictions = self.model(circuit_specs)
                    
                    # Move predictions to device
                    for key in predictions:
                        predictions[key] = predictions[key].to(self.device)
                    
                    # Calculate loss
                    losses = self.criterion(predictions, targets)
                    
                    # Accumulate losses
                    for key, loss in losses.items():
                        total_losses[key] += loss.item()
                    
                    num_batches += 1
                    
                    # Store for metrics calculation
                    all_predictions.append({k: v.cpu() for k, v in predictions.items()})
                    all_targets.append({k: v.cpu() for k, v in targets.items()})
                    
                except Exception as e:
                    print(f"⚠️ 검증 배치 오류: {e}")
                    continue
        
        # Average losses
        avg_losses = {key: total_loss / max(num_batches, 1) for key, total_loss in total_losses.items()}
        
        # Calculate additional metrics
        metrics = self._calculate_metrics(all_predictions, all_targets)
        avg_losses.update(metrics)
        
        return avg_losses
    
    def _calculate_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
        """추가 메트릭 계산"""
        if not predictions:
            return {}
        
        metrics = {}
        
        # Concatenate all predictions and targets
        for prop in ['entanglement', 'fidelity', 'robust_fidelity']:
            pred_values = torch.cat([pred[prop] for pred in predictions])
            target_values = torch.cat([target[prop] for target in targets])
            
            # MAE (Mean Absolute Error)
            mae = torch.mean(torch.abs(pred_values - target_values)).item()
            metrics[f'{prop}_mae'] = mae
            
            # R² score approximation
            ss_res = torch.sum((target_values - pred_values) ** 2)
            ss_tot = torch.sum((target_values - torch.mean(target_values)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            metrics[f'{prop}_r2'] = r2.item()
        
        return metrics
    
    def train(self, num_epochs: int = 100):
        """전체 학습 프로세스"""
        print(f"🚀 Property Prediction Transformer 학습 시작")
        print(f"   - 에폭 수: {num_epochs}")
        print(f"   - 학습 샘플: {len(self.train_dataset)}")
        print(f"   - 검증 샘플: {len(self.val_dataset)}")
        print(f"   - 배치 크기: {self.train_loader.batch_size}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_losses = self.train_epoch()
            
            # Validation
            val_losses = self.validate()
            
            # Save best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_checkpoint('best_model.pt')
                print(f"✅ 최고 모델 저장 (val_loss: {self.best_val_loss:.4f})")
            
            # Log results
            epoch_results = {
                'epoch': epoch,
                'train_loss': train_losses['total'],
                'val_loss': val_losses['total'],
                'train_entanglement': train_losses['entanglement'],
                'val_entanglement': val_losses['entanglement'],
                'train_fidelity': train_losses['fidelity'],
                'val_fidelity': val_losses['fidelity'],
                'train_robust_fidelity': train_losses['robust_fidelity'],
                'val_robust_fidelity': val_losses['robust_fidelity'],
                'learning_rate': self.scheduler.get_last_lr()[0]
            }
            
            # Add validation metrics
            for key, value in val_losses.items():
                if key.endswith('_mae') or key.endswith('_r2'):
                    epoch_results[f'val_{key}'] = value
            
            self.training_history.append(epoch_results)
            
            # Print progress
            print(f"Epoch {epoch:3d} | "
                  f"Train: {train_losses['total']:.4f} | "
                  f"Val: {val_losses['total']:.4f} | "
                  f"LR: {self.scheduler.get_last_lr()[0]:.2e}")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        print("🎉 학습 완료!")
        
        # Save final results
        self.save_training_history()
    
    def save_checkpoint(self, filename: str):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config),
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, self.save_dir / filename)
    
    def save_training_history(self):
        """학습 기록 저장"""
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"📊 학습 기록 저장: {history_path}")


def create_datasets(data_path: str, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[PropertyPredictionDataset, PropertyPredictionDataset, PropertyPredictionDataset]:
    """데이터셋 분할 생성"""
    # Load all data
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Handle different data structures
    if 'circuits' in raw_data:
        # 더미 데이터셋 구조: {"circuits": {circuit_id: circuit_data, ...}}
        circuits_data = raw_data['circuits']
        all_data = list(circuits_data.values())
    elif isinstance(raw_data, list):
        # 리스트 구조
        all_data = raw_data
    else:
        # 단일 딕셔너리
        all_data = [raw_data]
    
    # Shuffle data (이제 리스트이므로 안전)
    np.random.shuffle(all_data)
    
    # Split data
    total_size = len(all_data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]
    
    # Save split data
    data_dir = Path(data_path).parent
    
    train_path = data_dir / 'train_data.json'
    val_path = data_dir / 'val_data.json'
    test_path = data_dir / 'test_data.json'
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=2)
    with open(test_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # Create datasets
    train_dataset = PropertyPredictionDataset(train_path, 'train')
    val_dataset = PropertyPredictionDataset(val_path, 'val')
    test_dataset = PropertyPredictionDataset(test_path, 'test')
    
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Configuration
    config = PropertyPredictionConfig(
        d_model=256,
        n_heads=8,
        n_layers=6,
        dropout=0.1,
        learning_rate=1e-4
    )
    
    # Create model
    model = create_property_prediction_model(config)
    
    # Load datasets (예시 경로)
    data_path = "path/to/your/quantum_circuit_data.json"
    
    try:
        train_dataset, val_dataset, test_dataset = create_datasets(data_path)
        
        # Create trainer
        trainer = PropertyPredictionTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            save_dir="property_prediction_checkpoints"
        )
        
        # Start training
        trainer.train(num_epochs=100)
        
    except FileNotFoundError:
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
        print("실제 데이터 경로로 수정해주세요.")
    except Exception as e:
        print(f"❌ 학습 중 오류 발생: {e}")
