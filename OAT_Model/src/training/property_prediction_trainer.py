"""
Property Prediction Transformer Training Pipeline

CircuitSpec으로부터 얽힘도, fidelity, robust fidelity를 예측하는 
트랜스포머 모델의 학습 파이프라인
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Tuple
import wandb
import json
import numpy as np
import math
from dataclasses import asdict
import time
import os

# Import model components
from models.property_prediction_transformer import (
    PropertyPredictionTransformer,
    PropertyPredictionConfig,
    PropertyPredictionLoss,
    create_property_prediction_model
)

# Import quantum dataset
from data.quantum_circuit_dataset import (
    DatasetManager,
    QuantumCircuitDataset,
    CircuitData,
    create_dataloaders
)

# Import circuit interface
import sys
sys.path.append(str(Path(__file__).parent.parent.parent/ "quantumcommon"))
from circuit_interface import CircuitSpec
from gates import GateOperation


class PropertyPredictionDataset:
    """양자 회로 특성 예측을 위한 데이터셋 래퍼"""
    
    def __init__(self, quantum_dataset: QuantumCircuitDataset):
        """
        Args:
            quantum_dataset: QuantumCircuitDataset 인스턴스
        """
        self.quantum_dataset = quantum_dataset
        
        print(f"[INIT] Property Prediction 데이터셋 초기화: {len(self.quantum_dataset)} 샘플")
    
    def __len__(self) -> int:
        return len(self.quantum_dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        """CircuitData를 Property Prediction 형식으로 변환"""
        circuit_data: CircuitData = self.quantum_dataset[idx]
        
        # Check if measurement result exists
        if circuit_data.measurement_result is None:
            raise ValueError(f"No measurement result for circuit {circuit_data.circuit_id}")
            
        measurement = circuit_data.measurement_result
        
        # Validate required fields
        if measurement.fidelity is None:
            raise ValueError(f"Missing fidelity for circuit {circuit_data.circuit_id}")
        
        # Extract expressibility (KL divergence only)
        expressibility_value = 0.0
        if measurement.expressibility and isinstance(measurement.expressibility, dict):
            kl_div = measurement.expressibility.get('kl_divergence', 0.0)
            # Use KL divergence directly as expressibility
            expressibility_value = float(kl_div)
        
        # Extract robust fidelity (디버그 추가)
        robust_fidelity_value = 0.0
        if hasattr(measurement, 'robust_fidelity') and measurement.robust_fidelity is not None:
            robust_fidelity_value = float(measurement.robust_fidelity)
        else:
            raise ValueError(f"Missing robust_fidelity for circuit {circuit_data.circuit_id}")
        
        targets = {
            'entanglement': float(measurement.entanglement) if measurement.entanglement is not None else 0.0,
            'fidelity': float(measurement.fidelity),
            'expressibility': float(expressibility_value),
            'robust_fidelity': robust_fidelity_value
        }
        
        # Combined target vector
        targets['combined'] = torch.tensor([
            targets['entanglement'],
            targets['fidelity'], 
            targets['expressibility'],
            targets['robust_fidelity']
        ], dtype=torch.float32)
        
        return {
            'circuit_spec': circuit_data.circuit_spec,
            'targets': targets,
            'metadata': {
                'num_qubits': circuit_data.num_qubits,
                'num_gates': len(circuit_data.gates),
                'circuit_id': circuit_data.circuit_id,
                'depth': measurement.depth
            }
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """배치 데이터 collation"""
    # Filter out None items from batch
    valid_batch = [item for item in batch if item is not None]
    
    if not valid_batch:
        raise ValueError("[EMPTY] - No valid items in batch")
    
    circuit_specs = [item['circuit_spec'] for item in valid_batch]
    
    # 타겟 값들을 텐서로 변환
    targets = {}
    for key in ['entanglement', 'fidelity', 'expressibility', 'robust_fidelity']:
        targets[key] = torch.tensor([item['targets'][key] for item in valid_batch], dtype=torch.float32)
    
    targets['combined'] = torch.stack([item['targets']['combined'] for item in valid_batch])
    
    # 메타데이터
    metadata = [item['metadata'] for item in valid_batch]
    
    return {
        'circuit_specs': circuit_specs,
        'targets': targets,
        'metadata': metadata
    }


class PropertyPredictionTrainer:
    """Property Prediction Transformer 학습기"""
    
    def __init__(
        self,
        config: PropertyPredictionConfig,
        model: nn.Module,
        train_dataset: PropertyPredictionDataset,
        val_dataset: PropertyPredictionDataset,
        save_dir: str = './OAT_Model/checkpoints'
    ):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 학습 파라미터 설정
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.train_batch_size,
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=0,  # 멀티프로세싱 비활성화 (안정성)
            pin_memory=True if self.device.type == 'cuda' else False  # GPU 메모리 최적화
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Create optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay if hasattr(config, 'weight_decay') else 0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Create loss criterion
        self.criterion = PropertyPredictionLoss(
            entanglement_weight=getattr(config, 'weight_entanglement', 1.0),
            fidelity_weight=getattr(config, 'weight_fidelity', 1.0),
            expressibility_weight=getattr(config, 'weight_expressibility', 1.0),
            combined_weight=getattr(config, 'weight_combined', 1.0)
        )
        
        # Learning rate scheduler - 더 안정적인 스케줄러 사용
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # GPU 메모리 최적화 설정
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()  # 초기 메모리 정리
            torch.backends.cudnn.benchmark = True  # cuDNN 최적화
            torch.backends.cudnn.deterministic = False  # 성능 우선
            print(f"[GPU] 메모리 최적화 활성화 - 사용 가능: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        
        # 디버깅을 위한 예측/정답 추적
        self.debug_predictions = []
        
        # Early stopping 설정
        self.patience = getattr(config, 'early_stopping_patience', 15)  # 기본값 15 에폭
        self.early_stopping_counter = 0
        self.min_delta = getattr(config, 'early_stopping_delta', 0.001)  # 최소 개선 필요치
        self.early_stopped = False
        
        # 학습률 최소값 설정 (너무 작아지면 학습이 진행되지 않음)
        self.min_lr = getattr(config, 'min_learning_rate', 1e-7)
        
        # 메모리 관리 설정
        self.memory_cleanup_frequency = getattr(config, 'memory_cleanup_frequency', 10)
        self.debug_targets = []
    
    def train_epoch(self) -> Dict[str, float]:
        """한 에폭 학습"""
        self.model.train()
        
        total_losses = {
            'total': 0.0,
            'entanglement': 0.0,
            'fidelity': 0.0,
            'expressibility': 0.0,
            'robust_fidelity': 0.0,
            'combined': 0.0
        }
        
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # GPU 메모리 최적화
                if self.device.type == 'cuda' and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()  # 주기적 메모리 정리
                
                # Forward pass
                circuit_specs = batch['circuit_specs']
                targets = {k: v.to(self.device, non_blocking=True).float() for k, v in batch['targets'].items()}
                
                # Model prediction - AMP 제거, 모든 텐서를 float으로 통일
                predictions = self.model(circuit_specs)
                
                # Move predictions to device and ensure float type
                for key in predictions:
                    predictions[key] = predictions[key].to(self.device, non_blocking=True).float()
                
                # 디버깅: 첫 번째 배치의 예측과 정답 비교
                if batch_idx == 0:
                    self._debug_predictions_vs_targets(predictions, targets, batch_idx)
                
                # Calculate loss with NaN check
                losses = self.criterion(predictions, targets)
                
                # NaN loss 체크 및 스킵
                if torch.isnan(losses['total']) or torch.isinf(losses['total']):
                    print(f"[WARNING] 배치 {batch_idx}: NaN/Inf loss 감지, 스킵")
                    continue
                
                # Backward pass - float 타입 보장
                self.optimizer.zero_grad()
                total_loss = losses['total'].float()  # 명시적으로 float으로 변환
                total_loss.backward()
                
                # 그래디언트 NaN 체크
                has_nan_grad = False
                for name, param in self.model.named_parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        print(f"[WARNING] 배치 {batch_idx}: {name}에서 NaN 그래디언트 감지")
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    self.optimizer.zero_grad()
                    continue
                
                # Gradient clipping (더 강한 클리핑)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                self.optimizer.step()
                # ReduceLROnPlateau는 validation loss로 step
                
                # Accumulate losses
                for key, loss in losses.items():
                    total_losses[key] += loss.item()
                
                num_batches += 1
                
                # Update progress bar with GPU 메모리 정보
                current_lr = self.optimizer.param_groups[0]['lr']
                postfix = {
                    'loss': f"{losses['total'].item():.4f}",
                    'lr': f"{current_lr:.2e}"
                }
                
                if self.device.type == 'cuda':
                    gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                    postfix['GPU'] = f"{gpu_memory:.1f}GB"
                
                progress_bar.set_postfix(postfix)
                
            except Exception as e:
                print(f"[ERROR] 배치 {batch_idx} 학습 오류: {e}")
                import traceback
                traceback.print_exc()
                
                # GPU 메모리 정리
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                continue
        
        # Average losses
        avg_losses = {key: total_loss / max(num_batches, 1) for key, total_loss in total_losses.items()}
        
        return avg_losses
    
    def _debug_predictions_vs_targets(self, predictions: Dict[str, torch.Tensor], 
                                    targets: Dict[str, torch.Tensor], batch_idx: int):
        """예측값과 정답 레이블 비교 디버깅"""
        print(f"\n[DEBUG] [배치 {batch_idx}] 예측 vs 정답 디버깅:")
        
        # 첫 번째 샘플만 분석
        sample_idx = 0
        
        for property_name in ['entanglement', 'fidelity', 'expressibility', 'robust_fidelity']:
            if property_name in predictions and property_name in targets:
                pred_val = predictions[property_name][sample_idx].item()
                target_val = targets[property_name][sample_idx].item()
                diff = abs(pred_val - target_val)
                
                print(f"  [DATA] {property_name:15s}: 예측={pred_val:7.4f}, 정답={target_val:7.4f}, 차이={diff:7.4f}")
        
        # Combined 예측 (4차원 벡터)
        if 'combined' in predictions and 'combined' in targets:
            pred_combined = predictions['combined'][sample_idx]
            target_combined = targets['combined'][sample_idx]
            
            print(f"  [DATA] {'combined':15s}:")
            property_names = ['entanglement', 'fidelity', 'expressibility', 'robust_fidelity']
            for i, prop_name in enumerate(property_names):
                if i < len(pred_combined) and i < len(target_combined):
                    pred_val = pred_combined[i].item()
                    target_val = target_combined[i].item()
                    diff = abs(pred_val - target_val)
                    print(f"    - {prop_name:13s}: 예측={pred_val:7.4f}, 정답={target_val:7.4f}, 차이={diff:7.4f}")
        
        # 예측값 범위 체크
        print(f"  [RANGE] 예측값 범위 체크:")
        for property_name, pred_tensor in predictions.items():
            if torch.is_tensor(pred_tensor):
                min_val = pred_tensor.min().item()
                max_val = pred_tensor.max().item()
                mean_val = pred_tensor.mean().item()
                print(f"    - {property_name:13s}: min={min_val:7.4f}, max={max_val:7.4f}, mean={mean_val:7.4f}")
        
        # NaN/Inf 체크
        nan_found = False
        for property_name, pred_tensor in predictions.items():
            if torch.is_tensor(pred_tensor):
                if torch.isnan(pred_tensor).any() or torch.isinf(pred_tensor).any():
                    print(f"  [WARNING] {property_name}에서 NaN/Inf 감지!")
                    nan_found = True
        
        if not nan_found:
            print(f"  [OK] 모든 예측값이 정상 범위 내에 있습니다.")
        
        print()  # 빈 줄 추가
    
    def validate(self) -> Dict[str, float]:
        """검증"""
        self.model.eval()
        
        total_losses = {
            'total': 0.0,
            'entanglement': 0.0,
            'fidelity': 0.0,
            'expressibility': 0.0,
            'robust_fidelity': 0.0,
            'combined': 0.0
        }
        
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # GPU 메모리 최적화 - 검증에도 적용
                    if self.device.type == 'cuda' and batch_idx % 10 == 0:
                        torch.cuda.empty_cache()  # 주기적 메모리 정리
                    
                    circuit_specs = batch['circuit_specs']
                    targets = {k: v.to(self.device, non_blocking=True).float() for k, v in batch['targets'].items()}
                    
                    # Model prediction with float consistency
                    predictions = self.model(circuit_specs)
                    
                    # Move predictions to device and ensure float type
                    for key in predictions:
                        predictions[key] = predictions[key].to(self.device, non_blocking=True).float()
                    
                    # NaN 체크 (검증 과정에서도 필요)
                    has_nan = False
                    for key, pred in predictions.items():
                        if torch.isnan(pred).any() or torch.isinf(pred).any():
                            has_nan = True
                            print(f"[WARNING] 검증 배치 {batch_idx}: {key}에서 NaN/Inf 예측값 감지")
                            break
                    
                    if has_nan:
                        continue
                    
                    # Calculate loss
                    losses = self.criterion(predictions, targets)
                    
                    # NaN loss 체크
                    if torch.isnan(losses['total']) or torch.isinf(losses['total']):
                        print(f"[WARNING] 검증 배치 {batch_idx}: NaN/Inf loss 감지, 스킵")
                        continue
                    
                    # Accumulate losses
                    for key, loss in losses.items():
                        total_losses[key] += loss.item()
                    
                    # 디버깅용 예측/정답 저장 (일부만)
                    if len(all_predictions) < 100:  # 최대 100개 샘플만 저장
                        all_predictions.append({k: v.detach().cpu() for k, v in predictions.items()})
                        all_targets.append({k: v.detach().cpu() for k, v in targets.items()})
                    
                    num_batches += 1
                    
                    # Update progress bar with GPU info
                    postfix = {'loss': f"{losses['total'].item():.4f}"}
                    if self.device.type == 'cuda':
                        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                        postfix['GPU'] = f"{gpu_memory:.1f}GB"
                    
                    progress_bar.set_postfix(postfix)
                
                except Exception as e:
                    print(f"[ERROR] 검증 배치 {batch_idx} 오류: {e}")
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                        
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Average losses
        avg_losses = {key: total_loss / max(num_batches, 1) for key, total_loss in total_losses.items()}
        
        # Calculate additional metrics
        metrics = self._calculate_metrics(all_predictions, all_targets)
        avg_losses.update(metrics)
        
        return avg_losses
    
    def _calculate_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
        """추가 메트릭 계산 - 개선된 에러 처리와 통계"""
        if not predictions or not targets:
            print("\n[WARNING] 메트릭 계산을 위한 데이터가 없습니다.")
            return {}
            
        try:
            metrics = {}
            
            # 분석할 프로퍼티 지정
            properties = ['entanglement', 'fidelity', 'expressibility']
            available_props = [p for p in properties if all(p in pred and p in target for pred, target in zip(predictions, targets))]
            
            if not available_props:
                print("\n[WARNING] 메트릭 계산을 위한 프로퍼티가 없습니다.")
                return {}
                
            print(f"\n[INFO] 분석할 프로퍼티: {available_props}")
            
            # 프로퍼티별 메트릭 계산
            for prop in available_props:
                try:
                    # 예측/타겟값 모으기 (유효한 값만)
                    pred_values = []
                    target_values = []
                    
                    for pred, target in zip(predictions, targets):
                        if prop in pred and prop in target:
                            # NaN/Inf 체크
                            p_vals = pred[prop]
                            t_vals = target[prop]
                            
                            valid_indices = ~(torch.isnan(p_vals) | torch.isinf(p_vals) | torch.isnan(t_vals) | torch.isinf(t_vals))
                            if valid_indices.any():
                                pred_values.append(p_vals[valid_indices])
                                target_values.append(t_vals[valid_indices])
                    
                    if not pred_values:
                        print(f"  - {prop}: 유효한 값이 없습니다")
                        continue
                    
                    # 유효한 값만 모아서 텐서로 변환
                    pred_tensor = torch.cat(pred_values)
                    target_tensor = torch.cat(target_values)
                    
                    # 값 범위 확인
                    pred_min, pred_max = pred_tensor.min().item(), pred_tensor.max().item()
                    target_min, target_max = target_tensor.min().item(), target_tensor.max().item()
                    
                    # MAE (Mean Absolute Error)
                    mae = torch.mean(torch.abs(target_tensor - pred_tensor)).item()
                    metrics[f'{prop}_mae'] = mae
                    
                    # MSE (Mean Squared Error)
                    mse = torch.mean((target_tensor - pred_tensor) ** 2).item()
                    metrics[f'{prop}_mse'] = mse
                    
                    # RMSE (Root Mean Squared Error)
                    rmse = math.sqrt(mse)
                    metrics[f'{prop}_rmse'] = rmse
                    
                    # R² score 계산
                    ss_res = torch.sum((target_tensor - pred_tensor) ** 2)
                    ss_tot = torch.sum((target_tensor - torch.mean(target_tensor)) ** 2)
                    r2 = 1 - (ss_res / (ss_tot + 1e-8))
                    r2 = max(min(r2.item(), 1.0), -1.0)  # R2는 -∞ ~ 1 범위지만, 표시를 위해 제한
                    metrics[f'{prop}_r2'] = r2
                    
                    # 평균 편향 (Mean Bias)
                    mean_bias = torch.mean(pred_tensor - target_tensor).item()
                    metrics[f'{prop}_bias'] = mean_bias
                    
                    # 상관계수 (Pearson correlation)
                    if len(pred_tensor) > 1:  # 상관계수는 2개 이상의 샘플 필요
                        pred_std = torch.std(pred_tensor)
                        target_std = torch.std(target_tensor)
                        if pred_std > 0 and target_std > 0:  # 0으로 나누는 것 방지
                            cov = torch.mean((pred_tensor - torch.mean(pred_tensor)) * (target_tensor - torch.mean(target_tensor)))
                            corr = cov / (pred_std * target_std)
                            metrics[f'{prop}_corr'] = corr.item()
                    
                    print(f"  - {prop}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, "
                          f"bias={mean_bias:.4f}, range=({pred_min:.2f}-{pred_max:.2f})")
                    
                except Exception as e:
                    print(f"  - {prop} 메트릭 계산 중 오류: {e}")
            
            return metrics
            
        except Exception as e:
            print(f"\n[ERROR] 메트릭 계산 오류: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def train(self, num_epochs: int = 100, resume_checkpoint: str = None) -> Dict[str, Any]:
        """
        전체 학습 프로세스 - Early stopping 및 학습 재개 기능 추가
        
        Args:
            num_epochs: 최대 에폭 수
            resume_checkpoint: 재개할 체크포인트 경로, None이면 처음부터 학습
            
        Returns:
            Dict[str, Any]: 학습 기록 및 상태 정보 포함
        """
        # 학습 재개 처리
        start_epoch = 0
        if resume_checkpoint:
            if self.load_checkpoint(resume_checkpoint):
                start_epoch = self.current_epoch + 1  # 다음 에폭부터 시작
                print(f"\n[RESUME] 학습 재개 준비 완료: 에폭 {start_epoch}부터 {num_epochs}까지 학습 진행")
            else:
                print(f"\n[WARNING] 체크포인트 로딩 실패, 처음부터 학습 진행")
        
        print(f"\n[START] Property Prediction Transformer 학습 시작")
        print(f"   - 에폭 수: {start_epoch} 시작, {num_epochs} 까지 (최대 {num_epochs - start_epoch} 에폭)")
        print(f"   - 학습 샘플: {len(self.train_dataset)}")
        print(f"   - 검증 샘플: {len(self.val_dataset)}")
        print(f"   - 배치 크기: {self.train_loader.batch_size}")
        print(f"   - Early stopping 인내: {self.patience} 에폭 (최소 개선치: {self.min_delta:.6f})")
        print(f"   - 현재 최적 검증 손실: {self.best_val_loss:.6f}")
        print(f"   - Early stopping 카운터: {self.early_stopping_counter}/{self.patience}")
        
        # 이미 Early stopping 조건에 도달한 경우
        if self.early_stopping_counter >= self.patience:
            print(f"\n[WARNING] Early stopping 임계치({self.patience})에 이미 도달했습니다. 학습을 진행하려면 early_stopping_counter를 수동으로 리셋해야 합니다.")
            return False
        
        start_time = time.time()
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            self.current_epoch = epoch
            
            # 메모리 정리 (주기적)
            if self.device.type == 'cuda' and epoch % 3 == 0:
                torch.cuda.empty_cache()
            
            # Training
            train_losses = self.train_epoch()
            
            # Validation
            val_losses = self.validate()
            
            epoch_duration = time.time() - epoch_start_time
            
            # Early stopping 처리
            improved = False
            
            # 스케줄러 업데이트 (validation loss 기반)
            self.scheduler.step(val_losses['total'])
            
            # 최적 모델 저장
            if val_losses['total'] < self.best_val_loss - self.min_delta:
                improved = True
                self.early_stopping_counter = 0  # 개선되었으니 카운터 리셋
                self.best_val_loss = val_losses['total']
                self.save_checkpoint('best_model.pt')
                print(f"[SAVE] 최고 모델 저장 (val_loss: {self.best_val_loss:.4f})")
            else:
                self.early_stopping_counter += 1
                print(f"   [WAIT] 개선되지 않음: {self.early_stopping_counter}/{self.patience} (최고: {self.best_val_loss:.4f})")
            
            # 학습률 감소 체크
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr <= self.min_lr:
                print(f"[WARNING] 학습률이 최소값({self.min_lr:.8f})에 도달했습니다. 학습 중지.")
                break
            
            # Log results
            epoch_results = {
                'epoch': epoch,
                'train_loss': train_losses['total'],
                'val_loss': val_losses['total'],
                'train_entanglement': train_losses['entanglement'],
                'val_entanglement': val_losses['entanglement'],
                'train_fidelity': train_losses['fidelity'],
                'val_fidelity': val_losses['fidelity'],
                'train_expressibility': train_losses['expressibility'],
                'val_expressibility': val_losses['expressibility'],
                'learning_rate': current_lr,
                'duration_sec': epoch_duration,
                'improved': improved
            }
            
            # 추가 검증 메트릭들 기록
            for key, value in val_losses.items():
                if key.endswith('_mae') or key.endswith('_r2') or key.endswith('_corr') or key.endswith('_rmse'):
                    epoch_results[f'val_{key}'] = value
            
            self.training_history.append(epoch_results)
            
            # 상세 진행률 출력 (4가지 정확도 메트릭 포함)
            metrics_str = ""
            for prop in ['entanglement', 'fidelity', 'expressibility']:
                mae_key = f'val_{prop}_mae'
                r2_key = f'val_{prop}_r2'
                if mae_key in val_losses and r2_key in val_losses:
                    metrics_str += f" | {prop[:3].upper()}: MAE={val_losses[mae_key]:.3f}, R²={val_losses[r2_key]:.3f}"
            
            print(f"Epoch {epoch:3d}/{num_epochs-1} | "
                  f"Train: {train_losses['total']:.4f} | "
                  f"Val: {val_losses['total']:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_duration:.1f}s{metrics_str}")
            
            # 주기적 체크포인트 저장
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
                
            # Early stopping 효과
            if self.early_stopping_counter >= self.patience:
                print(f"[WARNING] Early stopping 활성화: {self.patience} 에폭 동안 개선 없음")
                self.early_stopped = True
                break
        
        # 학습 완료 메시지
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if self.early_stopped:
            print(f"[DONE] Early stopping으로 학습 조기 종료! (총 {self.current_epoch+1} 에폭, {int(hours)}h {int(minutes)}m {int(seconds)}s)")
        else:
            print(f"[DONE] 계획된 학습 완료! (총 {self.current_epoch+1} 에폭, {int(hours)}h {int(minutes)}m {int(seconds)}s)")
        
        # 최종 결과 저장 및 시각화용 데이터 생성
        self.save_training_history()
        self.save_metrics_for_visualization()
        
        # 학습 통계 계산
        total_duration = total_time  # 이미 계산된 total_time 사용
        epoch_count = self.current_epoch - start_epoch + 1
        epoch_mean_time = total_duration / max(1, epoch_count)
        best_epoch = 0
        best_val_loss = float('inf')
        
        # 최적 에폭 찾기
        for i, epoch_data in enumerate(self.training_history):
            if epoch_data.get('val_total', float('inf')) < best_val_loss:
                best_val_loss = epoch_data.get('val_total', float('inf'))
                best_epoch = i
        
        # 최종 학습률 확인
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # 결과 반환
        return {
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': best_epoch,
            'early_stopped': self.early_stopped,
            'early_stopping_counter': self.early_stopping_counter,
            'last_epoch': self.current_epoch,
            'total_epochs': epoch_count,
            'total_duration': total_duration,
            'epoch_mean_time': epoch_mean_time,
            'final_lr': current_lr,
            'best_model_path': str(self.save_dir / 'best_model.pt'),
            'device': str(self.device),
            'metrics_file': str(self.save_dir / 'training_metrics.json'),
            'visualization_data': str(self.save_dir / 'visualization_data.json')
        }
    
    def save_checkpoint(self, filename: str):
        """체크포인트 저장 - 메타데이터 추가 및 보안 개선"""
        try:
            # 현재 GPU 메모리 사용량 확인
            gpu_memory_info = None
            if self.device.type == 'cuda':
                gpu_memory_info = {
                    'allocated': torch.cuda.memory_allocated() / (1024**3),
                    'reserved': torch.cuda.memory_reserved() / (1024**3),
                    'max_allocated': torch.cuda.max_memory_allocated() / (1024**3)
                }
            
            # 체크포인트 데이터 구성
            checkpoint = {
                # 학습 상태
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss,
                
                # 구성 및 기록
                'config': asdict(self.config),
                'training_history': self.training_history,
                
                # 추가 메타데이터
                'early_stopping': {
                    'counter': self.early_stopping_counter,
                    'patience': self.patience,
                    'min_delta': self.min_delta,
                    'stopped_early': self.early_stopped
                },
                
                # 시스템 정보
                'timestamp': time.time(),
                'save_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'device': str(self.device),
                'gpu_memory': gpu_memory_info
            }
            
            # 체크포인트 저장 경로 구성
            checkpoint_path = self.save_dir / filename
            
            # 임시 파일로 저장 후 이동 (파일 손상 방지)
            temp_path = self.save_dir / f"temp_{filename}"
            torch.save(checkpoint, temp_path)
            
            # 이미 파일이 있는 경우 백업
            if checkpoint_path.exists():
                backup_path = self.save_dir / f"backup_{filename}"
                if backup_path.exists():
                    backup_path.unlink()  # 기존 백업 삭제
                checkpoint_path.rename(backup_path)  # 기존 파일을 백업으로 이동
            
            # 임시 파일을 최종 파일로 이름 변경
            temp_path.rename(checkpoint_path)
            
            # 성공 메시지 (상세 출력 옵션
            if 'best_model' in filename:
                print(f"[DONE] 체크포인트 저장 완료: {checkpoint_path.name} ")
            
            return True
            
        except Exception as e:
            print(f"\n[ERROR] 체크포인트 저장 오류: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_training_history(self):
        """학습 기록 저장"""
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"[SAVE] 학습 기록 저장: {history_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        체크포인트를 불러와 모델과 학습 상태를 복원
        
        Args:
            checkpoint_path: 체크포인트 파일 경로
            
        Returns:
            bool: 로딩 성공 여부
            
        """
        try:
            # Path 객체로 변환
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                print(f"\n[ERROR] 체크포인트 파일이 없습니다: {checkpoint_path}")
                return False
                
            # CPU에서 로딩 (안정성을 위해)
            print(f"\n[LOAD] 체크포인트 로딩 중: {checkpoint_path.name}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 기본 필수 필드 검색
            required_fields = ['model_state_dict', 'optimizer_state_dict', 'epoch']
            for field in required_fields:
                if field not in checkpoint:
                    print(f"\n[ERROR] 체크포인트에 필수 필드 '{field}'가 없습니다.")
                    return False
            
            # 상세 정보 출력 (메타데이터)
            if 'save_date' in checkpoint:
                print(f"  - 저장 시점: {checkpoint['save_date']}")
            print(f"  - 에폭: {checkpoint['epoch']}")
            if 'best_val_loss' in checkpoint:
                print(f"  - 최적 검증 손실: {checkpoint['best_val_loss']:.6f}")
            
            # GPU 메모리 정리
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # 모델 가중치 로딩
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 추가 상태 복원
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 스케줄러 복원
            if 'scheduler_state_dict' in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                    print(f"  ⚠️ 스케줄러 복원 오류 (skip): {e}")
            
            # 학습 기록 복원
            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']
            
            # 기타 필드 복원
            self.current_epoch = checkpoint['epoch']
            if 'best_val_loss' in checkpoint:
                self.best_val_loss = checkpoint['best_val_loss']
            
            # Early stopping 관련 필드 복원
            if 'early_stopping' in checkpoint:
                es_info = checkpoint['early_stopping']
                if 'counter' in es_info:
                    self.early_stopping_counter = es_info['counter']
                if 'patience' in es_info:
                    self.patience = es_info['patience']
                if 'min_delta' in es_info:
                    self.min_delta = es_info['min_delta']
                if 'stopped_early' in es_info:
                    self.early_stopped = es_info['stopped_early']
            
            # 모델을 적절한 기기로 이동
            self.model = self.model.to(self.device)
            
            print(f"\n✅ 체크포인트 로딩 성공! 학습을 에폭 {self.current_epoch+1}부터 계속합니다.")
            return True
            
        except Exception as e:
            print(f"\n⚠️ 체크포인트 로딩 오류: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_metrics_for_visualization(self):
        """시각화를 위한 메트릭 데이터 저장"""
        try:
            import json
            from datetime import datetime
            
            # 시각화용 데이터 구조 생성
            visualization_data = {
                'metadata': {
                    'experiment_name': 'property_prediction_training',
                    'timestamp': datetime.now().isoformat(),
                    'total_epochs': len(self.training_history),
                    'device': str(self.device),
                    'model_config': {
                        'd_model': getattr(self.config, 'd_model', 512),
                        'n_heads': getattr(self.config, 'n_heads', 8),
                        'n_layers': getattr(self.config, 'n_layers', 6),
                        'attention_mode': getattr(self.config, 'attention_mode', 'advanced')
                    }
                },
                'metrics': {
                    'epochs': [],
                    'train_loss': [],
                    'val_loss': [],
                    'learning_rate': [],
                    'duration_sec': [],
                    'properties': {
                        'entanglement': {
                            'train_loss': [], 'val_loss': [],
                            'val_mae': [], 'val_rmse': [], 'val_r2': [], 'val_corr': []
                        },
                        'fidelity': {
                            'train_loss': [], 'val_loss': [],
                            'val_mae': [], 'val_rmse': [], 'val_r2': [], 'val_corr': []
                        },
                        'expressibility': {
                            'train_loss': [], 'val_loss': [],
                            'val_mae': [], 'val_rmse': [], 'val_r2': [], 'val_corr': []
                        }
                    }
                }
            }
            
            # 에포크별 데이터 추출
            for epoch_data in self.training_history:
                visualization_data['metrics']['epochs'].append(epoch_data.get('epoch', 0))
                visualization_data['metrics']['train_loss'].append(epoch_data.get('train_loss', 0.0))
                visualization_data['metrics']['val_loss'].append(epoch_data.get('val_loss', 0.0))
                visualization_data['metrics']['learning_rate'].append(epoch_data.get('learning_rate', 0.0))
                visualization_data['metrics']['duration_sec'].append(epoch_data.get('duration_sec', 0.0))
                
                # 프로퍼티별 메트릭 추출
                for prop in ['entanglement', 'fidelity', 'expressibility']:
                    prop_data = visualization_data['metrics']['properties'][prop]
                    prop_data['train_loss'].append(epoch_data.get(f'train_{prop}', 0.0))
                    prop_data['val_loss'].append(epoch_data.get(f'val_{prop}', 0.0))
                    
                    # 정확도 메트릭들
                    for metric in ['mae', 'rmse', 'r2', 'corr']:
                        key = f'val_{prop}_{metric}'
                        prop_data[f'val_{metric}'].append(epoch_data.get(key, 0.0))
            
            # 시각화 데이터 저장
            viz_file = self.save_dir / 'visualization_data.json'
            with open(viz_file, 'w', encoding='utf-8') as f:
                json.dump(visualization_data, f, indent=2, ensure_ascii=False)
            
            # 요약 통계 계산 및 저장
            summary_stats = self._calculate_training_summary()
            summary_file = self.save_dir / 'training_summary.json'
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_stats, f, indent=2, ensure_ascii=False)
            
            print(f"📊 시각화 데이터 저장 완료:")
            print(f"  - 메트릭 데이터: {viz_file}")
            print(f"  - 학습 요약: {summary_file}")
            
        except Exception as e:
            print(f"[ERROR] 시각화 데이터 저장 실패: {e}")
            import traceback
            traceback.print_exc()
    
    def _calculate_training_summary(self) -> dict:
        """학습 요약 통계 계산"""
        if not self.training_history:
            return {}
        
        summary = {
            'training_overview': {
                'total_epochs': len(self.training_history),
                'best_epoch': 0,
                'best_val_loss': self.best_val_loss,
                'early_stopped': getattr(self, 'early_stopped', False),
                'final_learning_rate': self.training_history[-1].get('learning_rate', 0.0)
            },
            'loss_progression': {
                'initial_train_loss': self.training_history[0].get('train_loss', 0.0),
                'final_train_loss': self.training_history[-1].get('train_loss', 0.0),
                'initial_val_loss': self.training_history[0].get('val_loss', 0.0),
                'final_val_loss': self.training_history[-1].get('val_loss', 0.0)
            },
            'property_performance': {}
        }
        
        # 최적 에포크 찾기
        best_val_loss = float('inf')
        for i, epoch_data in enumerate(self.training_history):
            if epoch_data.get('val_loss', float('inf')) < best_val_loss:
                best_val_loss = epoch_data.get('val_loss', float('inf'))
                summary['training_overview']['best_epoch'] = i
        
        # 프로퍼티별 최종 성능
        final_epoch = self.training_history[-1]
        for prop in ['entanglement', 'fidelity', 'expressibility']:
            prop_summary = {}
            for metric in ['mae', 'rmse', 'r2', 'corr']:
                key = f'val_{prop}_{metric}'
                if key in final_epoch:
                    prop_summary[f'final_{metric}'] = final_epoch[key]
            
            if prop_summary:
                summary['property_performance'][prop] = prop_summary
        
        return summary


def create_datasets(data_path: str, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                   enable_augmentation: bool = True) -> Tuple[PropertyPredictionDataset, PropertyPredictionDataset, PropertyPredictionDataset]:
    """merged_data.json을 사용한 데이터셋 분할 생성 (증강 지원)"""
    # Create dataset manager
    manager = DatasetManager(unified_data_path=data_path)
    
    # Split quantum datasets
    train_quantum, val_quantum, test_quantum = manager.split_dataset(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=1.0 - train_ratio - val_ratio
    )
    
    # Apply augmentation to training set if enabled
    if enable_augmentation:
        from data.augmented_dataset import create_augmented_datasets
        train_quantum, val_quantum, test_quantum = create_augmented_datasets(
            train_quantum, val_quantum, test_quantum,
            mixup_samples=500,
            noise_samples=500,
            param_random_samples=1000
        )
    
    # Wrap with PropertyPredictionDataset
    train_dataset = PropertyPredictionDataset(train_quantum)
    val_dataset = PropertyPredictionDataset(val_quantum)
    test_dataset = PropertyPredictionDataset(test_quantum)
    
    print(f"📊 데이터셋 분할 완료:")
    print(f"  - Train: {len(train_dataset)} 샘플")
    print(f"  - Validation: {len(val_dataset)} 샘플")
    print(f"  - Test: {len(test_dataset)} 샘플")
    
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Configuration
    config = PropertyPredictionConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        attention_mode="advanced",
        dropout=0.1,
        learning_rate=1e-4,
        property_dim=3  # entanglement, fidelity, expressibility
    )
    
    # Create model
    from models.property_prediction_transformer import create_property_prediction_model
    model = create_property_prediction_model(config)
    
    # Load datasets using merged_data.json
    data_path = r"C:\Users\jungh\Documents\GitHub\Kaist\OAT_Model\raw_data\merged_data.json"
    
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
        print("🚀 Starting Property Prediction Training...")
        results = trainer.train(num_epochs=100)
        
        print(f"✅ Training completed!")
        print(f"📊 Best validation loss: {results['best_val_loss']:.4f}")
        print(f"📁 Best model saved at: {results['best_model_path']}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
