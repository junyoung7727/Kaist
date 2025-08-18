"""
Decision Transformer Training Pipeline
간단하고 확장성 높은 학습 파이프라인 (에포크 캐시 시스템 통합)
"""
import gc
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
from typing import Dict, Any, Optional, List
import numpy as np
from pathlib import Path

from .epoch_cache import EpochCache
from typing import Dict, Any
import wandb
from torch.optim import AdamW
import random
import numpy as np
import sys
from typing import Dict, List, Optional, Any, Tuple, List, Tuple, Any, Union, Dict  
import os
from dataclasses import dataclass, asdict
# NEW: 게이트 레지스트리 싱글톤 임포트
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))
from gates import QuantumGateRegistry

import time
# wandb 선택적으로 임포트
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Logging will be disabled.")
    
# 경로 설정 추가
sys.path.append(str(Path(__file__).parent.parent))

# 절대 경로 임포트
try:
    from models.decision_transformer import DecisionTransformer
    from data.embedding_pipeline import EmbeddingPipeline, EmbeddingConfig
    from data.quantum_circuit_dataset import CircuitSpec, CircuitData
except ImportError:
    # 상대 경로 임포트 시도
    from ..models.decision_transformer import DecisionTransformer
    from ..data.embedding_pipeline import EmbeddingPipeline, EmbeddingConfig
    from ..data.quantum_circuit_dataset import CircuitSpec


@dataclass
class TrainingConfig:
    """학습 설정"""
    # 모델 설정
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    n_gate_types: int = None  # 🎆 NEW: gate vocab 싱글톤에서 자동 설정
    dropout: float = 0.1
    attention_mode: str = "standard"  # "standard", "advanced", "hybrid"
    
    def __post_init__(self):
        """초기화 후 gate 수를 싱글톤에서 가져오기"""
        if self.n_gate_types is None:
            self.n_gate_types = QuantumGateRegistry.get_singleton_gate_count()
            print(f"🎆 TrainingConfig: Using gate vocab singleton, n_gate_types = {self.n_gate_types}")
    
    # 학습 설정
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    num_epochs: int = 1
    warmup_steps: int = 1000
    
    # 검증 설정
    eval_every: int = 500
    save_every: int = 1000
    
    # 기타
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # 로깅
    use_wandb: bool = True
    project_name: str = "quantum-decision-transformer"
    run_name: Optional[str] = None
    
    # GPU 메모리 최적화 설정
    use_amp: bool = True
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    memory_cleanup_interval: int = 50
    
    # 데이터셋 분할 설정
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # 기타 설정
    enable_filtering: bool = True
    save_dir: str = "./OAT_Model/checkpoints"


def dict_to_config(config_dict: dict) -> TrainingConfig:
    """딕셔너리를 TrainingConfig 클래스로 변환"""
    # TrainingConfig의 기본값으로 시작
    config = TrainingConfig()
    
    # 딕셔너리의 값들로 업데이트
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


class QuantumCircuitCollator:
    """🚀 배치 콜레이터 - CircuitData를 모델 입력으로 변환 (캐싱 최적화)"""
    
    def __init__(self, embedding_pipeline: EmbeddingPipeline):
        self.embedding_pipeline = embedding_pipeline
        self._batch_count = 0
        self._total_circuits = 0
    
    def __call__(self, batch: List['CircuitData']) -> Dict[str, torch.Tensor]:
        """🚀 배치 처리 (캐싱 최적화)"""
        
        self._batch_count += 1
        self._total_circuits += len(batch)
        
        # CircuitData에서 CircuitSpec 추출
        circuit_specs = [circuit_data.circuit_spec for circuit_data in batch]
        
        # 측정 결과 정보 추가 (타겟 메트릭으로 사용)
        target_metrics = []
        for circuit_data in batch:
            result = circuit_data.measurement_result
            metrics = {
                'fidelity': result.fidelity,
                'entanglement': result.entanglement,
                'robust_fidelity': result.robust_fidelity or 0.0,
            }
            
            # Expressibility 정보 추가
            if result.expressibility:
                expr = result.expressibility
                metrics.update({
                    'expressibility': expr.get('expressibility', 0.0),
                    'kl_divergence': expr.get('kl_divergence', 0.0),
                })
            else:
                metrics.update({
                    'expressibility': 0.0,
                    'kl_divergence': 0.0,
                })
            
            target_metrics.append(metrics)

        # 임베딩 파이프라인을 통해 배치 처리 (캐싱 자동 적용)
        embedded_batch = self.embedding_pipeline.process_batch(circuit_specs)
        
        
        # 타겟 메트릭 정보 추가
        if embedded_batch:
            embedded_batch['target_metrics'] = target_metrics
        
        if not embedded_batch:
            return {}
        
        # 임베딩 파이프라인에서 이미 통합 액션 타겟이 생성됨
        # 추가 처리 없이 그대로 반환 (캐싱 최적화 완료)
        return embedded_batch
    
    
    def get_stats(self) -> Dict[str, int]:
        """ 콜레이터 통계 반환"""
        return {
            'total_batches': self._batch_count,
            'total_circuits': self._total_circuits,
            'avg_batch_size': self._total_circuits / max(self._batch_count, 1)
        }
            
    
    # 복잡한 액션 생성 메서드들 제거됨 - 이제 임베딩 파이프라인에서 통합 처리 (캐싱 최적화)


class DecisionTransformerTrainer:
    """Decision Transformer 트레이너"""
    
    def __init__(self, model, train_dataloader, val_dataloader, config: TrainingConfig, embedding_pipeline):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.embedding_pipeline = embedding_pipeline
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 에포크 캐시 시스템 초기화
        self.epoch_cache = EpochCache(cache_dir="cache/epochs", max_cache_size_gb=2.0)
        
        # 학습 진행 상태 추적
        self.global_step = 0
        
        # 모델을 디바이스로 이동
        self.model.to(self.device)
        
        # GPU 메모리 최적화 설정
        self.use_amp = self.config.use_amp and self.device.type == 'cuda'
        self.gradient_accumulation_steps = self.config.gradient_accumulation_steps
        self.gradient_checkpointing = self.config.gradient_checkpointing
        
        # Mixed Precision 스케일러
        if self.use_amp:
            self.scaler = GradScaler()
        else:
            # Dummy scaler for code consistency when not using AMP
            self.scaler = None
        
        # 그래디언트 체크포인팅 활성화
        if self.gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # 옵티마이저 설정
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=1e-6  # 수치 안정성 개선
        )
        
        # 스케줄러 설정 (웜업 + 코사인 어닐링)
        total_steps = len(self.train_dataloader) * self.config.num_epochs
        warmup_steps = total_steps // 10  # 전체 스텝의 10%를 웜업으로 사용
        
        from torch.optim.lr_scheduler import LinearLR, SequentialLR
        
        # 웜업 스케줄러 (0에서 target_lr까지 선형 증가)
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,  # 시작 학습률 = target_lr * 0.1
            end_factor=1.0,    # 끝 학습률 = target_lr * 1.0
            total_iters=warmup_steps
        )
        
        # 코사인 어닐링 스케줄러 (웜업 후 적용)
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.config.learning_rate * 0.01  # 최소 학습률 = target_lr * 0.01
        )
        
        # 순차적 스케줄러 (웜업 → 코사인)
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        # 🎯 손실 함수 설정: 모델의 compute_loss 메서드 사용
        # 예측과 손실 계산을 분리하여 깔끔한 구조 유지
        self.loss_fn = self.model.compute_loss
        
        # 로깅 설정
        self.use_wandb = self.config.use_wandb
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.project_name,
                name=self.config.run_name,
                config=asdict(self.config)
            )
        
        # 체크포인트 디렉토리 설정
        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 최고 성능 추적
        self.best_val_loss = float('inf')
        self.best_model_path = None
        
        # 메모리 정리 주기
        self.memory_cleanup_interval = self.config.memory_cleanup_interval

    def train_epoch(self):
        """한 에포크 학습 (캐시 시스템 적용)"""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        # 현재 에포크 확인 (전역 스텝으로부터 추정)
        current_epoch = getattr(self, 'current_epoch', 0)
        
        # 캐시된 데이터가 있는지 확인
        cached_batches = None
        if current_epoch > 0:  # 첫 번째 에포크가 아닌 경우
            cached_batches = self.epoch_cache.load_epoch_data(self.train_dataloader, 0)  # 첫 번째 에포크 데이터 재사용
        
        if cached_batches is not None:
            # 캐시된 데이터 사용
            print(f"[CACHE] 캐시된 데이터 사용 중... ({len(cached_batches)} 배치)")
            pbar = tqdm(cached_batches, desc="Training (Cached)")
            
            for batch_idx, batch in enumerate(pbar):
                # 캐시된 배치는 이미 처리된 상태이므로 바로 모델 학습에 사용
                loss, accuracy = self._train_single_batch_cached(batch, batch_idx)
                
                total_loss += loss
                total_accuracy += accuracy
                num_batches += 1
                
                # 프로그레스 바 업데이트
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'acc': f'{accuracy:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                })
        else:
            # 첫 번째 에포크 또는 캐시 없음 - 정상 처리 및 캐시 저장
            processed_batches = []
            pbar = tqdm(self.train_dataloader, desc="Training")
            
            # 각 배치 처리
            for batch_idx, batch in enumerate(pbar):
                # 순전파
                self.optimizer.zero_grad()
                
                # 입력 텐서를 모델과 같은 디바이스로 이동
                input_sequence = batch['input_sequence'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                action_prediction_mask = batch['action_prediction_mask'].to(self.device)
                
                # 어텐션 모드에 따라 추가 파라미터 전달
                model_kwargs = {
                    'input_sequence': input_sequence,
                    'attention_mask': attention_mask,
                    'action_prediction_mask': action_prediction_mask
                }
                
                # 고급 어텐션 모드인 경우 grid_structure와 edges 전달
                if hasattr(self.model, 'get_attention_mode') and self.model.get_attention_mode() in ['advanced', 'hybrid']:
                    # 배치에서 grid_structure와 edges 정보 추출 (있는 경우) 및 디바이스로 이동
                    if 'grid_structure' in batch:
                        model_kwargs['grid_structure'] = batch['grid_structure'].to(self.device)
                    if 'edges' in batch:
                        model_kwargs['edges'] = batch['edges'].to(self.device)
                    if 'circuit_constraints' in batch:
                        model_kwargs['circuit_constraints'] = batch['circuit_constraints'].to(self.device)
            
                # 순전파 및 손실 계산
                # 공통 전처리 - 모든 텐서를 디바이스로 이동
                squeezed_action_mask = action_prediction_mask.squeeze(1)  # 이미 디바이스로 이동된 action_prediction_mask 사용
                
                # 타겟 액션 처리
                if len(batch['target_actions'].shape) == 3:
                    squeezed_target_actions = batch['target_actions'].squeeze(1).to(self.device)
                else:
                    squeezed_target_actions = batch['target_actions'].to(self.device)
                
                # 타겟 큐빗 및 파라미터 처리
                if 'target_qubits' in batch and torch.is_tensor(batch['target_qubits']):
                    target_qubits = batch['target_qubits'].to(self.device)
                else:
                    target_qubits = batch.get('target_qubits', [])
                    
                if 'target_params' in batch and torch.is_tensor(batch['target_params']):
                    target_params = batch['target_params'].to(self.device)
                else:
                    target_params = batch.get('target_params', [])
                
                # 타겟 데이터 준비
                targets = {
                    'gate_targets': squeezed_target_actions,
                    'qubit_targets': target_qubits,
                    'parameter_targets': target_params
                }
                
                # Mixed Precision으로 forward pass
                # Forward pass and loss calculation with or without AMP
                if self.use_amp:
                    with autocast('cuda', enabled=True):
                        outputs = self.model(**model_kwargs)
                        # 손실 계산
                        loss_outputs = self.loss_fn(
                            outputs, 
                            targets, 
                            squeezed_action_mask,
                            num_qubits=batch.get('num_qubits', None),
                            num_gates=batch.get('num_gates', None)
                        )
                        loss = loss_outputs['loss']
                        accuracy = loss_outputs.get('gate_accuracy', 0.0)
                    
                    # 스케일링된 손실로 역전파
                    self.scaler.scale(loss).backward()
                else:
                    outputs = self.model(**model_kwargs)
                    loss_outputs = self.loss_fn(
                        outputs,
                        targets,
                        squeezed_action_mask,
                        num_qubits=batch.get('num_qubits', None),
                        num_gates=batch.get('num_gates', None)
                    )
                    loss = loss_outputs['loss']
                    accuracy = loss_outputs.get('gate_accuracy', 0.0)
                    
                    # Regular backward pass
                    loss.backward()
                
                # Optimizer step handling
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.use_amp:
                        # 그래디언트 클리핑
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        # 스케일러를 통한 업데이트
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # Regular gradient clipping and optimizer step
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
            
                # 통계 업데이트
                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1
                
                # 세부 손실 로깅 (WandB)
                if self.config.use_wandb and batch_idx % 10 == 0:  # 10배치마다
                    detailed_metrics = {
                        'train/batch_loss': loss.item(),
                        'train/batch_accuracy': accuracy,
                        'train/learning_rate': self.scheduler.get_last_lr()[0],
                        'global_step': self.global_step
                    }
                    
                    # 세부 손실 분해 (가능한 경우)
                    if 'gate_loss' in loss_outputs:
                        detailed_metrics['train/gate_loss'] = loss_outputs['gate_loss'].item()
                    if 'position_loss' in loss_outputs:
                        detailed_metrics['train/position_loss'] = loss_outputs['position_loss'].item()
                    if 'parameter_loss' in loss_outputs:
                        detailed_metrics['train/parameter_loss'] = loss_outputs['parameter_loss'].item()
                    
                    # F1, Precision, Recall 메트릭 추가
                    if 'gate_precision' in loss_outputs:
                        detailed_metrics['train/gate_precision'] = loss_outputs['gate_precision']
                    if 'gate_recall' in loss_outputs:
                        detailed_metrics['train/gate_recall'] = loss_outputs['gate_recall']
                    if 'gate_f1' in loss_outputs:
                        detailed_metrics['train/gate_f1'] = loss_outputs['gate_f1']
                    
                    wandb.log(detailed_metrics)
                
                # 에포크 캐시에 배치 저장
                processed_batches.append(batch)
                
                # 프로그레스 바 업데이트
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{accuracy:.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
                
                # 주기적 메모리 정리
                if batch_idx % self.memory_cleanup_interval == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            
            # 첫 번째 에포크인 경우, 캐시 저장
            if current_epoch == 0 and processed_batches:
                print(f"\n[CACHE] 첫번째 에포크 데이터 캐시 저장 중... ({len(processed_batches)} 배치)")
                self.epoch_cache.save_epoch_data(self.train_dataloader, 0, processed_batches)
        
        # 에포크 평균 메트릭 계산
        avg_metrics = {
            'loss': total_loss / max(num_batches, 1),
            'accuracy': total_accuracy / max(num_batches, 1)
        }
        
        # 마지막 배치의 세부 메트릭 추가 (loss_outputs에서 가져오기)
        if 'loss_outputs' in locals():
            if 'gate_loss' in loss_outputs:
                avg_metrics['gate_loss'] = loss_outputs['gate_loss'].item()
            if 'position_loss' in loss_outputs:
                avg_metrics['position_loss'] = loss_outputs['position_loss'].item()
            if 'parameter_loss' in loss_outputs:
                avg_metrics['parameter_loss'] = loss_outputs['parameter_loss'].item()
            if 'gate_precision' in loss_outputs:
                avg_metrics['precision'] = loss_outputs['gate_precision']
            if 'gate_recall' in loss_outputs:
                avg_metrics['recall'] = loss_outputs['gate_recall']
            if 'gate_f1' in loss_outputs:
                avg_metrics['f1'] = loss_outputs['gate_f1']
        
        return avg_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """검증 단계 - 4가지 정확도 메트릭과 3가지 손실 메트릭"""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_gate_loss = 0.0
        total_position_loss = 0.0
        total_parameter_loss = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                if not batch:
                    continue
                
                batch = self._move_batch_to_device(batch)
                
                # 모델 예측
                model_kwargs = {
                    'input_sequence': batch['input_sequence'],
                    'attention_mask': batch['attention_mask'],
                    'action_prediction_mask': batch['action_prediction_mask']
                }
                
                outputs = self.model(**model_kwargs)
                
                # 타겟 준비 (훈련과 동일한 방식)
                squeezed_action_mask = batch['action_prediction_mask']
                squeezed_target_actions = batch['target_actions']
                
                # 타겟 큐빗 및 파라미터 처리 (훈련과 동일하게)
                if 'target_qubits' in batch and torch.is_tensor(batch['target_qubits']):
                    target_qubits = batch['target_qubits'].to(self.device)
                else:
                    target_qubits = batch.get('target_qubits', [])
                    
                if 'target_params' in batch and torch.is_tensor(batch['target_params']):
                    target_params = batch['target_params'].to(self.device)
                else:
                    target_params = batch.get('target_params', [])
                
                # 차원 수정
                if len(squeezed_action_mask.shape) == 3 and squeezed_action_mask.shape[1] == 1:
                    squeezed_action_mask = squeezed_action_mask.squeeze(1)
                if len(squeezed_target_actions.shape) == 3 and squeezed_target_actions.shape[1] == 1:
                    squeezed_target_actions = squeezed_target_actions.squeeze(1)
                
                # 타겟 구조 준비 (훈련과 동일한 구조)
                targets_dict = {
                    'gate_targets': squeezed_target_actions,
                    'qubit_targets': target_qubits,  # position_loss에서 사용하는 키
                    'parameter_targets': target_params,
                    'target_actions': squeezed_target_actions,
                    'target_qubits': target_qubits,
                    'target_params': target_params,
                    'action_targets': batch.get('action_targets', {})
                }
                
                # 손실 계산
                loss_outputs = self.loss_fn(
                    predictions=outputs,
                    targets=targets_dict,
                    action_prediction_mask=squeezed_action_mask,
                    num_qubits=batch.get('num_qubits', None),
                    num_gates=batch.get('num_gates', None)  # 🔧 검증에서도 게이트 수 정보 추가
                )
                
                # 메트릭 누적
                total_loss += loss_outputs['loss'].item()
                total_accuracy += loss_outputs.get('gate_accuracy', 0.0)
                
                # 세부 손실 누적
                if 'gate_loss' in loss_outputs:
                    total_gate_loss += loss_outputs['gate_loss'].item()
                if 'position_loss' in loss_outputs:
                    total_position_loss += loss_outputs['position_loss'].item()
                if 'parameter_loss' in loss_outputs:
                    total_parameter_loss += loss_outputs['parameter_loss'].item()
                
                # 분류 메트릭 누적
                total_precision += loss_outputs.get('gate_precision', 0.0)
                total_recall += loss_outputs.get('gate_recall', 0.0)
                total_f1 += loss_outputs.get('gate_f1', 0.0)
                
                num_batches += 1
        
        val_metrics = {
            'val_loss': total_loss / max(num_batches, 1),
            'val_accuracy': total_accuracy / max(num_batches, 1),
            'val_gate_loss': total_gate_loss / max(num_batches, 1),
            'val_position_loss': total_position_loss / max(num_batches, 1),
            'val_parameter_loss': total_parameter_loss / max(num_batches, 1),
            'val_precision': total_precision / max(num_batches, 1),
            'val_recall': total_recall / max(num_batches, 1),
            'val_f1': total_f1 / max(num_batches, 1)
        }
        
        # 최고 성능 모델 저장
        if val_metrics['val_loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['val_loss']
            self.save_checkpoint(is_best=True)
        
        # 로깅
        if self.config.use_wandb:
            wandb.log({**{f'val/{k}': v for k, v in val_metrics.items()}, 'global_step': self.global_step})
        
        print(f"Validation - Loss: {val_metrics['val_loss']:.4f}, Accuracy: {val_metrics['val_accuracy']:.4f}")
        
        return val_metrics
    
    def _move_batch_to_device(self, batch):
        """배치 데이터를 현재 디바이스로 이동"""
        result = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                result[k] = v.to(self.device)
            else:
                result[k] = v
        return result
    
    def train(self):
        """전체 학습 루프"""
        print(f"Starting training for {self.config.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 학습 시작 준비 - 캐시는 이미 epoch_cache 초기화 시 정리됨
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # 학습
            train_metrics = self.train_epoch()
            
            # 검증
            val_metrics = self.validate_epoch()
            
            # 에포크 로깅 - 4가지 정확도 메트릭과 3가지 손실 메트릭
            print(f"\n[STATS] Epoch {epoch + 1}/{self.config.num_epochs} Results:")
            print(f"[TRAIN] Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            if 'gate_loss' in train_metrics:
                print(f"         Gate: {train_metrics.get('gate_loss', 0):.4f}, Pos: {train_metrics.get('position_loss', 0):.4f}, Param: {train_metrics.get('parameter_loss', 0):.4f}")
            if 'precision' in train_metrics:
                print(f"         Prec: {train_metrics.get('precision', 0):.4f}, Rec: {train_metrics.get('recall', 0):.4f}, F1: {train_metrics.get('f1', 0):.4f}")
            
            print(f"[VAL]   Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['val_accuracy']:.4f}")
            print(f"         Gate: {val_metrics['val_gate_loss']:.4f}, Pos: {val_metrics['val_position_loss']:.4f}, Param: {val_metrics['val_parameter_loss']:.4f}")
            print(f"         Prec: {val_metrics['val_precision']:.4f}, Rec: {val_metrics['val_recall']:.4f}, F1: {val_metrics['val_f1']:.4f}")
            
            if self.config.use_wandb:
                wandb_metrics = {
                    'epoch': epoch + 1,
                    'train/epoch_loss': train_metrics['loss'],
                    'train/epoch_accuracy': train_metrics['accuracy'],
                    'val/epoch_loss': val_metrics['val_loss'],
                    'val/epoch_accuracy': val_metrics['val_accuracy'],
                    'val/epoch_gate_loss': val_metrics['val_gate_loss'],
                    'val/epoch_position_loss': val_metrics['val_position_loss'],
                    'val/epoch_parameter_loss': val_metrics['val_parameter_loss'],
                    'val/epoch_precision': val_metrics['val_precision'],
                    'val/epoch_recall': val_metrics['val_recall'],
                    'val/epoch_f1': val_metrics['val_f1']
                }
                
                # 훈련 세부 메트릭 추가 (가능한 경우)
                if 'gate_loss' in train_metrics:
                    wandb_metrics.update({
                        'train/epoch_gate_loss': train_metrics['gate_loss'],
                        'train/epoch_position_loss': train_metrics.get('position_loss', 0),
                        'train/epoch_parameter_loss': train_metrics.get('parameter_loss', 0),
                        'train/epoch_precision': train_metrics.get('precision', 0),
                        'train/epoch_recall': train_metrics.get('recall', 0),
                        'train/epoch_f1': train_metrics.get('f1', 0)
                    })
                
                wandb.log(wandb_metrics)
        
        print("Training completed!")
        
        self._debug_last_batch_analysis()
    
    def _debug_last_batch_analysis(self):
        """마지막 배치의 샘플 분석 디버그 출력"""
        # 검증 데이터에서 첫 번째 배치 가져오기
        if not hasattr(self, 'val_dataloader') or len(self.val_dataloader) == 0:
            print("No validation data available for debug analysis.")
            return
        
        try:
            # 첫 번째 배치와 샘플 가져오기
            batch = next(iter(self.val_dataloader))
            batch_idx = 0
            sample_idx = 0
            
            # 디바이스로 데이터 이동
            batch = self._move_batch_to_device(batch)
            
            # 모델 예측 실행
            model_kwargs = {
            'input_sequence': batch['input_sequence'],
            'attention_mask': batch['attention_mask'],
            'action_prediction_mask': batch['action_prediction_mask']
        }
        
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**model_kwargs)
            
            print(f"\n ===== 배치 {batch_idx} 샘플 {sample_idx} 시퀀스 분석 =====\n")
            
            # 기본 정보
            input_seq = batch['input_sequence'][sample_idx]  # [seq_len, d_model]
            action_mask = batch['action_prediction_mask'][sample_idx]  # [seq_len] or [1, seq_len]
            if len(action_mask.shape) > 1:
                action_mask = action_mask.squeeze(0)
            
            # 액션 예측 위치 찾기
            action_positions = torch.where(action_mask > 0)[0]
            num_actions = len(action_positions)
            
            print(f"🎯 기본 정보:")
            print(f"   - 입력 시퀀스 길이: {input_seq.shape[0]}")
            print(f"   - 액션 예측 위치 수: {num_actions}")
            print(f"   - 액션 위치들: {action_positions.tolist()[:10]}{'...' if num_actions > 10 else ''}")
            
            # 모델 예측 추출
            gate_logits = outputs['gate_logits'][sample_idx]  # [num_actions, num_gate_types]
            position_preds = outputs['position_preds'][sample_idx]  # [num_actions, max_qubits, num_positions]
            parameter_preds = outputs['parameter_preds'][sample_idx]  # [num_actions]
            
            # 예측된 게이트 타입 (최고 확률)
            predicted_gates = torch.argmax(gate_logits, dim=-1)  # [num_actions]
            
            #  정답 레이블 추출 (안전한 텐서 접근)
            target_actions = batch.get('target_actions', torch.tensor([]))
            target_qubits = batch.get('target_qubits', [])
            target_params = batch.get('target_params', [])
            
            # [DEBUG] 텐서 차원 디버깅
            print(f"[DEBUG] 타겟 텐서 차원 분석:")
            print(f"   - target_actions.shape: {target_actions.shape if hasattr(target_actions, 'shape') else 'N/A'}")
            print(f"   - target_qubits type: {type(target_qubits)}, len: {len(target_qubits) if hasattr(target_qubits, '__len__') else 'N/A'}")
            print(f"   - target_params type: {type(target_params)}, len: {len(target_params) if hasattr(target_params, '__len__') else 'N/A'}")
            
            # 🚀 안전한 타겟 게이트 추출
            if hasattr(target_actions, 'shape'):
                if len(target_actions.shape) == 3:  # [32, 1, 164] 형태
                    if target_actions.shape[0] > sample_idx:
                        target_gates = target_actions[sample_idx, 0]  # [164] - 중간 차원 제거
                    else:
                        target_gates = torch.tensor([])
                elif len(target_actions.shape) == 2:  # [32, 164] 형태
                    if target_actions.shape[0] > sample_idx:
                        target_gates = target_actions[sample_idx]  # [164]
                    else:
                        target_gates = torch.tensor([])
                elif len(target_actions.shape) == 1:  # [164] 형태
                    target_gates = target_actions  # 이미 1D 텐서
                else:
                    target_gates = torch.tensor([])
            else:
                target_gates = torch.tensor([])
            
            print(f"   - target_gates.shape: {target_gates.shape if hasattr(target_gates, 'shape') else 'N/A'}")
            
            print(f"\n 모델 예측 vs 정답 (처음 5개 액션):")
            
            # 게이트 예측 로짓 추출 (샘플당)
            sample_gate_logits = outputs['gate_logits'][sample_idx]  # [seq_len, num_gate_types]
            
            for i in range(min(5, num_actions)):
                action_pos = action_positions[i].item()
                pred_gate = predicted_gates[i].item()
                
                # 게이트 로짓 분석
                gate_logits_at_pos = gate_logits[i]
                top3_gate_values, top3_gate_indices = torch.topk(gate_logits_at_pos, k=min(3, gate_logits_at_pos.size(-1)))
                top3_gate_probs = torch.softmax(top3_gate_values, dim=-1)
                gate_confidence = top3_gate_probs[0].item()
                
                # 안전한 정답 게이트 추출
                if hasattr(target_gates, 'shape') and i < target_gates.shape[0]:
                    if target_gates.dim() == 0:  # 스칼라 텐서
                        true_gate = int(target_gates.item())
                    elif target_gates.dim() == 1 and i < target_gates.size(0):  # 1D 텐서
                        true_gate = int(target_gates[i].item())
                    else:  # 다차원 텐서
                        if target_gates.dim() > 1 and target_gates.size(0) > i:
                            true_gate = int(target_gates[i].item() if target_gates[i].dim() == 0 else target_gates[i][0].item())
                        else:
                            true_gate = "N/A"
                else:
                    true_gate = "N/A"
                
                # 위치 예측 (최적화 예측 방식)
                position_logits_at_pos = position_preds[i]  # [max_qubits, 2] or similar
                
                # 게이트 타입에 따라 필요한 큐빗 수 결정 (예: 1큐빗 또는 2큐빗 게이트)
                # 여기서는 단순화를 위해 핵심 위치 1~2개만 추출
                qubit_positions = []
                
                # 위치 예측 논리
                if len(position_logits_at_pos.shape) >= 2 and position_logits_at_pos.shape[0] > 0:
                    # 첫 번째 큐빗 위치 (모든 게이트에 필요)
                    pos1 = torch.argmax(position_logits_at_pos[0]).item()
                    qubit_positions.append(pos1)
                    
                    # 2큐빗 게이트인 경우 두 번째 위치 추가 
                    # (간단한 구현을 위해 여기서는 모든 게이트에 대해 첫 두 위치 표시)
                    if position_logits_at_pos.shape[0] > 1:
                        pos2 = torch.argmax(position_logits_at_pos[1]).item()
                        qubit_positions.append(pos2)
                
                # 파라미터 예측
                pred_param = parameter_preds[i].item()
                
                # 🚀 안전한 정답 위치/파라미터 추출 (샘플 내 게이트별)
                true_positions = []
                if isinstance(target_qubits, torch.Tensor):
                    # 텐서 형태 분석
                    if len(target_qubits.shape) == 3:  # [batch, gate, pos]
                        if target_qubits.shape[0] > sample_idx and target_qubits.shape[1] > i:
                            # 모든 유효한 큐빗 위치 추출
                            for j in range(target_qubits.shape[2]):
                                if j < target_qubits[sample_idx][i].shape[0]:
                                    qpos = target_qubits[sample_idx][i][j].item()
                                    if qpos >= 0:  # 유효한 큐빗 위치만 포함
                                        true_positions.append(int(qpos))
                    elif len(target_qubits.shape) == 2:  # [batch, pos]
                        if target_qubits.shape[0] > sample_idx:
                            qpos = target_qubits[sample_idx][i].item() if i < target_qubits.shape[1] else -1
                            if qpos >= 0:
                                true_positions.append(int(qpos))
            
            # 파라미터 값
            true_param = None
            if isinstance(target_params, torch.Tensor):
                if len(target_params.shape) == 2:  # [batch, gate]
                    if target_params.shape[0] > sample_idx and target_params.shape[1] > i:
                        param_val = target_params[sample_idx][i].item()
                        if not torch.isnan(target_params[sample_idx][i]):
                            true_param = float(param_val)
                elif len(target_params.shape) == 1:  # [batch]
                    if i < target_params.shape[0]:
                        param_val = target_params[i].item()
                        if not torch.isnan(target_params[i]):
                            true_param = float(param_val)
                            
                # 최종 출력
                print(f"   [{i}] 시퀀스 위치 {action_pos}:")
                
                # 게이트 정보 출력 (예측 vs 실제)
                gate_match = "✓" if str(pred_gate) == str(true_gate) else "✗"
                print(f"       게이트: 예측={pred_gate} vs 정답={true_gate} {gate_match}")
                print(f"       게이트 확률: {gate_confidence:.3f}")
                
                # 큐빗 위치 출력 (예측 vs 실제)
                pos_match = "✓" if qubit_positions == true_positions else "✗"
                print(f"       위치: 예측={qubit_positions} vs 정답={true_positions} {pos_match}")
                
                # 파라미터 출력 (예측 vs 실제)
                param_match = ""
                if true_param is not None and isinstance(true_param, (int, float)):
                    param_diff = abs(pred_param - true_param)
                    param_match = "✓" if param_diff < 0.1 else "✗"
                    print(f"       파라미터: 예측={pred_param:.4f} vs 정답={true_param:.4f} {param_match}")
                else:
                    print(f"       파라미터: 예측={pred_param:.4f} vs 정답={true_param}")
            
            print(f"[DEBUG] ===== 분석 완료 =====\n")
            
            # 모델을 다시 학습 모드로 변경
            self.model.train()
            
        except Exception as e:
            print(f"Debug analysis failed: {e}")
            import traceback
            traceback.print_exc()
            # 모델 상태 복원
            self.model.train()
    
    def _debug_sequence_details(self, batch, outputs, batch_idx, sample_idx=0):
        """[DEBUG] 배치당 1개 샘플의 상세한 시퀀스 분석"""
        print(f"\n[DEBUG] ===== 배치 {batch_idx} 샘플 {sample_idx} 시퀀스 분석 =====")
        
        # 기본 정보
        input_seq = batch['input_sequence'][sample_idx]  # [seq_len, d_model]
        action_mask = batch['action_prediction_mask'][sample_idx]  # [seq_len] or [1, seq_len]
        if len(action_mask.shape) > 1:
            action_mask = action_mask.squeeze(0)
        
        # 액션 예측 위치 찾기
        action_positions = torch.where(action_mask > 0)[0]
        num_actions = len(action_positions)
        
        print(f" 기본 정보:")
        print(f"   - 입력 시퀀스 길이: {input_seq.shape[0]}")
        print(f"   - 액션 예측 위치 수: {num_actions}")
        print(f"   - 액션 위치들: {action_positions.tolist()[:10]}{'...' if num_actions > 10 else ''}")
        
        # 모델 예측 추출
        gate_logits = outputs['gate_logits'][sample_idx]  # [seq_len, num_gate_types]
        position_preds = outputs.get('position_preds', None)
        parameter_preds = outputs.get('parameter_preds', None)
        
        print(f" 모델 출력 텐서 차원:")
        print(f"   - gate_logits.shape: {gate_logits.shape if gate_logits is not None else 'None'}")
        print(f"   - position_preds.shape: {position_preds[sample_idx].shape if position_preds is not None else 'None'}")
        print(f"   - parameter_preds.shape: {parameter_preds[sample_idx].shape if parameter_preds is not None else 'None'}")
        
        # 예측된 게이트 타입 (최고 확률)
        if gate_logits is not None:
            predicted_gates = torch.argmax(gate_logits, dim=-1)  # [seq_len]
        else:
            predicted_gates = torch.tensor([])
        
        #  정답 레이블 추출 (안전한 텐서 접근)
        target_actions = batch.get('target_actions', torch.tensor([]))
        target_qubits = batch.get('target_qubits', [])
        target_params = batch.get('target_params', [])
        
        # 🚀 텐서 차원 디버깅
        print(f" 타겟 텐서 차원 분석:")
        print(f"   - target_actions.shape: {target_actions.shape if hasattr(target_actions, 'shape') else 'N/A'}")
        print(f"   - target_qubits type: {type(target_qubits)}, shape: {target_qubits.shape if isinstance(target_qubits, torch.Tensor) else 'N/A'}")
        print(f"   - target_params type: {type(target_params)}, shape: {target_params.shape if isinstance(target_params, torch.Tensor) else 'N/A'}")
        
        # 🚀 안전한 타겟 게이트 추출
        if isinstance(target_actions, torch.Tensor):
            if len(target_actions.shape) == 3:  # [batch, 1, seq_len] 형태
                if target_actions.shape[0] > sample_idx:
                    target_gates = target_actions[sample_idx, 0]  # [seq_len] - 중간 차원 제거
                else:
                    target_gates = torch.tensor([])
            elif len(target_actions.shape) == 2:  # [batch, seq_len] 형태
                if target_actions.shape[0] > sample_idx:
                    target_gates = target_actions[sample_idx]  # [seq_len]
                else:
                    target_gates = torch.tensor([])
            elif len(target_actions.shape) == 1:  # [seq_len] 형태
                target_gates = target_actions  # 이미 1D 텐서
            else:
                target_gates = torch.tensor([])
        else:
            target_gates = torch.tensor([])
        
        print(f"   - 추출된 target_gates.shape: {target_gates.shape if hasattr(target_gates, 'shape') else 'N/A'}")
        
        # 실제 값 분포 분석
        if isinstance(target_gates, torch.Tensor) and target_gates.numel() > 0:
            # 유니크 값 추출
            unique_targets, target_counts = torch.unique(target_gates, return_counts=True)
            print(f"   - unique targets: {unique_targets}")
            print(f"   - target distribution: {target_counts}")
        
        print(f"\n 모델 예측 vs 정답 (처음 5개 액션):")
        
        # 유효한 위치에서만 샘플링하기
        if len(action_positions) == 0:
            print("   [!] 액션 위치가 없습니다!")
            return
            
        for i in range(min(5, num_actions)):
            action_pos = action_positions[i].item()
            
            # 해당 위치의 게이트 로짓 추출
            if gate_logits is None or action_pos >= gate_logits.shape[0]:
                print(f"   [{i}] 시퀀스 위치 {action_pos}: 게이트 로짓 없음!")
                continue
                
            gate_logits_at_pos = gate_logits[action_pos]
            pred_gate = predicted_gates[action_pos].item()
            
            # 게이트 로짓 분석
            top3_gate_values, top3_gate_indices = torch.topk(gate_logits_at_pos, k=min(3, gate_logits_at_pos.size(-1)))
            top3_gate_probs = torch.softmax(top3_gate_values, dim=-1)
            gate_confidence = top3_gate_probs[0].item()
            
            # 정답 값 추출 - 안전하게 접근
            true_gate = "N/A"
            if isinstance(target_gates, torch.Tensor) and target_gates.numel() > 0:
                if action_pos < target_gates.shape[0]:
                    true_gate = int(target_gates[action_pos].item())
            
            # 위치 예측 (있는 경우에만 추출)
            position_info = "N/A"
            if position_preds is not None and sample_idx < position_preds.shape[0]:
                if action_pos < position_preds[sample_idx].shape[0]:
                    pos_pred = position_preds[sample_idx][action_pos]
                    # 위치 예측 구조에 따라 다르게 처리 (2D 또는 3D)
                    if pos_pred.dim() == 1:
                        position_info = f"{pos_pred.tolist()}"
                    elif pos_pred.dim() == 2:
                        # 가장 높은 확률의 위치 추출
                        if pos_pred.shape[1] >= 2:  # [qubit_idx, pos]
                            positions = []
                            for q_idx in range(min(2, pos_pred.shape[0])):
                                if pos_pred[q_idx].numel() > 0:
                                    pos = torch.argmax(pos_pred[q_idx]).item()
                                    positions.append(pos)
                            position_info = f"{positions}"
            
            # 파라미터 예측 (있는 경우에만)
            param_info = "N/A"
            if parameter_preds is not None and sample_idx < parameter_preds.shape[0]:
                if action_pos < parameter_preds[sample_idx].shape[0]:
                    param_pred = parameter_preds[sample_idx][action_pos]
                    param_info = f"{param_pred.item():.4f}"
            
            # 최종 출력
            print(f"   [{i}] 시퀀스 위치 {action_pos}:")
            print(f"       - 예측 게이트: {pred_gate} (확률: {gate_confidence:.4f})")
            print(f"       - 정답 게이트: {true_gate}")
            print(f"       - 예측 위치: {position_info}")
            print(f"       - 예측 파라미터: {param_info}")
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """배치를 디바이스로 이동"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def save_checkpoint(self, is_best: bool = False):
        """체크포인트 저장 (원자적 저장으로 손상 방지)"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config)
        }
        
        # 일반 체크포인트 (원자적 저장)
        checkpoint_path = self.save_dir / f"checkpoint_step_{self.global_step}.pt"
        temp_path = self.save_dir / f"checkpoint_step_{self.global_step}.pt.tmp"
        
        try:
            torch.save(checkpoint, temp_path)
            temp_path.rename(checkpoint_path)  # 원자적 이동
            print(f"✅ Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()  # 임시 파일 삭제
        
        # 최고 성능 모델 (원자적 저장)
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            temp_best_path = self.save_dir / "best_model.pt.tmp"
            
            try:
                torch.save(checkpoint, temp_best_path)
                temp_best_path.rename(best_path)  # 원자적 이동
                print(f"✅ New best model saved! Val loss: {self.best_val_loss:.4f}")
            except Exception as e:
                print(f"❌ Failed to save best model: {e}")
                if temp_best_path.exists():
                    temp_best_path.unlink()  # 임시 파일 삭제
    
    def load_checkpoint(self, checkpoint_path: str):
        """체크포인트 로딩"""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded from {checkpoint_path}")


def set_seed(seed: int):
    """시드 설정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# 사용 예시
if __name__ == "__main__":
    from ..data.quantum_circuit_dataset import DatasetManager, create_dataloaders
    from ..data.embedding_pipeline import create_embedding_pipeline, EmbeddingConfig
    from ..models.decision_transformer import create_decision_transformer
    
    # 설정
    config = TrainingConfig(
        d_model=256,
        n_layers=4,
        n_heads=8,
        batch_size=8,
        num_epochs=10,
        use_wandb=False  # 테스트용
    )
    
    # 시드 설정
    set_seed(config.seed)
    
    # 데이터셋 준비
    manager = DatasetManager("../data/unified_batch_experiment_results_with_circuits.json")
    train_ds, val_ds, test_ds = manager.split_dataset()
    
    # 임베딩 파이프라인 (캐싱 활성화)
    embed_config = EmbeddingConfig(d_model=config.d_model, n_gate_types=config.n_gate_types)
    embedding_pipeline = create_embedding_pipeline(embed_config)
    
    # 캐싱 활성화 (성능 최적화)
    if hasattr(embedding_pipeline, 'enable_cache'):
        embedding_pipeline.enable_cache = True
        print(" 임베딩 파이프라인 캐싱이 활성화되었습니다!")
    
    # 콜레이터
    collator = QuantumCircuitCollator(embedding_pipeline)
    
    # 데이터로더
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, 
        batch_size=config.batch_size,
        num_workers=0
    )
    
    # 콜레이터 적용
    train_loader.collate_fn = collator
    val_loader.collate_fn = collator
    
    # NEW: 모듈러 어텐션을 지원하는 모델 생성 (gate 수는 싱글톤에서 자동 설정)
    model = DecisionTransformer(
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        n_gate_types=config.n_gate_types,  # 이미 __post_init__에서 설정됨
        dropout=config.dropout,
        attention_mode=config.attention_mode
    )  
    
    # 트레이너
    trainer = DecisionTransformerTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # 학습 전 캐시 통계 초기화
    if hasattr(embedding_pipeline, 'clear_cache'):
        embedding_pipeline.clear_cache()
        print(" 학습 시작 전 캐시를 초기화했습니다.")
    
    # 학습 시작
    print(" 캐싱 최적화된 학습을 시작합니다!")
    trainer.train()
    
    # 학습 후 캐시 통계 출력
    if hasattr(embedding_pipeline, 'print_cache_stats'):
        print("\n" + "="*50)
        print(" 학습 완료! 캐시 성능 통계:")
        embedding_pipeline.print_cache_stats()
        print("="*50)
