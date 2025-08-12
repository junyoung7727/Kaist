"""
Decision Transformer Training Pipeline
간단하고 확장성 높은 학습 파이프라인
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any
import wandb
import random
import numpy as np
import sys
from typing import Optional, List, Tuple, Any, Union, Dict  
import os
from dataclasses import dataclass, asdict

from src.models.decision_transformer import DecisionTransformerLoss

# 공통 디버그 유틸리티 사용
from utils.debug_utils import debug_print, debug_tensor_info

# 🎆 NEW: 게이트 레지스트리 싱글톤 임포트
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
    from models.decision_transformer import DecisionTransformer, DecisionTransformerLoss
    from data.embedding_pipeline import EmbeddingPipeline, EmbeddingConfig
    from data.quantum_circuit_dataset import CircuitSpec
except ImportError:
    # 상대 경로 임포트 시도
    from ..models.decision_transformer import DecisionTransformer, DecisionTransformerLoss
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


class QuantumCircuitCollator:
    """배치 콜레이터 - CircuitSpec을 모델 입력으로 변환"""
    
    def __init__(self, embedding_pipeline: EmbeddingPipeline):
        self.embedding_pipeline = embedding_pipeline
    
    def __call__(self, batch: List[CircuitSpec]) -> Dict[str, torch.Tensor]:
        """배치 처리"""

        # 임베딩 파이프라인을 통해 배치 처리
        embedded_batch = self.embedding_pipeline.process_batch(batch)
        
        if not embedded_batch:
            return {}
        
        # 타겟 액션 생성 (다음 게이트 예측)
        target_actions = self._create_target_actions(embedded_batch)
        embedded_batch['target_actions'] = target_actions
        
        return embedded_batch
            
    
    def _create_target_actions(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """임베딩된 시퀀스에서 다음 액션을 추출하여 타겟 생성"""
        # action_prediction_mask의 형태 확인
        action_mask_shape = batch_data['action_prediction_mask'].shape
        debug_print(f"Debug: action_prediction_mask shape: {action_mask_shape}")
        
        batch_size = len(batch_data['circuit_id'])
        # input_sequence: [batch, 1, seq_len, d_model] -> seq_len은 shape[2]
        max_seq_len = batch_data['input_sequence'].shape[2]
        
        debug_print(f"Debug: Creating target_actions with shape [{batch_size}, {max_seq_len}]")
        target_actions = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        
        # 이미 임베딩 파이프라인에서 계산된 target_actions 사용
        if 'target_actions' in batch_data:
            debug_print("Debug: Using pre-computed target_actions from embedding pipeline")
            # 임베딩 파이프라인에서 이미 계산된 타겟 사용
            pipeline_targets = batch_data['target_actions']
            
            # 차원 맞추기
            if len(pipeline_targets.shape) == 3:  # [batch, 1, seq_len]
                pipeline_targets = pipeline_targets.squeeze(1)  # [batch, seq_len]
            
            # 배치 크기와 시퀀스 길이 맞추기
            target_batch_size = min(batch_size, pipeline_targets.shape[0])
            target_seq_len = min(max_seq_len, pipeline_targets.shape[1])
            
            target_actions[:target_batch_size, :target_seq_len] = pipeline_targets[:target_batch_size, :target_seq_len]
            debug_print(f"Debug: Copied target_actions shape: [{target_batch_size}, {target_seq_len}]")
            
        else:
            debug_print("Warning: No pre-computed target_actions found, using fallback")
            # 폴백: State-Action-Reward 시퀀스에서 다음 Action 추출
            for i in range(batch_size):
                # action_prediction_mask 처리
                if len(action_mask_shape) == 3:  # [batch_size, 1, seq_len] 형태
                    action_mask = batch_data['action_prediction_mask'][i].squeeze(0)
                elif len(action_mask_shape) == 2:  # [batch_size, seq_len] 형태
                    action_mask = batch_data['action_prediction_mask'][i]
                else:  # 1차원 형태
                    action_mask = batch_data['action_prediction_mask']
                
                debug_print(f"Debug: action_mask[{i}] shape: {action_mask.shape}")
                
                # State-Action-Reward 패턴에서 다음 Action 위치 찾기
                action_positions = torch.where(action_mask)[0]
                for pos_idx, seq_pos in enumerate(action_positions):
                    # 다음 Action 위치 계산 (S-A-R 패턴에서 Action은 1, 4, 7, 10... 위치)
                    if seq_pos % 3 == 1:  # Action 위치인 경우
                        next_action_pos = seq_pos + 3  # 다음 Action 위치
                        if next_action_pos < max_seq_len:
                            # 다음 Action의 게이트 ID를 타겟으로 (더미 구현)
                            target_actions[i][seq_pos] = (seq_pos // 3) % 20  # 임시 게이트 ID
                        else:
                            # 시퀀스 끝이면 EOS 토큰 (19번)
                            target_actions[i][seq_pos] = 19
                
                debug_print(f"Debug: Fallback - set targets for batch {i}")
        
        return target_actions


class DecisionTransformerTrainer:
    """Decision Transformer 트레이너"""
    
    def __init__(
        self,
        config: TrainingConfig,
        model: DecisionTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: str = "OAT_Model/checkpoints"
    ):
        self.config = config
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 문제 1 해결: 학습률 복원 
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=1e-8,  # 수치 안정성
            betas=(0.9, 0.95)  # 안정적인 모멘텀
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs * len(train_loader)
        )
        
        # 손실 함수
        self.loss_fn = DecisionTransformerLoss()
        
        # 학습 상태
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # 로깅 초기화
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=config.project_name,
                name=config.run_name,
                config=asdict(config)
            )
        elif config.use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not available. Install with 'pip install wandb'.")
    
    def train_epoch(self) -> Dict[str, float]:
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
            if not batch:  # 빈 배치 스킵
                continue
            
            # 배치를 디바이스로 이동
            batch = self._move_batch_to_device(batch)
            
            # 순전파
            self.optimizer.zero_grad()
            
            outputs = self.model(
                input_sequence=batch['input_sequence'],
                attention_mask=batch['attention_mask'],
                action_prediction_mask=batch['action_prediction_mask']
            )
            
            # 손실 계산
            # action_prediction_mask와 target_actions를 squeeze해서 [batch, seq_len] 형태로 만들기
            squeezed_action_mask = batch['action_prediction_mask'].squeeze(1)
            squeezed_target_actions = batch['target_actions'].squeeze(1)
            loss_outputs = self.loss_fn(
                action_logits=outputs['action_logits'],  # 손실 함수에서 마스킹 처리
                target_actions=squeezed_target_actions,
                action_prediction_mask=squeezed_action_mask
            )
            
            loss = loss_outputs['loss']
            accuracy = loss_outputs['accuracy']
            
            # 역전파 및 그래디언트 클리핑
            loss.backward()
            
            # 문제 3 해결: 그래디언트 클리핑 완화 (0.1 → 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # 통계 업데이트
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1
            
            # 프로그레스 바 업데이트
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{accuracy.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # 로깅
            if self.config.use_wandb:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/accuracy': accuracy.item(),
                    'train/learning_rate': self.scheduler.get_last_lr()[0],
                    'global_step': self.global_step
                })
            
            self.global_step += 1
            
            # 검증
            if self.global_step % self.config.eval_every == 0:
                val_metrics = self.validate()
                self.model.train()  # 다시 학습 모드로
            
            # 체크포인트 저장
            if self.global_step % self.config.save_every == 0:
                self.save_checkpoint()
        
        return {
            'loss': total_loss / max(num_batches, 1),
            'accuracy': total_accuracy / max(num_batches, 1)
        }
    
    def validate(self) -> Dict[str, float]:
        """검증"""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                if not batch:
                    continue
                
                batch = self._move_batch_to_device(batch)
                
                outputs = self.model(
                    input_sequence=batch['input_sequence'],
                    attention_mask=batch['attention_mask'],
                    action_prediction_mask=batch['action_prediction_mask']
                )
                
                # 🚨 CRITICAL FIX: 검증 단계에서도 마스크 차원 수정 필요
                # action_prediction_mask와 target_actions를 squeeze해서 [batch, seq_len] 형태로 만들기
                squeezed_action_mask = batch['action_prediction_mask']
                squeezed_target_actions = batch['target_actions']
                
                # 차원 수정
                if len(squeezed_action_mask.shape) == 3 and squeezed_action_mask.shape[1] == 1:
                    squeezed_action_mask = squeezed_action_mask.squeeze(1)  # [batch, 1, seq_len] -> [batch, seq_len]
                if len(squeezed_target_actions.shape) == 3 and squeezed_target_actions.shape[1] == 1:
                    squeezed_target_actions = squeezed_target_actions.squeeze(1)  # [batch, 1, seq_len] -> [batch, seq_len]
                
                loss_outputs = self.loss_fn(
                    action_logits=outputs['action_logits'],  # [batch, seq_len, n_gate_types]
                    target_actions=squeezed_target_actions,  # [batch, seq_len]
                    action_prediction_mask=squeezed_action_mask  # [batch, seq_len]
                )
                
                total_loss += loss_outputs['loss'].item()
                total_accuracy += loss_outputs['accuracy'].item()
                num_batches += 1
        
        val_metrics = {
            'val_loss': total_loss / max(num_batches, 1),
            'val_accuracy': total_accuracy / max(num_batches, 1)
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
    
    def train(self):
        """전체 학습 루프"""
        print(f"Starting training for {self.config.num_epochs} epochs...")
        print(f"Device: {self.config.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # 학습
            train_metrics = self.train_epoch()
            
            # 에포크 로깅
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
            
            if self.config.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/epoch_loss': train_metrics['loss'],
                    'train/epoch_accuracy': train_metrics['accuracy']
                })
        
        print("Training completed!")
        
        # 최종 검증
        final_val_metrics = self.validate()
        print(f"Final validation - Loss: {final_val_metrics['val_loss']:.4f}, Accuracy: {final_val_metrics['val_accuracy']:.4f}")
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """배치를 디바이스로 이동"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.config.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def save_checkpoint(self, is_best: bool = False):
        """체크포인트 저장"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config)
        }
        
        # 일반 체크포인트
        checkpoint_path = self.save_dir / f"checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # 최고 성능 모델
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"New best model saved! Val loss: {self.best_val_loss:.4f}")
    
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
    
    # 임베딩 파이프라인
    embed_config = EmbeddingConfig(d_model=config.d_model, n_gate_types=config.n_gate_types)
    embedding_pipeline = create_embedding_pipeline(embed_config)
    
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
    
    # 🎆 NEW: 모듈러 어텐션을 지원하는 모델 생성 (gate 수는 싱글톤에서 자동 설정)
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
    
    # 학습 시작
    trainer.train()
