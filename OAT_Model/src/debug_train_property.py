"""
Property Prediction Training with Focused Debugging
수렴 문제 분석을 위한 핵심 디버깅이 포함된 훈련 스크립트
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import sys
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import time
import os

# Add project paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "quantumcommon"))

# Import models and config
from models.property_prediction_transformer import (
    PropertyPredictionTransformer, 
    PropertyPredictionConfig,
    PropertyPredictionLoss
)

# Import data
from data.quantum_circuit_dataset import DatasetManager, create_dataloaders

# Import debugging system
from debug.training_debug import create_training_debugger, PropertyTrainingDebugger

# Import gates
from gates import QuantumGateRegistry


class PropertyTrainingAnalyzer:
    """Property Prediction 훈련 분석기"""
    
    def __init__(self, 
                 model_config: PropertyPredictionConfig = None,
                 debug_mode: str = "focused"):  # "focused", "detailed", "minimal"
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Device: {self.device}")
        
        # 모델 설정
        self.config = model_config or PropertyPredictionConfig(
            d_model=512,
            n_heads=8, 
            n_layers=6,
            dropout=0.3,
            learning_rate=1e-4,
            weight_decay=1e-3
        )
        
        # 디버거 설정
        enable_all_debug = (debug_mode == "detailed")
        self.debugger = create_training_debugger(enable_all=enable_all_debug)
        
        # 모델 초기화
        self.model = PropertyPredictionTransformer(self.config)
        self.model.to(self.device)
        
        # 손실 함수 (가중치 조정)
        self.criterion = PropertyPredictionLoss(
            entanglement_weight=1.0,
            fidelity_weight=5.0,      # Fidelity 중요도 증가
            expressibility_weight=0.1, # Expressibility 가중치 감소
            combined_weight=0.5
        )
        
        # 옵티마이저 (더 보수적인 설정)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 스케줄러 (Plateau 기반으로 변경)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        
        # 훈련 상태
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 15
        
        print(f"📊 모델 파라미터: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"🎯 디버그 모드: {debug_mode}")
    
    def analyze_data_quality(self, dataloader: DataLoader):
        """데이터 품질 분석"""
        print("\n🔍 데이터 품질 분석...")
        
        total_samples = 0
        property_stats = {
            'entanglement': [],
            'fidelity': [], 
            'expressibility': []
        }
        
        # 샘플 수집 (최대 1000개)
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 10:  # 10 배치만 분석
                break
                
            if isinstance(batch, dict) and 'targets' in batch:
                targets = batch['targets']
                for prop in property_stats.keys():
                    if prop in targets:
                        values = targets[prop].cpu().numpy()
                        property_stats[prop].extend(values)
                        
                total_samples += len(targets.get('entanglement', []))
        
        # 통계 출력
        print(f"   총 분석 샘플: {total_samples}")
        for prop, values in property_stats.items():
            if values:
                import numpy as np
                values = np.array(values)
                print(f"   {prop}:")
                print(f"     범위: [{values.min():.4f}, {values.max():.4f}]")
                print(f"     평균: {values.mean():.4f} ± {values.std():.4f}")
                
                # 이상치 체크
                q1, q3 = np.percentile(values, [25, 75])
                iqr = q3 - q1
                outliers = np.sum((values < q1 - 1.5*iqr) | (values > q3 + 1.5*iqr))
                if outliers > 0:
                    print(f"     ⚠️ 이상치: {outliers}개 ({outliers/len(values)*100:.1f}%)")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int):
        """훈련 에포크"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # 배치 처리
                if isinstance(batch, dict):
                    circuit_specs = batch.get('circuit_specs', [])
                    targets = batch.get('targets', {})
                else:
                    circuit_specs, targets = batch
                
                # GPU로 이동
                if isinstance(targets, dict):
                    targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(circuit_specs)
                
                # 손실 계산
                if isinstance(self.criterion, PropertyPredictionLoss):
                    loss_dict = self.criterion(predictions, targets)
                    loss = loss_dict['total']
                else:
                    loss = self.criterion(predictions, targets)
                
                # Backward pass
                loss.backward()
                
                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # 디버깅 로깅
                self.debugger.log_training_step(
                    model=self.model,
                    loss=loss,
                    predictions=predictions,
                    targets=targets,
                    optimizer=self.optimizer,
                    batch_idx=batch_idx,
                    epoch=epoch
                )
                
                # 진행률 업데이트
                total_loss += loss.item()
                num_batches += 1
                avg_loss = total_loss / num_batches
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'Avg': f'{avg_loss:.6f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                
            except Exception as e:
                print(f"❌ 배치 {batch_idx} 처리 중 오류: {e}")
                continue
        
        return total_loss / max(num_batches, 1)
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int):
        """검증 에포크"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = {'entanglement': [], 'fidelity': [], 'expressibility': []}
        all_targets = {'entanglement': [], 'fidelity': [], 'expressibility': []}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    # 배치 처리
                    if isinstance(batch, dict):
                        circuit_specs = batch.get('circuit_specs', [])
                        targets = batch.get('targets', {})
                    else:
                        circuit_specs, targets = batch
                    
                    # GPU로 이동
                    if isinstance(targets, dict):
                        targets = {k: v.to(self.device) for k, v in targets.items()}
                    
                    # Forward pass
                    predictions = self.model(circuit_specs)
                    
                    # 손실 계산
                    if isinstance(self.criterion, PropertyPredictionLoss):
                        loss_dict = self.criterion(predictions, targets)
                        loss = loss_dict['total']
                    else:
                        loss = self.criterion(predictions, targets)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # 예측값 수집 (분석용)
                    for prop in all_predictions.keys():
                        if prop in predictions and prop in targets:
                            all_predictions[prop].append(predictions[prop].cpu())
                            all_targets[prop].append(targets[prop].cpu())
                
                except Exception as e:
                    print(f"❌ 검증 배치 {batch_idx} 처리 중 오류: {e}")
                    continue
        
        avg_val_loss = total_loss / max(num_batches, 1)
        
        # 예측값 결합
        combined_predictions = {}
        combined_targets = {}
        for prop in all_predictions.keys():
            if all_predictions[prop]:
                combined_predictions[prop] = torch.cat(all_predictions[prop])
                combined_targets[prop] = torch.cat(all_targets[prop])
        
        # 검증 디버깅 로깅
        self.debugger.log_validation_epoch(
            val_loss=avg_val_loss,
            val_predictions=combined_predictions,
            val_targets=combined_targets,
            epoch=epoch
        )
        
        return avg_val_loss
    
    def train(self, 
              data_path: str,
              num_epochs: int = 50,
              batch_size: int = 32,
              save_dir: str = "checkpoints"):
        """메인 훈련 루프"""
        print(f"\n🚀 Property Prediction 훈련 시작")
        print(f"   데이터: {data_path}")
        print(f"   에포크: {num_epochs}")
        print(f"   배치 크기: {batch_size}")
        
        # 데이터 로더 생성
        try:
            dataset_manager = DatasetManager(data_path)
            train_loader, val_loader, test_loader = create_dataloaders(
                dataset_manager,
                train_batch_size=batch_size,
                val_batch_size=batch_size,
                test_batch_size=batch_size
            )
            
            print(f"   훈련 배치: {len(train_loader)}")
            print(f"   검증 배치: {len(val_loader)}")
            
        except Exception as e:
            print(f"❌ 데이터 로더 생성 실패: {e}")
            return
        
        # 데이터 품질 분석
        self.analyze_data_quality(train_loader)
        
        # 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # 훈련 루프
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # 훈련
            train_loss = self.train_epoch(train_loader, epoch)
            
            # 검증
            val_loss = self.validate_epoch(val_loader, epoch)
            
            # 스케줄러 업데이트
            self.scheduler.step(val_loss)
            
            # 베스트 모델 저장
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # 모델 저장
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }, os.path.join(save_dir, 'best_model.pt'))
                
                print(f"💾 베스트 모델 저장 (val_loss: {val_loss:.6f})")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.max_patience:
                print(f"🛑 Early stopping (patience: {self.max_patience})")
                break
        
        # 디버그 요약 저장
        debug_summary_path = os.path.join(save_dir, 'debug_summary.json')
        self.debugger.save_debug_summary(debug_summary_path)
        
        print(f"\n✅ 훈련 완료!")
        print(f"   최고 검증 손실: {self.best_val_loss:.6f}")
        print(f"   총 에포크: {self.current_epoch + 1}")


def main():
    parser = argparse.ArgumentParser(description='Property Prediction Training with Debug')
    parser.add_argument('--data_path', type=str, required=True, help='Dataset path')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--debug_mode', type=str, default='focused', 
                       choices=['minimal', 'focused', 'detailed'], help='Debug level')
    parser.add_argument('--save_dir', type=str, default='debug_checkpoints', help='Save directory')
    
    args = parser.parse_args()
    
    # 모델 설정
    config = PropertyPredictionConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        dropout=0.3,
        learning_rate=1e-4,
        weight_decay=1e-3
    )
    
    # 훈련 분석기 생성
    analyzer = PropertyTrainingAnalyzer(
        model_config=config,
        debug_mode=args.debug_mode
    )
    
    # 훈련 실행
    analyzer.train(
        data_path=args.data_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
