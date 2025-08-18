"""
Property Prediction Training Debug System
핵심 훈련 정보만 디버깅하여 수렴 문제 분석
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List
import time
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class TrainingDebugConfig:
    """디버깅 설정"""
    # 핵심 메트릭만 활성화
    enable_loss_analysis: bool = True
    enable_gradient_analysis: bool = True
    enable_prediction_analysis: bool = True
    enable_data_analysis: bool = False  # 데이터 분석은 선택적
    
    # 출력 빈도 (너무 많은 정보 방지)
    log_every_n_steps: int = 10
    detailed_log_every_n_epochs: int = 5
    
    # 분석 깊이 제한
    max_samples_to_analyze: int = 5
    gradient_norm_threshold: float = 10.0
    loss_plateau_threshold: float = 0.001  # 손실 정체 임계값


class PropertyTrainingDebugger:
    """Property Prediction 훈련 핵심 디버거"""
    
    def __init__(self, config: TrainingDebugConfig = None):
        self.config = config or TrainingDebugConfig()
        self.step_count = 0
        self.epoch_count = 0
        
        # 핵심 메트릭 추적
        self.loss_history = []
        self.val_loss_history = []
        self.gradient_norms = []
        self.learning_rates = []
        
        # 수렴 분석
        self.plateau_detection = {
            'consecutive_no_improvement': 0,
            'best_val_loss': float('inf'),
            'last_improvement_epoch': 0
        }
        
        print("🔍 Property Training Debugger 초기화")
        print(f"   - 손실 분석: {self.config.enable_loss_analysis}")
        print(f"   - 그래디언트 분석: {self.config.enable_gradient_analysis}")
        print(f"   - 예측 분석: {self.config.enable_prediction_analysis}")
    
    def log_training_step(self, 
                         model: nn.Module,
                         loss: torch.Tensor,
                         predictions: Dict[str, torch.Tensor],
                         targets: Dict[str, torch.Tensor],
                         optimizer: torch.optim.Optimizer,
                         batch_idx: int,
                         epoch: int):
        """훈련 스텝 핵심 정보 로깅"""
        self.step_count += 1
        
        # 매 N 스텝마다만 로깅
        if self.step_count % self.config.log_every_n_steps != 0:
            return
        
        debug_info = {}
        
        # 1. 핵심 손실 분석
        if self.config.enable_loss_analysis:
            debug_info['loss'] = self._analyze_loss(loss, predictions, targets)
        
        # 2. 핵심 그래디언트 분석
        if self.config.enable_gradient_analysis:
            debug_info['gradients'] = self._analyze_gradients(model)
        
        # 3. 핵심 예측 분석 (샘플링)
        if self.config.enable_prediction_analysis:
            debug_info['predictions'] = self._analyze_predictions(predictions, targets)
        
        # 4. 학습률 추적
        debug_info['learning_rate'] = optimizer.param_groups[0]['lr']
        self.learning_rates.append(debug_info['learning_rate'])
        
        # 핵심 정보만 출력
        self._print_core_debug_info(debug_info, batch_idx, epoch)
    
    def log_validation_epoch(self, 
                           val_loss: float,
                           val_predictions: Dict[str, torch.Tensor],
                           val_targets: Dict[str, torch.Tensor],
                           epoch: int):
        """검증 에포크 핵심 분석"""
        self.epoch_count = epoch
        self.val_loss_history.append(val_loss)
        
        # 수렴 분석
        convergence_info = self._analyze_convergence(val_loss, epoch)
        
        # 상세 분석 (N 에포크마다)
        detailed_analysis = {}
        if epoch % self.config.detailed_log_every_n_epochs == 0:
            detailed_analysis = self._detailed_validation_analysis(
                val_predictions, val_targets, epoch
            )
        
        # 핵심 검증 정보 출력
        self._print_validation_summary(val_loss, convergence_info, detailed_analysis, epoch)
    
    def _analyze_loss(self, loss: torch.Tensor, predictions: Dict, targets: Dict) -> Dict:
        """핵심 손실 분석"""
        self.loss_history.append(loss.item())
        
        analysis = {
            'current_loss': loss.item(),
            'loss_trend': 'stable'
        }
        
        # 손실 트렌드 분석 (최근 10스텝)
        if len(self.loss_history) >= 10:
            recent_losses = self.loss_history[-10:]
            if recent_losses[-1] < recent_losses[0] * 0.95:
                analysis['loss_trend'] = 'decreasing'
            elif recent_losses[-1] > recent_losses[0] * 1.05:
                analysis['loss_trend'] = 'increasing'
        
        # 개별 속성 손실 (가능한 경우)
        if isinstance(loss, dict):
            analysis['property_losses'] = {
                k: v.item() if torch.is_tensor(v) else v 
                for k, v in loss.items() 
                if k in ['entanglement', 'fidelity', 'expressibility']
            }
        
        return analysis
    
    def _analyze_gradients(self, model: nn.Module) -> Dict:
        """핵심 그래디언트 분석"""
        total_norm = 0.0
        param_count = 0
        max_grad = 0.0
        min_grad = float('inf')
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                grad_max = param.grad.data.abs().max().item()
                grad_min = param.grad.data.abs().min().item()
                
                max_grad = max(max_grad, grad_max)
                min_grad = min(min_grad, grad_min)
        
        total_norm = total_norm ** (1. / 2)
        self.gradient_norms.append(total_norm)
        
        analysis = {
            'total_norm': total_norm,
            'max_grad': max_grad,
            'min_grad': min_grad if min_grad != float('inf') else 0.0,
            'param_count': param_count
        }
        
        # 그래디언트 문제 감지
        if total_norm > self.config.gradient_norm_threshold:
            analysis['warning'] = 'gradient_explosion'
        elif total_norm < 1e-6:
            analysis['warning'] = 'gradient_vanishing'
        
        return analysis
    
    def _analyze_predictions(self, predictions: Dict, targets: Dict) -> Dict:
        """핵심 예측 분석 (샘플링)"""
        analysis = {}
        
        for prop_name in ['entanglement', 'fidelity', 'expressibility']:
            if prop_name in predictions and prop_name in targets:
                pred = predictions[prop_name]
                target = targets[prop_name]
                
                # 기본 통계
                analysis[prop_name] = {
                    'pred_mean': pred.mean().item(),
                    'pred_std': pred.std().item(),
                    'target_mean': target.mean().item(),
                    'target_std': target.std().item(),
                    'mae': torch.abs(pred - target).mean().item()
                }
                
                # 예측 범위 체크
                if prop_name in ['entanglement', 'fidelity']:
                    out_of_range = ((pred < 0) | (pred > 1)).sum().item()
                    if out_of_range > 0:
                        analysis[prop_name]['out_of_range_count'] = out_of_range
        
        return analysis
    
    def _analyze_convergence(self, val_loss: float, epoch: int) -> Dict:
        """수렴 분석"""
        convergence_info = {
            'current_val_loss': val_loss,
            'best_val_loss': self.plateau_detection['best_val_loss'],
            'epochs_since_improvement': epoch - self.plateau_detection['last_improvement_epoch']
        }
        
        # 개선 여부 확인
        if val_loss < self.plateau_detection['best_val_loss'] - self.config.loss_plateau_threshold:
            self.plateau_detection['best_val_loss'] = val_loss
            self.plateau_detection['last_improvement_epoch'] = epoch
            self.plateau_detection['consecutive_no_improvement'] = 0
            convergence_info['status'] = 'improving'
        else:
            self.plateau_detection['consecutive_no_improvement'] += 1
            convergence_info['status'] = 'plateau'
        
        # 정체 경고
        if self.plateau_detection['consecutive_no_improvement'] >= 10:
            convergence_info['warning'] = 'long_plateau'
        
        return convergence_info
    
    def _detailed_validation_analysis(self, predictions: Dict, targets: Dict, epoch: int) -> Dict:
        """상세 검증 분석 (N 에포크마다)"""
        analysis = {}
        
        for prop_name in ['entanglement', 'fidelity', 'expressibility']:
            if prop_name in predictions and prop_name in targets:
                pred = predictions[prop_name]
                target = targets[prop_name]
                
                # 상세 통계
                analysis[prop_name] = {
                    'correlation': torch.corrcoef(torch.stack([pred, target]))[0, 1].item(),
                    'mse': torch.mean((pred - target) ** 2).item(),
                    'r2_score': self._calculate_r2(pred, target),
                    'prediction_range': [pred.min().item(), pred.max().item()],
                    'target_range': [target.min().item(), target.max().item()]
                }
        
        return analysis
    
    def _calculate_r2(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """R² 점수 계산"""
        ss_res = torch.sum((targets - predictions) ** 2)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        return r2.item()
    
    def _print_core_debug_info(self, debug_info: Dict, batch_idx: int, epoch: int):
        """핵심 디버그 정보 출력"""
        print(f"\n📊 [Epoch {epoch:3d}, Batch {batch_idx:3d}] 핵심 훈련 상태")
        
        # 손실 정보
        if 'loss' in debug_info:
            loss_info = debug_info['loss']
            print(f"   Loss: {loss_info['current_loss']:.6f} ({loss_info['loss_trend']})")
            
            if 'property_losses' in loss_info:
                for prop, loss_val in loss_info['property_losses'].items():
                    print(f"     {prop}: {loss_val:.6f}")
        
        # 그래디언트 정보
        if 'gradients' in debug_info:
            grad_info = debug_info['gradients']
            print(f"   Gradient Norm: {grad_info['total_norm']:.6f}")
            
            if 'warning' in grad_info:
                print(f"   ⚠️  Gradient Warning: {grad_info['warning']}")
        
        # 예측 정보 (간략)
        if 'predictions' in debug_info:
            pred_info = debug_info['predictions']
            for prop, stats in pred_info.items():
                if isinstance(stats, dict) and 'mae' in stats:
                    print(f"   {prop} MAE: {stats['mae']:.6f}")
        
        # 학습률
        print(f"   Learning Rate: {debug_info['learning_rate']:.2e}")
    
    def _print_validation_summary(self, val_loss: float, convergence_info: Dict, 
                                detailed_analysis: Dict, epoch: int):
        """검증 요약 출력"""
        print(f"\n🎯 [Epoch {epoch:3d}] 검증 결과")
        print(f"   Validation Loss: {val_loss:.6f}")
        print(f"   Best Loss: {convergence_info['best_val_loss']:.6f}")
        print(f"   Status: {convergence_info['status']}")
        print(f"   No Improvement: {convergence_info['epochs_since_improvement']} epochs")
        
        if 'warning' in convergence_info:
            print(f"   ⚠️  {convergence_info['warning']}")
        
        # 상세 분석 (N 에포크마다)
        if detailed_analysis:
            print(f"\n📈 상세 분석:")
            for prop, stats in detailed_analysis.items():
                if isinstance(stats, dict):
                    corr = stats.get('correlation', 0)
                    r2 = stats.get('r2_score', 0)
                    print(f"   {prop}: Corr={corr:.3f}, R²={r2:.3f}")
    
    def save_debug_summary(self, save_path: str):
        """디버그 요약 저장"""
        summary = {
            'loss_history': self.loss_history[-100:],  # 최근 100개만
            'val_loss_history': self.val_loss_history,
            'gradient_norms': self.gradient_norms[-100:],
            'learning_rates': self.learning_rates[-100:],
            'plateau_detection': self.plateau_detection,
            'final_analysis': {
                'total_steps': self.step_count,
                'total_epochs': self.epoch_count,
                'convergence_status': 'plateau' if self.plateau_detection['consecutive_no_improvement'] > 5 else 'training'
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 디버그 요약 저장: {save_path}")


def create_training_debugger(enable_all: bool = False) -> PropertyTrainingDebugger:
    """훈련 디버거 생성 헬퍼"""
    if enable_all:
        config = TrainingDebugConfig(
            enable_loss_analysis=True,
            enable_gradient_analysis=True,
            enable_prediction_analysis=True,
            enable_data_analysis=True,
            log_every_n_steps=5,
            detailed_log_every_n_epochs=2
        )
    else:
        # 핵심 정보만
        config = TrainingDebugConfig(
            enable_loss_analysis=True,
            enable_gradient_analysis=True,
            enable_prediction_analysis=True,
            enable_data_analysis=False,
            log_every_n_steps=10,
            detailed_log_every_n_epochs=5
        )
    
    return PropertyTrainingDebugger(config)


if __name__ == "__main__":
    # 테스트
    debugger = create_training_debugger()
    print("✅ Property Training Debugger 생성 완료")
