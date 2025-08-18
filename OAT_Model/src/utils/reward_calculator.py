"""
Unified Reward Calculator for Quantum Circuit Generation

통합된 보상 계산 시스템:
- Property predictor를 사용하여 현재 상태에서 속성 예측
- 목표값과의 차이를 바탕으로 수학적 보상 계산
- 학습과 추론에서 동일한 환경 제공
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np
from pathlib import Path
import sys

# Add quantumcommon to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))
from circuit_interface import CircuitSpec

# Import property prediction model
from models.property_prediction_transformer import PropertyPredictionTransformer, PropertyPredictionConfig


class RewardCalculator(nn.Module):
    """통합 보상 계산기 - Property Predictor 기반"""
    
    def __init__(
        self,
        property_predictor: PropertyPredictionTransformer,
        target_properties: Dict[str, float] = None,
        reward_weights: Dict[str, float] = None,
        convergence_threshold: float = 0.05,
        reward_scale: float = 10.0
    ):
        super().__init__()
        
        self.property_predictor = property_predictor
        self.property_predictor.eval()  # Always in eval mode for reward calculation
        
        # Default target properties
        self.target_properties = target_properties or {
            'entanglement': 0.8,
            'fidelity': 0.9,
            'expressibility': 2.0
        }
        
        # Reward weights for different properties
        self.reward_weights = reward_weights or {
            'entanglement': 1.0,
            'fidelity': 1.0,
            'expressibility': 0.5
        }
        
        self.convergence_threshold = convergence_threshold
        self.reward_scale = reward_scale
        
        # Cache for efficiency
        self._prediction_cache = {}
        
    def set_target_properties(self, targets: Dict[str, float]):
        """목표 속성값 설정"""
        self.target_properties.update(targets)
        self._prediction_cache.clear()  # Clear cache when targets change
        
    def predict_properties_from_state(self, state_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        상태 임베딩으로부터 속성 예측
        
        Args:
            state_embedding: [batch_size, d_model] 또는 [batch_size, seq_len, d_model]
            
        Returns:
            예측된 속성들
        """
        with torch.no_grad():
            # Handle different input dimensions
            if len(state_embedding.shape) == 3:
                # [batch_size, seq_len, d_model] -> [batch_size, d_model]
                state_embedding = state_embedding.mean(dim=1)
            
            # Use property predictor's prediction head directly
            predictions = self.property_predictor.prediction_head(state_embedding)
            
            return predictions
    
    def predict_properties_from_circuit(self, circuit_spec: Union[CircuitSpec, List[CircuitSpec]]) -> Dict[str, torch.Tensor]:
        """
        CircuitSpec으로부터 속성 예측
        
        Args:
            circuit_spec: 단일 회로 또는 회로 리스트
            
        Returns:
            예측된 속성들
        """
        with torch.no_grad():
            predictions = self.property_predictor(circuit_spec)
            return predictions
    
    def calculate_reward_from_predictions(
        self, 
        predictions: Dict[str, torch.Tensor],
        use_convergence_bonus: bool = True
    ) -> torch.Tensor:
        """
        예측된 속성으로부터 보상 계산
        
        수학적 보상 함수:
        - 각 속성에 대해 목표값과의 거리 계산
        - 가우시안 기반 보상 (목표에 가까울수록 높은 보상)
        - 수렴 보너스 (모든 속성이 임계값 내에 있을 때)
        
        Args:
            predictions: 예측된 속성값들
            use_convergence_bonus: 수렴 보너스 사용 여부
            
        Returns:
            계산된 보상 [batch_size]
        """
        device = next(iter(predictions.values())).device
        batch_size = next(iter(predictions.values())).shape[0]
        
        total_reward = torch.zeros(batch_size, device=device)
        convergence_count = torch.zeros(batch_size, device=device)
        
        for prop_name, weight in self.reward_weights.items():
            if prop_name in predictions and prop_name in self.target_properties:
                pred_values = predictions[prop_name]  # [batch_size]
                target_value = self.target_properties[prop_name]
                
                # Calculate distance from target
                distance = torch.abs(pred_values - target_value)
                
                # Gaussian reward function: exp(-distance^2 / (2 * sigma^2))
                # sigma는 속성별로 다르게 설정
                if prop_name == 'entanglement':
                    sigma = 0.2  # 0-1 범위, 더 엄격
                elif prop_name == 'fidelity':
                    sigma = 0.15  # 0-1 범위, 매우 엄격
                elif prop_name == 'expressibility':
                    sigma = 0.5   # 0-10 범위, 상대적으로 관대
                else:
                    sigma = 0.2
                
                # Gaussian reward
                gaussian_reward = torch.exp(-distance.pow(2) / (2 * sigma**2))
                
                # Apply weight and add to total
                weighted_reward = weight * gaussian_reward
                total_reward += weighted_reward
                
                # Count properties within convergence threshold
                within_threshold = distance < self.convergence_threshold
                convergence_count += within_threshold.float()
        
        # Convergence bonus: extra reward when all properties are close to target
        if use_convergence_bonus:
            num_properties = len(self.reward_weights)
            all_converged = convergence_count >= num_properties
            convergence_bonus = all_converged.float() * 2.0  # 2x bonus
            total_reward += convergence_bonus
        
        # Scale reward
        total_reward *= self.reward_scale
        
        return total_reward
    
    def calculate_reward_from_state(
        self, 
        state_embedding: torch.Tensor,
        use_convergence_bonus: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        상태 임베딩으로부터 보상 계산 (예측 + 보상 계산)
        
        Args:
            state_embedding: 상태 임베딩
            use_convergence_bonus: 수렴 보너스 사용 여부
            
        Returns:
            (보상값, 예측된 속성들)
        """
        predictions = self.predict_properties_from_state(state_embedding)
        rewards = self.calculate_reward_from_predictions(predictions, use_convergence_bonus)
        
        return rewards, predictions
    
    def calculate_reward_from_circuit(
        self, 
        circuit_spec: Union[CircuitSpec, List[CircuitSpec]],
        use_convergence_bonus: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        CircuitSpec으로부터 보상 계산
        
        Args:
            circuit_spec: 회로 스펙
            use_convergence_bonus: 수렴 보너스 사용 여부
            
        Returns:
            (보상값, 예측된 속성들)
        """
        predictions = self.predict_properties_from_circuit(circuit_spec)
        rewards = self.calculate_reward_from_predictions(predictions, use_convergence_bonus)
        
        return rewards, predictions
    
    def get_reward_breakdown(
        self, 
        predictions: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        보상의 세부 분해 (디버깅용)
        
        Returns:
            각 속성별 보상 기여도
        """
        device = next(iter(predictions.values())).device
        batch_size = next(iter(predictions.values())).shape[0]
        
        breakdown = {}
        
        for prop_name, weight in self.reward_weights.items():
            if prop_name in predictions and prop_name in self.target_properties:
                pred_values = predictions[prop_name]
                target_value = self.target_properties[prop_name]
                
                distance = torch.abs(pred_values - target_value)
                
                # Property-specific sigma
                if prop_name == 'entanglement':
                    sigma = 0.2
                elif prop_name == 'fidelity':
                    sigma = 0.15
                elif prop_name == 'expressibility':
                    sigma = 0.5
                else:
                    sigma = 0.2
                
                gaussian_reward = torch.exp(-distance.pow(2) / (2 * sigma**2))
                weighted_reward = weight * gaussian_reward * self.reward_scale
                
                breakdown[f'{prop_name}_reward'] = weighted_reward
                breakdown[f'{prop_name}_distance'] = distance
                breakdown[f'{prop_name}_target'] = torch.full_like(pred_values, target_value)
        
        return breakdown
    
    def update_targets_dynamically(
        self, 
        current_predictions: Dict[str, torch.Tensor],
        adaptation_rate: float = 0.1
    ):
        """
        동적 목표 조정 (선택적 기능)
        
        현재 예측값을 바탕으로 목표값을 점진적으로 조정
        """
        with torch.no_grad():
            for prop_name in self.target_properties:
                if prop_name in current_predictions:
                    current_mean = current_predictions[prop_name].mean().item()
                    current_target = self.target_properties[prop_name]
                    
                    # Exponential moving average
                    new_target = (1 - adaptation_rate) * current_target + adaptation_rate * current_mean
                    self.target_properties[prop_name] = new_target


class StateEmbeddingExtractor(nn.Module):
    """상태 임베딩 추출기 - Decision Transformer와 호환"""
    
    def __init__(self, decision_transformer):
        super().__init__()
        self.decision_transformer = decision_transformer
        
    def extract_state_embedding(
        self, 
        input_sequence: torch.Tensor,
        attention_mask: torch.Tensor,
        position: int = -1
    ) -> torch.Tensor:
        """
        Decision Transformer로부터 상태 임베딩 추출
        
        Args:
            input_sequence: [batch_size, seq_len, d_model]
            attention_mask: [batch_size, seq_len, seq_len]
            position: 추출할 위치 (-1은 마지막 위치)
            
        Returns:
            상태 임베딩 [batch_size, d_model]
        """
        with torch.no_grad():
            # Get hidden states from decision transformer
            action_prediction_mask = torch.zeros(
                input_sequence.shape[:2], 
                dtype=torch.bool, 
                device=input_sequence.device
            )
            
            outputs = self.decision_transformer(
                input_sequence, 
                attention_mask, 
                action_prediction_mask
            )
            
            hidden_states = outputs['hidden_states']  # [batch_size, seq_len, d_model]
            
            # Extract state at specified position
            if position == -1:
                state_embedding = hidden_states[:, -1, :]  # Last position
            else:
                state_embedding = hidden_states[:, position, :]
            
            return state_embedding


def create_reward_calculator(
    property_predictor_path: str,
    target_properties: Dict[str, float] = None,
    reward_weights: Dict[str, float] = None,
    device: str = 'cuda'
) -> RewardCalculator:
    """
    보상 계산기 생성 헬퍼 함수
    
    Args:
        property_predictor_path: 학습된 property predictor 모델 경로
        target_properties: 목표 속성값들
        reward_weights: 보상 가중치들
        device: 디바이스
        
    Returns:
        설정된 보상 계산기
    """
    # Load property predictor
    config = PropertyPredictionConfig()
    property_predictor = PropertyPredictionTransformer(config)
    
    if Path(property_predictor_path).exists():
        checkpoint = torch.load(property_predictor_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            property_predictor.load_state_dict(checkpoint['model_state_dict'])
        else:
            property_predictor.load_state_dict(checkpoint)
        print(f"✅ Property predictor loaded from {property_predictor_path}")
    else:
        print(f"⚠️ Property predictor path not found: {property_predictor_path}")
        print("Using randomly initialized property predictor")
    
    property_predictor.to(device)
    property_predictor.eval()
    
    # Create reward calculator
    reward_calculator = RewardCalculator(
        property_predictor=property_predictor,
        target_properties=target_properties,
        reward_weights=reward_weights
    )
    
    return reward_calculator


if __name__ == "__main__":
    # Test the reward calculator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create a dummy property predictor for testing
    config = PropertyPredictionConfig(d_model=256)
    property_predictor = PropertyPredictionTransformer(config).to(device)
    
    # Create reward calculator
    reward_calc = RewardCalculator(property_predictor)
    
    # Test with dummy state embedding
    batch_size = 4
    d_model = 256
    dummy_state = torch.randn(batch_size, d_model, device=device)
    
    rewards, predictions = reward_calc.calculate_reward_from_state(dummy_state)
    
    print(f"✅ Reward Calculator Test:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Rewards shape: {rewards.shape}")
    print(f"  - Rewards: {rewards}")
    print(f"  - Predictions keys: {list(predictions.keys())}")
    
    # Test reward breakdown
    breakdown = reward_calc.get_reward_breakdown(predictions)
    print(f"  - Breakdown keys: {list(breakdown.keys())}")
