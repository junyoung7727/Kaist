"""
Decision Transformer Model
간단하고 확장성 높은 Decision Transformer 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math
import os
import sys
from pathlib import Path

# 공통 디버그 유틸리티 사용
from utils.debug_utils import debug_print, debug_tensor_info

# 모듈러 어텐션 시스템 임포트
from models.modular_attention import ModularAttention, AttentionMode, create_modular_attention

# 🎆 NEW: 게이트 레지스트리 싱글톤 임포트
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))
from gates import QuantumGateRegistry

# 🗑️ REMOVED: Legacy MultiHeadAttention class - now using ModularAttention system


class TransformerBlock(nn.Module):
    """트랜스포머 블록 (모듈러 어텐션 지원)"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, 
                 attention_mode: str = "standard"):
        super().__init__()
        
        # 🎆 NEW: 모듈러 어텐션 사용
        self.attention = create_modular_attention(d_model, n_heads, dropout, attention_mode)
        self.attention_mode = attention_mode
        
        # 문제 6 해결: 피드포워드 정규화 최적화 (과도한 정규화 제거)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),  # 중간 dropout만 유지
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Pre-norm 구조 (안정적 epsilon)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 문제 4 해결: 학습 가능한 스케일 파라미터 복원
        self.scale = nn.Parameter(torch.ones(1) * 0.5)  # 학습 가능한 파라미터
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor, 
                grid_structure: Optional[Dict] = None, edges: Optional[List[Dict]] = None) -> torch.Tensor:
        debug_print(f"  TransformerBlock input - NaN: {torch.isnan(x).any()}, min/max: {x.min().item():.4f}/{x.max().item():.4f}")
        debug_print(f"  Using attention mode: {self.attention_mode}")
        
        # 🎆 NEW: 모듈러 어텐션 (고급 모드용 추가 인자 지원)
        norm_x = self.norm1(x)
        debug_print(f"    After norm1 - NaN: {torch.isnan(norm_x).any()}, min/max: {norm_x.min().item():.4f}/{norm_x.max().item():.4f}")
        
        attn_out = self.attention(norm_x, mask, grid_structure, edges)
        debug_print(f"    After {self.attention_mode} attention - NaN: {torch.isnan(attn_out).any()}, min/max: {attn_out.min().item():.4f}/{attn_out.max().item():.4f}")
        
        dropout_attn = self.dropout1(attn_out)
        debug_print(f"    After dropout1 - NaN: {torch.isnan(dropout_attn).any()}, min/max: {dropout_attn.min().item():.4f}/{dropout_attn.max().item():.4f}")
        
        scaled_attn = self.scale * dropout_attn
        debug_print(f"    After scale*dropout1 - NaN: {torch.isnan(scaled_attn).any()}, scale: {self.scale.item():.4f}")
        
        x = x + scaled_attn
        debug_print(f"    After residual1 - NaN: {torch.isnan(x).any()}, min/max: {x.min().item():.4f}/{x.max().item():.4f}")
        
        # Pre-norm + 피드포워드 + 스케일링된 잔차 연결
        norm_x = self.norm2(x)
        debug_print(f"    After norm2 - NaN: {torch.isnan(norm_x).any()}, min/max: {norm_x.min().item():.4f}/{norm_x.max().item():.4f}")
        
        ff_out = self.feed_forward(norm_x)
        debug_print(f"    After feedforward - NaN: {torch.isnan(ff_out).any()}, min/max: {ff_out.min().item():.4f}/{ff_out.max().item():.4f}")
        
        dropout_ff = self.dropout2(ff_out)
        debug_print(f"    After dropout2 - NaN: {torch.isnan(dropout_ff).any()}, min/max: {dropout_ff.min().item():.4f}/{dropout_ff.max().item():.4f}")
        
        scaled_ff = self.scale * dropout_ff
        debug_print(f"    After scale*dropout2 - NaN: {torch.isnan(scaled_ff).any()}")
        
        x = x + scaled_ff
        debug_print(f"    TransformerBlock output - NaN: {torch.isnan(x).any()}, min/max: {x.min().item():.4f}/{x.max().item():.4f}")
        
        return x
    
    def set_attention_mode(self, mode: str):
        """어텐션 모드 변경"""
        self.attention.set_mode(AttentionMode(mode.lower()))
        self.attention_mode = mode
        debug_print(f"TransformerBlock attention mode changed to: {mode}")


class DecisionTransformer(nn.Module):
    """Decision Transformer 모델"""
    
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        n_gate_types: int = None,  # 🎆 NEW: gate vocab 싱글톤에서 자동 설정
        dropout: float = 0.1,
        attention_mode: str = "standard"  # 🎆 NEW: 어텐션 모드 선택
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 🎆 NEW: gate vocab 싱글톤에서 gate 수 가져오기
        if n_gate_types is None:
            self.n_gate_types = QuantumGateRegistry.get_singleton_gate_count()
            debug_print(f"🎆 DecisionTransformer: Using gate vocab singleton, n_gate_types = {self.n_gate_types}")
        else:
            self.n_gate_types = n_gate_types
            debug_print(f"⚠️ DecisionTransformer: Using manual n_gate_types = {self.n_gate_types}")
        
        self.attention_mode = attention_mode  # 🎆 NEW: 어텐션 모드 저장
        
        # 🎆 NEW: 모듈러 어텐션을 사용하는 트랜스포머 레이어들
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, attention_mode)
            for _ in range(n_layers)
        ])
        
        # 액션 예측 헤드 (게이트 타입 예측)
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_gate_types)
        )
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # 모델 초기화 (매우 보수적)
        self.apply(self._conservative_init_weights)
        
    def _conservative_init_weights(self, module):
        """문제 5 해결: 최적화된 가중치 초기화 (안정성 + 학습 능력)"""
        if isinstance(module, nn.Linear):
            # 문제 5 해결: gain 0.1 → 0.5로 증가 (그래디언트 소실 방지)
            torch.nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # 임베딩 초기화도 약간 증가
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _init_weights(self, module):
        """가중치 초기화"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self, 
        input_sequence: torch.Tensor,
        attention_mask: torch.Tensor,
        action_prediction_mask: torch.Tensor,
        grid_structure: Optional[Dict] = None,  # 🎆 NEW: 고급 어텐션용
        edges: Optional[List[Dict]] = None       # 🎆 NEW: 고급 어텐션용
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_sequence: [batch, seq_len, d_model]
            attention_mask: [batch, seq_len, seq_len] 
            action_prediction_mask: [batch, seq_len]
        
        Returns:
            Dict with predictions and logits
        """
        # 디버그: 입력 테서 체크
        debug_print(f"Debug: input_sequence shape: {input_sequence.shape}")
        debug_print(f"Debug: attention_mask shape: {attention_mask.shape}")
        debug_print(f"Debug: action_prediction_mask shape: {action_prediction_mask.shape}")
        
        # 디버그: NaN 체크
        debug_print(f"Debug: input_sequence contains NaN: {torch.isnan(input_sequence).any()}")
        debug_print(f"Debug: input_sequence contains Inf: {torch.isinf(input_sequence).any()}")
        debug_print(f"Debug: input_sequence min/max: {input_sequence.min().item():.4f}/{input_sequence.max().item():.4f}")
        
        # 차원 조정: [batch, 1, seq_len, ...] -> [batch, seq_len, ...]
        input_sequence = input_sequence.squeeze(1)  # [batch, seq_len, d_model]
        attention_mask = attention_mask.squeeze(1)  # [batch, seq_len, seq_len]
        action_prediction_mask = action_prediction_mask.squeeze(1)  # [batch, seq_len]
        
        debug_print(f"Debug: squeezed input_sequence: {input_sequence.shape}")
        debug_print(f"Debug: squeezed attention_mask: {attention_mask.shape}")
        debug_print(f"Debug: squeezed action_prediction_mask: {action_prediction_mask.shape}")
        
        batch_size, seq_len, d_model = input_sequence.shape
        
        # 입력 드롭아웃
        x = self.dropout(input_sequence)
        debug_print(f"Debug: After dropout - contains NaN: {torch.isnan(x).any()}")
        
        # 🎆 NEW: 모듈러 어텐션을 사용하는 트랜스포머 레이어들 통과
        for i, transformer_block in enumerate(self.transformer_blocks):
            x = transformer_block(x, attention_mask, grid_structure, edges)
            debug_print(f"Debug: After transformer block {i} ({self.attention_mode}) - contains NaN: {torch.isnan(x).any()}")
            if torch.isnan(x).any():
                debug_print(f"Debug: NaN detected at transformer block {i}!")
                break
        
        # 액션 예측 헤드
        debug_print(f"Debug: Before action_head - contains NaN: {torch.isnan(x).any()}")
        action_logits = self.action_head(x)  # [batch, seq_len, n_gate_types]
        debug_print(f"Debug: After action_head - contains NaN: {torch.isnan(action_logits).any()}")
        
        # 액션 위치에서만 로짓 추출
        action_predictions = torch.zeros_like(action_logits)
        action_predictions[action_prediction_mask] = action_logits[action_prediction_mask]
        
        return {
            'action_logits': action_logits,
            'action_predictions': action_predictions,
            'hidden_states': x
        }
    
    def set_attention_mode(self, mode: str):
        """모든 트랜스포머 블록의 어텐션 모드 변경"""
        self.attention_mode = mode
        for block in self.transformer_blocks:
            block.set_attention_mode(mode)
        debug_print(f"DecisionTransformer attention mode changed to: {mode}")
    
    def get_attention_mode(self) -> str:
        """현재 어텐션 모드 반환"""
        return self.attention_mode
    
    def compare_attention_modes(self, input_sequence: torch.Tensor, attention_mask: torch.Tensor, 
                              action_prediction_mask: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """어텐션 모드별 결과 비교"""
        original_mode = self.attention_mode
        results = {}
        
        for mode in ["standard", "advanced", "hybrid"]:
            self.set_attention_mode(mode)
            with torch.no_grad():
                output = self.forward(input_sequence, attention_mask, action_prediction_mask)
                results[mode] = {
                    'action_logits': output['action_logits'].clone(),
                    'action_predictions': output['action_predictions'].clone()
                }
        
        # 원래 모드로 복구
        self.set_attention_mode(original_mode)
        return results
    
    def predict_next_action(
        self,
        input_sequence: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """다음 액션 예측 (추론용)"""
        with torch.no_grad():
            outputs = self.forward(
                input_sequence, 
                attention_mask,
                torch.ones(input_sequence.shape[:2], dtype=torch.bool, device=input_sequence.device)
            )
            
            # 마지막 위치의 예측 반환
            last_logits = outputs['action_logits'][:, -1, :]  # [batch, n_gate_types]
            return F.softmax(last_logits, dim=-1)


class DecisionTransformerLoss(nn.Module):
    """Decision Transformer 손실 함수"""
    
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(
        self,
        action_logits: torch.Tensor,
        target_actions: torch.Tensor,
        action_prediction_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            action_logits: [batch, seq_len, n_gate_types]
            target_actions: [batch, seq_len] - 정답 액션 인덱스
            action_prediction_mask: [batch, seq_len] - 액션 예측 위치
        """
        
        # 디버그: 형태 확인
        debug_print(f"Debug: action_logits shape: {action_logits.shape}")
        debug_print(f"Debug: target_actions shape: {target_actions.shape}")
        debug_print(f"Debug: action_prediction_mask shape: {action_prediction_mask.shape}")
        
        # 디버그: 실제 값들 확인
        debug_print(f"Debug: action_prediction_mask sum: {action_prediction_mask.sum().item()}")
        debug_print(f"Debug: target_actions unique values: {torch.unique(target_actions)}")
        debug_print(f"Debug: action_logits contains NaN: {torch.isnan(action_logits).any()}")
        debug_print(f"Debug: target_actions contains invalid: {(target_actions < 0).any() or (target_actions >= 20).any()}")
        
        # 액션 위치에서만 손실 계산
        masked_logits = action_logits[action_prediction_mask]  # [n_actions, n_gate_types]
        masked_targets = target_actions[action_prediction_mask]  # [n_actions]
        
        debug_print(f"Debug: masked_logits shape: {masked_logits.shape}")
        debug_print(f"Debug: masked_targets shape: {masked_targets.shape}")
        debug_print(f"Debug: masked_targets values: {masked_targets}")
        debug_print(f"Debug: masked_logits contains NaN: {torch.isnan(masked_logits).any()}")
        debug_print(f"Debug: masked_logits min/max: {masked_logits.min().item():.4f}/{masked_logits.max().item():.4f}")
        
        if masked_logits.numel() == 0:
            # 예측할 액션이 없는 경우
            return {
                'loss': torch.tensor(0.0, device=action_logits.device, requires_grad=True),
                'accuracy': torch.tensor(0.0, device=action_logits.device)
            }
        
        # 크로스 엔트로피 손실
        loss = self.cross_entropy(masked_logits, masked_targets)
        
        # 정확도 계산
        with torch.no_grad():
            predictions = torch.argmax(masked_logits, dim=-1)
            accuracy = (predictions == masked_targets).float().mean()
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }


# 모델 팩토리 함수
def create_decision_transformer(
    d_model: int = 512,
    n_layers: int = 6,
    n_heads: int = 8,
    n_gate_types: int = 20,
    dropout: float = 0.1
) -> DecisionTransformer:
    """Decision Transformer 모델 생성"""
    
    d_ff = d_model * 4  # 표준 비율
    
    return DecisionTransformer(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        n_gate_types=n_gate_types,
        dropout=dropout
    )


# 사용 예시
if __name__ == "__main__":
    # 모델 생성
    model = create_decision_transformer(
        d_model=256,
        n_layers=4,
        n_heads=8,
        n_gate_types=20  # 🔧 FIXED: 통일된 게이트 타입 수
    )
    
    # 더미 데이터로 테스트
    batch_size, seq_len, d_model = 2, 10, 256
    
    input_sequence = torch.randn(batch_size, seq_len, d_model)
    attention_mask = torch.tril(torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool))
    action_prediction_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    action_prediction_mask[:, 1::3] = True  # 액션 위치
    
    # 순전파
    outputs = model(input_sequence, attention_mask, action_prediction_mask)
    
    debug_print(f"Action logits shape: {outputs['action_logits'].shape}")
    debug_print(f"Hidden states shape: {outputs['hidden_states'].shape}")
    
    # 손실 계산 테스트
    loss_fn = DecisionTransformerLoss()
    target_actions = torch.randint(0, 16, (batch_size, seq_len))
    
    loss_outputs = loss_fn(
        outputs['action_logits'],
        target_actions,
        action_prediction_mask
    )
    
    print(f"Loss: {loss_outputs['loss'].item():.4f}")
    print(f"Accuracy: {loss_outputs['accuracy'].item():.4f}")
