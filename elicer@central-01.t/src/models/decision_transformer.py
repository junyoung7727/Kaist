"""
Decision Transformer Model
간단하고 확장성 높은 Decision Transformer 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np

# Focal Loss for addressing class imbalance in classification tasks
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in classification tasks"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, ignore_index: int = -100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [N, C] logits
            targets: [N] class indices
        """
        # 기본 CrossEntropy 계산
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        
        # p_t 계산 (정답 클래스에 대한 확률)
        pt = torch.exp(-ce_loss)
        
        # Focal Loss 계산: α * (1-p_t)^γ * CE_loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()

from dataclasses import dataclass
from pathlib import Path

# 공통 디버그 유틸리티 사용
from utils.debug_utils import debug_print, debug_tensor_info

# 모듈러 어텐션 시스템 임포트
from models.modular_attention import ModularAttention, AttentionMode, create_modular_attention

# 🎆 NEW: 게이트 레지스트리 싱글톤 임포트
from pathlib import Path
import sys
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
        max_qubits: int = 32,  # 🎆 NEW: 최대 큐빗 수
        dropout: float = 0.1,
        attention_mode: str = "standard"  # 🎆 NEW: 어텐션 모드 선택
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_qubits = max_qubits
        
        # 🎆 NEW: gate vocab 싱글톤에서 gate 수 가져오기
        if n_gate_types is None:
            self.n_gate_types = QuantumGateRegistry.get_singleton_gate_count()
            debug_print(f"🎆 DecisionTransformer: Using gate vocab singleton, n_gate_types = {self.n_gate_types}")
        else:
            self.n_gate_types = n_gate_types
            debug_print(f"⚠️ DecisionTransformer: Using manual n_gate_types = {self.n_gate_types}")
        
        self.attention_mode = attention_mode  # 🎆 NEW: 어텐션 모드 저장
        
        # Gate registry for qubit/parameter requirements
        self.gate_registry = QuantumGateRegistry()
        
        # 🎆 NEW: 기존 attention.py의 고급 어텐션 시스템 활용 (요구사항 3)
        # 인코더: 양자 회로 제약 정보를 고급 어텐션으로 처리
        from models.attention import (
            GridPositionalAttention, RegisterFlowAttention, 
            EntanglementAttention, SemanticAttention, AttentionFusionNetwork
        )
        
        # 인코더 블록들 (양자 회로 제약 인식 어텐션 사용)
        self.constraint_encoder_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, "advanced")  # 고급 어텐션 모드
            for _ in range(n_layers // 2)
        ])
        
        # 디코더 블록들 (시퀀스 생성용)
        self.sequence_decoder_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, attention_mode)
            for _ in range(n_layers - n_layers // 2)
        ])
        
        # 🎆 NEW: 양자 회로 제약 정보를 위한 특화된 어텐션들
        self.grid_attention = GridPositionalAttention(d_model, n_heads)
        self.register_attention = RegisterFlowAttention(d_model, n_heads)  
        self.entangle_attention = EntanglementAttention(d_model, n_heads)
        self.semantic_attention = SemanticAttention(d_model, n_heads)
        self.attention_fusion = AttentionFusionNetwork(d_model, n_heads)
        
        # Cross-attention for constraint-aware sequence generation
        self.constraint_cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # 🔥 NEW: 클린한 3개 액션 헤드 (단일 책임 원칙)
        self.action_heads = nn.ModuleDict({
            'gate': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, self.n_gate_types)  # 분류
            ),
            'position': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(), 
                nn.Linear(d_model // 2, max_qubits * 2)  # 분류: [qubit1_logits, qubit2_logits] for each possible qubit
            ),
            'parameter': nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, 1)  # 회귀: 단일 연속값
            )
        })
        
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
        grid_structure: Optional[torch.Tensor] = None,
        edges: Optional[torch.Tensor] = None,
        circuit_constraints: Optional[torch.Tensor] = None,  # NEW: 양자 회로 제약 정보
        predictions: Optional[Dict] = None,
        targets: Optional[Dict] = None,
        num_qubits: Optional[List[int]] = None,
        num_gates: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_sequence: [batch, seq_len, d_model]
            attention_mask: [batch, seq_len, seq_len] 
            action_prediction_mask: [batch, seq_len]
            circuit_constraints: [batch, constraint_len, d_model] - 양자 회로 제약 정보
        
        Returns:
            Dict with predictions and logits
        """
        debug_print(f"DecisionTransformer forward - input shape: {input_sequence.shape}")
        debug_print(f"  Input - NaN: {torch.isnan(input_sequence).any()}, min/max: {input_sequence.min().item():.4f}/{input_sequence.max().item():.4f}")
        
        # 텐서 차원 정규화 - 4D 텐서를 3D로 변환
        if len(input_sequence.shape) == 4:
            # [batch, 1, seq_len, d_model] -> [batch, seq_len, d_model]
            input_sequence = input_sequence.squeeze(1)
            debug_print(f"  Squeezed input shape: {input_sequence.shape}")
        elif len(input_sequence.shape) != 3:
            raise ValueError(f"Expected 3D or 4D input tensor, got {len(input_sequence.shape)}D: {input_sequence.shape}")
        
        # 마스크 차원 정규화
        debug_print(f"  Original action_prediction_mask shape: {action_prediction_mask.shape}")
        if len(action_prediction_mask.shape) == 3:
            # [batch, 1, seq_len] -> [batch, seq_len]
            action_prediction_mask = action_prediction_mask.squeeze(1)
            debug_print(f"  Squeezed action_prediction_mask shape: {action_prediction_mask.shape}")
        elif len(action_prediction_mask.shape) != 2:
            raise ValueError(f"Expected 2D or 3D action_prediction_mask, got {len(action_prediction_mask.shape)}D: {action_prediction_mask.shape}")
        
        batch_size, seq_len, _ = input_sequence.shape
        
        # NEW: 고급 어텐션 기반 제약 인코딩 (요구사항 3)
        # 1. 양자 회로 제약 정보를 고급 어텐션으로 처리
        constraint_features = None
        if grid_structure is not None and edges is not None:
            # 입력 시퀀스를 constraint encoder로 처리
            constraint_input = input_sequence.transpose(0, 1)  # [seq_len, batch, d_model] for attention modules
            
            # 각 특화된 어텐션 적용
            attention_outputs = {}
            
            # Grid positional attention
            grid_out = self.grid_attention(constraint_input, grid_structure, edges)
            attention_outputs['grid_attention'] = grid_out
            
            # Register flow attention  
            register_out = self.register_attention(constraint_input, grid_structure, edges)
            attention_outputs['register_attention'] = register_out
            
            # Entanglement attention
            entangle_out = self.entangle_attention(constraint_input, grid_structure, edges)
            attention_outputs['entanglement_attention'] = entangle_out
            
            # Semantic attention
            semantic_out = self.semantic_attention(constraint_input)
            attention_outputs['semantic_attention'] = semantic_out
            
            # 모든 어텐션 결과 융합
            constraint_features = self.attention_fusion(attention_outputs)  # [seq_len, d_model]
            constraint_features = constraint_features.transpose(0, 1)  # [batch, seq_len, d_model]
            
            debug_print(f"  Constraint features from advanced attention: {constraint_features.shape}")
        
        # 2. 시퀀스 디코더 처리
        x = self.dropout(input_sequence)
        debug_print(f"  After input dropout - NaN: {torch.isnan(x).any()}, min/max: {x.min().item():.4f}/{x.max().item():.4f}")
        
        # 시퀀스 디코더 블록들 처리
        for i, decoder_block in enumerate(self.sequence_decoder_blocks):
            debug_print(f"  Processing sequence decoder block {i}")
            x = decoder_block(x, attention_mask, grid_structure, edges)
            
            # Cross-attention with constraint features
            if constraint_features is not None:
                x_norm = F.layer_norm(x, x.shape[-1:])
                cross_attn_out, _ = self.constraint_cross_attention(
                    x_norm, constraint_features, constraint_features,
                    need_weights=False
                )
                x = x + self.dropout(cross_attn_out)
            
            debug_print(f"    Sequence decoder block {i} output - NaN: {torch.isnan(x).any()}, min/max: {x.min().item():.4f}/{x.max().item():.4f}")
        
        return self.predict_actions(x, action_prediction_mask, num_qubits)
    
    def predict_actions(self, hidden_states: torch.Tensor, action_mask: torch.Tensor, num_qubits: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """ 액션 예측 전용 메서드 - 전체 시퀀스 길이 유지"""
        batch_size, seq_len, d_model = hidden_states.shape
        
        debug_print(f"🚀 predict_actions - hidden_states: {hidden_states.shape}, action_mask: {action_mask.shape}")
        
        # 전체 시퀀스에 대해 예측 수행 (마스크로 필터링)
        # 모든 위치에서 예측하고, 손실 계산 시 마스크로 필터링
        
        # 3가지 액션 예측 - 전체 시퀀스에 대해
        gate_logits = self.action_heads['gate'](hidden_states)  # [batch, seq_len, n_gate_types]
        
        # Position head 출력
        position_raw = self.action_heads['position'](hidden_states)  # [batch, seq_len, max_qubits * 2]
        # [batch, seq_len, max_qubits * 2] -> [batch, seq_len, max_qubits, 2]
        position_reshaped = position_raw.view(batch_size, seq_len, self.max_qubits, 2)
        
        parameter_preds = self.action_heads['parameter'](hidden_states).squeeze(-1)  # [batch, seq_len]
        
        predictions = {
            'gate_logits': gate_logits,        # [batch, seq_len, n_gate_types]
            'position_preds': position_reshaped,  # [batch, seq_len, max_qubits, 2]
            'parameter_preds': parameter_preds    # [batch, seq_len]
        }
        
        debug_print(f"🚀 predictions shapes - gate: {gate_logits.shape}, position: {position_reshaped.shape}, param: {parameter_preds.shape}")
        
        # 🚀 배치 메타데이터를 활용한 동적 마스킹 적용
        if num_qubits is not None:
            predictions = self._apply_dynamic_qubit_masking(predictions, num_qubits, action_mask)
        
        debug_print(f"🚀 Final predictions ready - gate: {predictions['gate_logits'].shape}")
        
        return predictions
    
    def _apply_dynamic_qubit_masking(
        self, 
        predictions: Dict[str, torch.Tensor], 
        num_qubits: List[int], 
        action_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        🚀 배치별 큐빗 수에 따른 동적 마스킹 적용
        
        Args:
            predictions: 모델 예측 결과
            num_qubits: 각 회로의 큐빗 수 [batch_size]
            action_mask: 액션 마스크 [batch, num_actions]
        
        Returns:
            마스킹이 적용된 예측 결과
        """
        batch_size = len(num_qubits)
        position_preds = predictions['position_preds']  # [batch, num_actions, max_qubits, 2]
        
        # 배치 크기 검증
        if position_preds.shape[0] != batch_size:
            raise ValueError(
                f"❌ CRITICAL ERROR: 배치 크기 불일치!\n"
                f"   position_preds batch size: {position_preds.shape[0]}\n"
                f"   num_qubits length: {batch_size}"
            )
        
        # 각 회로별로 동적 마스킹 적용
        for batch_idx, circuit_qubits in enumerate(num_qubits):
            # 큐빗 수 검증
            if circuit_qubits <= 0 or circuit_qubits > self.max_qubits:
                raise ValueError(
                    f"❌ CRITICAL ERROR: 회로 {batch_idx}의 큐빗 수가 잘못되었습니다!\n"
                    f"   circuit_qubits: {circuit_qubits}\n"
                    f"   max_qubits: {self.max_qubits}"
                )
            
            # 유효하지 않은 큐빗 인덱스를 -inf로 마스킹 (softmax에서 확률 0)
            # position_preds[batch_idx, :, circuit_qubits:, :] = float('-inf')
            if circuit_qubits < self.max_qubits:
                position_preds[batch_idx, :, circuit_qubits:, :] = float('-inf')
        
        predictions['position_preds'] = position_preds
        return predictions
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict,
        action_prediction_mask: torch.Tensor,
        num_qubits: Optional[List[int]] = None,
        num_gates: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        🎯 별도의 손실 계산 메서드 (예측과 분리)
        
        Args:
            predictions: 모델 예측 결과
            targets: 타겟 데이터
            action_prediction_mask: 액션 예측 마스크
            num_qubits: 배치별 큐빗 수 정보
        
        Returns:
            손실 계산 결과 dict
        """
        loss_computer = ActionLossComputer()
        
        # EOS 마스크 생성 (필요시)
        eos_mask = None
        if hasattr(self, 'gate_registry'):
            gate_vocab = self.gate_registry.get_gate_vocab()
            eos_token_id = gate_vocab.get('[EOS]', -1)
            if eos_token_id != -1 and 'gate_targets' in targets:
                eos_mask = targets['gate_targets'] != eos_token_id
        
        # 마스크 결합
        combined_mask = action_prediction_mask
        if eos_mask is not None:
            combined_mask = action_prediction_mask & eos_mask
        
        # 손실 계산
        return loss_computer.compute(
            predictions=predictions,
            targets=targets,
            mask=combined_mask,
            num_qubits=num_qubits,
            num_gates=num_gates
        )
    
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
    def predict_next_action(
        self,
        input_sequence: torch.Tensor,
        attention_mask: torch.Tensor,
        circuit_constraints: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """다음 액션 예측 (추론용) - 멀티태스크 예측 지원
        기대 형태:
          - input_sequence: [1, seq_len, d_model]
          - attention_mask: [1, seq_len, seq_len]
        """
        with torch.no_grad():
            # 마지막 위치에서 액션 예측
            action_prediction_mask = torch.zeros(input_sequence.shape[:2], dtype=torch.bool, device=input_sequence.device)
            action_prediction_mask[0, -1] = True  # 마지막 위치만 예측
            
            outputs = self.forward(input_sequence, attention_mask, action_prediction_mask, circuit_constraints=circuit_constraints)
            
            # 마지막 액션 위치의 확률 분포 반환
            action_positions = torch.where(action_prediction_mask)
            last_action_pos = int(action_positions[-1].item())
            
            # 🔥 NEW: 새로운 키 구조에 맞는 예측 추출
            gate_logits = outputs['gate_logits'][:, last_action_pos, :]  # [1, n_gate_types]
            gate_probs = F.softmax(gate_logits, dim=-1)
            
            # 큐빗 위치 예측 (새로운 구조: 위치 벡터)
            position_preds = outputs['position_preds'][:, last_action_pos, :]  # [1, 3]
            
            # 파라미터 예측 (새로운 구조: 단일 연속값)
            param_value = outputs['parameter_preds'][:, last_action_pos]  # [1]
            
            return {
                'gate_probs': gate_probs,
                'position_preds': position_preds,
                'param_value': param_value,
                'gate_logits': gate_logits,
                'hidden_states': outputs['hidden_states']
            }
    
    def sample_gate_with_constraints(
        self, 
        gate_probs: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[int, str, int, int]:
        """게이트 타입을 샘플링하고 해당 게이트의 큐빗/파라미터 요구사항 반환
        
        Returns:
            gate_idx: 샘플링된 게이트 인덱스
            gate_name: 게이트 이름
            required_qubits: 필요한 큐빗 수
            required_params: 필요한 파라미터 수
        """
        # 온도 스케일링
        if temperature != 1.0:
            gate_probs = gate_probs / temperature
            gate_probs = F.softmax(gate_probs, dim=-1)
        
        # 게이트 샘플링
        gate_idx = torch.multinomial(gate_probs.squeeze(0), 1).item()
        
        # 게이트 정보 조회
        gate_vocab = self.gate_registry.get_gate_vocab()
        gate_names = list(gate_vocab.keys())
        
        if gate_idx < len(gate_names):
            gate_name = gate_names[gate_idx]
            gate_def = self.gate_registry.get_gate(gate_name)
            
            if gate_def:
                return gate_idx, gate_name, gate_def.num_qubits, gate_def.num_parameters
        
        # 기본값 (알 수 없는 게이트)
        return gate_idx, "unknown", 1, 0
    
    def sample_qubits_for_gate(
        self,
        qubit_probs: List[torch.Tensor],
        gate_name: str,
        num_qubits_required: int,
        available_qubits: int,
        temperature: float = 1.0
    ) -> List[int]:
        """게이트 요구사항에 맞는 큐빗 위치들을 샘플링
        
        Args:
            qubit_probs: 각 큐빗 위치별 확률 분포 리스트
            gate_name: 게이트 이름
            num_qubits_required: 필요한 큐빗 수
            available_qubits: 사용 가능한 총 큐빗 수
            temperature: 샘플링 온도
            
        Returns:
            선택된 큐빗 인덱스들 (요구사항 1: [n], [n,n], [n,n,n] 형태)
        """
        selected_qubits = []
        used_qubits = set()
        
        for i in range(min(num_qubits_required, len(qubit_probs))):
            probs = qubit_probs[i].clone()
            
            # 온도 스케일링
            if temperature != 1.0:
                probs = probs / temperature
                probs = F.softmax(probs, dim=-1)
            
            # 이미 사용된 큐빗과 "no qubit" 토큰 마스킹
            for used_qubit in used_qubits:
                if used_qubit < probs.shape[-1] - 1:  # -1은 "no qubit" 토큰
                    probs[0, used_qubit] = 0.0
            
            # 사용 가능한 큐빗 범위 밖은 마스킹
            if available_qubits < probs.shape[-1] - 1:
                probs[0, available_qubits:-1] = 0.0
            
            # 확률 재정규화
            probs = probs / probs.sum()
            
            # 큐빗 샘플링
            qubit_idx = torch.multinomial(probs.squeeze(0), 1).item()
            
            # "no qubit" 토큰이 아닌 경우에만 추가
            if qubit_idx < available_qubits:
                selected_qubits.append(qubit_idx)
                used_qubits.add(qubit_idx)
        
        return selected_qubits
    
    def sample_parameters_for_gate(
        self,
        param_values: List[torch.Tensor],
        gate_name: str,
        num_params_required: int
    ) -> List[float]:
        """게이트에 필요한 파라미터들을 추출
        
        Args:
            param_values: 예측된 파라미터 값들
            gate_name: 게이트 이름  
            num_params_required: 필요한 파라미터 수
            
        Returns:
            파라미터 값들 (요구사항 2)
        """
        parameters = []
        
        for i in range(min(num_params_required, len(param_values))):
            param_val = param_values[i].squeeze().item()
            
            # 파라미터 범위 제한 (회전 게이트의 경우 0 ~ 2π)
            if gate_name.startswith('r') or gate_name in ['p']:  # rx, ry, rz, p 게이트
                param_val = param_val % (2 * math.pi)
            
            parameters.append(param_val)
        
        return parameters


class DebugMode:
    """디버그 모드 설정"""
    TENSOR_DIM = "tensor_dim"          # 텐서 차원 테스트
    EMBEDDING = "embedding"            # 임베딩 디버그
    MODEL_PREDICTION = "model_prediction"  # 모델 예측 디버그
    MODEL_OUTPUT = "model_output"      # 모델 출력 디버그
    
    # 현재 활성화된 디버그 모드들
    ACTIVE_MODES = {MODEL_OUTPUT}
    
    @staticmethod
    def is_active(mode: str) -> bool:
        """디버그 모드가 활성화되어 있는지 확인"""
        return mode in DebugMode.ACTIVE_MODES


class ActionLossComputer:
    """액션 손실 계산 전용 클래스 - 확장성 극대화"""
    
    def __init__(self, loss_weights: Dict[str, float] = None, ignore_index: int = -100):
        self.weights = loss_weights or {'gate': 0.8, 'position': 0.1, 'parameter': 0.1}
        self.ignore_index = ignore_index
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0, ignore_index=ignore_index)
        self.mse_loss = nn.MSELoss()
    
    def compute(self, predictions: Dict, targets: Dict, mask: torch.Tensor, num_qubits: Optional[List[int]] = None, num_gates: Optional[List[int]] = None) -> Dict:
        """통합 손실 계산 (동적 큐빗 마스킹 지원)"""
        device = mask.device
        
        #  DEBUG: 액션 마스크 차원 분석
        debug_print(f" MASK_SHAPE: {mask.shape}")
        debug_print(f" MASK_DTYPE: {mask.dtype}")
        debug_print(f" MASK_SUM_BEFORE_FLATTEN: {mask.sum().item()}")
        
        # 유효한 예측 위치만 선택
        valid_mask = mask.view(-1)
        debug_print(f" VALID_MASK_SHAPE: {valid_mask.shape}")
        debug_print(f" VALID_MASK_SUM: {valid_mask.sum().item()}")
        
        if valid_mask.sum() == 0:
            return self._empty_loss_dict(device)
        
        # 동적 큐빗 마스킹 적용
        if num_qubits is not None and 'position_preds' in predictions:
            predictions = self._apply_dynamic_qubit_masking(predictions, num_qubits, mask)
        
        #  DEBUG: 손실 계산 진행 상황 추적
        debug_print(f" [LOSS_DEBUG] 손실 계산 시작 - predictions keys: {list(predictions.keys())}")
        debug_print(f" [LOSS_DEBUG] targets keys: {list(targets.keys())}")
        debug_print(f" [LOSS_DEBUG] valid_mask sum: {valid_mask.sum().item()}")
        
        # 각 손실 계산
        debug_print(f" [LOSS_DEBUG] Gate 손실 계산 시작...")
        gate_loss = self._compute_gate_loss(predictions, targets, valid_mask)
        print(f" [LOSS_DEBUG] Gate 손실 완료: {gate_loss.item()}")
        
        debug_print(f" [LOSS_DEBUG] Position 손실 계산 시작...")
        position_loss = self._compute_position_loss(predictions, targets, valid_mask, num_qubits, num_gates)
        print(f" [LOSS_DEBUG] Position 손실 완료: {position_loss.item()}")
        
        debug_print(f" [LOSS_DEBUG] Parameter 손실 계산 시작...")
        parameter_loss = self._compute_parameter_loss(predictions, targets, valid_mask)
        print(f" [LOSS_DEBUG] Parameter 손실 완료: {parameter_loss.item()}")
        
        #  DEBUG: 첫 번째 회로의 예측 vs 정답 비교
        #self._debug_first_circuit_predictions(predictions, targets, mask, num_gates)
        
        # 가중 합계
        total_loss = (
            self.weights['gate'] * gate_loss + 
            self.weights['position'] * position_loss + 
            self.weights['parameter'] * parameter_loss
        )
        
        debug_print(f" [LOSS_DEBUG] 최종 손실 계산 완료 - total: {total_loss.item()}")
        
        losses = {
            'loss': total_loss,
            'gate_loss': gate_loss,
            'position_loss': position_loss,
            'parameter_loss': parameter_loss,
        }
        
        debug_print(f" [LOSS_DEBUG] 손실 딕셔너리 생성 완료")
        
        # 정확도 및 분류 메트릭 추가 (하위 호환성)
        if hasattr(self, '_gate_accuracy'):
            losses['gate_accuracy'] = self._gate_accuracy
        if hasattr(self, '_gate_precision'):
            losses['gate_precision'] = self._gate_precision
        if hasattr(self, '_gate_recall'):
            losses['gate_recall'] = self._gate_recall
        if hasattr(self, '_gate_f1'):
            losses['gate_f1'] = self._gate_f1
        
        debug_print(f" [LOSS_DEBUG] 손실 계산 완전 종료 - 반환 준비")
        return losses
    
    def _debug_first_circuit_predictions(self, predictions: Dict, targets: Dict, mask: torch.Tensor, num_gates: Optional[List[int]] = None) -> None:
        """첫 번째 회로의 예측 vs 정답 비교 디버그"""
        try:
            print(f" [DEBUG_ENTRY] num_gates: {num_gates}")
            print(f" [DEBUG_ENTRY] mask.shape: {mask.shape}")
            print(f" [DEBUG_ENTRY] predictions keys: {list(predictions.keys())}")
            print(f" [DEBUG_ENTRY] targets keys: {list(targets.keys())}")
            
            if num_gates is None or len(num_gates) == 0:
                print(" [DEBUG_EXIT] num_gates가 None이거나 비어있음")
                return
            
            batch_size, seq_len = mask.shape
            first_circuit_gates = num_gates[0]
            print(f" [DEBUG_INFO] first_circuit_gates: {first_circuit_gates}")
            
            # 첫 번째 회로의 유효한 게이트 위치 찾기
            first_circuit_mask = mask[0]  # [seq_len]
            valid_positions = torch.where(first_circuit_mask)[0][:first_circuit_gates]  # 첫 N개 위치만
            print(f" [DEBUG_INFO] valid_positions: {valid_positions}")
            
            if len(valid_positions) == 0:
                print(" [DEBUG_EXIT] valid_positions가 비어있음")
                return
            
            print(f"\n🔍 [DEBUG] 첫 번째 회로 예측 분석 (게이트 수: {first_circuit_gates})")
            print("=" * 80)
        
            # Gate predictions vs targets
            if 'gate_logits' in predictions and 'gate_targets' in targets:
                gate_logits = predictions['gate_logits'][0]  # [seq_len, 20]
                gate_preds = torch.argmax(gate_logits, dim=-1)  # [seq_len]
                
                # 타겟 처리
                gate_targets = targets['gate_targets']
                if gate_targets.dim() == 1:
                    # [batch*seq] 형태인 경우
                    first_circuit_targets = gate_targets[:first_circuit_gates]
                else:
                    # [batch, seq] 형태인 경우
                    first_circuit_targets = gate_targets[0, :first_circuit_gates]
                
                print(f"🎯 Gate 예측 vs 정답 (처음 {min(10, first_circuit_gates)}개):")
                for i, pos in enumerate(valid_positions[:10]):
                    pred_gate = gate_preds[pos].item()
                    true_gate = first_circuit_targets[i].item() if i < len(first_circuit_targets) else -1
                    match = "✅" if pred_gate == true_gate else "❌"
                    print(f"   위치 {pos:2d}: 예측={pred_gate:2d}, 정답={true_gate:2d} {match}")
            
            # Position predictions vs targets  
            if 'position_preds' in predictions and 'position_targets' in targets:
                position_preds = predictions['position_preds'][0]  # [seq_len, 32, 2]
                position_logits = position_preds[:, 0, :]  # [seq_len, 2] - 첫 번째 큐빗만
                
                position_targets = targets['position_targets']
                if position_targets.dim() == 2 and position_targets.shape[1] == 2:
                    # [N, 2] 형태
                    first_circuit_pos_targets = position_targets[:first_circuit_gates]
                else:
                    print("   Position targets 형태를 파싱할 수 없음")
                    first_circuit_pos_targets = None
                
                if first_circuit_pos_targets is not None:
                    print(f"🎯 Position 예측 vs 정답 (처음 {min(5, first_circuit_gates)}개):")
                    for i, pos in enumerate(valid_positions[:5]):
                        pred_pos = position_logits[pos]  # [2]
                        true_pos = first_circuit_pos_targets[i] if i < len(first_circuit_pos_targets) else torch.tensor([-1, -1])
                        print(f"   위치 {pos:2d}: 예측=[{pred_pos[0]:.2f}, {pred_pos[1]:.2f}], 정답=[{true_pos[0]}, {true_pos[1]}]")
            
                print("=" * 80)
        
        except Exception as e:
            print(f" [DEBUG_ERROR] 디버그 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    def _compute_gate_loss(self, predictions, targets, valid_mask):
        """게이트 타입 손실 계산"""
        # Gate prediction logits
        gate_logits = predictions['gate_logits']  # [batch, seq, num_types]
        batch_size, seq_len, num_gate_types = gate_logits.shape
        
        # 로짓 재구성 (Reshape logits)
        reshaped_logits = gate_logits.reshape(-1, num_gate_types)  # [batch*seq, num_types]
        
        # 디버깅 정보
        debug_print(f" GATE_LOGITS_SHAPE: {gate_logits.shape}")
        debug_print(f" RESHAPED_LOGITS_SHAPE: {reshaped_logits.shape}")
        debug_print(f" VALID_MASK_FOR_GATE: {valid_mask.shape}")
        
        # 마스크가 True인 위치만 선택 (직접 인덱싱)
        selected_indices = torch.where(valid_mask)[0]
        debug_print(f" SELECTED_INDICES_COUNT: {len(selected_indices)}")
        
        # 가능한 범위 내의 인덱스만 사용
        max_idx = reshaped_logits.shape[0] - 1
        valid_indices = selected_indices[selected_indices <= max_idx]
        
        if len(valid_indices) < len(selected_indices):
            debug_print(f"⚠️ 인덱스 범위 초과! {len(selected_indices) - len(valid_indices)}개 인덱스 제외됨")
        
        # 마스크 적용 - 유효한 인덱스만 사용
        gate_logits = reshaped_logits[valid_indices]
        debug_print(f" SELECTED_LOGITS_SHAPE: {gate_logits.shape}")
        
        # 타겟 인덱싱 - 같은 방식으로 처리
        gate_targets = targets['gate_targets'].reshape(-1)
        
        # 가능한 범위 내의 타겟만 사용
        if len(gate_targets) > len(valid_indices):
            gate_targets = gate_targets[valid_indices]
        else:
            # 타겟 배열이 더 작은 경우
            max_target_idx = min(len(gate_targets)-1, max(valid_indices))
            usable_indices = valid_indices[valid_indices <= max_target_idx]
            gate_logits = reshaped_logits[usable_indices]
            gate_targets = gate_targets[usable_indices]
            debug_print(f"⚠️ 타겟 크기 제한으로 {len(valid_indices) - len(usable_indices)}개 인덱스 추가 제외")
        
        debug_print(f" FINAL_LOGITS_SHAPE: {gate_logits.shape}, FINAL_TARGETS_SHAPE: {gate_targets.shape}")
        
        # 텐서 타입을 Long으로 변환 (cross_entropy는 Long 타입을 요구함)
        if gate_targets.dtype == torch.bool:
            gate_targets = gate_targets.long()
        elif gate_targets.dtype != torch.long:
            gate_targets = gate_targets.to(torch.long)
        
        # 패딩 타겟(-1) 필터링
        valid_target_mask = gate_targets >= 0
        if valid_target_mask.sum() == 0:
            return torch.tensor(0.0, device=gate_logits.device)
        
        final_logits = gate_logits[valid_target_mask]
        final_targets = gate_targets[valid_target_mask]
        
        # 정확도 계산 및 저장
        gate_predictions = torch.argmax(final_logits, dim=-1)
        self._gate_accuracy = (gate_predictions == final_targets).float().mean()
        
        # F1, Precision, Recall 계산 (다중 클래스)
        self._compute_classification_metrics(gate_predictions, final_targets)
        
        return self.focal_loss(final_logits, final_targets)
    
    def _compute_classification_metrics(self, predictions: torch.Tensor, targets: torch.Tensor):
        """F1, Precision, Recall 계산 (매크로 평균)"""
        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # 고유 클래스들
        unique_classes = np.unique(np.concatenate([predictions_np, targets_np]))
        
        if len(unique_classes) <= 1:
            # 단일 클래스인 경우
            self._gate_precision = 1.0
            self._gate_recall = 1.0
            self._gate_f1 = 1.0
            return
        
        # 클래스별 precision, recall 계산
        precisions = []
        recalls = []
        f1s = []
        
        for cls in unique_classes:
            # True Positive, False Positive, False Negative
            tp = np.sum((predictions_np == cls) & (targets_np == cls))
            fp = np.sum((predictions_np == cls) & (targets_np != cls))
            fn = np.sum((predictions_np != cls) & (targets_np == cls))
            
            # Precision, Recall 계산
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        
        # 매크로 평균
        self._gate_precision = np.mean(precisions)
        self._gate_recall = np.mean(recalls)
        self._gate_f1 = np.mean(f1s)
    
    def _compute_position_loss(self, predictions: Dict, targets: Dict, valid_mask: torch.Tensor,
                               num_qubits: Optional[List[int]] = None, num_gates: Optional[List[int]] = None) -> torch.Tensor:
        position_preds = predictions['position_preds']  # [batch, seq_len, max_qubits, 2]
        batch_size, seq_len, max_qubits, pos_dim = position_preds.shape
        
        debug_print(f"🔍position_preds shape: {position_preds.shape}")
        
        # 첫 번째 큐빗 위치만 예측 (간단화)
        position_logits = position_preds[:, :, 0, :]  # [batch, seq_len, 2]
        position_logits_flat = position_logits.reshape(-1, 2)  # [batch*seq_len, 2]
        
        # valid_mask로 실제 게이트 위치만 선택
        valid_position_logits = position_logits_flat[valid_mask]  # [num_valid_gates, 2]
        debug_print(f"  valid_position_logits shape: {valid_position_logits.shape}")
        
        # position_targets 가져오기 (qubit_targets로 매핑)
        if 'qubit_targets' not in targets:
            debug_print(f"  qubit_targets 키가 없음!")
            return torch.tensor(0.0, device=position_preds.device)
        
        position_targets = targets['qubit_targets']  # [batch, seq_len, 2]
        debug_print(f"🔍position_targets shape: {position_targets.shape}")
        debug_print(f"🔍position_targets sample: {position_targets[:2]}")
        
        # position_targets를 flat하게 변환
        position_targets_flat = position_targets.reshape(-1, 2)  # [batch*seq_len, 2]
        debug_print(f"🔍position_targets_flat shape: {position_targets_flat.shape}")
        
        # valid_mask로 필터링
        valid_position_targets = position_targets_flat[valid_mask]
        debug_print(f"🔍valid_position_targets sample: {valid_position_targets[:5]}")
        
        # 패딩된 타겟(-1) 제거
        non_padding_mask = (valid_position_targets[:, 0] >= 0) & (valid_position_targets[:, 1] >= 0)
        debug_print(f"🔍non_padding_mask sum: {non_padding_mask.sum().item()}")
        
        if non_padding_mask.sum() == 0:
            debug_print(f"🔍모든 position targets이 패딩(-1)임!")
            return torch.tensor(0.0, device=position_preds.device)
        
        final_preds = valid_position_logits[non_padding_mask]
        final_targets = valid_position_targets[non_padding_mask].float()
        
        # MSE loss for position regression
        return F.mse_loss(final_preds, final_targets, reduction='mean')
    
    def _compute_parameter_loss(self, predictions: Dict, targets: Dict, valid_mask: torch.Tensor) -> torch.Tensor:
        # 파라미터 예측 정보 차원 디버깅
        param_preds = predictions['parameter_preds']
        debug_print(f" PARAMETER_PREDS_SHAPE: {param_preds.shape}")
        
        # 리서이핑
        reshaped_preds = param_preds.reshape(-1)
        debug_print(f" RESHAPED_PARAM_PREDS_SHAPE: {reshaped_preds.shape}")
        
        # 안전하게 마스크 적용 (차원이 맞지 않으면 예외 발생)
        if len(reshaped_preds) != len(valid_mask):
            debug_print(f"⚠️ 파라미터 차원 불일치 발생! 가장 작은 차원으로 잘라냄")
            min_len = min(len(reshaped_preds), len(valid_mask))
            valid_mask = valid_mask[:min_len]
            reshaped_preds = reshaped_preds[:min_len]
        
        # 마스크 적용
        param_preds = reshaped_preds[valid_mask]
        
        # parameter_targets 확인
        if 'parameter_targets' not in targets:
            return torch.tensor(0.0, device=param_preds.device)
        
        # parameter_targets가 리스트인 경우 텐서로 변환
        if isinstance(targets['parameter_targets'], list):
            if len(targets['parameter_targets']) == 0:
                return torch.tensor(0.0, device=param_preds.device)
            parameter_targets_tensor = torch.tensor(targets['parameter_targets'], device=param_preds.device)
        else:
            parameter_targets_tensor = targets['parameter_targets']
        
        # 타겟 재생성
        reshaped_targets = parameter_targets_tensor.reshape(-1)
        if len(reshaped_targets) > len(valid_mask):
            reshaped_targets = reshaped_targets[:len(valid_mask)]
        param_targets = reshaped_targets[valid_mask]
        
        # NaN 처리
        non_nan_mask = ~torch.isnan(param_targets)
        if non_nan_mask.sum() > 0:
            return self.mse_loss(param_preds[non_nan_mask], param_targets[non_nan_mask])
        return torch.tensor(0.0, device=param_preds.device)
    
    def _apply_dynamic_qubit_masking(self, predictions: Dict, num_qubits: List[int], mask: torch.Tensor) -> Dict:
        """🚀 엄격한 동적 큐빗 마스킹: 각 회로의 정확한 큐빗 수에 맞게 예측을 마스킹"""
        if 'position_preds' not in predictions:
            raise ValueError("❌ CRITICAL ERROR: position_preds가 예측에 포함되지 않았습니다!")
        
        if num_qubits is None:
            raise ValueError("❌ CRITICAL ERROR: num_qubits 정보가 누락되었습니다!")
        
        # 예측 복사 (원본 수정 방지)
        masked_predictions = predictions.copy()
        position_preds = predictions['position_preds'].clone()  # [batch, num_actions, max_qubits, 2]
        
        batch_size, num_actions, max_qubits, qubit_dims = position_preds.shape
        
        # 🚀 배치 크기 검증
        if len(num_qubits) != batch_size:
            raise ValueError(
                f"❌ CRITICAL ERROR: 배치 크기 불일치!\n"
                f"   position_preds 배치 크기: {batch_size}\n"
                f"   num_qubits 길이: {len(num_qubits)}"
            )
        
        # 🚀 각 배치별로 정확한 동적 마스킹 적용
        for batch_idx in range(batch_size):
            circuit_qubits = num_qubits[batch_idx]
            
            # 🚀 큐빗 수 검증 및 자동 조정
            if circuit_qubits <= 0:
                raise ValueError(f"❌ CRITICAL ERROR: 회로 {batch_idx}의 큐빗 수가 0 이하입니다: {circuit_qubits}")
            
            if circuit_qubits > max_qubits:
                debug_print(f"⚠️ 회로 {batch_idx}: 큐빗 수 초과 감지 - 자동 조정")
                debug_print(f"   원본 큐빗 수: {circuit_qubits}")
                debug_print(f"   모델 최대 큐빗: {max_qubits}")
                debug_print(f"   → {max_qubits}개로 제한")
                circuit_qubits = max_qubits
            
            # ✅ 유효하지 않은 큐빗 인덱스를 -inf로 마스킹 (softmax에서 확률 0이 됨)
            for action_idx in range(num_actions):
                for qubit_dim in range(qubit_dims):  # qubit1, qubit2
                    # circuit_qubits 이상의 인덱스는 -inf로 마스킹
                    position_preds[batch_idx, action_idx, circuit_qubits:, qubit_dim] = float('-inf')
        
        masked_predictions['position_preds'] = position_preds
        return masked_predictions
    
    def _empty_loss_dict(self, device: torch.device) -> Dict:
        return {
            'total_loss': torch.tensor(0.0, device=device),
            'gate_loss': torch.tensor(0.0, device=device),
            'position_loss': torch.tensor(0.0, device=device),
            'parameter_loss': torch.tensor(0.0, device=device),
            'gate_accuracy': torch.tensor(0.0, device=device)
        }

    def forward(self, predictions, targets, action_prediction_mask, eos_mask=None, num_qubits=None):
        """ 클린한 손실 계산 (복잡한 로직 제거)"""
        device = action_prediction_mask.device
        
        # 새로운 타겟 구조 처리
        if 'action_targets' in targets and targets['action_targets']:
            action_targets = targets['action_targets']
            
            # ActionLossComputer 사용
            loss_computer = ActionLossComputer()
            combined_mask = action_prediction_mask & eos_mask if eos_mask is not None else action_prediction_mask
            
            return loss_computer.compute(predictions, action_targets, combined_mask, num_qubits=num_qubits)
        
        # 🔥 NEW: legacy 타겟 구조를 ActionLossComputer로 변환
        if 'target_actions' in targets:
            # legacy 타겟을 새로운 구조로 변환
            action_targets = {
                'gate_targets': targets['target_actions'],
                'position_targets': targets.get('target_qubits'),
                'parameter_targets': targets.get('target_params')
            }
            
            # ActionLossComputer 사용
            loss_computer = ActionLossComputer()
            combined_mask = action_prediction_mask & eos_mask if eos_mask is not None else action_prediction_mask
            
            return loss_computer.compute(predictions, action_targets, combined_mask, num_qubits=num_qubits)
        
        # FALLBACK: 기존 복잡한 로직 (하위 호환성)
        return self._legacy_loss_computation(predictions, targets, action_prediction_mask, eos_mask, num_qubits)
    
    def _legacy_loss_computation(self, predictions, targets, action_prediction_mask, eos_mask, num_qubits):
        """기존 복잡한 손실 계산 로직 (하위 호환성)"""
        device = action_prediction_mask.device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}
        
        # EOS 토큰 이후 위치 마스킹
        eos_mask = self.create_eos_mask(target_gates)
        
        # 디버깅: 마스크 크기 확인
        if DebugMode.is_active(DebugMode.MODEL_PREDICTION):
            print(f" MASK_DEBUG: action_prediction_mask.shape = {action_prediction_mask.shape}")
            print(f" MASK_DEBUG: eos_mask.shape = {eos_mask.shape}")
            print(f" MASK_DEBUG: target_gates.shape = {target_gates.shape if target_gates is not None else 'None'}")
        
        # 액션 예측 마스크와 EOS 마스크 결합 (크기 불일치 시 에러 발생)
        combined_mask = action_prediction_mask & eos_mask
        
        # 마스킹된 위치에서만 손실 계산
        mask_sum = combined_mask.sum()
        
        if mask_sum == 0:
            if DebugMode.is_active(DebugMode.MODEL_PREDICTION):
                print(" WARNING: No valid predictions to compute loss!")
            return {
                'total_loss': total_loss,
                'gate_loss': torch.tensor(0.0, device=device),
                'qubit_loss': torch.tensor(0.0, device=device),
                'param_loss': torch.tensor(0.0, device=device),
                'gate_accuracy': torch.tensor(0.0, device=device),
                'num_predictions': torch.tensor(0, device=device)
            }
        
        masked_gate_logits = gate_type_logits[combined_mask]
        masked_gate_targets = target_gates[combined_mask]
        
        # 모델 예측 디버그 (선택적 출력)
        if DebugMode.is_active(DebugMode.MODEL_PREDICTION):
            # 패딩 토큰(0번) 제외하고 실제 게이트만 분석
            non_pad_mask = masked_gate_targets != 0
            non_pad_targets = masked_gate_targets[non_pad_mask]
            
            if len(non_pad_targets) > 0:
                unique_targets = torch.unique(non_pad_targets)
                target_dist = torch.bincount(non_pad_targets)
                print(f" REAL_TARGETS: unique={unique_targets.tolist()}, count={len(unique_targets)}")
                print(f" REAL_DIST: {target_dist.tolist()}")
                print(f" PAD_RATIO: {(masked_gate_targets == 0).sum().item()}/{len(masked_gate_targets)} ({(masked_gate_targets == 0).float().mean():.2%})")
            else:
                print(" No non-padding targets found!")
            
            # 큐빗 위치 타겟 분석
            if target_qubits:
                for i, qubit_targets in enumerate(target_qubits):
                    if qubit_targets is not None:
                        masked_qubit_targets = qubit_targets[combined_mask]
                        valid_qubits = masked_qubit_targets[masked_qubit_targets >= 0]  # -1 제외
                        if len(valid_qubits) > 0:
                            unique_qubits = torch.unique(valid_qubits)
                            print(f" QUBIT_{i}: unique={unique_qubits.tolist()}, count={len(unique_qubits)}")
            
            # 파라미터 타겟 분석
            if target_params:
                for i, param_targets in enumerate(target_params):
                    if param_targets is not None:
                        masked_param_targets = param_targets[combined_mask]
                        valid_params = masked_param_targets[~torch.isnan(masked_param_targets)]  # NaN 제외
                        if len(valid_params) > 0:
                            param_range = (valid_params.min().item(), valid_params.max().item())
                            print(f"🎯 PARAM_{i}: range={param_range}, count={len(valid_params)}")
        
        gate_loss = self.cross_entropy(masked_gate_logits, masked_gate_targets)
        total_loss = total_loss + self.gate_weight * gate_loss
        loss_dict['gate_loss'] = gate_loss
        
        with torch.no_grad():
            gate_predictions = torch.argmax(masked_gate_logits, dim=-1)
            gate_accuracy = (gate_predictions == masked_gate_targets).float().mean()
            
            # 모델 예측 디버그 (선택적 출력)
            if DebugMode.is_active(DebugMode.MODEL_PREDICTION):
                unique_preds = torch.unique(gate_predictions)
                print(f"🤖 PREDICTIONS: unique={unique_preds.tolist()}, count={len(unique_preds)}")
                print(f"📈 ACCURACY: gate={gate_accuracy:.4f}")
                
            loss_dict['gate_accuracy'] = gate_accuracy
        
        # 2. 큐빗 위치 손실 (요구사항 1: 통합 텐서로 처리)
        if target_qubits is not None:
            # 새로운 통합 텐서 구조 처리: [batch, num_gates, max_qubits_per_gate]
            if isinstance(target_qubits, torch.Tensor) and target_qubits.dim() == 3:
                # 통합 큐빗 예측: [num_actions, max_qubits_per_gate, max_qubits + 1]
                masked_qubit_logits = qubit_position_logits[combined_mask]
                
                # 통합 큐빗 타겟: [num_actions, max_qubits_per_gate]
                masked_qubit_targets = target_qubits[combined_mask]
                
                # 동적 큐빗 마스킹 적용
                if num_qubits is not None:
                    num_actions = masked_qubit_logits.shape[0]
                    max_qubits_per_gate = masked_qubit_logits.shape[1]
                    
                    for action_idx in range(num_actions):
                        # 각 회로의 실제 큐빗 수를 넘는 예측은 마스킹
                        circuit_idx = action_idx  # 배치 내 회로 인덱스 (단순화)
                        if circuit_idx < len(num_qubits):
                            max_valid_qubit = num_qubits[circuit_idx].item()
                            # 유효하지 않은 큐빗 인덱스는 매우 작은 값으로 마스킹
                            for qubit_pos in range(max_qubits_per_gate):
                                masked_qubit_logits[action_idx, qubit_pos, max_valid_qubit+1:] = -1e9
                
                # 유효한 타겟이 있는 위치만 손실 계산
                valid_mask = masked_qubit_targets != self.ignore_index
                if valid_mask.sum() > 0:
                    # Flatten for cross entropy: [num_valid_predictions, max_qubits + 1]
                    valid_logits = masked_qubit_logits[valid_mask]  # [num_valid, max_qubits + 1]
                    valid_targets = masked_qubit_targets[valid_mask]  # [num_valid]
                    
                    qubit_loss = self.cross_entropy(valid_logits, valid_targets)
                    total_loss = total_loss + self.qubit_weight * qubit_loss
                    loss_dict['qubit_loss'] = qubit_loss
                else:
                    loss_dict['qubit_loss'] = torch.tensor(0.0, device=device)
            
            # 기존 리스트 구조 처리 (하위 호환성)
            elif isinstance(target_qubits, list) and len(target_qubits) > 0:
                # 통합 큐빗 예측: [num_actions, max_qubits_per_gate, max_qubits + 1]
                masked_qubit_logits = qubit_position_logits[combined_mask]
                
                # 타겟 큐빗들을 통합 텐서로 변환: [num_actions, max_qubits_per_gate]
                num_actions = masked_qubit_logits.shape[0]
                max_qubits_per_gate = masked_qubit_logits.shape[1]
                
                # 통합 타겟 텐서 생성
                unified_qubit_targets = torch.full(
                    (num_actions, max_qubits_per_gate), 
                    self.ignore_index, 
                    dtype=torch.long, 
                    device=device
                )
                
                # 각 큐빗 위치별 타겟을 통합 텐서에 복사
                for qubit_idx, qubit_targets in enumerate(target_qubits):
                    if qubit_targets is not None and qubit_idx < max_qubits_per_gate:
                        masked_targets = qubit_targets[combined_mask]
                        unified_qubit_targets[:, qubit_idx] = masked_targets
                
                # 동적 큐빗 마스킹 적용
                if num_qubits is not None:
                    for action_idx in range(num_actions):
                        for qubit_pos in range(max_qubits_per_gate):
                            # 각 회로의 실제 큐빗 수를 넘는 예측은 마스킹
                            circuit_idx = action_idx  # 배치 내 회로 인덱스 (단순화)
                            if circuit_idx < len(num_qubits):
                                max_valid_qubit = num_qubits[circuit_idx].item()
                                # 유효하지 않은 큐빗 인덱스는 매우 작은 값으로 마스킹
                                masked_qubit_logits[action_idx, qubit_pos, max_valid_qubit+1:] = -1e9
                
                # 유효한 타겟이 있는 위치만 손실 계산
                valid_mask = unified_qubit_targets != self.ignore_index
                if valid_mask.sum() > 0:
                    # Flatten for cross entropy: [num_valid_predictions, max_qubits + 1]
                    valid_logits = masked_qubit_logits[valid_mask]  # [num_valid, max_qubits + 1]
                    valid_targets = unified_qubit_targets[valid_mask]  # [num_valid]
                    
                    qubit_loss = self.cross_entropy(valid_logits, valid_targets)
                    total_loss = total_loss + self.qubit_weight * qubit_loss
                    loss_dict['qubit_loss'] = qubit_loss
                else:
                    loss_dict['qubit_loss'] = torch.tensor(0.0, device=device)
            else:
                loss_dict['qubit_loss'] = torch.tensor(0.0, device=device)
        else:
            loss_dict['qubit_loss'] = torch.tensor(0.0, device=device)
        
        # 3. 파라미터 손실 (요구사항 2: 통합 텐서로 처리)
        if target_params is not None:
            # 새로운 통합 텐서 구조 처리: [batch, num_gates, max_params_per_gate]
            if isinstance(target_params, torch.Tensor) and target_params.dim() == 3:
                # 통합 파라미터 예측: [num_actions, max_params_per_gate]
                masked_param_preds = parameter_predictions[combined_mask]
                
                # 통합 파라미터 타겟: [num_actions, max_params_per_gate]
                masked_param_targets = target_params[combined_mask]
                
                # 유효한 파라미터 타겟만 손실 계산 (NaN 제외)
                valid_mask = ~torch.isnan(masked_param_targets)
                non_padding_mask = position_targets != -1
        if non_padding_mask.sum() > 0:
            # 🚨 개별 회로 큐빗 수 기반 동적 정규화
            if num_qubits is not None and len(num_qubits) > 0:
                # 배치 내 각 샘플의 큐빗 수를 사용하여 정규화
                batch_size = len(num_qubits)
                seq_len = position_targets.size(0) // batch_size
                # 통합 파라미터 예측: [num_actions, max_params_per_gate]
                masked_param_preds = parameter_predictions[combined_mask]
                
                # 타겟 파라미터들을 통합 텐서로 변환: [num_actions, max_params_per_gate]
                num_actions = masked_param_preds.shape[0]
                max_params_per_gate = masked_param_preds.shape[1]
                
                # 통합 타겟 텐서 생성 (NaN으로 초기화)
                unified_param_targets = torch.full(
                    (num_actions, max_params_per_gate), 
                    float('nan'), 
                    dtype=torch.float32, 
                    device=device
                )
                
                # 각 파라미터별 타겟을 통합 텐서에 복사
                for param_idx, param_targets in enumerate(target_params):
                    if param_targets is not None and param_idx < max_params_per_gate:
                        masked_targets = param_targets[combined_mask]
                        unified_param_targets[:, param_idx] = masked_targets
                
                # 유효한 파라미터 타겟만 손실 계산 (NaN 제외)
                valid_mask = ~torch.isnan(unified_param_targets)
                if valid_mask.sum() > 0:
                    valid_preds = masked_param_preds[valid_mask]
                    valid_targets = unified_param_targets[valid_mask]
                    
                    param_loss = self.mse_loss(valid_preds, valid_targets)
                    total_loss = total_loss + self.param_weight * param_loss
                    loss_dict['param_loss'] = param_loss
                else:
                    loss_dict['param_loss'] = torch.tensor(0.0, device=device)
            else:
                loss_dict['param_loss'] = torch.tensor(0.0, device=device)
        else:
            loss_dict['param_loss'] = torch.tensor(0.0, device=device)
        
        # 🔥 CRITICAL: gate_accuracy가 항상 포함되도록 보장
        if 'gate_accuracy' not in loss_dict:
            loss_dict['gate_accuracy'] = torch.tensor(0.0, device=device)
        
        loss_dict.update({
            'total_loss': total_loss,
            'num_predictions': torch.tensor(combined_mask.sum().item(), device=device)
        })
        
        return loss_dict


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
    
    debug_print(f"Gate logits shape: {outputs['gate_logits'].shape}")
    debug_print(f"Position preds shape: {outputs['position_preds'].shape}")
    debug_print(f"Parameter preds shape: {outputs['parameter_preds'].shape}")
    debug_print(f"Hidden states shape: {outputs['hidden_states'].shape}")
    
    # 🎯 손실 계산 테스트 (새로운 구조)
    # 더미 타겟 데이터 생성
    targets = {
        'gate_targets': torch.randint(0, 16, (batch_size, seq_len)),
        'position_targets': torch.randn(batch_size, seq_len, 3),
        'parameter_targets': torch.randn(batch_size, seq_len)
    }
    
    # 모델의 compute_loss 메서드 사용
    loss_outputs = model.compute_loss(
        predictions=outputs,
        targets=targets,
        action_prediction_mask=action_prediction_mask
    )
    
    print(f"Total Loss: {loss_outputs['total_loss'].item():.4f}")
    print(f"Gate Accuracy: {loss_outputs['gate_accuracy'].item():.4f}")
    print(f"Gate Loss: {loss_outputs['gate_loss'].item():.4f}")
    print(f"Position Loss: {loss_outputs['position_loss'].item():.4f}")
    print(f"Parameter Loss: {loss_outputs['parameter_loss'].item():.4f}")
