"""
Quantum Circuit Attention Mechanisms

This module contains all attention-related components for quantum circuit processing:
- Multi-Head Attention with RoPE support
- Grid Positional Attention
- Register Flow Attention
- Entanglement Attention
- Semantic Attention
- Attention Fusion Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional


class RotaryPositionalEmbedding(nn.Module):
    """RoPE (Rotary Positional Embedding) - 최신 positional encoding"""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency matrix
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for efficiency
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
    
    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cos/sin cache if sequence length changed"""
        if seq_len != self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
    
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        """Apply rotary positional embedding to queries and keys"""
        seq_len = q.shape[-2]
        self._update_cos_sin_cache(seq_len, q.device, q.dtype)
        
        cos = self._cos_cached[:seq_len, :]
        sin = self._sin_cached[:seq_len, :]
        
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)
        
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        
        return q_embed, k_embed

class GridPositionalAttention(nn.Module):
    """그리드 위치 기반 어텐션 (거리 기반 bias)"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Distance bias (learnable parameter)
        self.distance_bias = nn.Parameter(torch.ones(n_heads))
        
        # Layer normalization and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
        # Improved initialization
        self._init_weights()
    
    def _init_weights(self):
        """Xavier/Glorot initialization for better gradient flow"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
        
        # Distance bias initialization with smaller variance
        nn.init.normal_(self.distance_bias, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, distance_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 입력 임베딩 텐서
            distance_matrix: 노드 간 거리 행렬
        """
        if x.size(0) == 0:
            return {
                'output': torch.zeros(0, self.d_model),
                'attention_weights': torch.zeros(0, 0)
            }
            
        seq_len = x.size(0)
        
        # Pre-Layer Normalization for better training stability
        x_norm = self.norm(x)
        
        # Q, K, V 계산 (normalized input 사용)
        Q = self.q_proj(x_norm).view(seq_len, self.n_heads, self.head_dim).transpose(0, 1)
        K = self.k_proj(x_norm).view(seq_len, self.n_heads, self.head_dim).transpose(0, 1)
        V = self.v_proj(x_norm).view(seq_len, self.n_heads, self.head_dim).transpose(0, 1)
        
        # 어텐션 스코어
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 거리 기반 bias 추가 (가까운 게이트일수록 높은 어텐션)
        # FP16 호환 마스크 값 사용
        mask_value = -65504.0 if scores.dtype == torch.float16 else -1e9
        mask = (distance_matrix == -1).float() * mask_value
        distance_bias = -torch.clamp(distance_matrix, min=0).unsqueeze(0) * self.distance_bias
        scores = scores + distance_bias + mask
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)  # Dropout for regularization
        attended = torch.matmul(attention_weights, V)
        
        # 헤드 결합
        attended = attended.transpose(0, 1).contiguous().view(seq_len, self.d_model)
        output = self.out_proj(attended)
        
        # Residual connection for better gradient flow
        output = output + x  # Skip connection
        
        return {
            'output': output,
            'attention_weights': attention_weights.mean(0)  # 헤드 평균
        }


class RegisterFlowAttention(nn.Module):
    """레지스터 연결 기반 어텐션 (같은 큐빗 시간 흐름)"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=False)
        
    def forward(self, x: torch.Tensor, grid_structure: Dict, edges: List[Dict]) -> Dict[str, torch.Tensor]:
        if x.size(0) == 0:
            return {
                'output': torch.zeros(0, x.size(-1) if x.dim() > 1 else 1),
                'attention_weights': torch.zeros(0, 0)
            }
            
        # 레지스터 연결 마스크 생성
        seq_len = x.size(0)
        register_mask = torch.zeros(seq_len, seq_len)
        
        for edge in edges:
            if edge['type'] == 'REGISTER_CONNECTION':
                src_pos = edge['source']
                tgt_pos = edge['target']
                
                # 포지션에서 노드 인덱스 찾기
                src_idx = self._find_node_by_position(src_pos, grid_structure)
                tgt_idx = self._find_node_by_position(tgt_pos, grid_structure)
                
                if src_idx is not None and tgt_idx is not None:
                    register_mask[src_idx, tgt_idx] = 1
                    register_mask[tgt_idx, src_idx] = 1  # 양방향
        
        # 마스크를 어텐션에 적용
        attn_mask = register_mask.bool()
        output, attention_weights = self.multihead_attn(
            x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1), 
            attn_mask=~attn_mask if attn_mask.any() else None
        )
        
        return {
            'output': output.squeeze(1),
            'attention_weights': attention_weights.squeeze(0)
        }
    
    def _find_node_by_position(self, position: List[int], grid_structure: Dict) -> Optional[int]:
        """그리드 포지션으로 노드 인덱스 찾기"""
        positions = grid_structure['positions']
        if positions.size(0) == 0:
            return None
            
        target = torch.tensor(position)
        
        for i, pos in enumerate(positions):
            if torch.equal(pos, target):
                return i
        return None


class EntanglementAttention(nn.Module):
    """엔탱글먼트 연결 기반 어텐션"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=False)
        
    def forward(self, x: torch.Tensor, grid_structure: Dict, edges: List[Dict]) -> Dict[str, torch.Tensor]:
        if x.size(0) == 0:
            return {
                'output': torch.zeros(0, x.size(-1) if x.dim() > 1 else 1),
                'attention_weights': torch.zeros(0, 0)
            }
            
        # 엔탱글먼트 연결 마스크 생성
        seq_len = x.size(0)
        entangle_mask = torch.zeros(seq_len, seq_len)
        
        for edge in edges:
            if edge['type'] == 'ENTANGLE_CONNECTION':
                src_pos = edge['source']
                tgt_pos = edge['target']
                
                src_idx = self._find_node_by_position(src_pos, grid_structure)
                tgt_idx = self._find_node_by_position(tgt_pos, grid_structure)
                
                if src_idx is not None and tgt_idx is not None:
                    entangle_mask[src_idx, tgt_idx] = 1
                    entangle_mask[tgt_idx, src_idx] = 1
        
        # 강한 어텐션 가중치 적용
        attn_mask = entangle_mask.bool()
        output, attention_weights = self.multihead_attn(
            x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1),
            attn_mask=~attn_mask if attn_mask.any() else None
        )
        
        return {
            'output': output.squeeze(1),
            'attention_weights': attention_weights.squeeze(0)
        }
    
    def _find_node_by_position(self, position: List[int], grid_structure: Dict) -> Optional[int]:
        positions = grid_structure['positions']
        if positions.size(0) == 0:
            return None
            
        target = torch.tensor(position)
        
        for i, pos in enumerate(positions):
            if torch.equal(pos, target):
                return i
        return None


class SemanticAttention(nn.Module):
    """게이트 타입 의미적 유사성 어텐션"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=False)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.size(0) == 0:
            return {
                'output': torch.zeros(0, x.size(-1) if x.dim() > 1 else 1),
                'attention_weights': torch.zeros(0, 0)
            }
            
        # 표준 셀프 어텐션 (의미적 유사성에 의존)
        output, attention_weights = self.multihead_attn(
            x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1)
        )
        
        return {
            'output': output.squeeze(1),
            'attention_weights': attention_weights.squeeze(0)
        }


class AttentionFusionNetwork(nn.Module):
    """다층 어텐션 결과 융합 네트워크"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        
        # 각 어텐션 타입별 가중치 학습
        self.attention_weights = nn.Parameter(torch.ones(4) / 4)  # 4개 어텐션 타입
        
        # GLU (Gated Linear Unit) for better gating mechanism
        self.fusion_proj = nn.Linear(d_model * 4, d_model * 2)  # 2x for gate and value
        
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, attention_outputs: Dict[str, Dict]) -> torch.Tensor:
        # 모든 어텐션 출력 결합
        outputs = [
            attention_outputs['grid_attention']['output'],
            attention_outputs['register_attention']['output'],
            attention_outputs['entangle_attention']['output'],
            attention_outputs['semantic_attention']['output']
        ]
        
        # 빈 출력 처리
        if outputs[0].size(0) == 0:
            return torch.zeros(0, self.d_model)
        
        # 가중합 및 융합
        weighted_outputs = []
        weights = F.softmax(self.attention_weights, dim=0)
        
        for i, output in enumerate(outputs):
            weighted_outputs.append(weights[i] * output)
        
        # 연결 후 GLU (Gated Linear Unit) 적용
        concatenated = torch.cat(outputs, dim=-1)
        
        # GLU: 더 효과적인 gating mechanism
        fusion_output = self.fusion_proj(concatenated)  # [batch, d_model * 2]
        gate, value = fusion_output.chunk(2, dim=-1)    # Split into gate and value
        
        # GLU gating: sigmoid(gate) * value
        gated_output = torch.sigmoid(gate) * value
        
        # 가중합과 게이트 출력 결합
        weighted_sum = sum(weighted_outputs)
        final_output = self.layer_norm(gated_output + weighted_sum)
        
        return final_output
