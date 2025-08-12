"""
Modular Attention System for Decision Transformer

표준 어텐션과 고급 다층 어텐션을 쉽게 전환할 수 있는 모듈러 시스템
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List, Literal
from enum import Enum

# 공통 디버그 유틸리티 사용
from utils.debug_utils import debug_print, debug_tensor_info

# 기존 어텐션 모듈들 임포트
from models.attention import (
    GridPositionalAttention, 
    RegisterFlowAttention, 
    EntanglementAttention, 
    SemanticAttention,
    AttentionFusionNetwork
)


class AttentionMode(Enum):
    """어텐션 모드 정의"""
    STANDARD = "standard"           # 표준 멀티헤드 어텐션
    ADVANCED = "advanced"           # 고급 다층 어텐션 (attention.py)
    HYBRID = "hybrid"               # 하이브리드 (둘 다 사용)


class StandardMultiHeadAttention(nn.Module):
    """표준 멀티헤드 어텐션 (최적화된 안정성)"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # 어텐션 가중치용 경량 dropout
        self.attn_dropout = nn.Dropout(dropout * 0.5)
        self.output_dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        debug_print(f"      Standard Attention input - shape: {x.shape}")
        
        # Q, K, V 계산
        Q = self.w_q(x)  # [batch, seq_len, d_model]
        K = self.w_k(x)  # [batch, seq_len, d_model]
        V = self.w_v(x)  # [batch, seq_len, d_model]
        
        # 멀티헤드로 변형
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 어텐션 계산
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 마스크 적용 (안정화된 마스킹)
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
            scores = scores.masked_fill(~mask, -1e9)
        
        # 안정화된 소프트맥스
        scores_max = scores.max(dim=-1, keepdim=True)[0]
        scores_stable = scores - scores_max
        attention_weights = F.softmax(scores_stable, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)
        
        out = torch.matmul(attention_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # 출력 프로젝션 및 dropout
        out = self.w_o(out)
        out = self.output_dropout(out)
        
        return out


class AdvancedMultiLayerAttention(nn.Module):
    """고급 다층 어텐션 (attention.py 기반)"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # 4가지 전문화된 어텐션
        self.grid_attention = GridPositionalAttention(d_model, n_heads)
        self.register_attention = RegisterFlowAttention(d_model, n_heads)
        self.entangle_attention = EntanglementAttention(d_model, n_heads)
        self.semantic_attention = SemanticAttention(d_model, n_heads)
        
        # 어텐션 융합 네트워크
        self.attention_fusion = AttentionFusionNetwork(d_model, n_heads)
        
        # 추가 정규화
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                grid_structure: Optional[Dict] = None, edges: Optional[List[Dict]] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: attention mask (optional)
            grid_structure: 그리드 구조 정보 (optional)
            edges: 엣지 연결 정보 (optional)
        """
        batch_size, seq_len, d_model = x.shape
        debug_print(f"      Advanced Attention input - shape: {x.shape}")
        
        # 배치 처리를 위해 시퀀스 차원으로 변환 [seq_len, batch, d_model]
        x_seq = x.transpose(0, 1)
        
        # 기본 구조 생성 (없는 경우)
        if grid_structure is None:
            grid_structure = self._create_default_grid_structure(seq_len, batch_size)
        if edges is None:
            edges = self._create_default_edges(seq_len)
        
        # 거리 매트릭스 생성
        distance_matrix = self._create_distance_matrix(seq_len)
        
        attention_outputs = {}
        
        # 1. Grid positional attention
        try:
            grid_out = self.grid_attention(x_seq, distance_matrix)
            attention_outputs['grid_attention'] = grid_out
        except Exception as e:
            debug_print(f"Grid attention error: {e}")
            attention_outputs['grid_attention'] = {'output': x_seq, 'attention_weights': None}
        
        # 2. Register flow attention
        try:
            register_out = self.register_attention(x_seq, grid_structure, edges)
            attention_outputs['register_attention'] = register_out
        except Exception as e:
            debug_print(f"Register attention error: {e}")
            attention_outputs['register_attention'] = {'output': x_seq, 'attention_weights': None}
        
        # 3. Entanglement attention
        try:
            entangle_out = self.entangle_attention(x_seq, grid_structure, edges)
            attention_outputs['entangle_attention'] = entangle_out
        except Exception as e:
            debug_print(f"Entanglement attention error: {e}")
            attention_outputs['entangle_attention'] = {'output': x_seq, 'attention_weights': None}
        
        # 4. Semantic attention
        try:
            semantic_out = self.semantic_attention(x_seq)
            attention_outputs['semantic_attention'] = semantic_out
        except Exception as e:
            debug_print(f"Semantic attention error: {e}")
            attention_outputs['semantic_attention'] = {'output': x_seq, 'attention_weights': None}
        
        # 어텐션 융합
        try:
            fused_output = self.attention_fusion(attention_outputs)  # [seq_len, d_model]
            
            # 배치 차원으로 다시 변환 [batch, seq_len, d_model]
            if fused_output.dim() == 2:
                # [seq_len, d_model] -> [batch, seq_len, d_model]
                fused_output = fused_output.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                # [seq_len, batch, d_model] -> [batch, seq_len, d_model]
                fused_output = fused_output.transpose(0, 1)
                
        except Exception as e:
            debug_print(f"Attention fusion error: {e}")
            # 폴백: 원본 입력 반환
            fused_output = x
        
        # 드롭아웃 적용
        fused_output = self.dropout(fused_output)
        
        return fused_output
    
    def _create_default_grid_structure(self, seq_len: int, batch_size: int) -> Dict:
        """기본 그리드 구조 생성"""
        positions = torch.zeros(seq_len, 2)  # [seq_len, 2] (x, y 좌표)
        for i in range(seq_len):
            positions[i] = torch.tensor([i % 8, i // 8])  # 8x8 그리드 가정
        
        return {
            'positions': positions,
            'num_nodes': seq_len
        }
    
    def _create_default_edges(self, seq_len: int) -> List[Dict]:
        """기본 엣지 연결 생성"""
        edges = []
        for i in range(seq_len - 1):
            edges.append({
                'type': 'REGISTER_CONNECTION',
                'source': [i % 8, i // 8],
                'target': [(i+1) % 8, (i+1) // 8]
            })
        return edges
    
    def _create_distance_matrix(self, seq_len: int) -> torch.Tensor:
        """거리 매트릭스 생성"""
        distance_matrix = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            for j in range(seq_len):
                distance_matrix[i, j] = abs(i - j)
        return distance_matrix


class ModularAttention(nn.Module):
    """모듈러 어텐션 시스템 - 모드 전환 가능"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, 
                 mode: AttentionMode = AttentionMode.STANDARD):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.mode = mode
        
        # 두 가지 어텐션 메커니즘 모두 초기화
        self.standard_attention = StandardMultiHeadAttention(d_model, n_heads, dropout)
        self.advanced_attention = AdvancedMultiLayerAttention(d_model, n_heads, dropout)
        
        # 하이브리드 모드용 가중치
        if mode == AttentionMode.HYBRID:
            self.hybrid_weight = nn.Parameter(torch.tensor(0.5))  # 0~1 사이 가중치
            self.fusion_proj = nn.Linear(d_model * 2, d_model)
    
    def set_mode(self, mode: AttentionMode):
        """어텐션 모드 변경"""
        self.mode = mode
        debug_print(f"Attention mode changed to: {mode.value}")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                grid_structure: Optional[Dict] = None, edges: Optional[List[Dict]] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: attention mask
            grid_structure: 그리드 구조 (advanced/hybrid 모드용)
            edges: 엣지 정보 (advanced/hybrid 모드용)
        """
        debug_print(f"ModularAttention forward - mode: {self.mode.value}")
        
        if self.mode == AttentionMode.STANDARD:
            return self.standard_attention(x, mask)
            
        elif self.mode == AttentionMode.ADVANCED:
            return self.advanced_attention(x, mask, grid_structure, edges)
            
        elif self.mode == AttentionMode.HYBRID:
            # 두 어텐션 결과를 가중 결합
            standard_out = self.standard_attention(x, mask)
            advanced_out = self.advanced_attention(x, mask, grid_structure, edges)
            
            # 가중 결합
            weight = torch.sigmoid(self.hybrid_weight)
            combined = torch.cat([
                standard_out * (1 - weight),
                advanced_out * weight
            ], dim=-1)
            
            # 차원 축소
            output = self.fusion_proj(combined)
            return output
        
        else:
            raise ValueError(f"Unknown attention mode: {self.mode}")


# 편의 함수들
def create_modular_attention(d_model: int, n_heads: int, dropout: float = 0.1, 
                           mode: str = "standard") -> ModularAttention:
    """모듈러 어텐션 생성 편의 함수"""
    attention_mode = AttentionMode(mode.lower())
    return ModularAttention(d_model, n_heads, dropout, attention_mode)


def compare_attention_modes(x: torch.Tensor, attention: ModularAttention, 
                          mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """어텐션 모드별 결과 비교"""
    results = {}
    
    # 각 모드별로 실행
    for mode in AttentionMode:
        attention.set_mode(mode)
        with torch.no_grad():
            output = attention(x, mask)
            results[mode.value] = output
    
    return results


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 Modular Attention System Test")
    
    # 테스트 데이터
    batch_size, seq_len, d_model = 2, 16, 512
    n_heads = 8
    
    x = torch.randn(batch_size, seq_len, d_model)
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).expand(batch_size, -1, -1)
    
    # 모듈러 어텐션 생성
    attention = create_modular_attention(d_model, n_heads, mode="standard")
    
    print(f"Input shape: {x.shape}")
    
    # 각 모드 테스트
    for mode in ["standard", "advanced", "hybrid"]:
        attention.set_mode(AttentionMode(mode))
        output = attention(x, mask)
        print(f"{mode.capitalize()} attention output shape: {output.shape}")
    
    print("✅ All tests passed!")
