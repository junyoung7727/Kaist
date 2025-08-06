"""
Property Prediction Transformer

CircuitSpec으로부터 양자 회로의 물리적 특성을 예측하는 트랜스포머 모델:
- Entanglement (얽힘도)
- Fidelity (충실도)
- Robust Fidelity (견고한 충실도)

Predictor_Embed + attention.py 모듈들을 활용한 완전한 예측 시스템
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import math

# Import embedding and attention modules
from ..encoding.Predictor_Embed import QuantumCircuitEmbedding
from .attention import (
    GridPositionalAttention,
    RegisterFlowAttention, 
    EntanglementAttention,
    SemanticAttention,
    AttentionFusionNetwork
)

# Import grid encoder
from ..encoding.grid_graph_encoder import GridGraphEncoder

# Add quantumcommon to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))


@dataclass
class PropertyPredictionConfig:
    """Property Prediction Transformer 설정"""
    # Model architecture
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    
    # Embedding settings
    max_grid_size: int = 64
    max_qubits: int = 32
    use_rotary_pe: bool = True
    
    # Output settings
    property_dim: int = 3  # entanglement, fidelity, robust_fidelity
    
    # Training settings
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000


class PropertyPredictionHead(nn.Module):
    """물리적 특성 예측을 위한 출력 헤드"""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        # 각 특성별 독립적인 예측 헤드
        self.entanglement_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()  # 0-1 범위로 정규화
        )
        
        self.fidelity_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()  # 0-1 범위로 정규화
        )
        
        self.robust_fidelity_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()  # 0-1 범위로 정규화
        )
        
        # 통합 특성 벡터 (선택적)
        self.combined_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),  # [entanglement, fidelity, robust_fidelity]
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch_size, d_model] - 회로 표현 벡터
        
        Returns:
            각 특성별 예측값
        """
        return {
            'entanglement': self.entanglement_head(x).squeeze(-1),  # [batch_size]
            'fidelity': self.fidelity_head(x).squeeze(-1),         # [batch_size]
            'robust_fidelity': self.robust_fidelity_head(x).squeeze(-1),  # [batch_size]
            'combined': self.combined_head(x)  # [batch_size, 3]
        }


class TransformerBlock(nn.Module):
    """트랜스포머 블록 (어텐션 + FFN)"""
    
    def __init__(self, config: PropertyPredictionConfig):
        super().__init__()
        self.config = config
        
        # Multi-head attention components
        self.grid_attention = GridPositionalAttention(config.d_model, config.n_heads)
        self.register_attention = RegisterFlowAttention(config.d_model, config.n_heads)
        self.entanglement_attention = EntanglementAttention(config.d_model, config.n_heads)
        self.semantic_attention = SemanticAttention(config.d_model, config.n_heads)
        
        # Attention fusion
        self.attention_fusion = AttentionFusionNetwork(config.d_model, config.n_heads)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 4, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
        # Residual dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        distance_matrix: torch.Tensor,
        grid_structure: Dict,
        edges: List[Dict]
    ) -> torch.Tensor:
        """
        Args:
            x: [seq_len, batch_size, d_model]
            distance_matrix: [seq_len, seq_len]
            grid_structure: 그리드 구조 정보
            edges: 엣지 연결 정보
        """
        # Multi-head attention with different mechanisms
        attention_outputs = {}
        
        # 1. Grid positional attention
        attention_outputs['grid'] = {
            'output': self.grid_attention(x, distance_matrix),
            'weights': None  # 필요시 attention weights 저장
        }
        
        # 2. Register flow attention
        attention_outputs['register'] = {
            'output': self.register_attention(x, grid_structure, edges),
            'weights': None
        }
        
        # 3. Entanglement attention
        attention_outputs['entanglement'] = {
            'output': self.entanglement_attention(x, grid_structure, edges),
            'weights': None
        }
        
        # 4. Semantic attention
        attention_outputs['semantic'] = {
            'output': self.semantic_attention(x),
            'weights': None
        }
        
        # Attention fusion
        fused_attention = self.attention_fusion(attention_outputs)
        
        # Residual connection + layer norm
        x = self.norm1(x + self.dropout(fused_attention))
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x


class PropertyPredictionTransformer(nn.Module):
    """양자 회로 특성 예측 트랜스포머"""
    
    def __init__(self, config: PropertyPredictionConfig):
        super().__init__()
        self.config = config
        
        # Grid encoder for CircuitSpec
        self.grid_encoder = GridGraphEncoder()
        
        # Circuit embedding
        self.circuit_embedding = QuantumCircuitEmbedding(
            d_model=config.d_model,
            n_heads=config.n_heads,
            max_grid_size=config.max_grid_size,
            max_qubits=config.max_qubits,
            dropout=config.dropout,
            use_rotary_pe=config.use_rotary_pe
        )
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Global pooling for circuit-level representation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Property prediction heads
        self.property_head = PropertyPredictionHead(config.d_model, config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """가중치 초기화"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, circuit_spec) -> Dict[str, torch.Tensor]:
        """
        Args:
            circuit_spec: CircuitSpec 객체 또는 배치
        
        Returns:
            각 특성별 예측값
        """
        # Handle batch input
        if isinstance(circuit_spec, list):
            # Batch processing
            batch_predictions = []
            for spec in circuit_spec:
                pred = self._forward_single(spec)
                batch_predictions.append(pred)
            
            # Stack batch results
            batch_result = {}
            for key in batch_predictions[0].keys():
                batch_result[key] = torch.stack([pred[key] for pred in batch_predictions])
            
            return batch_result
        else:
            # Single circuit
            return self._forward_single(circuit_spec)
    
    def _forward_single(self, circuit_spec) -> Dict[str, torch.Tensor]:
        """단일 회로 처리"""
        # 1. Grid encoding
        encoded_circuit = self.grid_encoder.encode(circuit_spec)
        
        # 2. Circuit embedding
        embedding_result = self.circuit_embedding(encoded_circuit)
        
        # Extract components
        node_embeddings = embedding_result['node_embeddings']  # [seq_len, d_model]
        distance_matrix = embedding_result['distance_matrix']  # [seq_len, seq_len]
        grid_structure = embedding_result['grid_structure']
        edges = encoded_circuit['edges']
        
        # 3. Add batch dimension and transpose for transformer
        x = node_embeddings.unsqueeze(1)  # [seq_len, 1, d_model]
        
        # 4. Transformer layers
        for layer in self.transformer_layers:
            x = layer(x, distance_matrix, grid_structure, edges)
        
        # 5. Global pooling for circuit representation
        # x: [seq_len, 1, d_model] -> [1, d_model, seq_len] -> [1, d_model, 1] -> [1, d_model]
        x = x.transpose(0, 2)  # [1, d_model, seq_len]
        circuit_repr = self.global_pool(x).squeeze(-1)  # [1, d_model]
        
        # 6. Property prediction
        predictions = self.property_head(circuit_repr)
        
        # Remove batch dimension for single circuit
        for key, value in predictions.items():
            predictions[key] = value.squeeze(0)
        
        return predictions


class PropertyPredictionLoss(nn.Module):
    """특성 예측을 위한 손실 함수"""
    
    def __init__(self, 
                 entanglement_weight: float = 1.0,
                 fidelity_weight: float = 1.0, 
                 robust_fidelity_weight: float = 1.0,
                 combined_weight: float = 0.5):
        super().__init__()
        self.entanglement_weight = entanglement_weight
        self.fidelity_weight = fidelity_weight
        self.robust_fidelity_weight = robust_fidelity_weight
        self.combined_weight = combined_weight
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: 모델 예측값
            targets: 실제 값
        
        Returns:
            각 손실 및 총 손실
        """
        losses = {}
        
        # Individual property losses
        if 'entanglement' in targets:
            losses['entanglement'] = self.mse_loss(
                predictions['entanglement'], 
                targets['entanglement']
            )
        
        if 'fidelity' in targets:
            losses['fidelity'] = self.mse_loss(
                predictions['fidelity'], 
                targets['fidelity']
            )
        
        if 'robust_fidelity' in targets:
            losses['robust_fidelity'] = self.mse_loss(
                predictions['robust_fidelity'], 
                targets['robust_fidelity']
            )
        
        # Combined loss (optional)
        if 'combined' in targets:
            losses['combined'] = self.mse_loss(
                predictions['combined'], 
                targets['combined']
            )
        
        # Total weighted loss
        total_loss = 0.0
        if 'entanglement' in losses:
            total_loss += self.entanglement_weight * losses['entanglement']
        if 'fidelity' in losses:
            total_loss += self.fidelity_weight * losses['fidelity']
        if 'robust_fidelity' in losses:
            total_loss += self.robust_fidelity_weight * losses['robust_fidelity']
        if 'combined' in losses:
            total_loss += self.combined_weight * losses['combined']
        
        losses['total'] = total_loss
        
        return losses


def create_property_prediction_model(config: PropertyPredictionConfig = None) -> PropertyPredictionTransformer:
    """Property Prediction 모델 생성 헬퍼 함수"""
    if config is None:
        config = PropertyPredictionConfig()
    
    model = PropertyPredictionTransformer(config)
    
    print(f"Property Prediction Transformer 생성:")
    print(f"  - d_model: {config.d_model}")
    print(f"  - n_layers: {config.n_layers}")
    print(f"  - n_heads: {config.n_heads}")
    print(f"  - dropout: {config.dropout}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  - 총 파라미터: {total_params:,}")
    print(f"  - 학습 가능 파라미터: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    # 테스트 실행
    config = PropertyPredictionConfig(d_model=256, n_layers=4, n_heads=8)
    model = create_property_prediction_model(config)
    
    print("\n✅ Property Prediction Transformer 모델 생성 완료!")
