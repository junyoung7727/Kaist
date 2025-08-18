"""
Property Prediction Transformer

CircuitSpec으로부터 양자 회로의 물리적 특성을 예측하는 트랜스포머 모델:
- Entanglement (얽힘도)
- Fidelity (충실도)
- Expressibility (표현력)

Predictor_Embed + attention.py 모듈들을 활용한 완전한 예측 시스템
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import math

# Import embedding and attention modules
from encoding.Decision_Transformer_Embed import QuantumGateSequenceEmbedding
from models.attention import (
    GridPositionalAttention,
    RegisterFlowAttention, 
    EntanglementAttention,
    SemanticAttention,
    AttentionFusionNetwork
)

# Import grid encoder
from encoding.grid_graph_encoder import GridGraphEncoder

# Add quantumcommon to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))
from gates import QuantumGateRegistry


@dataclass
class PropertyPredictionConfig:
    """Property Prediction Transformer 설정 (Unified Config 호환)"""
    # Model architecture (unified with Decision Transformer)
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.3  # 과적합 방지를 위해 증가
    
    # Unified attention configuration
    attention_mode: str = "advanced"  # "standard", "advanced", "grid", "semantic"
    use_rotary_pe: bool = True
    
    # Embedding settings
    max_grid_size: int = 64
    max_qubits: int = 50
    
    # Output settings
    property_dim: int = 4  # entanglement, fidelity, expressibility, robust_fidelity
    
    # Training settings
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3  # 과적합 방지를 위해 증가
    warmup_steps: int = 1000

    # Data settings
    train_batch_size: int = 64
    val_batch_size: int = 64
    test_batch_size: int = 64


class PropertyPredictionHead(nn.Module):
    """물리적 특성 예측을 위한 출력 헤드"""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        # 안정적인 공유 특성 추출기 (그래디언트 안정성 개선)
        self.shared_feature_extractor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # 더 낮은 dropout
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model), 
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2)  # 출력 정규화 추가
        )
        
        # 안정적인 전문화된 헤드 (그래디언트 안정성 개선)
        def create_stable_head(input_dim, use_activation=True):
            layers = [
                nn.Linear(input_dim, input_dim // 2),
                nn.LayerNorm(input_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(input_dim // 2, 1)
            ]
            if use_activation:
                layers.append(nn.Sigmoid())  # (0, 1) 범위로 변경
            return nn.Sequential(*layers)
        
        self.entanglement_head = create_stable_head(d_model // 2)  # Sigmoid: (0,1)
        self.fidelity_head = create_stable_head(d_model // 2)     # Sigmoid: (0,1)
        self.expressibility_head = create_stable_head(d_model // 2, use_activation=False)  # 무제한 범위
        self.robust_fidelity_head = create_stable_head(d_model // 2)  # Sigmoid: (0,1)
        
        # 안정적인 통합 헤드 (활성화 함수 제거)
        self.combined_head = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(d_model // 4, 4)  # [entanglement, fidelity, expressibility, robust_fidelity]
            # 활성화 함수 제거 - 각 헤드에서 개별 처리
        )
        
        # 가중치 초기화 개선
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """안정적인 가중치 초기화"""
        if isinstance(module, nn.Linear):
            # Xavier 초기화로 그래디언트 안정성 향상
            torch.nn.init.xavier_uniform_(module.weight, gain=0.01)  # 더 작은 gain
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def _register_gradient_hooks(self):
        """그래디언트 클리핑 훅 등록"""
        def gradient_hook(grad):
            if grad is not None:
                # NaN 체크 및 클리핑
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    return torch.zeros_like(grad)
                return torch.clamp(grad, -1.0, 1.0)
            return grad
        
        for param in self.parameters():
            if param.requires_grad:
                param.register_hook(gradient_hook)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Circuit representation [batch_size, d_model]
            
        Returns:
            Dict containing predictions for each property
        """
        # 공유 특성 추출 (더 깊은 representation)
        shared_features = self.shared_feature_extractor(x)  # [batch_size, d_model//2]
        
        # NaN 체크 및 안전한 예측
        if torch.isnan(shared_features).any() or torch.isinf(shared_features).any():
            batch_size = shared_features.size(0)
            device = shared_features.device
            return {
                'entanglement': torch.zeros(batch_size, device=device),
                'fidelity': torch.zeros(batch_size, device=device),
                'expressibility': torch.zeros(batch_size, device=device),
                'robust_fidelity': torch.zeros(batch_size, device=device),
                'combined': torch.zeros(batch_size, 4, device=device)
            }
        
        # 각 특성별 전문화된 예측
        combined_raw = self.combined_head(shared_features)  # [batch, 4]
        
        predictions = {
            'entanglement': self.entanglement_head(shared_features).squeeze(-1),
            'fidelity': self.fidelity_head(shared_features).squeeze(-1),
            'expressibility': self.expressibility_head(shared_features).squeeze(-1),
            'robust_fidelity': self.robust_fidelity_head(shared_features).squeeze(-1),
            'combined': combined_raw
        }
        
        # 출력 NaN 체크 및 적절한 클리핑
        for key, value in predictions.items():
            if torch.isnan(value).any() or torch.isinf(value).any():
                predictions[key] = torch.zeros_like(value)
            elif key == 'expressibility':
                # Expressibility는 더 넓은 범위 허용 (0~50)
                predictions[key] = torch.clamp(value, 0.0, 50.0)
            elif key == 'combined':
                # Combined는 각 요소별로 다르게 처리
                continue  # 별도 처리 없음
            else:
                # Entanglement, Fidelity, Robust_fidelity는 (0,1) 범위
                predictions[key] = torch.clamp(value, 0.0, 1.0)
        
        return predictions


class TransformerBlock(nn.Module):
    """트랜스포머 블록 (Unified Attention System)"""
    
    def __init__(self, config: PropertyPredictionConfig):
        super().__init__()
        self.config = config
        
        # Unified ModularAttention 사용 (Decision Transformer와 동일)
        from models.modular_attention import create_modular_attention
        
        # 통합 어텐션 모드 사용
        self.attention = create_modular_attention(
            d_model=config.d_model,
            n_heads=config.n_heads, 
            dropout=config.dropout,
            mode=config.attention_mode.upper()  # 통합 설정에서 가져옴
        )
        
        # 피드포워드 네트워크 (Decision Transformer와 동일)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        # Pre-norm 구조 (Decision Transformer와 동일)
        self.norm1 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        
        # 학습 가능한 스케일 파라미터 (더 작은 초기값)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # Residual dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, 
                grid_structure: Dict, edges: List[Dict], device=None) -> torch.Tensor:
        """
        Decision Transformer 호환 forward pass
        
        Args:
            x: [batch_size, seq_len, d_model] (Decision Transformer 표준)
            attention_mask: [batch_size, seq_len, seq_len]
            grid_structure: 그리드 구조 정보
            edges: 엣지 정보
            device: 텐서를 생성할 디바이스 (선택사항)
        """
        # Pre-norm + 어텐션 + 스케일링된 잔차 연결 (NaN 체크 포함)
        norm_x = self.norm1(x)
        attn_out = self.attention(norm_x, attention_mask, grid_structure, edges)
        
        # NaN 체크
        if torch.isnan(attn_out).any() or torch.isinf(attn_out).any():
            attn_out = torch.zeros_like(attn_out)
        
        dropout_attn = self.dropout1(attn_out)
        scaled_attn = self.scale * dropout_attn
        x = x + scaled_attn
        
        # Pre-norm + 피드포워드 + 스케일링된 잔차 연결 (NaN 체크 포함)
        norm_x = self.norm2(x)
        ff_out = self.feed_forward(norm_x)
        
        # NaN 체크
        if torch.isnan(ff_out).any() or torch.isinf(ff_out).any():
            ff_out = torch.zeros_like(ff_out)
        
        dropout_ff = self.dropout2(ff_out)
        scaled_ff = self.scale * dropout_ff
        x = x + scaled_ff
        
        return x


class PropertyPredictionTransformer(nn.Module):
    """양자 회로 특성 예측 트랜스포머"""
    
    def __init__(self, config: PropertyPredictionConfig):
        super().__init__()
        self.config = config
        
        # Grid encoder for CircuitSpec
        self.grid_encoder = GridGraphEncoder()
        
        # State-based embedding system (unified with Decision Transformer)
        gate_registry = QuantumGateRegistry()
        n_gate_types = len(gate_registry.get_gate_vocab())
        
        self.circuit_embedding = QuantumGateSequenceEmbedding(
            d_model=config.d_model,
            n_gate_types=n_gate_types,
            max_pos=1024,
            dropout=config.dropout,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Global pooling for circuit-level representation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Property prediction heads
        self.prediction_head = PropertyPredictionHead(config.d_model, config.dropout)
        
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
    
    def forward(self, circuit_specs) -> Dict[str, torch.Tensor]:
        """
        Forward pass for property prediction with padding support
        
        Args:
            circuit_specs: Single circuit spec dict or list of circuit specs
            
        Returns:
            Dict containing predictions for each property
        """
        if isinstance(circuit_specs, list):
            # Batch processing with padding
            return self._forward_batch_with_padding(circuit_specs)
        else:
            # Single circuit
            return self._forward_single(circuit_specs)
    
    def _forward_single(self, circuit_spec) -> Dict[str, torch.Tensor]:
        """단일 회로 처리 (State-based embedding)"""
        # 1. Grid encoding
        encoded_circuit = self.grid_encoder.encode(circuit_spec)
        
        # 2. Convert to state-based format [gate_type, qubit1, qubit2, parameter]
        state_sequence = self._convert_to_state_sequence(encoded_circuit)
        
        # 3. State-based embedding
        if len(state_sequence) > 0:
            state_embeddings = self.circuit_embedding.state(state_sequence)  # [seq_len, d_model]
            x = state_embeddings.unsqueeze(0)  # [1, seq_len, d_model]
        else:
            # Empty circuit
            x = torch.zeros(1, 1, self.config.d_model, device=next(self.parameters()).device)
        
        # 4. Attention mask 생성
        seq_len = x.size(1)
        attention_mask = torch.ones(1, seq_len, seq_len, device=x.device)
        
        # 5. Create minimal grid structure for compatibility
        grid_structure = {'distance_matrix': torch.eye(seq_len, device=x.device)}
        edges = []
        
        # 6. Transformer layers
        for layer in self.transformer_layers:
            x = layer(x, attention_mask, grid_structure, edges, device=x.device)
        
        # 7. Global pooling for circuit representation
        circuit_repr = x.mean(dim=1)  # [1, d_model]
        
        # 8. Property predictions
        predictions = self.prediction_head(circuit_repr)
        
        return predictions
    
    def _convert_to_state_sequence(self, encoded_circuit: Dict) -> torch.Tensor:
        """Convert grid-encoded circuit to state sequence format [gate_type, qubit1, qubit2, parameter]"""
        nodes = encoded_circuit.get('nodes', [])
        if not nodes:
            return torch.zeros(0, 4)  # Empty sequence
        
        gate_registry = QuantumGateRegistry()
        gate_vocab = gate_registry.get_gate_vocab()
        
        state_sequence = []
        for node in nodes:
            # Get gate type index
            gate_name = node.get('gate_name', 'I')
            gate_type_id = gate_vocab.get(gate_name, gate_vocab.get('I', 0))
            
            # Get qubit positions (default to 0 if not specified)
            qubits = node.get('qubits', [0])
            if isinstance(qubits, int):
                qubits = [qubits]
            
            qubit1 = qubits[0] if len(qubits) > 0 else 0
            qubit2 = qubits[1] if len(qubits) > 1 else qubit1  # For single-qubit gates, use same qubit
            
            # Get parameter value
            parameter = node.get('parameter_value', 0.0)
            if parameter is None or (isinstance(parameter, float) and math.isnan(parameter)):
                parameter = 0.0
            
            # Create state vector [gate_type, qubit1, qubit2, parameter]
            state_vector = [float(gate_type_id), float(qubit1), float(qubit2), float(parameter)]
            state_sequence.append(state_vector)
        
        return torch.tensor(state_sequence, dtype=torch.float32)
    
    def _forward_batch_with_padding(self, circuit_specs: List[Dict]) -> Dict[str, torch.Tensor]:
        """배치 처리 (Decision Transformer 호환 방식)"""
        if not circuit_specs:
            raise ValueError("Empty circuit specs list")
        
        # 1. 각 회로에 대해 임베딩 계산
        embeddings_list = []
        seq_lengths = []
        
        for circuit_spec in circuit_specs:
            encoded_circuit = self.grid_encoder.encode(circuit_spec)
            state_sequence = self._convert_to_state_sequence(encoded_circuit)
            
            if len(state_sequence) > 0:
                node_embeddings = self.circuit_embedding.state(state_sequence)
                seq_len = len(state_sequence)
            else:
                node_embeddings = torch.zeros(1, self.config.d_model, device=next(self.parameters()).device)
                seq_len = 1
            
            embeddings_list.append(node_embeddings)
            seq_lengths.append(seq_len)
        
        # 2. 최대 시퀀스 길이로 패딩
        max_seq_len = max(seq_lengths) if seq_lengths else 1
        batch_size = len(circuit_specs)
        
        # 3. Decision Transformer 표준 차원: [batch_size, seq_len, d_model]
        device = next(self.parameters()).device  # Get model device
        padded_embeddings = torch.zeros(batch_size, max_seq_len, self.config.d_model, device=device)
        attention_masks = torch.zeros(batch_size, max_seq_len, max_seq_len, device=device)  # [batch_size, seq_len, seq_len]
        
        # 4. 패딩 및 마스크 생성
        for i, (embeddings, seq_len) in enumerate(zip(embeddings_list, seq_lengths)):
            if seq_len > 0:
                padded_embeddings[i, :seq_len, :] = embeddings
                # Causal mask for valid tokens
                attention_masks[i, :seq_len, :seq_len] = 1.0
        
        # 5. Transformer 레이어 적용 (Decision Transformer 방식)
        x = padded_embeddings
        
        for layer in self.transformer_layers:
            # 모든 배치를 한 번에 처리 (simplified structure)
            grid_structure = {'distance_matrix': torch.eye(max_seq_len, device=x.device)}
            edges = []
            x = layer(x, attention_masks, grid_structure, edges, device=x.device)
        
        # 6. Global pooling (마스크 고려)
        # 유효한 토큰들의 평균 계산
        device = x.device  # 현재 x의 디바이스 가져오기
        valid_lengths = torch.tensor(seq_lengths, device=device).float().unsqueeze(-1)  # [batch_size, 1]
        circuit_reprs = x.sum(dim=1) / torch.clamp(valid_lengths, min=1.0)  # [batch_size, d_model]
        
        # 7. Property predictions
        batch_predictions = self.prediction_head(circuit_reprs)
        
        return batch_predictions
    
    def _pad_distance_matrices(self, distance_matrices: List[torch.Tensor], max_seq_len: int) -> torch.Tensor:
        """거리 매트릭스들을 최대 길이로 패딩"""
        batch_size = len(distance_matrices)
        device = next(self.parameters()).device  # Get model device
        padded_matrices = torch.zeros(batch_size, max_seq_len, max_seq_len, device=device)
        
        for i, dist_matrix in enumerate(distance_matrices):
            seq_len = dist_matrix.size(0)
            if seq_len > 0:
                # Move the distance matrix to the same device if needed
                if dist_matrix.device != device:
                    dist_matrix = dist_matrix.to(device)
                padded_matrices[i, :seq_len, :seq_len] = dist_matrix
        
        return padded_matrices
    
    def _apply_layer_with_batch_info(self, layer, x: torch.Tensor, 
                                   padded_distance_matrices: torch.Tensor,
                                   grid_structures: List[Dict],
                                   edges_list: List[List[Dict]],
                                   attention_masks: torch.Tensor,
                                   layer_idx: int) -> torch.Tensor:
        """배치 정보를 고려하여 레이어 적용"""
        batch_size = x.size(1)
        outputs = []
        
        for b in range(batch_size):
            # 각 배치 아이템에 대해 개별 처리
            seq_len = attention_masks[b].sum().item()
            if seq_len == 0:
                # 빈 시퀀스인 경우
                outputs.append(torch.zeros(x.size(0), 1, x.size(2), device=x.device))
                continue
            
            # 해당 배치의 데이터 추출
            batch_x = x[:seq_len, b:b+1, :]  # [seq_len, 1, d_model]
            batch_dist_matrix = padded_distance_matrices[b, :seq_len, :seq_len]  # [seq_len, seq_len]
            batch_grid_structure = grid_structures[b]
            batch_edges = edges_list[b]
            
            # 레이어 적용
            batch_output = layer(batch_x, batch_dist_matrix, batch_grid_structure, batch_edges)
            
            # 최대 길이로 패딩
            padded_output = torch.zeros(x.size(0), 1, x.size(2), device=x.device)
            padded_output[:seq_len, :, :] = batch_output
            outputs.append(padded_output)
        
        return torch.cat(outputs, dim=1)  # [max_seq_len, batch_size, d_model]
    
    def _masked_global_pooling(self, x: torch.Tensor, attention_masks: torch.Tensor) -> torch.Tensor:
        """마스크를 고려한 글로벌 풀링"""
        # x: [max_seq_len, batch_size, d_model]
        # attention_masks: [batch_size, max_seq_len]
        
        x = x.transpose(0, 1)  # [batch_size, max_seq_len, d_model]
        
        # 마스크 적용
        mask_expanded = attention_masks.unsqueeze(-1)  # [batch_size, max_seq_len, 1]
        masked_x = x * mask_expanded.float()  # [batch_size, max_seq_len, d_model]
        
        # 유효한 토큰들의 평균 계산
        seq_lengths = attention_masks.sum(dim=1, keepdim=True).float()  # [batch_size, 1]
        seq_lengths = torch.clamp(seq_lengths, min=1.0)  # 0으로 나누기 방지
        
        pooled = masked_x.sum(dim=1) / seq_lengths  # [batch_size, d_model]
        
        return pooled

class PropertyPredictionLoss(nn.Module):
    """특성 예측을 위한 손실 함수"""
    
    def __init__(self, 
                 entanglement_weight: float = 1.0,
                 fidelity_weight: float = 10.0,  # Fidelity 가중치 증가 (좁은 범위)
                 expressibility_weight: float = 0.1,  # Expressibility 가중치 감소 (큰 값)
                 combined_weight: float = 0.5):
        super().__init__()
        self.entanglement_weight = entanglement_weight
        self.fidelity_weight = fidelity_weight
        self.expressibility_weight = expressibility_weight
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
        
        # Individual property losses with normalization
        if 'entanglement' in targets:
            losses['entanglement'] = self.mse_loss(
                predictions['entanglement'], 
                targets['entanglement']
            )
        
        if 'fidelity' in targets:
            # Fidelity: 동적 정규화 (배치별 범위 기반)
            fidelity_min = targets['fidelity'].min()
            fidelity_max = targets['fidelity'].max()
            fidelity_range = fidelity_max - fidelity_min + 1e-8  # 0으로 나누기 방지
            
            fidelity_pred_norm = (predictions['fidelity'] - fidelity_min) / fidelity_range
            fidelity_target_norm = (targets['fidelity'] - fidelity_min) / fidelity_range
            losses['fidelity'] = self.mse_loss(fidelity_pred_norm, fidelity_target_norm)
        
        if 'expressibility' in targets:
            # Expressibility: 큰 값이므로 로그 스케일 적용
            expr_pred_log = torch.log1p(predictions['expressibility'])  # log(1+x)
            expr_target_log = torch.log1p(targets['expressibility'])
            losses['expressibility'] = self.mse_loss(expr_pred_log, expr_target_log)
        
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
        if 'expressibility' in losses:
            total_loss += self.expressibility_weight * losses['expressibility']
        if 'robust_fidelity' in losses:
            total_loss += self.fidelity_weight * losses['robust_fidelity']  # Same weight as fidelity
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
