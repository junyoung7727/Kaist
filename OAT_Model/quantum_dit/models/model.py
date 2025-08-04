"""
Decision Transformer for Quantum Circuit Generation
State-of-the-art implementation with modern Transformer techniques
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Add quantumcommon to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))

from gates import QuantumGateRegistry


@dataclass
class DecisionTransformerConfig:
    """Decision Transformer model configuration"""
    # Model architecture
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    mlp_ratio: float = 4.0
    
    # Quantum circuit specific
    n_gate_types: int = 10  # rx, ry, rz, cx, h, x, y, z, i, measure
    n_qubits: int = 8
    max_seq_length: int = 256
    
    # Training
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    
    # Advanced features
    use_rotary_pe: bool = True
    use_swiglu: bool = True
    gradient_checkpointing: bool = False
    
    # Special tokens
    eos_token: int = 0  # End of sequence token


class QuantumInputEmbedding(nn.Module):
    """목표 메트릭, 현재 상태, 게이트 액션을 임베딩"""
    
    def __init__(self, d_model: int, n_gate_types: int, n_qubits: int):
        super().__init__()
        self.d_model = d_model
        
        # Target metrics embedding (fidelity, entanglement, expressibility, depth, n_qubits)
        self.metric_proj = nn.Linear(5, d_model)
        
        # Current state embedding (current_fidelity, current_entanglement, gate_count, validity)
        self.state_proj = nn.Linear(4, d_model)
        
        # Gate action embedding
        self.gate_embed = nn.Embedding(n_gate_types, d_model)  # rx, ry, cx, h, etc.
        self.qubit_embed = nn.Embedding(n_qubits, d_model)     # q0, q1, q2, ...
        self.param_proj = nn.Linear(1, d_model)                # angle parameters
        
        # Projection to combine embeddings
        self.combine_proj = nn.Linear(d_model * 5, d_model)  # 5 components
        
    def forward(self, target_metrics: torch.Tensor, current_states: torch.Tensor, 
                gate_actions: torch.Tensor, qubit1_actions: torch.Tensor, 
                qubit2_actions: torch.Tensor, param_actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target_metrics: [batch, seq, 5] - target fidelity, entanglement, expressibility, depth, n_qubits
            current_states: [batch, seq, 4] - current fidelity, entanglement, gate_count, validity
            gate_actions: [batch, seq] - gate type indices
            qubit1_actions: [batch, seq] - first qubit indices
            qubit2_actions: [batch, seq] - second qubit indices (for 2-qubit gates)
            param_actions: [batch, seq, 1] - gate parameters
        """
        # Embed each component
        metric_emb = self.metric_proj(target_metrics)  # [batch, seq, d_model]
        state_emb = self.state_proj(current_states)    # [batch, seq, d_model]
        gate_emb = self.gate_embed(gate_actions)       # [batch, seq, d_model]
        qubit1_emb = self.qubit_embed(qubit1_actions)  # [batch, seq, d_model]
        qubit2_emb = self.qubit_embed(qubit2_actions)  # [batch, seq, d_model]
        param_emb = self.param_proj(param_actions)     # [batch, seq, d_model]
        
        # Combine all embeddings
        combined = torch.cat([
            metric_emb, state_emb, gate_emb, qubit1_emb, param_emb
        ], dim=-1)  # [batch, seq, d_model * 5]
        
        # Project to final dimension
        output = self.combine_proj(combined)  # [batch, seq, d_model]
        
        return output


class RotaryPositionalEmbedding(nn.Module):
    """RoPE (Rotary Positional Embedding) - 최신 위치 인코딩"""
    
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        self.d_model = d_model
        
        # Precompute rotation matrices
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for efficiency
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cos_sin_tables(self, seq_len: int, device: torch.device):
        """Update cached cos/sin tables if needed"""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]
    
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary positional embedding to queries and keys"""
        seq_len = q.shape[2]
        self._update_cos_sin_tables(seq_len, q.device)
        
        cos = self._cos_cached[:, :, :seq_len, :]
        sin = self._sin_cached[:, :, :seq_len, :]
        
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)
        
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        
        return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU activation function - 최신 FFN 활성화 함수"""
    
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Attention(nn.Module):
    """Multi-head attention module reused from existing DiT implementation"""
    
    def __init__(self, hidden_size: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        if mask is not None:
            # Apply causal mask for autoregressive generation
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    """MLP module reused from existing implementation"""
    
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None, 
                 act_layer=nn.GELU, drop: float = 0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DecisionTransformerBlock(nn.Module):
    """기존 Attention + MLP를 재활용한 Transformer 블록"""
    
    def __init__(self, config: DecisionTransformerConfig):
        super().__init__()
        self.config = config
        
        # 기존 Attention 모듈 재활용
        self.attn = Attention(config.d_model, config.n_heads, qkv_bias=True)
        
        # 기존 MLP 구조 재활용
        if config.use_swiglu:
            self.mlp = SwiGLU(config.d_model, int(config.d_model * config.mlp_ratio))
        else:
            self.mlp = Mlp(in_features=config.d_model, 
                          hidden_features=int(config.d_model * config.mlp_ratio),
                          drop=config.dropout)
        
        # LayerNorm 재활용
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm structure (기존 DiT와 유사)
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class QuantumOutputHeads(nn.Module):
    """양자 게이트 생성을 위한 멀티태스크 출력"""
    
    def __init__(self, d_model: int, n_gate_types: int, n_qubits: int):
        super().__init__()
        
        # 게이트 타입 분류
        self.gate_head = nn.Linear(d_model, n_gate_types)
        
        # 큐빗 선택 (첫 번째, 두 번째)
        self.qubit1_head = nn.Linear(d_model, n_qubits)
        self.qubit2_head = nn.Linear(d_model, n_qubits)
        
        # 파라미터 회귀 (각도 값)
        self.param_head = nn.Linear(d_model, 1)
        
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'gate_logits': self.gate_head(hidden_states),
            'qubit1_logits': self.qubit1_head(hidden_states),
            'qubit2_logits': self.qubit2_head(hidden_states),
            'param_values': torch.sigmoid(self.param_head(hidden_states)) * 2 * math.pi
        }


class QuantumDecisionTransformer(nn.Module):
    """양자회로 생성을 위한 Decision Transformer"""
    
    def __init__(self, config: DecisionTransformerConfig):
        super().__init__()
        self.config = config
        
        # Get gate registry for vocabulary
        gate_registry = QuantumGateRegistry()
        gate_vocab = gate_registry.get_gate_vocab()
        self.vocab_size = len(gate_vocab)
        
        # 입력 임베딩
        self.input_embedding = QuantumInputEmbedding(
            config.d_model, config.n_gate_types, config.n_qubits
        )
        
        # 위치 임베딩 (기존 DiT의 pos_embed 로직 재활용)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_length, config.d_model))
        
        # RoPE for advanced positional encoding
        if config.use_rotary_pe:
            self.rope = RotaryPositionalEmbedding(config.d_model)
        
        # Transformer 블록들 (기존 Attention 재활용)
        self.blocks = nn.ModuleList([
            DecisionTransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # 출력 헤드
        self.output_heads = QuantumOutputHeads(config.d_model, config.n_gate_types, config.n_qubits)
        
        # Layer norm (기존 재활용)
        self.norm_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with proper scaling"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive generation"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    
    def forward(self, target_metrics: torch.Tensor, current_states: torch.Tensor,
                gate_actions: torch.Tensor, qubit1_actions: torch.Tensor, 
                qubit2_actions: torch.Tensor, param_actions: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Decision Transformer
        
        Args:
            target_metrics: [batch, seq, 5] - target fidelity, entanglement, expressibility, depth, n_qubits
            current_states: [batch, seq, 4] - current fidelity, entanglement, gate_count, validity
            gate_actions: [batch, seq] - gate type indices
            qubit1_actions: [batch, seq] - first qubit indices
            qubit2_actions: [batch, seq] - second qubit indices
            param_actions: [batch, seq, 1] - gate parameters
            attention_mask: [batch, seq] - attention mask
        
        Returns:
            Dict with gate_logits, qubit1_logits, qubit2_logits, param_values
        """
        batch_size, seq_len = gate_actions.shape
        device = gate_actions.device
        
        # 1. Input embedding
        x = self.input_embedding(
            target_metrics, current_states, gate_actions, 
            qubit1_actions, qubit2_actions, param_actions
        )  # [batch, seq, d_model]
        
        # 2. Add positional embedding
        if seq_len <= self.config.max_seq_length:
            pos_emb = self.pos_embed[:, :seq_len, :]
            x = x + pos_emb
        
        # 3. Create causal mask for autoregressive generation
        if attention_mask is None:
            attention_mask = self._create_causal_mask(seq_len, device)
        
        # 4. Apply transformer blocks
        for i, block in enumerate(self.blocks):
            if self.config.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, attention_mask)
            else:
                x = block(x, attention_mask)
        
        # 5. Final norm
        x = self.norm_f(x)
        
        # 6. Multi-task output
        outputs = self.output_heads(x)
        
        return outputs
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """멀티태스크 loss 계산"""
        outputs = self.forward(
            batch['target_metrics'], 
            batch['current_states'],
            batch['gate_actions'],
            batch['qubit1_actions'],
            batch['qubit2_actions'], 
            batch['param_actions']
        )
        
        # Gate type classification loss
        gate_loss = F.cross_entropy(
            outputs['gate_logits'].view(-1, self.config.n_gate_types), 
            batch['target_gates'].view(-1)
        )
        
        # Qubit selection losses  
        qubit1_loss = F.cross_entropy(
            outputs['qubit1_logits'].view(-1, self.config.n_qubits),
            batch['target_qubit1'].view(-1)
        )
        qubit2_loss = F.cross_entropy(
            outputs['qubit2_logits'].view(-1, self.config.n_qubits), 
            batch['target_qubit2'].view(-1)
        )
        
        # Parameter regression loss
        param_loss = F.mse_loss(outputs['param_values'], batch['target_params'])
        
        # 가중합
        total_loss = gate_loss + 0.5 * (qubit1_loss + qubit2_loss) + 0.3 * param_loss
        
        return {
            'loss': total_loss,
            'gate_loss': gate_loss, 
            'qubit_loss': (qubit1_loss + qubit2_loss) / 2,
            'param_loss': param_loss
        }
    
    def sample_from_logits(self, logits: torch.Tensor, temperature: float = 1.0, 
                          top_k: int = None, top_p: float = None) -> torch.Tensor:
        """Sample from logits with temperature, top-k, and top-p"""
        logits = logits / temperature
        
        if top_k is not None:
            # Top-k sampling
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        if top_p is not None:
            # Top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)
    
    @torch.no_grad()
    def generate_circuit(self, target_metrics: torch.Tensor, max_length: int = 50, 
                        temperature: float = 1.0, top_k: int = None, top_p: float = None) -> List[Dict]:
        """조건부 양자회로 생성"""
        self.eval()
        batch_size = target_metrics.size(0)
        device = target_metrics.device
        
        # 초기 상태
        current_states = torch.zeros(batch_size, 1, 4, device=device)  # [batch, seq, state_dim]
        gate_actions = torch.zeros(batch_size, 1, dtype=torch.long, device=device)    # [batch, seq]
        qubit1_actions = torch.zeros(batch_size, 1, dtype=torch.long, device=device)  # [batch, seq]
        qubit2_actions = torch.zeros(batch_size, 1, dtype=torch.long, device=device)  # [batch, seq]
        param_actions = torch.zeros(batch_size, 1, 1, device=device)   # [batch, seq, 1]
        
        # Expand target metrics to sequence dimension
        target_metrics_expanded = target_metrics.unsqueeze(1)  # [batch, 1, 5]
        
        generated_sequences = [[] for _ in range(batch_size)]
        
        for step in range(max_length):
            # Forward pass
            outputs = self.forward(
                target_metrics_expanded.expand(-1, step + 1, -1),
                current_states,
                gate_actions,
                qubit1_actions, 
                qubit2_actions,
                param_actions
            )
            
            # 마지막 위치의 예측 사용
            last_outputs = {k: v[:, -1:] for k, v in outputs.items()}
            
            # 샘플링
            next_gate = self.sample_from_logits(last_outputs['gate_logits'], temperature, top_k, top_p)
            next_qubit1 = self.sample_from_logits(last_outputs['qubit1_logits'], temperature, top_k, top_p)
            next_qubit2 = self.sample_from_logits(last_outputs['qubit2_logits'], temperature, top_k, top_p)
            next_param = last_outputs['param_values']
            
            # 생성된 액션 저장
            for b in range(batch_size):
                generated_action = {
                    'gate': next_gate[b].item(),
                    'qubit1': next_qubit1[b].item(), 
                    'qubit2': next_qubit2[b].item(),
                    'param': next_param[b, 0].item()
                }
                generated_sequences[b].append(generated_action)
                
                # 종료 조건 체크
                if next_gate[b].item() == self.config.eos_token:
                    break
            
            # 다음 스텝을 위한 상태 업데이트
            # Concatenate new actions to sequences
            gate_actions = torch.cat([gate_actions, next_gate.unsqueeze(1)], dim=1)
            qubit1_actions = torch.cat([qubit1_actions, next_qubit1.unsqueeze(1)], dim=1)
            qubit2_actions = torch.cat([qubit2_actions, next_qubit2.unsqueeze(1)], dim=1)
            param_actions = torch.cat([param_actions, next_param], dim=1)
            
            # Update current states (simplified - would need actual circuit evaluation)
            new_states = torch.zeros(batch_size, 1, 4, device=device)
            new_states[:, :, 2] = step + 1  # gate_count
            current_states = torch.cat([current_states, new_states], dim=1)
        
        return generated_sequences
    
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_decision_transformer_model(config: DecisionTransformerConfig) -> QuantumDecisionTransformer:
    """Factory function to create Decision Transformer model for quantum circuit generation"""
    model = QuantumDecisionTransformer(config)
    
    print(f"Created Decision Transformer model with {model.get_num_params():,} parameters")
    print(f"Model configuration: {config}")
    
    return model


def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Helper function for sampling from logits"""
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


if __name__ == "__main__":
    # Test model creation
    config = DecisionTransformerConfig(
        d_model=512,
        n_layers=6,
        n_heads=8,
        n_gate_types=10,
        n_qubits=8,
        max_seq_length=256
    )
    
    model = create_decision_transformer_model(config)
    print(f"Decision Transformer model created successfully!")
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    target_metrics = torch.randn(batch_size, seq_len, 5)
    current_states = torch.randn(batch_size, seq_len, 4)
    gate_actions = torch.randint(0, config.n_gate_types, (batch_size, seq_len))
    qubit1_actions = torch.randint(0, config.n_qubits, (batch_size, seq_len))
    qubit2_actions = torch.randint(0, config.n_qubits, (batch_size, seq_len))
    param_actions = torch.randn(batch_size, seq_len, 1)
    
    outputs = model(target_metrics, current_states, gate_actions, 
                   qubit1_actions, qubit2_actions, param_actions)
    
    print(f"Output shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    print("\nDecision Transformer replacement completed successfully!")
