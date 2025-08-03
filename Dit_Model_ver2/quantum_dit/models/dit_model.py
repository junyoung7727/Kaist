"""
DiT (Diffusion Transformer) for Quantum Circuit Generation
State-of-the-art implementation with modern Transformer techniques
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Import attention modules
from .attention import (
    QuantumMultiHeadAttention,
    GridPositionalAttention,
    RegisterFlowAttention,
    EntanglementAttention,
    SemanticAttention,
    AttentionFusionNetwork
)

# Import diffusion scheduler
from ..utils.diffusion import DiffusionScheduler

# Import existing grid embedding
from ..encoding.Embeding import QuantumCircuitAttentionEmbedding

# Add quantumcommon to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))

from gates import QuantumGateRegistry


@dataclass
class DiTConfig:
    """DiT model configuration"""
    # Model architecture
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    d_ff: int = 1024
    
    # Quantum circuit specific
    max_circuit_length: int = 128
    max_qubits: int = 16
    
    # Training
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    
    # Diffusion settings
    timesteps: int = 1000
    noise_schedule: str = "cosine"  # "linear", "cosine", "sigmoid"
    diffusion_mode: bool = False  # Enable diffusion mode
    
    # Advanced features
    use_flash_attention: bool = True
    use_rotary_pe: bool = True
    use_swiglu: bool = True
    gradient_checkpointing: bool = False


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding for diffusion"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Sinusoidal timestep embedding
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        
        return self.proj(emb)


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

class DiTBlock(nn.Module):
    """DiT Transformer Block with modern optimizations and timestep conditioning"""
    
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        
        # Pre-Layer Normalization
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Quantum-Specialized Multi-Layer Attention
        self.grid_attention = GridPositionalAttention(config.d_model, config.n_heads)
        self.register_attention = RegisterFlowAttention(config.d_model, config.n_heads)
        self.entangle_attention = EntanglementAttention(config.d_model, config.n_heads)
        self.semantic_attention = SemanticAttention(config.d_model, config.n_heads)
        self.attention_fusion = AttentionFusionNetwork(config.d_model, config.n_heads)
        
        # Feed-Forward Network
        if config.use_swiglu:
            self.ffn = SwiGLU(config.d_model, config.d_ff)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(config.d_model, config.d_ff),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_ff, config.d_model)
            )
        
        # AdaLN for timestep conditioning (only in diffusion mode)
        if config.diffusion_mode:
            self.adaln_linear = nn.Sequential(
                nn.SiLU(),
                nn.Linear(config.d_model, 6 * config.d_model)
            )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def _apply_quantum_attention(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply quantum-specialized multi-layer attention fusion"""
        batch_size, seq_len, d_model = x.shape
        
        # Create simplified grid structure for attention
        # For efficiency, we'll create a representative structure
        grid_structure = self._create_grid_structure(seq_len)
        edges = self._create_edges(seq_len)
        
        # Apply each quantum-specialized attention layer
        attention_outputs = {}
        
        # Grid positional attention (distance-based)
        distance_matrix = self._create_distance_matrix(seq_len)
        grid_out = self.grid_attention(x.view(-1, d_model), distance_matrix)
        attention_outputs['grid_attention'] = {
            'output': grid_out['output'].view(batch_size, seq_len, d_model),
            'attention_weights': grid_out['attention_weights']
        }
        
        # Register flow attention
        register_out = self.register_attention(x.view(-1, d_model), grid_structure, edges)
        attention_outputs['register_attention'] = {
            'output': register_out['output'].view(batch_size, seq_len, d_model),
            'attention_weights': register_out['attention_weights']
        }
        
        # Entanglement attention
        entangle_out = self.entangle_attention(x.view(-1, d_model), grid_structure, edges)
        attention_outputs['entangle_attention'] = {
            'output': entangle_out['output'].view(batch_size, seq_len, d_model),
            'attention_weights': entangle_out['attention_weights']
        }
        
        # Semantic attention
        semantic_out = self.semantic_attention(x.view(-1, d_model))
        attention_outputs['semantic_attention'] = {
            'output': semantic_out['output'].view(batch_size, seq_len, d_model),
            'attention_weights': semantic_out['attention_weights']
        }
        
        # Fusion network
        fused_output = self.attention_fusion(attention_outputs)
        
        return fused_output.view(batch_size, seq_len, d_model)
    
    def _create_grid_structure(self, seq_len: int) -> Dict:
        """Create simplified grid structure for attention"""
        positions = torch.stack([
            torch.arange(seq_len) % 8,  # time dimension (mod 8 for efficiency)
            torch.arange(seq_len) // 8  # qubit dimension
        ], dim=1)
        
        return {
            'positions': positions,
            'distance_matrix': self._create_distance_matrix(seq_len)
        }
    
    def _create_edges(self, seq_len: int) -> List[Dict]:
        """Create simplified edge connections for attention"""
        edges = []
        
        # Register connections (sequential)
        for i in range(seq_len - 1):
            edges.append({
                'type': 'REGISTER_CONNECTION',
                'source': [i % 8, i // 8],
                'target': [(i + 1) % 8, (i + 1) // 8]
            })
        
        # Entanglement connections (every 2 positions)
        for i in range(0, seq_len - 1, 2):
            edges.append({
                'type': 'ENTANGLE_CONNECTION',
                'source': [i % 8, i // 8],
                'target': [(i + 1) % 8, (i + 1) // 8]
            })
        
        return edges
    
    def _create_distance_matrix(self, seq_len: int) -> torch.Tensor:
        """Create distance matrix for grid attention"""
        positions = torch.arange(seq_len).unsqueeze(0)
        distances = torch.abs(positions - positions.T)
        return distances.float()
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.config.diffusion_mode and t_emb is not None:
            # Diffusion mode with timestep conditioning (AdaLN)
            adaln_out = self.adaln_linear(t_emb)  # [batch, 6 * d_model]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln_out.chunk(6, dim=1)
            
            # Quantum-Specialized Multi-Layer Attention with AdaLN
            norm_x = self.norm1(x)
            norm_x = norm_x * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
            
            # Apply quantum-specialized attention layers
            attn_out = self._apply_quantum_attention(norm_x, mask)
            x = x + gate_msa.unsqueeze(1) * self.dropout(attn_out)
            
            # Feed-forward with AdaLN
            norm_x = self.norm2(x)
            norm_x = norm_x * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
            ffn_out = self.ffn(norm_x)
            x = x + gate_mlp.unsqueeze(1) * self.dropout(ffn_out)
        else:
            # Standard mode (property prediction)
            # Pre-LN with residual connection
            norm_x = self.norm1(x)
            attn_out = self._apply_quantum_attention(norm_x, mask)
            x = x + self.dropout(attn_out)
            
            # Pre-LN with residual connection
            ffn_out = self.ffn(self.norm2(x))
            x = x + self.dropout(ffn_out)
        
        return x


class TimestepEmbedding(nn.Module):
    """Timestep embedding for diffusion process"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Sinusoidal timestep embedding
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        
        return self.proj(emb)


class QuantumDiT(nn.Module):
    """
    DiT (Diffusion Transformer) for Quantum Circuit Generation and Property Prediction
    
    Supports both:
    1. Diffusion-based circuit generation (diffusion_mode=True)
    2. Property prediction (diffusion_mode=False)
    
    State-of-the-art implementation with:
    - RoPE (Rotary Positional Embedding)
    - SwiGLU activation
    - Pre-Layer Normalization
    - Flash Attention support
    - Gradient checkpointing
    - AdaLN timestep conditioning
    """
    
    def __init__(self, config: DiTConfig, num_targets: int = 3):
        super().__init__()
        self.config = config
        self.num_targets = num_targets
        
        # Circuit feature embeddings
        gate_registry = QuantumGateRegistry()
        self.vocab_size = len(gate_registry.get_gate_vocab())
        
        # Gate sequence embedding
        self.gate_embedding = nn.Embedding(self.vocab_size + 1, config.d_model)  # +1 for padding
        
        # Circuit property embeddings
        self.qubit_embedding = nn.Embedding(65, config.d_model // 4)  # Up to 64 qubits
        self.gate_count_embedding = nn.Linear(1, config.d_model // 4)
        self.depth_embedding = nn.Linear(1, config.d_model // 4)
        self.two_qubit_ratio_embedding = nn.Linear(1, config.d_model // 4)
        
        # Grid-based embedding with integrated RoPE positional encoding
        self.grid_embedding = QuantumCircuitAttentionEmbedding(
            d_model=config.d_model,
            n_heads=config.n_heads,
            max_grid_size=config.max_circuit_length,
            max_qubits=64,
            dropout=config.dropout,
            use_rotary_pe=config.use_rotary_pe
        )
        
        # Diffusion components (only if diffusion_mode is enabled)
        if config.diffusion_mode:
            self.timestep_embedding = TimestepEmbedding(config.d_model)
            self.diffusion_scheduler = DiffusionScheduler(
                timesteps=config.timesteps,
                noise_schedule=config.noise_schedule
            )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Output heads (dual purpose)
        if config.diffusion_mode:
            # For diffusion: predict noise
            self.noise_predictor = nn.Linear(config.d_model, self.vocab_size)
        else:
            # For property prediction
            self.property_predictor = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model // 2, num_targets)
            )
        
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
    
    def _create_encoded_circuit_from_gates(self, gates: torch.Tensor, num_qubits: torch.Tensor = None) -> Dict:
        """Convert gate sequences to encoded circuit format for grid embedding"""
        batch_size, seq_len = gates.shape
        device = gates.device
        
        # Get gate registry for gate name mapping
        gate_registry = QuantumGateRegistry()
        gate_vocab = gate_registry.get_gate_vocab()
        idx_to_gate = {v: k for k, v in gate_vocab.items()}
        
        # Default number of qubits if not provided
        if num_qubits is None:
            max_qubits = 4  # Default assumption
        else:
            max_qubits = num_qubits.max().item() if num_qubits.dim() > 0 else num_qubits.item()
        
        # Create a simplified encoded circuit structure
        # For batch processing, we'll create a representative structure
        nodes = []
        edges = []
        
        # Create nodes for each gate in the sequence
        for t in range(min(seq_len, 16)):  # Limit to prevent memory issues
            for q in range(min(max_qubits, 4)):  # Limit qubits for efficiency
                gate_idx = gates[0, t].item()  # Use first batch item as representative
                gate_name = idx_to_gate.get(gate_idx, 'i')  # Default to identity
                
                node = {
                    'id': f'{gate_name}_q{q}_{t}',
                    'gate_name': gate_name,
                    'grid_position': [t, q],  # [time, qubit]
                    'role': 'single',
                    'parameter_value': 0.0,
                    'has_parameter': 0.0,
                    'is_hermitian': gate_name in ['h', 'x', 'y', 'z', 'i']
                }
                nodes.append(node)
                
                # Add register connections (time flow)
                if t > 0:
                    edges.append({
                        'type': 'REGISTER_CONNECTION',
                        'source': [t-1, q],
                        'target': [t, q]
                    })
        
        # Add some entanglement connections for two-qubit gates
        for t in range(min(seq_len, 16)):
            if max_qubits > 1:
                edges.append({
                    'type': 'ENTANGLE_CONNECTION',
                    'source': [t, 0],
                    'target': [t, 1]
                })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'grid_shape': [min(seq_len, 16), min(max_qubits, 4)]
        }
    
    def forward(self, 
                gates: torch.Tensor = None,
                timesteps: torch.Tensor = None,
                num_qubits: torch.Tensor = None,
                gate_count: torch.Tensor = None,
                depth: torch.Tensor = None,
                two_qubit_ratio: torch.Tensor = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of DiT model - supports both diffusion and property prediction modes
        
        Diffusion Mode (diffusion_mode=True):
            Args:
                gates: Noisy gate sequence [batch_size, seq_len]
                timesteps: Diffusion timesteps [batch_size]
                mask: Attention mask [batch_size, seq_len]
            Returns:
                Predicted noise [batch_size, seq_len, vocab_size]
                
        Property Prediction Mode (diffusion_mode=False):
            Args:
                gates: Gate sequence tensor [batch_size, seq_len]
                num_qubits: Number of qubits [batch_size]
                gate_count: Gate count [batch_size]
                depth: Circuit depth [batch_size]
                two_qubit_ratio: Two-qubit gate ratio [batch_size]
                mask: Attention mask [batch_size, seq_len]
            Returns:
                Predicted properties [batch_size, num_targets]
        """
        batch_size, seq_len = gates.shape
        
        # 1. Gate sequence embedding
        gate_emb = self.gate_embedding(gates)  # [batch_size, seq_len, d_model]
        
        # 2. Apply grid embedding for semantic features (no positional encoding)
        # Create encoded circuit structure for grid embedding
        encoded_circuit = self._create_encoded_circuit_from_gates(gates, num_qubits)
        grid_results = self.grid_embedding(encoded_circuit)
        
        # Use grid embedding for semantic circuit features
        grid_emb = grid_results['circuit_embedding']  # [batch_size, d_model]
        
        # Expand grid embedding to sequence length and add to gate embeddings
        grid_emb_expanded = grid_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, d_model]
        x = gate_emb + grid_emb_expanded
        
        # RoPE will handle all positional encoding in attention layers
        
        # 3. Mode-specific conditioning
        if self.config.diffusion_mode:
            # Diffusion mode: use timestep embedding
            if timesteps is None:
                raise ValueError("timesteps must be provided in diffusion mode")
            t_emb = self.timestep_embedding(timesteps)  # [batch_size, d_model]
        else:
            # Property prediction mode: use circuit properties
            if any(x is None for x in [num_qubits, gate_count, depth, two_qubit_ratio]):
                raise ValueError("Circuit properties must be provided in property prediction mode")
            
            qubit_emb = self.qubit_embedding(num_qubits)  # [batch_size, d_model//4]
            gate_count_emb = self.gate_count_embedding(gate_count.float().unsqueeze(-1))  # [batch_size, d_model//4]
            depth_emb = self.depth_embedding(depth.float().unsqueeze(-1))  # [batch_size, d_model//4]
            two_qubit_emb = self.two_qubit_ratio_embedding(two_qubit_ratio.unsqueeze(-1))  # [batch_size, d_model//4]
            
            # Combine circuit properties
            circuit_props = torch.cat([qubit_emb, gate_count_emb, depth_emb, two_qubit_emb], dim=-1)  # [batch_size, d_model]
            
            # Add circuit properties to each position
            circuit_props_expanded = circuit_props.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, d_model]
            x = x + circuit_props_expanded
        
        # 4. Apply transformer blocks with appropriate conditioning
        for i, block in enumerate(self.blocks):
            if self.config.gradient_checkpointing and self.training:
                if self.config.diffusion_mode:
                    x = torch.utils.checkpoint.checkpoint(block, x, mask, t_emb)
                else:
                    x = torch.utils.checkpoint.checkpoint(block, x, mask)
            else:
                if self.config.diffusion_mode:
                    x = block(x, mask, t_emb)
                else:
                    x = block(x, mask)
        
        # 5. Final processing based on mode
        if self.config.diffusion_mode:
            # Diffusion mode: predict noise for each position
            x = self.final_norm(x)  # [batch_size, seq_len, d_model]
            noise_pred = self.noise_predictor(x)  # [batch_size, seq_len, vocab_size]
            return noise_pred
        else:
            # Property prediction mode: global pooling and prediction
            if mask is not None:
                # Masked average pooling
                mask_expanded = mask.unsqueeze(-1).expand_as(x)
                x_masked = x * mask_expanded
                seq_lengths = mask.sum(dim=1, keepdim=True).float()
                x_pooled = x_masked.sum(dim=1) / seq_lengths.clamp(min=1)
            else:
                # Simple average pooling
                x_pooled = x.mean(dim=1)  # [batch_size, d_model]
            
            # Final normalization and property prediction
            x_pooled = self.final_norm(x_pooled)
        properties = self.property_predictor(x_pooled)  # [batch_size, num_targets]
        
        return properties
    
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def estimate_flops(self, seq_len: int, batch_size: int = 1) -> int:
        """Estimate FLOPs for forward pass"""
        # Rough estimation for transformer
        attention_flops = 4 * batch_size * seq_len * self.config.d_model * seq_len
        ffn_flops = 8 * batch_size * seq_len * self.config.d_model * self.config.d_ff
        
        total_flops = self.config.n_layers * (attention_flops + ffn_flops)
        return total_flops


def create_dit_model(config: DiTConfig, num_targets: int = 3, target_names: List[str] = None) -> QuantumDiT:
    """Factory function to create DiT model for property prediction"""
    model = QuantumDiT(config, num_targets=num_targets)
    
    if target_names is None:
        target_names = ['expressibility', 'two_qubit_ratio', 'simulator_error_fidelity']
    
    print(f"Created DiT model with {model.get_num_params():,} parameters")
    print(f"Target properties: {target_names}")
    print(f"Model configuration: {config}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    config = DiTConfig(
        d_model=512,
        n_layers=6,
        n_heads=8,
        max_circuit_length=128
    )
    
    model = create_dit_model(config)
    print(f"DiT model created successfully!")
