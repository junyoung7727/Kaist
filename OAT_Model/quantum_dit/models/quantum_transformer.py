"""
Quantum Gate Prediction Transformer
Transformer model for predicting the next quantum gate based on circuit sequence and user requirements
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

# Import attention modules
from .attention import (
    QuantumMultiHeadAttention,
    GridPositionalAttention,
    RegisterFlowAttention,
    EntanglementAttention,
    SemanticAttention,
    AttentionFusionNetwork
)

# Import existing grid embedding
from ..encoding.Embeding import QuantumCircuitAttentionEmbedding

# Add quantumcommon to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))

from gates import QuantumGateRegistry


@dataclass
class QuantumTransformerConfig:
    """Quantum Transformer model configuration for next-gate prediction"""
    # Model architecture
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    
    # Quantum circuit specific
    max_circuit_length: int = 256
    max_qubits: int = 32
    max_parameters: int = 8  # Maximum number of parameters per gate
    
    # Training
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Positional encoding
    use_rotary_pe: bool = True
    use_learned_pe: bool = False
    
    # Activation
    use_swiglu: bool = True
    
    # Requirements conditioning
    requirement_dim: int = 128
    
    # Gate prediction
    gate_vocab_size: int = None  # Will be set from QuantumGateRegistry
    predict_parameters: bool = True
    predict_qubits: bool = True


class RequirementEncoder(nn.Module):
    """Encode user requirements into embeddings"""
    
    def __init__(self, config: QuantumTransformerConfig):
        super().__init__()
        self.config = config
        
        # Target property embeddings
        self.fidelity_embedding = nn.Linear(1, config.requirement_dim // 4)
        self.expressibility_embedding = nn.Linear(1, config.requirement_dim // 4)
        self.entanglement_embedding = nn.Linear(1, config.requirement_dim // 4)
        self.depth_embedding = nn.Linear(1, config.requirement_dim // 4)
        
        # Circuit constraints
        self.num_qubits_embedding = nn.Embedding(config.max_qubits + 1, config.requirement_dim // 4)
        self.max_depth_embedding = nn.Linear(1, config.requirement_dim // 4)
        self.two_qubit_ratio_embedding = nn.Linear(1, config.requirement_dim // 4)
        
        # Requirement fusion
        self.requirement_fusion = nn.Sequential(
            nn.Linear(config.requirement_dim * 2, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model)
        )
        
    def forward(self, requirements: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode user requirements
        
        Args:
            requirements: Dictionary containing:
                - target_fidelity: [batch_size, 1]
                - target_expressibility: [batch_size, 1] 
                - target_entanglement: [batch_size, 1]
                - target_depth: [batch_size, 1]
                - num_qubits: [batch_size]
                - max_depth: [batch_size, 1]
                - two_qubit_ratio: [batch_size, 1]
                
        Returns:
            Requirement embeddings [batch_size, d_model]
        """
        batch_size = requirements['target_fidelity'].size(0)
        
        # Target property embeddings
        fidelity_emb = self.fidelity_embedding(requirements['target_fidelity'])
        expressibility_emb = self.expressibility_embedding(requirements['target_expressibility'])
        entanglement_emb = self.entanglement_embedding(requirements['target_entanglement'])
        depth_emb = self.depth_embedding(requirements['target_depth'])
        
        target_emb = torch.cat([fidelity_emb, expressibility_emb, entanglement_emb, depth_emb], dim=-1)
        
        # Circuit constraint embeddings
        num_qubits_emb = self.num_qubits_embedding(requirements['num_qubits'])
        max_depth_emb = self.max_depth_embedding(requirements['max_depth'])
        ratio_emb = self.two_qubit_ratio_embedding(requirements['two_qubit_ratio'])
        
        constraint_emb = torch.cat([num_qubits_emb, max_depth_emb, ratio_emb], dim=-1)
        
        # Fuse requirements
        combined_emb = torch.cat([target_emb, constraint_emb], dim=-1)
        requirement_emb = self.requirement_fusion(combined_emb)
        
        return requirement_emb


class GateEncoder(nn.Module):
    """Encode quantum gates into embeddings"""
    
    def __init__(self, config: QuantumTransformerConfig):
        super().__init__()
        self.config = config
        
        # Gate type embedding
        self.gate_embedding = nn.Embedding(config.gate_vocab_size + 1, config.d_model // 2)  # +1 for padding
        
        # Qubit position embeddings
        self.qubit_embedding = nn.Embedding(config.max_qubits, config.d_model // 4)
        
        # Parameter embeddings
        self.parameter_embedding = nn.Linear(config.max_parameters, config.d_model // 4)
        
        # Gate fusion
        self.gate_fusion = nn.Linear(config.d_model, config.d_model)
        
    def forward(self, gates: torch.Tensor, qubits: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        """
        Encode gate sequence
        
        Args:
            gates: Gate type indices [batch_size, seq_len]
            qubits: Qubit indices [batch_size, seq_len, max_qubits]
            parameters: Gate parameters [batch_size, seq_len, max_parameters]
            
        Returns:
            Gate embeddings [batch_size, seq_len, d_model]
        """
        # Gate type embeddings
        gate_emb = self.gate_embedding(gates)  # [batch_size, seq_len, d_model//2]
        
        # Qubit embeddings (average over qubits for each gate)
        qubit_emb = self.qubit_embedding(qubits)  # [batch_size, seq_len, max_qubits, d_model//4]
        qubit_emb = qubit_emb.mean(dim=2)  # [batch_size, seq_len, d_model//4]
        
        # Parameter embeddings
        param_emb = self.parameter_embedding(parameters)  # [batch_size, seq_len, d_model//4]
        
        # Combine embeddings
        combined_emb = torch.cat([gate_emb, qubit_emb, param_emb], dim=-1)
        gate_emb = self.gate_fusion(combined_emb)
        
        return gate_emb


class RotaryPositionalEmbedding(nn.Module):
    """RoPE (Rotary Positional Embedding) for sequence modeling"""
    
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
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]
            
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor):
        """Apply rotary positional embedding to queries and keys"""
        seq_len = q.size(-2)
        self._update_cos_sin_tables(seq_len, q.device)
        
        cos, sin = self._cos_cached[:, :, :seq_len, :], self._sin_cached[:, :, :seq_len, :]
        
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)
            
        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        
        return q_rot, k_rot


class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Transformer block with quantum-aware attention and modern optimizations"""
    
    def __init__(self, config: QuantumTransformerConfig):
        super().__init__()
        self.config = config
        
        # Pre-Layer Normalization
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Multi-head attention with RoPE
        self.attention = nn.MultiheadAttention(
            config.d_model, 
            config.n_heads, 
            dropout=config.dropout,
            batch_first=True
        )
        
        # Rotary positional embedding
        if config.use_rotary_pe:
            self.rope = RotaryPositionalEmbedding(config.d_model // config.n_heads)
        
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
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of transformer block
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [seq_len, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Pre-norm
        x_norm = self.norm1(x)
        
        # Multi-head attention with RoPE
        if self.config.use_rotary_pe and hasattr(self, 'rope'):
            # Apply RoPE to queries and keys
            batch_size, seq_len, d_model = x_norm.shape
            head_dim = d_model // self.config.n_heads
            
            # Reshape for RoPE
            q = x_norm.view(batch_size, seq_len, self.config.n_heads, head_dim)
            k = x_norm.view(batch_size, seq_len, self.config.n_heads, head_dim)
            
            # Apply RoPE
            q, k = self.rope.apply_rotary_pos_emb(q, k)
            
            # Reshape back
            q = q.view(batch_size, seq_len, d_model)
            k = k.view(batch_size, seq_len, d_model)
            v = x_norm
            
            attn_out, _ = self.attention(q, k, v, attn_mask=mask)
        else:
            attn_out, _ = self.attention(x_norm, x_norm, x_norm, attn_mask=mask)
        
        # Residual connection
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual
        x = x + self.dropout(self.ffn(self.norm2(x)))
        
        return x


class QuantumTransformer(nn.Module):
    """
    Quantum Gate Prediction Transformer
    
    Predicts the next quantum gate based on:
    1. Current circuit sequence
    2. User requirements (target properties)
    3. Circuit constraints
    """
    
    def __init__(self, config: QuantumTransformerConfig):
        super().__init__()
        self.config = config
        
        # Initialize gate registry
        gate_registry = QuantumGateRegistry()
        self.gate_vocab = gate_registry.get_gate_vocab()
        self.config.gate_vocab_size = len(self.gate_vocab)
        
        # Create gate name to index mapping
        self.gate_to_idx = {gate: idx for idx, gate in enumerate(self.gate_vocab)}
        self.idx_to_gate = {idx: gate for gate, idx in self.gate_to_idx.items()}
        
        # Requirement encoder
        self.requirement_encoder = RequirementEncoder(config)
        
        # Gate encoder
        self.gate_encoder = GateEncoder(config)
        
        # Positional encoding
        if config.use_learned_pe:
            self.pos_embedding = nn.Parameter(torch.randn(1, config.max_circuit_length, config.d_model))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Output heads
        self.gate_predictor = nn.Linear(config.d_model, config.gate_vocab_size)
        
        if config.predict_qubits:
            self.qubit_predictor = nn.Linear(config.d_model, config.max_qubits)
            
        if config.predict_parameters:
            self.parameter_predictor = nn.Linear(config.d_model, config.max_parameters)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights with proper scaling"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
        
    def forward(self, 
                gates: torch.Tensor,
                qubits: torch.Tensor,
                parameters: torch.Tensor,
                requirements: Dict[str, torch.Tensor],
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for next-gate prediction
        
        Args:
            gates: Gate type indices [batch_size, seq_len]
            qubits: Qubit indices [batch_size, seq_len, max_qubits]
            parameters: Gate parameters [batch_size, seq_len, max_parameters]
            requirements: User requirements dictionary
            mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary containing:
                - gate_logits: [batch_size, seq_len, gate_vocab_size]
                - qubit_logits: [batch_size, seq_len, max_qubits] (if predict_qubits)
                - parameter_logits: [batch_size, seq_len, max_parameters] (if predict_parameters)
        """
        batch_size, seq_len = gates.shape
        
        # Encode requirements
        req_emb = self.requirement_encoder(requirements)  # [batch_size, d_model]
        req_emb = req_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, d_model]
        
        # Encode gates
        gate_emb = self.gate_encoder(gates, qubits, parameters)  # [batch_size, seq_len, d_model]
        
        # Combine gate and requirement embeddings
        x = gate_emb + req_emb
        
        # Add positional encoding
        if self.config.use_learned_pe:
            x = x + self.pos_embedding[:, :seq_len, :]
        
        # Create causal mask
        if mask is None:
            mask = self.create_causal_mask(seq_len, x.device)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Predictions
        outputs = {}
        outputs['gate_logits'] = self.gate_predictor(x)
        
        if self.config.predict_qubits:
            outputs['qubit_logits'] = self.qubit_predictor(x)
            
        if self.config.predict_parameters:
            outputs['parameter_logits'] = self.parameter_predictor(x)
        
        return outputs
    
    def generate_next_gate(self, 
                          gates: torch.Tensor,
                          qubits: torch.Tensor,
                          parameters: torch.Tensor,
                          requirements: Dict[str, torch.Tensor],
                          temperature: float = 1.0,
                          top_k: int = None,
                          top_p: float = None) -> Tuple[int, List[int], List[float]]:
        """
        Generate next gate using sampling
        
        Args:
            gates: Current gate sequence [1, seq_len]
            qubits: Current qubit sequence [1, seq_len, max_qubits]
            parameters: Current parameter sequence [1, seq_len, max_parameters]
            requirements: User requirements
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            
        Returns:
            Tuple of (gate_idx, qubit_indices, parameters)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(gates, qubits, parameters, requirements)
            
            # Get logits for the last position
            gate_logits = outputs['gate_logits'][0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = gate_logits < torch.topk(gate_logits, top_k)[0][..., -1, None]
                gate_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(gate_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                gate_logits[indices_to_remove] = float('-inf')
            
            # Sample gate
            gate_probs = F.softmax(gate_logits, dim=-1)
            gate_idx = torch.multinomial(gate_probs, 1).item()
            
            # Predict qubits and parameters if enabled
            predicted_qubits = []
            predicted_params = []
            
            if self.config.predict_qubits and 'qubit_logits' in outputs:
                qubit_logits = outputs['qubit_logits'][0, -1, :]
                qubit_probs = F.softmax(qubit_logits / temperature, dim=-1)
                # Sample based on gate requirements (this is simplified)
                gate_name = self.idx_to_gate[gate_idx]
                # For now, just sample top qubits
                predicted_qubits = torch.topk(qubit_probs, k=2).indices.tolist()
            
            if self.config.predict_parameters and 'parameter_logits' in outputs:
                param_logits = outputs['parameter_logits'][0, -1, :]
                predicted_params = param_logits.tolist()
        
        return gate_idx, predicted_qubits, predicted_params
    
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())


def create_quantum_transformer(config: QuantumTransformerConfig = None) -> QuantumTransformer:
    """Factory function to create Quantum Transformer model"""
    if config is None:
        config = QuantumTransformerConfig()
    
    model = QuantumTransformer(config)
    
    print(f"Created Quantum Transformer with {model.get_num_params():,} parameters")
    print(f"Gate vocabulary size: {config.gate_vocab_size}")
    print(f"Model configuration: {config}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    config = QuantumTransformerConfig(
        d_model=256,
        n_layers=4,
        n_heads=4,
        max_circuit_length=64,
        max_qubits=8
    )
    
    model = create_quantum_transformer(config)
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    
    # Dummy inputs
    gates = torch.randint(0, config.gate_vocab_size, (batch_size, seq_len))
    qubits = torch.randint(0, config.max_qubits, (batch_size, seq_len, config.max_qubits))
    parameters = torch.randn(batch_size, seq_len, config.max_parameters)
    
    requirements = {
        'target_fidelity': torch.tensor([[0.8], [0.9]]),
        'target_expressibility': torch.tensor([[0.5], [0.7]]),
        'target_entanglement': torch.tensor([[0.3], [0.4]]),
        'target_depth': torch.tensor([[20.0], [30.0]]),
        'num_qubits': torch.tensor([4, 6]),
        'max_depth': torch.tensor([[50.0], [60.0]]),
        'two_qubit_ratio': torch.tensor([[0.3], [0.4]])
    }
    
    outputs = model(gates, qubits, parameters, requirements)
    print(f"Gate logits shape: {outputs['gate_logits'].shape}")
    if 'qubit_logits' in outputs:
        print(f"Qubit logits shape: {outputs['qubit_logits'].shape}")
    if 'parameter_logits' in outputs:
        print(f"Parameter logits shape: {outputs['parameter_logits'].shape}")
