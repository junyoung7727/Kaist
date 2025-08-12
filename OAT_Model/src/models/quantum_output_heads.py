"""
Multi-task Output Heads for Quantum Decision Transformer
Handles gate type classification, qubit selection, and parameter regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional
import math

# Add quantumcommon to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))

from gates import QuantumGateRegistry


class QuantumOutputHeads(nn.Module):
    """양자 게이트 생성을 위한 멀티태스크 출력 헤드"""
    
    def __init__(self, 
                 d_model: int, 
                 n_gate_types: int, 
                 n_qubits: int,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_gate_types = n_gate_types
        self.n_qubits = n_qubits
        
        # Gate type classification head
        self.gate_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_gate_types)
        )
        
        # Qubit selection heads (first and second qubits)
        self.qubit1_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_qubits)
        )
        
        self.qubit2_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_qubits)
        )
        
        # Parameter regression head (for parameterized gates)
        self.param_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Gate validity head (whether this position should have a gate)
        self.validity_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with proper scaling"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, hidden_states: torch.Tensor, 
                extract_actions_only: bool = False,
                token_types: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-task prediction
        
        Args:
            hidden_states: [batch, seq_len, d_model] or [batch, seq_len*3, d_model]
            extract_actions_only: If True, only process action tokens (type 2)
            token_types: [batch, seq_len*3] token type indicators
        
        Returns:
            Dictionary with prediction logits and values
        """
        
        if extract_actions_only and token_types is not None:
            # Extract only action embeddings (token type 2)
            action_mask = (token_types == 2)  # [batch, seq_len*3]
            
            # Reshape to get action positions
            batch_size, full_seq_len = token_types.shape
            action_positions = action_mask.nonzero(as_tuple=False)  # [num_actions, 2]
            
            if action_positions.size(0) == 0:
                # No actions found, return empty predictions
                return self._empty_predictions(batch_size, 0, hidden_states.device)
            
            # Extract action embeddings
            action_embeddings = hidden_states[action_positions[:, 0], action_positions[:, 1]]  # [num_actions, d_model]
            
            # Process predictions
            predictions = self._process_embeddings(action_embeddings)
            
            # Reshape back to batch format
            seq_len = full_seq_len // 3
            predictions = self._reshape_to_batch(predictions, action_positions, batch_size, seq_len)
            
        else:
            # Process all embeddings (standard mode)
            batch_size, seq_len, d_model = hidden_states.shape
            embeddings = hidden_states.view(-1, d_model)  # [batch*seq_len, d_model]
            
            predictions = self._process_embeddings(embeddings)
            
            # Reshape back to batch format
            for key in predictions:
                if predictions[key].dim() == 1:
                    predictions[key] = predictions[key].view(batch_size, seq_len)
                else:
                    predictions[key] = predictions[key].view(batch_size, seq_len, -1)
        
        return predictions
    
    def _process_embeddings(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process embeddings through all heads"""
        # Gate type classification
        gate_logits = self.gate_head(embeddings)  # [N, n_gate_types]
        
        # Qubit selection
        qubit1_logits = self.qubit1_head(embeddings)  # [N, n_qubits]
        qubit2_logits = self.qubit2_head(embeddings)  # [N, n_qubits]
        
        # Parameter regression (normalized to [0, 2π])
        param_raw = self.param_head(embeddings)  # [N, 1]
        param_values = torch.sigmoid(param_raw) * 2 * math.pi
        
        # Gate validity
        validity_raw = self.validity_head(embeddings)  # [N, 1]
        validity_logits = validity_raw.squeeze(-1)  # [N]
        
        return {
            'gate_logits': gate_logits,
            'qubit1_logits': qubit1_logits,
            'qubit2_logits': qubit2_logits,
            'param_values': param_values,
            'validity_logits': validity_logits
        }
    
    def _reshape_to_batch(self, predictions: Dict[str, torch.Tensor], 
                         action_positions: torch.Tensor,
                         batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
        """Reshape action predictions back to batch format"""
        device = predictions['gate_logits'].device
        
        # Initialize batch tensors
        batch_predictions = {
            'gate_logits': torch.zeros(batch_size, seq_len, self.n_gate_types, device=device),
            'qubit1_logits': torch.zeros(batch_size, seq_len, self.n_qubits, device=device),
            'qubit2_logits': torch.zeros(batch_size, seq_len, self.n_qubits, device=device),
            'param_values': torch.zeros(batch_size, seq_len, 1, device=device),
            'validity_logits': torch.zeros(batch_size, seq_len, device=device)
        }
        
        # Fill in action predictions
        for i, (batch_idx, pos_idx) in enumerate(action_positions):
            step_idx = pos_idx // 3  # Convert from full sequence position to step
            if step_idx < seq_len:
                batch_predictions['gate_logits'][batch_idx, step_idx] = predictions['gate_logits'][i]
                batch_predictions['qubit1_logits'][batch_idx, step_idx] = predictions['qubit1_logits'][i]
                batch_predictions['qubit2_logits'][batch_idx, step_idx] = predictions['qubit2_logits'][i]
                batch_predictions['param_values'][batch_idx, step_idx] = predictions['param_values'][i]
                batch_predictions['validity_logits'][batch_idx, step_idx] = predictions['validity_logits'][i]
        
        return batch_predictions
    
    def _empty_predictions(self, batch_size: int, seq_len: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Return empty predictions when no actions are found"""
        return {
            'gate_logits': torch.zeros(batch_size, seq_len, self.n_gate_types, device=device),
            'qubit1_logits': torch.zeros(batch_size, seq_len, self.n_qubits, device=device),
            'qubit2_logits': torch.zeros(batch_size, seq_len, self.n_qubits, device=device),
            'param_values': torch.zeros(batch_size, seq_len, 1, device=device),
            'validity_logits': torch.zeros(batch_size, seq_len, device=device)
        }
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor],
                    mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            mask: Optional mask for valid positions
        
        Returns:
            Dictionary with individual and total losses
        """
        losses = {}
        
        # Flatten for loss computation
        if mask is not None:
            # Apply mask to select valid positions
            valid_positions = mask.view(-1)
            
            gate_pred = predictions['gate_logits'].view(-1, self.n_gate_types)[valid_positions]
            gate_target = targets['target_gates'].view(-1)[valid_positions]
            
            qubit1_pred = predictions['qubit1_logits'].view(-1, self.n_qubits)[valid_positions]
            qubit1_target = targets['target_qubit1'].view(-1)[valid_positions]
            
            qubit2_pred = predictions['qubit2_logits'].view(-1, self.n_qubits)[valid_positions]
            qubit2_target = targets['target_qubit2'].view(-1)[valid_positions]
            
            param_pred = predictions['param_values'].view(-1, 1)[valid_positions]
            param_target = targets['target_params'].view(-1, 1)[valid_positions]
            
            if 'validity_logits' in predictions:
                validity_pred = predictions['validity_logits'].view(-1)[valid_positions]
                validity_target = targets.get('target_validity', torch.ones_like(validity_pred))
        else:
            # Use all positions
            gate_pred = predictions['gate_logits'].view(-1, self.n_gate_types)
            gate_target = targets['target_gates'].view(-1)
            
            qubit1_pred = predictions['qubit1_logits'].view(-1, self.n_qubits)
            qubit1_target = targets['target_qubit1'].view(-1)
            
            qubit2_pred = predictions['qubit2_logits'].view(-1, self.n_qubits)
            qubit2_target = targets['target_qubit2'].view(-1)
            
            param_pred = predictions['param_values'].view(-1, 1)
            param_target = targets['target_params'].view(-1, 1)
            
            if 'validity_logits' in predictions:
                validity_pred = predictions['validity_logits'].view(-1)
                validity_target = targets.get('target_validity', torch.ones_like(validity_pred))
        
        # Gate type classification loss
        losses['gate_loss'] = F.cross_entropy(gate_pred, gate_target, ignore_index=-1)
        
        # Qubit selection losses
        losses['qubit1_loss'] = F.cross_entropy(qubit1_pred, qubit1_target, ignore_index=-1)
        losses['qubit2_loss'] = F.cross_entropy(qubit2_pred, qubit2_target, ignore_index=-1)
        
        # Parameter regression loss
        losses['param_loss'] = F.mse_loss(param_pred, param_target)
        
        # Validity loss (optional)
        if 'validity_logits' in predictions:
            losses['validity_loss'] = F.binary_cross_entropy_with_logits(validity_pred, validity_target)
        else:
            losses['validity_loss'] = torch.tensor(0.0, device=gate_pred.device)
        
        # Combined qubit loss
        losses['qubit_loss'] = (losses['qubit1_loss'] + losses['qubit2_loss']) / 2
        
        # Total weighted loss
        losses['total_loss'] = (
            losses['gate_loss'] + 
            0.5 * losses['qubit_loss'] + 
            0.3 * losses['param_loss'] + 
            0.1 * losses['validity_loss']
        )
        
        return losses
    
    def sample_actions(self, predictions: Dict[str, torch.Tensor],
                      temperature: float = 1.0,
                      top_k: Optional[int] = None,
                      top_p: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Sample actions from predictions
        
        Args:
            predictions: Model predictions
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
        
        Returns:
            Sampled actions
        """
        def sample_from_logits(logits: torch.Tensor) -> torch.Tensor:
            logits = logits / temperature
            
            if top_k is not None:
                top_k_actual = min(top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k_actual)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).squeeze(-1)
        
        # Sample from each head
        sampled_gates = sample_from_logits(predictions['gate_logits'])
        sampled_qubit1 = sample_from_logits(predictions['qubit1_logits'])
        sampled_qubit2 = sample_from_logits(predictions['qubit2_logits'])
        
        # Use predicted parameters directly (already continuous)
        sampled_params = predictions['param_values']
        
        # Sample validity
        validity_probs = torch.sigmoid(predictions['validity_logits'])
        sampled_validity = torch.bernoulli(validity_probs)
        
        return {
            'gates': sampled_gates,
            'qubit1': sampled_qubit1,
            'qubit2': sampled_qubit2,
            'params': sampled_params,
            'validity': sampled_validity
        }


if __name__ == "__main__":
    # Test the output heads
    d_model = 512
    n_gate_types = 20  # 🔧 FIXED: 통일된 게이트 타입 수
    n_qubits = 8
    batch_size = 4
    seq_len = 16
    
    # Create output heads
    output_heads = QuantumOutputHeads(d_model, n_gate_types, n_qubits)
    
    # Test standard mode
    hidden_states = torch.randn(batch_size, seq_len, d_model)
    predictions = output_heads(hidden_states)
    
    print("Standard mode predictions:")
    for key, value in predictions.items():
        print(f"  {key}: {value.shape}")
    
    # Test decision transformer mode (with token types)
    full_seq_len = seq_len * 3  # [target, state, action] triplets
    hidden_states_dt = torch.randn(batch_size, full_seq_len, d_model)
    token_types = torch.tensor([[0, 1, 2] * seq_len] * batch_size)  # Repeat pattern
    
    predictions_dt = output_heads(hidden_states_dt, extract_actions_only=True, token_types=token_types)
    
    print("\nDecision Transformer mode predictions:")
    for key, value in predictions_dt.items():
        print(f"  {key}: {value.shape}")
    
    # Test sampling
    sampled = output_heads.sample_actions(predictions, temperature=0.8, top_k=5)
    
    print("\nSampled actions:")
    for key, value in sampled.items():
        print(f"  {key}: {value.shape}")
    
    print("\nQuantum Output Heads test completed!")
