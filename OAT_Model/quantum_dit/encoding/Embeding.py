"""
수정된 양자 회로 그리드-그래프 구조적 어텐션 임베딩 시스템
차원 설계 문제를 해결한 버전
"""

# 표준 라이브러리
import os
import sys
import csv
import time
import math
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

# 써드파티 라이브러리
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Add quantumcommon to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))

from gates import QuantumGateRegistry

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
            self._cos_cached = emb.cos()  # [seq_len, d_model]
            self._sin_cached = emb.sin()  # [seq_len, d_model]
    
    def apply_grid_positional_encoding(self, x: torch.Tensor, grid_positions: torch.Tensor) -> torch.Tensor:
        """Apply 2D grid positional encoding for quantum circuits
        
        Args:
            x: Input tensor [seq_len, d_model]
            grid_positions: Grid positions [seq_len, 2] (time, qubit)
            
        Returns:
            Position encoded tensor [seq_len, d_model]
        """
        seq_len, d_model = x.shape
        device = x.device
        
        # 2D 그리드 위치 인코딩 생성
        pos_encoding = torch.zeros(seq_len, d_model, device=device)
        
        # 시간 축 인코딩 (d_model의 절반)
        time_dim = d_model // 2
        for i in range(time_dim):
            freq = 1.0 / (10000 ** (2 * i / time_dim))
            pos_encoding[:, 2*i] = torch.sin(grid_positions[:, 0] * freq)
            pos_encoding[:, 2*i + 1] = torch.cos(grid_positions[:, 0] * freq)
        
        # 큐빗 축 인코딩 (d_model의 나머지 절반)
        qubit_dim = d_model - time_dim
        for i in range(qubit_dim // 2):
            freq = 1.0 / (10000 ** (2 * i / qubit_dim))
            pos_encoding[:, time_dim + 2*i] = torch.sin(grid_positions[:, 1] * freq)
            if time_dim + 2*i + 1 < d_model:
                pos_encoding[:, time_dim + 2*i + 1] = torch.cos(grid_positions[:, 1] * freq)
        
        # 위치 인코딩을 임베딩에 더하기 (표준 Transformer 방식)
        return x + pos_encoding


class QuantumCircuitEmbedding(nn.Module):
    """양자 회로를 위한 다층 구조적 어텐션 임베딩 (차원 수정 버전)"""
    
    def __init__(self, 
                 d_model: int = 256,
                 n_heads: int = 8,
                 max_grid_size: int = 64,
                 max_qubits: int = 32,
                 dropout: float = 0.1,
                 use_rotary_pe: bool = True):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rotary_pe = use_rotary_pe
        
        # RoPE 초기화 (positional encoding 담당)
        if use_rotary_pe:
            self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)
        
        # 차원 안전성 체크
        assert d_model >= 64, f"d_model은 최소 64 이상이어야 합니다. 현재: {d_model}"
        assert d_model % 8 == 0, f"d_model은 8의 배수여야 합니다. 현재: {d_model}"
        
        # 1. 게이트 타입 임베딩
        self.gate_registry = QuantumGateRegistry()
        self.gate_vocab = self.gate_registry.get_gate_vocab()
        
        # 각 임베딩 차원을 명시적으로 계산 (RoPE가 positional encoding 담당)
        self.gate_dim = d_model // 2        # 예: 128
        self.role_dim = d_model // 4        # 예: 64
        self.param_dim = d_model // 4 - 1   # 예: 63 (parameter value만, indicator는 scalar)
                
        # 차원 검증 (positional encoding 제거됨, parameter indicator는 scalar +1)
        total_dim = self.gate_dim + self.role_dim + self.param_dim + 1  # +1 for scalar indicator
        print(f"임베딩 차원 분배: gate={self.gate_dim}, "
              f"role={self.role_dim}, param={self.param_dim}, indicator=1(scalar)")
        print(f"총 차원: {total_dim}, 목표 차원: {d_model}")
        
        # 2. 게이트 타입 임베딩
        self.gate_embedding = nn.Embedding(len(self.gate_vocab), self.gate_dim)
        
        # 4. 역할 임베딩
        self.role_embedding = nn.Embedding(4, self.role_dim)  # single, control, target, [PAD]
        
        # 3. 파라미터 임베딩 (indicator는 scalar로 처리)
        self.param_projection = nn.Linear(1, self.param_dim)
        
        # 7. 차원 조정 레이어 (총 차원이 d_model과 다를 경우)
        if total_dim != d_model:
            self.dimension_adapter = nn.Linear(total_dim, d_model)
        else:
            self.dimension_adapter = nn.Identity()
            
        
    def forward(self, encoded_circuit: Dict) -> Dict[str, torch.Tensor]:
        """
        순수 임베딩만 제공 (어텐션은 DiT 블록에서 처리)
        
        Args:
            encoded_circuit: GridGraphEncoder의 출력
                - nodes: List[Dict] - 노드 정보
                - edges: List[Dict] - 엣지 정보  
                - grid_shape: Tuple[int, int] - 그리드 크기
        
        Returns:
            순수 임베딩 결과
        """
        if not encoded_circuit.get('nodes'):
            return {
                'node_embeddings': torch.zeros(0, self.d_model),
                'circuit_embedding': torch.zeros(self.d_model)
            }
        
        # 순수 노드 임베딩 생성
        node_embeddings = self._embed_nodes(encoded_circuit['nodes'])
        
        # 2D 그리드 위치 인코딩 적용 (양자 회로 특화)
        if self.use_rotary_pe and len(node_embeddings) > 0:
            # 노드들의 그리드 위치 추출
            grid_positions = torch.tensor([
                node.get('grid_position', [0, 0]) for node in encoded_circuit['nodes']
            ], dtype=torch.float32, device=node_embeddings.device)
            
            node_embeddings = self.rotary_emb.apply_grid_positional_encoding(
                node_embeddings, grid_positions
            )
        
        return {
            'node_embeddings': node_embeddings,
            'circuit_embedding': torch.mean(node_embeddings, dim=0) if len(node_embeddings) > 0 else torch.zeros(self.d_model)
        }
    
    def _embed_nodes(self, nodes: List[Dict]) -> torch.Tensor:
        """노드 특성들을 임베딩으로 변환 (차원 안전성 보장)"""
        if not nodes:
            return torch.zeros(0, self.d_model)
            
        embeddings = []
        
        for node in nodes:
            # 1. 게이트 타입 임베딩
            gate_name = node.get('gate_name', '[EMPTY]')
            gate_idx = self.gate_vocab.get(gate_name, self.gate_vocab['[EMPTY]'])
            gate_emb = self.gate_embedding(torch.tensor(gate_idx))  # [gate_dim]
            
            # 2. 역할 임베딩 (RoPE가 positional encoding 담당)
            role = node.get('role', 'single')
            role_dict = {'single': 0, 'control': 1, 'target': 2, '[PAD]': 3}
            role_emb = self.role_embedding(torch.tensor(role_dict.get(role, 0)))  # [role_dim]
            
            # 3. 파라미터 임베딩 (단순화된 버전)
            param_val = node.get('parameter_value', 0.0)
            # 스칼라 입력을 명시적으로 1차원으로 만들어서 Linear 레이어에 전달
            param_input = torch.tensor(param_val, dtype=torch.float).unsqueeze(0)  # [1]
            param_emb = self.param_projection(param_input).squeeze(0)  # [param_dim]
            
            # 파라미터 존재 여부를 단순한 0/1 scalar로 처리
            has_param = float(node.get('has_parameter', 0.0))  # 0.0 or 1.0
            param_indicator_scalar = torch.tensor([has_param])  # [1] scalar tensor
            
            # 4. 모든 임베딩 결합 (positional embedding은 RoPE가 담당, indicator는 scalar)
            combined = torch.cat([
                gate_emb,                # [gate_dim]
                role_emb,                # [role_dim] 
                param_emb,               # [param_dim]
                param_indicator_scalar,  # [1] scalar indicator
            ], dim=-1)                   # [total_dim]
            
            # 7. 차원 조정
            final_emb = self.dimension_adapter(combined)  # [d_model]
            
            embeddings.append(final_emb)
        
        return torch.stack(embeddings)  # [num_nodes, d_model]
    
    def _create_grid_structure(self, encoded_circuit: Dict) -> Dict[str, torch.Tensor]:
        """그리드 구조 정보 생성"""
        nodes = encoded_circuit['nodes']
        grid_shape = encoded_circuit['grid_shape']
        
        if not nodes:
            return {
                'positions': torch.empty(0, 2),
                'distance_matrix': torch.empty(0, 0),
                'grid_shape': torch.tensor(grid_shape),
                'node_to_idx': {}
            }
        
        # 노드 ID to 인덱스 매핑
        node_to_idx = {node['id']: i for i, node in enumerate(nodes)}
        
        # 그리드 포지션 매트릭스
        positions = torch.tensor([node['grid_position'] for node in nodes])
        
        # 거리 매트릭스 (연결도 기반 hop 수)
        dist_matrix = self._compute_hop_distance_matrix(nodes, encoded_circuit['edges'], node_to_idx)
        
        return {
            'positions': positions,
            'distance_matrix': dist_matrix,
            'grid_shape': torch.tensor(grid_shape),
            'node_to_idx': node_to_idx
        }
    
    def _compute_hop_distance_matrix(self, nodes: List[Dict], edges: List[Dict], node_to_idx: Dict) -> torch.Tensor:
        """
        연결도 기반 hop 수를 계산하여 거리 매트릭스 생성
        그리드 좌표 기반 엣지 구조에서 최단 hop 수를 계산
        연결이 없으면 0, 연결이 있으면 최단 hop 수를 반환
        """
        n_nodes = len(nodes)
        if n_nodes == 0:
            return torch.empty(0, 0)
        
        # 무한대를 나타내는 수 (연결되지 않은 경우)
        INF = float('inf')
        
        # 초기 거리 매트릭스: 모두 무한대로 설정
        dist_matrix = [[INF for _ in range(n_nodes)] for _ in range(n_nodes)]
        
        # 자기 자신과의 거리는 0
        for i in range(n_nodes):
            dist_matrix[i][i] = 0
        
        # 그리드 좌표를 노드 ID로 변환하는 매핑 생성
        grid_to_node_id = {}
        for node in nodes:
            if 'grid_position' in node:
                grid_pos = tuple(node['grid_position'])  # [parallel_order, qubit_idx] -> (parallel_order, qubit_idx)
                grid_to_node_id[grid_pos] = node['id']
        
        # 엣지에서 직접 연결 정보 추출
        for edge in edges:
            edge_type = edge.get('type', '')
            
            # 다양한 연결 타입에 따라 hop 수 계산
            if edge_type in ['REGISTER_CONNECTION', 'ENTANGLE_CONNECTION']:
                # source와 target 그리드 좌표 가져오기
                src_grid = edge.get('source')  # [parallel_order, qubit_idx]
                tgt_grid = edge.get('target')   # [parallel_order, qubit_idx]
                
                if src_grid is not None and tgt_grid is not None:
                    # 그리드 좌표를 노드 ID로 변환
                    src_node_id = grid_to_node_id.get(tuple(src_grid))
                    tgt_node_id = grid_to_node_id.get(tuple(tgt_grid))
                    
                    if src_node_id and tgt_node_id:
                        # 노드 ID를 인덱스로 변환
                        src_idx = node_to_idx.get(src_node_id)
                        tgt_idx = node_to_idx.get(tgt_node_id)
                        
                        if src_idx is not None and tgt_idx is not None:
                            # 직접 연결은 hop 수 1
                            dist_matrix[src_idx][tgt_idx] = 1
                            dist_matrix[tgt_idx][src_idx] = 1  # 양방향
        
        # Floyd-Warshall 알고리즘으로 최단 경로 계산
        for k in range(n_nodes):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if dist_matrix[i][k] != INF and dist_matrix[k][j] != INF:
                        dist_matrix[i][j] = min(dist_matrix[i][j], dist_matrix[i][k] + dist_matrix[k][j])
        
        # 무한대 값을 -1으로 대체 (연결이 없는 경우)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if dist_matrix[i][j] == INF:
                    dist_matrix[i][j] = -1
        
        # torch 텐서로 변환
        return torch.tensor(dist_matrix, dtype=torch.float32)