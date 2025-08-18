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

# Import unified debug utilities
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from debug_utils import dt_debug_log, dt_debug_tensor

class ActionTargetBuilder:
    """ 액션 타겟 생성 전용 클래스 - 단일 책임 원칙"""
    
    @staticmethod
    def build_from_grid(grid_matrix_data, batch_size: int) -> Dict[str, torch.Tensor]:
        """그리드 매트릭스 데이터로부터 액션 타겟 생성"""
        if 'gates' not in grid_matrix_data:
            return ActionTargetBuilder._create_empty_targets(batch_size, 0)
        
        gates = grid_matrix_data['gates']
        num_gates = len(gates)
        
        # 벡터화된 타겟 생성 (단일 샘플용)
        gate_targets = torch.zeros(num_gates, dtype=torch.long)
        # NEW: 2큐빗 게이트 전용 형태로 변경 [qubit1, qubit2]
        position_targets = torch.full((num_gates, 2), -1, dtype=torch.long)
        parameter_targets = torch.zeros(num_gates, dtype=torch.float)
        
        # 각 게이트별로 타겟 생성 (배치는 현재 1개만 처리)
        for gate_idx, gate in enumerate(gates):
            if isinstance(gate, dict):
                # 게이트 타입 설정
                if 'gate_index' in gate:
                    gate_id = gate['gate_index']
                    if 0 <= gate_id < 20:  # 유효한 게이트만 (EOS/PAD 제외)
                        gate_targets[gate_idx] = gate_id
                
                # 큐빗 위치 설정 - 2큐빗 형태 지원
                if 'qubits' in gate and gate['qubits'] is not None:
                    qubits = gate['qubits']
                    if isinstance(qubits, list) and len(qubits) > 0:
                        # 첫 번째 큐빗 설정
                        if len(qubits) >= 1 and qubits[0] >= 0:
                            position_targets[gate_idx, 0] = qubits[0]
                        
                        # 두 번째 큐빗 설정
                        if len(qubits) >= 2 and qubits[1] >= 0:
                            position_targets[gate_idx, 1] = qubits[1]
                        elif len(qubits) == 1 and qubits[0] >= 0:
                            # 1큐빗 게이트의 경우: 같은 큐빗을 두 번 사용
                            position_targets[gate_idx, 1] = qubits[0]
                
                # 파라미터 설정 (그리드 데이터에서 직접 추출)
                if 'parameter_value' in gate and gate['parameter_value'] is not None:
                    param_val = gate['parameter_value']
                    if not (isinstance(param_val, float) and math.isnan(param_val)):
                        parameter_targets[gate_idx] = float(param_val)

        
        return {
            'gate_targets': gate_targets,
            'position_targets': position_targets,
            'parameter_targets': parameter_targets
        }
    
    @staticmethod
    def _create_empty_targets(batch_size: int, num_gates: int) -> Dict[str, torch.Tensor]:
        """빈 타겟 생성 (패딩용)"""
        return {
            'gate_targets': torch.zeros(num_gates, dtype=torch.long),
            'position_targets': torch.full((num_gates, 2), -1, dtype=torch.long),  # 2큐빗 형태로 변경
            'parameter_targets': torch.zeros(num_gates, dtype=torch.float)
        }

class QuantumGateSequenceEmbedding(nn.Module):
    def __init__(self, d_model: int = 512, n_gate_types: int = 20, max_pos: int = 1024, dropout: float = 0.1, device: str = 'cpu', property_prediction_model=None):
        """초기화
        
        Args:
            d_model: 모델 차원
            n_gate_types: 게이트 타입 수
            max_pos: 최대 위치
            dropout: 드롭아웃 비율
            device: 모델 디바이스
            property_prediction_model: 프로퍼티 예측 모델 (리워드 계산에 사용)
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_gate_types = n_gate_types
        self.max_pos = max_pos
        self.dropout = dropout
        self.property_prediction_model = property_prediction_model
        
        # 임베딩 레이어들 (정답레이블과 동일한 형태)
        # gate_type + position_vector + parameter
        gate_dim = d_model // 2      # 50% - 게이트 타입 (H, X, CNOT, RZ 등)
        position_dim = d_model // 4  # 25% - 포지션 벡터 (2큐빗 위치)
        param_dim = d_model - gate_dim - position_dim  # 나머지 25% - 파라미터
        
        self.gate_type_embed = nn.Embedding(n_gate_types, gate_dim)   # 게이트 타입 ID
        self.position_embed = nn.Linear(2, position_dim)              # 포지션 벡터 [qubit1, qubit2]
        self.param_embed = nn.Linear(1, param_dim)                    # 게이트 파라미터
        
        # 위치 인코딩 (학습 가능한 임베딩)
        self.positional_encoding = nn.Embedding(max_pos, d_model)
        
        # EOS (End-of-Sequence) 특수 토큰
        self.eos_embed = nn.Parameter(torch.randn(d_model))            # 큐빗 인덱스
        self.grid_position_embed = nn.Linear(2, d_model)              # (x, y) 좌표
        
        # Decision Transformer 컴포넌트들
        self.state_embed = nn.Linear(d_model, d_model)     # 상태 임베딩
        self.action_embed = nn.Linear(d_model, d_model)    # 액션 임베딩  
        self.reward_embed = nn.Linear(1, d_model)          # 리워드 임베딩
        self.return_embed = nn.Linear(1, d_model)          # Return-to-go 임베딩
        
        # 시퀀스 타입 임베딩 (state/action/reward 구분)
        self.type_embed = nn.Embedding(4, d_model)  # 0=state, 1=action, 2=reward, 3=return
        
        # 위치 인코딩
        self.register_buffer('pos_embed', self._create_positional_encoding())
        
        # 정규화
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def _create_positional_encoding(self) -> torch.Tensor:
        """올바른 위치 인코딩 생성"""
        pe = torch.zeros(self.max_pos, self.d_model)
        position = torch.arange(0, self.max_pos, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                           (-math.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, max_pos, d_model]
    
    def mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        시퀀스에 어텐션을 적용할 때, 자기보다 이후 스테이트의 게이트 배치를 가림
        
        Args:
            x: 입력 시퀀스 [batch_size, seq_len, hidden_dim]
        
        Returns:
            mask: 어텐션 마스크 [seq_len, seq_len] (bool)
        """
        seq_len = x.size(1)
        
        # 하삼각 마스크 생성 (causal mask)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        
        # 0은 마스킹, 1은 어텐션 허용
        # Decision Transformer에서는 현재와 이전 타임스텝만 참조 가능
        return mask

    def state(self, x: torch.Tensor, gate_type_indices: torch.Tensor = None) -> torch.Tensor:
        """
        에피소드 시퀀스에 대한 상태 임베딩
        
        Args:
            x: 에피소드 시퀀스 [episode_time_len, features]
            gate_type_indices: Long 타입 게이트 타입 인덱스 [episode_time_len]
        
        Returns:
            state: 인코딩된 상태 [episode_time_len, d_model]
        """
        # 🔍 CRITICAL FIX: x 차원 동적 처리
        device = x.device
        
        dt_debug_tensor("state_input", x, detailed=True)
        
        # 차원에 따른 동적 처리
        if x.dim() == 2:
            episode_time_len, features = x.shape
        elif x.dim() == 3:
            # [batch_or_time, episode_time_len_or_qubits, features] 형태
            # 실제로는 [time_steps, num_qubits, features] 형태일 가능성
            batch_or_time, episode_or_qubits, features = x.shape
            
            # 3D 텐서를 2D로 flatten
            x = x.view(-1, features)  # [batch_or_time * episode_or_qubits, features]
            episode_time_len, features = x.shape
        else:
            raise ValueError(f"Unsupported tensor dimensions in state method: {x.shape}")
        
        # 위치 인덱스 생성 (디바이스 일치 보장)
        position_indices = torch.arange(episode_time_len, device=device).long()
        # positional_encoding이 CUDA에 있는지 확인하고 인덱스를 같은 디바이스로 이동
        if next(self.positional_encoding.parameters()).device != device:
            position_indices = position_indices.to(next(self.positional_encoding.parameters()).device)
        position_emb = self.positional_encoding(position_indices)  # [episode_time_len, d_model]
        
        # 상태 인코딩 (새로운 형태: [gate_type_id, qubit1, qubit2, parameter_value])
        if features >= 4:
            # 피처 추출
            gate_type_ids = x[:, 0].long()      # [episode_time_len]
            positions = x[:, 1:3]               # [episode_time_len, 2] - [qubit1, qubit2]
            parameters = x[:, 3:4]              # [episode_time_len, 1]
            
            # 각각 임베딩 (디바이스 일치 보장)
            # embedding 레이어의 디바이스 직접 확인
            gate_embed_device = next(self.gate_type_embed.parameters()).device
            position_embed_device = next(self.position_embed.parameters()).device
            param_embed_device = next(self.param_embed.parameters()).device
            
            # 텐서를 각 임베딩 레이어의 디바이스로 명시적 이동
            gate_type_ids = gate_type_ids.to(gate_embed_device)
            positions = positions.to(position_embed_device)
            parameters = parameters.to(param_embed_device)
            
            gate_embedded = self.gate_type_embed(gate_type_ids)    # [episode_time_len, gate_dim]
            position_embedded = self.position_embed(positions)     # [episode_time_len, position_dim]
            param_embedded = self.param_embed(parameters)          # [episode_time_len, param_dim]
            
            # 임베딩 결합 (concatenation)
            state_encoded = torch.cat([
                gate_embedded, 
                position_embedded, 
                param_embedded
            ], dim=-1)  # [episode_time_len, d_model]
        
        # 위치 임베딩과 결합
        state = state_encoded + position_emb  # [episode_time_len, d_model]
        
        dt_debug_tensor("state_output", state, detailed=True)
        
        return state

    def action(self, episode_sequence: torch.Tensor, current_episode_time: int) -> torch.Tensor:
        """
        특정 에피소드타임에서의 액션 생성 (다음 에피소드타임의 게이트)
        미래 액션에 대한 마스킹 포함
        
        Args:
            episode_sequence: [episode_time_len, features] - 전체 에피소드 시퀀스
            current_episode_time: 현재 에피소드타임 인덱스
        
        Returns:
            action: 현재 에피소드타임에서의 액션 [d_model]
        """
        # 🔍 CRITICAL FIX: episode_sequence 차원 동적 처리
        device = episode_sequence.device
        
        dt_debug_tensor("action_input", episode_sequence, detailed=True)
        
        # 차원에 따른 동적 처리
        if episode_sequence.dim() == 2:
            episode_time_len, features = episode_sequence.shape
        elif episode_sequence.dim() == 3:
            # [batch_or_time, episode_time_len_or_qubits, features] 형태
            # 3D 텐서를 2D로 flatten
            batch_or_time, episode_or_qubits, features = episode_sequence.shape
            episode_sequence = episode_sequence.view(-1, features)  # [batch_or_time * episode_or_qubits, features]
            episode_time_len, features = episode_sequence.shape
        else:
            raise ValueError(f"Unsupported tensor dimensions in action method: {episode_sequence.shape}")
        
        if current_episode_time < episode_time_len - 1:
            # 다음 에피소드타임의 게이트를 액션으로 사용 (마스킹 없음)
            next_gate = episode_sequence[current_episode_time + 1].unsqueeze(0).unsqueeze(0)  # [1, 1, features]
            action_embedded = self._embed_episode_features(next_gate)  # [1, 1, d_model]
            action = self.action_embed(action_embedded).squeeze(0).squeeze(0)  # [d_model]
        else:
            # 마지막 에피소드타임: 비어있는 액션 (마스킹)
            # 디바이스 일관성을 위해 모델 디바이스 확인
            model_device = next(self.parameters()).device
            action = torch.zeros(self.d_model, device=model_device)
        
        dt_debug_tensor("action_output", action, detailed=True)
        
        return action
    
    def _embed_episode_features(self, episode_features: torch.Tensor) -> torch.Tensor:
        """
        에피소드타임 순서의 피처를 임베딩
        
        Args:
            episode_features: [batch, episode_time_len, features]
        
        Returns:
            embedded: [batch, episode_time_len, d_model]
        """
        batch_size, episode_time_len, features = episode_features.shape
        
        dt_debug_tensor("embed_episode_features_input", episode_features, detailed=True)
        
        if features >= 4:  # [gate_type_id, qubit1, qubit2, parameter_value, ...]
            # 각 피처 추출 (정답레이블과 동일한 형태)
            gate_type_ids = episode_features[:, :, 0].long()      # 게이트 타입 ID
            positions = episode_features[:, :, 1:3]               # 포지션 벡터 [qubit1, qubit2]
            parameters = episode_features[:, :, 3:4]              # 파라미터 값
            
            # 각각 임베딩 (디바이스 일치 보장)
            # embedding 레이어의 디바이스 직접 확인
            gate_embed_device = next(self.gate_type_embed.parameters()).device
            position_embed_device = next(self.position_embed.parameters()).device
            param_embed_device = next(self.param_embed.parameters()).device
            
            # 텐서를 각 임베딩 레이어의 디바이스로 명시적 이동
            gate_type_ids = gate_type_ids.to(gate_embed_device)
            positions = positions.to(position_embed_device)
            parameters = parameters.to(param_embed_device)
            
            gate_embedded = self.gate_type_embed(gate_type_ids)   # [batch, episode_time, gate_dim]
            position_embedded = self.position_embed(positions)    # [batch, episode_time, position_dim]
            param_embedded = self.param_embed(parameters)         # [batch, episode_time, param_dim]
            
            # 임베딩 결합 (concatenation)
            embedded = torch.cat([
                gate_embedded, 
                position_embedded, 
                param_embedded
            ], dim=-1)  # [batch, episode_time, d_model]
        else:
            # 다른 피처 차원의 경우 선형 변환
            linear_layer = nn.Linear(features, self.d_model).to(episode_features.device)
            embedded = linear_layer(episode_features)
        
        dt_debug_tensor("embed_episode_features_output", embedded, detailed=True)
        
        return embedded

    def reward(self, masked_state: torch.Tensor, action: torch.Tensor, current_episode_time: int, episode_time_len: int) -> torch.Tensor:
        """
        특정 에피소드타임에서의 리워드 생성
        미래 리워드에 대한 마스킹 포함
        
        Args:
            masked_state: 현재 에피소드타임에서의 마스킹 상태 [d_model]
            action: 현재 에피소드타임에서의 액션 [d_model]
            current_episode_time: 현재 에피소드타임 인덱스
            episode_time_len: 전체 에피소드 시간 길이
        
        Returns:
            reward: 현재 에피소드타임에서의 리워드 [d_model]
        """
        device = masked_state.device
        
        dt_debug_tensor("reward_input", masked_state, detailed=True)
        dt_debug_tensor("reward_action", action, detailed=True)
        
        # 미래 리워드 마스킹: 현재 시점에서만 리워드 계산 가능
        if current_episode_time < episode_time_len:
            # 상태-액션 쌍에서 리워드 계산
            state_action = torch.cat([masked_state, action], dim=0)  # [2*d_model]
            
            # 현재 모델 디바이스 확인
            model_device = next(self.parameters()).device
            
            # RTG Calculator를 사용한 실시간 리워드 계산
            # Property 모델로 현재 상태의 속성값을 예측하고 정답과의 거리를 기반으로 RTG 계산
            
            if hasattr(self, 'rtg_calculator') and self.rtg_calculator is not None:
                # 상태-액션을 Property 모델 입력 형식으로 변환
                input_seq = state_action.unsqueeze(0).unsqueeze(0)  # [1, 1, 2*d_model]
                attn_mask = torch.ones((1, 1), dtype=torch.bool, device=model_device)
                
                # RTG Calculator로 리워드 계산
                rtg_value = self.rtg_calculator.calculate_single_step_rtg(
                    state_action=input_seq,
                    attention_mask=attn_mask,
                    current_step=current_episode_time,
                    total_steps=episode_time_len
                )
                
                reward_normalized = torch.tensor([rtg_value], device=model_device)
            else:
                # RTG Calculator가 없는 경우 기본값
                reward_normalized = torch.tensor([0.5], device=model_device)
            
            # 리워드를 d_model 차원으로 임베딩
            # 현재 모델 디바이스에 맞는 레이어 생성
            reward_embed_layer = nn.Linear(1, self.d_model).to(model_device)
            reward = reward_embed_layer(reward_normalized.unsqueeze(0)).squeeze(0)  # [d_model]
        else:
            # 미래 리워드 마스킹: 비어있는 리워드
            # 디바이스 일관성을 위해 모델 디바이스 확인
            model_device = next(self.parameters()).device
            reward = torch.zeros(self.d_model, device=model_device)
        
        dt_debug_tensor("reward_output", reward, detailed=True)
        
        return reward

    def create_input_sequence(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        """
        state, action, reward를 순서대로 구성
        
        Args:
            state: 상태 텐서 [batch_size, seq_len, d_model]
            action: 액션 텐서 [batch_size, seq_len, d_model]
            reward: 리워드 텐서 [batch_size, seq_len, d_model]
        
        Returns:
            sequence: 결합된 시퀀스 [batch_size, seq_len * 3, d_model]
        """
        batch_size, seq_len, d_model = state.shape
        
        dt_debug_tensor("create_input_sequence_state", state, detailed=True)
        dt_debug_tensor("create_input_sequence_action", action, detailed=True)
        dt_debug_tensor("create_input_sequence_reward", reward, detailed=True)
        
        # 각 타임스텝에서 state, action, reward를 순서대로 배치
        sequence_list = []
        
        for t in range(seq_len):
            # 현재 타임스텝의 state, action, reward
            curr_state = state[:, t:t+1, :]   # [batch_size, 1, d_model]
            curr_action = action[:, t:t+1, :] # [batch_size, 1, d_model]
            curr_reward = reward[:, t:t+1, :] # [batch_size, 1, d_model]
            
            # state, action, reward 순서로 추가
            sequence_list.extend([curr_state, curr_action, curr_reward])
        
        # 시퀀스 결합
        sequence = torch.cat(sequence_list, dim=1)  # [batch_size, seq_len * 3, d_model]
        
        dt_debug_tensor("create_input_sequence_output", sequence, detailed=True)
        
        return sequence

    def create_input_sequence_batch(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        배치 시퀀스를 state, action, reward로 분리
        
        Args:
            sequence: 결합된 시퀀스 [batch_size, seq_len * 3, d_model]
        
        Returns:
            state: 상태 텐서 [batch_size, seq_len, d_model]
            action: 액션 텐서 [batch_size, seq_len, d_model] 
            reward: 리워드 텐서 [batch_size, seq_len, d_model]
        """
        batch_size, total_seq_len, d_model = sequence.shape
        seq_len = total_seq_len // 3
        
        dt_debug_tensor("create_input_sequence_batch_input", sequence, detailed=True)
        
        # 시퀀스를 3개씩 그룹으로 나누어 state, action, reward 추출
        sequence_reshaped = sequence.view(batch_size, seq_len, 3, d_model)
        
        # 각 타임스텝에서 state, action, reward 추출
        state = sequence_reshaped[:, :, 0, :]   # [batch_size, seq_len, d_model]
        action = sequence_reshaped[:, :, 1, :]  # [batch_size, seq_len, d_model]
        reward = sequence_reshaped[:, :, 2, :]  # [batch_size, seq_len, d_model]
        
        dt_debug_tensor("create_input_sequence_batch_state", state, detailed=True)
        dt_debug_tensor("create_input_sequence_batch_action", action, detailed=True)
        dt_debug_tensor("create_input_sequence_batch_reward", reward, detailed=True)
        
        return state, action, reward

    def forward(self, grid_states: torch.Tensor, 
                gate_actions: Optional[torch.Tensor] = None,
                rewards: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        단일 그리드 매트릭스 단위로 순차 처리 후 배치 합치기
        
        Args:
            grid_states: 그리드 상태 [batch_size, time_steps, num_qubits, features]
            gate_actions: 게이트 액션 (선택적)
            rewards: 리워드 (선택적)
        
        Returns:
            Dict containing embedded sequences and components
        """
        batch_size = grid_states.shape[0]
        batch_results = []
        
        dt_debug_tensor("forward_grid_states", grid_states, detailed=True)
        
        # 각 배치 샘플을 개별로 처리
        for b in range(batch_size):
            single_grid = grid_states[b]  # [time_steps, num_qubits, features]
            
            # 단일 그리드에 대한 전체 파이프라인 실행
            single_result = self._process_single_grid(single_grid)
            batch_results.append(single_result)
        
        # 배치 차원으로 합치기
        return self._combine_batch_results(batch_results)
    
    def _process_single_grid(self, single_grid: torch.Tensor, actual_gate_count: int, grid_matrix_data: Dict[str, Any] = None, max_seq_len: int = None) -> Dict[str, torch.Tensor]:
        """
        🚀 NEW: 순수 게이트 수 기반 단순 처리 (패딩 지원)
        
        Args:
            single_grid: [time_steps, num_qubits, features] 또는 [total_gates, features]
            actual_gate_count: 실제 게이트 수 (메타데이터에서 전달)
            grid_matrix_data: 원본 그리드 매트릭스 데이터 (타겟 생성에 필요)
            max_seq_len: 배치 내 최대 시퀀스 길이 (패딩용)
        
        Returns:
            Dict containing single sample results (padded if max_seq_len provided)
        """
        device = single_grid.device
        
        dt_debug_tensor("_process_single_grid_input", single_grid, detailed=True)
        
        # 실제 게이트 수 기반 SAR 시퀀스 길이 계산
        sar_sequence_len = actual_gate_count * 3
        actual_sequence_len = sar_sequence_len + 1  # EOS 토큰 포함
        
        # 패딩 길이 결정 (배치 레벨 최대 길이 사용)
        if max_seq_len is not None:
            sequence_len = max_seq_len
        else:
            sequence_len = actual_sequence_len
        
        # 디바이스 일관성을 위해 모델 디바이스 확인
        model_device = next(self.parameters()).device
        
        # 실제 게이트 데이터로부터 임베딩 생성
        # 입력 데이터를 실제 게이트 정보로 변환
        gate_features = single_grid
        
        # 입력 텐서 형태 확인 및 처리
        dt_debug_tensor("gate_features_shape", gate_features, detailed=True)
        
        # 각 에피소드 타임별 누적 상태 임베딩 생성
        state_emb = []
        
        for i in range(actual_gate_count):
            if i == 0:
                # 첫 번째 스테이트는 빈 상태 (아무 게이트도 없음)
                empty_state = torch.zeros(self.d_model, device=model_device)
                state_emb.append(empty_state)
            else:
                # i번째 스테이트는 0부터 i-1번째 게이트까지 추가된 회로 상태
                if grid_matrix_data is not None:
                    # 원본 그리드 데이터에서 i개 게이트만 사용한 부분 회로 생성
                    partial_circuit_data = self._create_partial_circuit_state(grid_matrix_data, i)
                    # 부분 회로 전체를 하나의 상태로 임베딩
                    circuit_state_tensor = self._convert_circuit_state_to_tensor(partial_circuit_data)
                    current_state = self.state(circuit_state_tensor).squeeze(0)  # [d_model]
                else:
                    # 그리드 데이터가 없으면 게이트 시퀀스를 회로 상태로 변환
                    cumulative_gates = gate_features[:i]  # [i, features] - 지금까지 추가된 게이트들
                    circuit_representation = self._build_circuit_state_from_gates(cumulative_gates)
                    current_state = self.state(circuit_representation.unsqueeze(0)).squeeze(0)  # [d_model]
                
                state_emb.append(current_state)
        
        state_emb = torch.stack(state_emb, dim=0)  # [actual_gate_count, d_model]
        
        dt_debug_tensor("state_embeddings", state_emb, detailed=True)
        
        # 액션 임베딩: 각 스텝에서 실제로 선택된 게이트 (정답 레이블)
        action_emb = []
        for i in range(actual_gate_count):
            current_gate = gate_features[i:i+1]  # [1, features] - i번째 게이트
            gate_embedded = self.state(current_gate)  # [1, d_model]
            action_emb.append(gate_embedded.squeeze(0))
        action_emb = torch.stack(action_emb, dim=0)  # [actual_gate_count, d_model]
        
        # 리워드 임베딩은 일단 초기화 (RTG 계산에서 채워짐)
        reward_emb = torch.zeros(actual_gate_count, self.d_model, device=model_device)
        
        # SAR 시퀀스 생성
        sar_sequence = torch.zeros(sar_sequence_len, self.d_model, device=model_device)
        for i in range(actual_gate_count):
            base_idx = i * 3
            sar_sequence[base_idx] = state_emb[i]      # State
            sar_sequence[base_idx + 1] = action_emb[i]  # Action
            sar_sequence[base_idx + 2] = reward_emb[i]  # Reward
        
        # EOS 토큰 추가
        actual_input_sequence = torch.cat([sar_sequence, self.eos_embed.unsqueeze(0)], dim=0)
        
        # 패딩 적용 (필요한 경우)
        if sequence_len > actual_sequence_len:
            # 패딩 토큰으로 채우기
            padding_len = sequence_len - actual_sequence_len
            # 모델 디바이스 사용
            padding = torch.zeros(padding_len, self.d_model, device=model_device)
            input_sequence = torch.cat([actual_input_sequence, padding], dim=0)
        else:
            input_sequence = actual_input_sequence
        
        dt_debug_tensor("_process_single_grid_output", input_sequence, detailed=True)
        
        # 어텐션 마스크 생성 (causal mask) - 모델 디바이스 사용
        attention_mask = torch.tril(torch.ones(sequence_len, sequence_len, device=model_device, dtype=torch.bool))
        
        # 액션 예측 마스크 생성 (1::3 패턴, 실제 길이만) - 모델 디바이스 사용
        action_prediction_mask = torch.zeros(sequence_len, dtype=torch.bool, device=model_device)
        # 실제 게이트 위치에만 True 설정
        for i in range(actual_gate_count):
            action_idx = i * 3 + 1  # 1, 4, 7, 10... 위치 (액션 위치)
            if action_idx < sequence_len:
                action_prediction_mask[action_idx] = True
        
        # 액션 타겟 생성 (학습에 필요)
        target_tensors = {}
        if grid_matrix_data is not None:
            # ActionTargetBuilder를 사용하여 타겟 텐서 생성
            action_targets = ActionTargetBuilder.build_from_grid(grid_matrix_data, batch_size=1)
            
            # 타겟 텐서 추출
            if 'gate_targets' in action_targets:
                target_tensors['target_actions'] = action_targets['gate_targets']
            if 'position_targets' in action_targets:
                target_tensors['target_qubits'] = action_targets['position_targets']
            if 'parameter_targets' in action_targets:
                target_tensors['target_params'] = action_targets['parameter_targets']
        
        # 타겟 텐서가 없는 경우 더미 생성 - 모델 디바이스 사용
        if 'target_actions' not in target_tensors:
            target_tensors['target_actions'] = torch.zeros(actual_gate_count, dtype=torch.long, device=model_device)
        if 'target_qubits' not in target_tensors:
            target_tensors['target_qubits'] = torch.full((actual_gate_count, 2), -1, dtype=torch.long, device=model_device)
        if 'target_params' not in target_tensors:
            target_tensors['target_params'] = torch.zeros(actual_gate_count, dtype=torch.float, device=model_device)
        
        result = {
            'input_sequence': input_sequence,           # [sequence_len, d_model]
            'attention_mask': attention_mask,           # [sequence_len, sequence_len]
            'action_prediction_mask': action_prediction_mask,  # [sequence_len]
            'state_embedded': state_emb,                # [actual_gate_count, d_model]
            'action_embedded': action_emb,              # [actual_gate_count, d_model]
            'reward_embedded': reward_emb,              # [actual_gate_count, d_model]
            'episode_time_len': torch.tensor(actual_gate_count, device=model_device),
            'sar_sequence_len': torch.tensor(sar_sequence_len, device=model_device),
            'target_actions': target_tensors['target_actions'],  # [actual_gate_count]
            'target_qubits': target_tensors['target_qubits'],    # [actual_gate_count, 2]
            'target_params': target_tensors['target_params']     # [actual_gate_count]
        }
        
        return result
    
    def _create_episode_mask(self, episode_sequence: torch.Tensor, current_episode_time: int) -> torch.Tensor:
        """
        에피소드타임 기준 마스킹: current_episode_time까지만 공개
        
        Args:
            episode_sequence: [episode_time_len, features]
            current_episode_time: 현재 에피소드 시간
        
        Returns:
            masked_sequence: [episode_time_len, features]
        """
        masked_sequence = torch.zeros_like(episode_sequence)
        if current_episode_time > 0:
            masked_sequence[:current_episode_time] = episode_sequence[:current_episode_time]
        return masked_sequence
    
    def _combine_batch_results(self, batch_results: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        개별 처리된 결과들을 배치 차원으로 합치기
        
        Args:
            batch_results: 각 샘플의 결과 리스트
        
        Returns:
            배치 차원으로 합쳐진 결과
        """
        if not batch_results:
            return {}
        
        combined = {}
        for key in batch_results[0].keys():
            combined[key] = torch.stack([result[key] for result in batch_results], dim=0)
        
        return combined
    
    def _build_circuit_state_from_gates(self, gate_sequence: torch.Tensor) -> torch.Tensor:
        """게이트 시퀀스로부터 회로 상태 표현 생성"""
        if len(gate_sequence) == 0:
            return torch.zeros(1, 4, device=next(self.parameters()).device)
        
        # 게이트 시퀀스를 회로 상태로 변환 (단순히 모든 게이트의 정보를 합성)
        # 각 게이트가 회로에 미치는 영향을 누적적으로 표현
        circuit_state = torch.zeros(4, device=gate_sequence.device)
        
        for gate in gate_sequence:
            # 각 게이트의 정보를 회로 상태에 누적
            circuit_state += gate
        
        # 정규화하여 안정적인 표현 생성
        circuit_state = circuit_state / len(gate_sequence)
        return circuit_state.unsqueeze(0)  # [1, 4]
    
    def _create_partial_circuit_state(self, grid_matrix_data: Dict[str, Any], num_gates: int) -> Dict[str, Any]:
        """부분 회로 상태 생성 (처음 num_gates개만 포함)"""
        if 'gates' not in grid_matrix_data:
            return {'gates': []}
        
        original_gates = grid_matrix_data['gates']
        partial_gates = original_gates[:num_gates]
        
        # 부분 회로 데이터 생성
        partial_data = {
            'gates': partial_gates,
            'num_qubits': grid_matrix_data.get('num_qubits', 10),
            'depth': min(grid_matrix_data.get('depth', 0), num_gates)
        }
        
        return partial_data
    
    def create_incremental_state_embedding(self, current_circuit_gates: List[Dict], predicted_gate: Dict, num_qubits: int = 10) -> torch.Tensor:
        """
        인퍼런스용: 기존 회로에 예측된 게이트를 추가한 새로운 상태 임베딩 생성
        
        Args:
            current_circuit_gates: 현재까지의 회로 게이트 리스트
            predicted_gate: 새로 예측된 게이트 정보 {'gate_name': str, 'qubits': List[int], 'parameter_value': float}
            num_qubits: 회로의 큐빗 수
            
        Returns:
            새로운 상태 임베딩 [d_model]
        """
        # 기존 게이트 리스트에 예측된 게이트 추가
        updated_gates = current_circuit_gates.copy()
        updated_gates.append(predicted_gate)
        
        # 업데이트된 회로 데이터 생성
        updated_circuit_data = {
            'gates': updated_gates,
            'num_qubits': num_qubits,
            'depth': len(updated_gates)
        }
        
        # 새로운 회로 상태를 텐서로 변환
        circuit_state_tensor = self._convert_circuit_state_to_tensor(updated_circuit_data)
        
        # 상태 임베딩 생성
        new_state_embedding = self.state(circuit_state_tensor).squeeze(0)  # [d_model]
        
        return new_state_embedding
    
    def update_sar_sequence_with_prediction(self, current_sar_sequence: torch.Tensor, 
                                          current_circuit_gates: List[Dict],
                                          predicted_gate: Dict,
                                          predicted_reward: float = 0.0,
                                          num_qubits: int = 10) -> torch.Tensor:
        """
        인퍼런스용: 예측된 게이트로 SAR 시퀀스 업데이트
        
        Args:
            current_sar_sequence: 현재 SAR 시퀀스 [seq_len, d_model]
            current_circuit_gates: 현재까지의 회로 게이트 리스트
            predicted_gate: 예측된 게이트 정보
            predicted_reward: 예측된 리워드 값
            num_qubits: 회로의 큐빗 수
            
        Returns:
            업데이트된 SAR 시퀀스 [seq_len+3, d_model]
        """
        device = current_sar_sequence.device
        
        # 1. 새로운 상태 임베딩 생성 (예측된 게이트가 추가된 회로 상태)
        new_state_emb = self.create_incremental_state_embedding(current_circuit_gates, predicted_gate, num_qubits)
        new_state_emb = new_state_emb.to(device)
        
        # 2. 예측된 액션 임베딩 생성
        gate_registry = QuantumGateRegistry()
        gate_vocab = gate_registry.get_gate_vocab()
        gate_type_id = gate_vocab.get(predicted_gate.get('gate_name', 'unknown'), 0)
        
        qubits = predicted_gate.get('qubits', [0, 0])
        qubit1 = qubits[0] if len(qubits) > 0 else 0
        qubit2 = qubits[1] if len(qubits) > 1 else qubit1
        parameter_value = predicted_gate.get('parameter_value', 0.0)
        
        # 게이트 정보를 텐서로 변환
        gate_tensor = torch.tensor([[gate_type_id, qubit1, qubit2, parameter_value]], 
                                 dtype=torch.float32, device=device)
        action_emb = self.state(gate_tensor).squeeze(0)  # [d_model]
        
        # 3. 리워드 임베딩 생성
        reward_tensor = torch.tensor([predicted_reward], device=device)
        reward_emb = self.reward_embed(reward_tensor.unsqueeze(0)).squeeze(0)  # [d_model]
        
        # 4. 새로운 SAR 트리플릿 생성
        new_sar_triplet = torch.stack([new_state_emb, action_emb, reward_emb], dim=0)  # [3, d_model]
        
        # 5. 기존 시퀀스에 추가
        updated_sequence = torch.cat([current_sar_sequence, new_sar_triplet], dim=0)
        
        return updated_sequence
    
    def _convert_circuit_state_to_tensor(self, circuit_data: Dict[str, Any]) -> torch.Tensor:
        """회로 상태를 텐서로 변환"""
        gates = circuit_data.get('gates', [])
        
        if not gates:
            # 빈 회로인 경우
            return torch.zeros(1, 4, device=next(self.parameters()).device)
        
        # 회로 상태를 그리드 형태로 표현
        gate_sequence = []
        for gate in gates:
            if isinstance(gate, dict):
                gate_name = gate.get('gate_name', 'unknown')
                gate_registry = QuantumGateRegistry()
                gate_vocab = gate_registry.get_gate_vocab()
                gate_type_id = gate_vocab.get(gate_name, gate_vocab.get('[EMPTY]', 0))
                
                qubits = gate.get('qubits', [])
                qubit1 = qubits[0] if len(qubits) > 0 else 0
                qubit2 = qubits[1] if len(qubits) > 1 else qubit1
                
                parameter_value = 0.0
                if 'parameter_value' in gate:
                    parameter_value = float(gate['parameter_value'])
                elif 'parameters' in gate and gate['parameters']:
                    parameter_value = float(gate['parameters'][0])
                
                gate_sequence.append([gate_type_id, qubit1, qubit2, parameter_value])
        
        if gate_sequence:
            circuit_tensor = torch.tensor(gate_sequence, dtype=torch.float32, device=next(self.parameters()).device)
            # 회로 전체 상태를 하나의 벡터로 요약 (평균)
            return circuit_tensor.mean(dim=0, keepdim=True)  # [1, 4]
        else:
            return torch.zeros(1, 4, device=next(self.parameters()).device)

    def convert_grid_matrix_to_tensor(self, grid_matrix_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        그리드 매트릭스 데이터를 순수 게이트 시퀀스로 변환 (분리하지 않음)
        
        Args:
            grid_matrix_data: to_grid_matrix()의 출력
        
        Returns:
            순수 게이트 시퀀스 텐서 [gate_type_id, qubit1, qubit2, parameter_value]
        """
        from gates import QuantumGateRegistry
        
        gates = grid_matrix_data.get('gates', [])
        
        # gates 모듈에서 gate_vocab 가져오기
        gate_registry = QuantumGateRegistry()
        gate_vocab = gate_registry.get_gate_vocab()
        
        # 순수 게이트 시퀀스 생성 (분리하지 않음)
        gate_sequence = []
        
        for gate in gates:
            if isinstance(gate, dict):
                gate_name = gate.get('gate_name', 'unknown')
                gate_type_id = gate_vocab.get(gate_name, gate_vocab.get('[EMPTY]', 0))
                
                # 큐빗 위치 정보
                qubits = gate.get('qubits', [])
                qubit1 = qubits[0] if len(qubits) > 0 else 0
                qubit2 = qubits[1] if len(qubits) > 1 else qubit1  # 1큐빗 게이트는 같은 값
                
                # 파라미터 정보
                parameter_value = 0.0
                if 'parameter_value' in gate:
                    parameter_value = float(gate['parameter_value'])
                elif 'parameters' in gate and gate['parameters']:
                    parameter_value = float(gate['parameters'][0])
                
                # 게이트 정보 추가 [gate_type_id, qubit1, qubit2, parameter_value]
                gate_sequence.append([gate_type_id, qubit1, qubit2, parameter_value])
        
        # 텐서로 변환
        if gate_sequence:
            gate_tensor = torch.tensor(gate_sequence, dtype=torch.float32).unsqueeze(0)  # [1, num_gates, 4]
            gate_type_tensor = torch.tensor([seq[0] for seq in gate_sequence], dtype=torch.long).unsqueeze(0)  # [1, num_gates]
        else:
            gate_tensor = torch.zeros(1, 0, 4, dtype=torch.float32)
            gate_type_tensor = torch.zeros(1, 0, dtype=torch.long)
        
        return {
            'grid_tensor': gate_tensor,
            'gate_type_indices': gate_type_tensor
        }
        

    def visualize_grid_tensor(self, grid_tensor: torch.Tensor) -> str:
        """
        그리드 텐서를 시각화 (x,y 구조)
        
        Args:
            grid_tensor: [1, time_steps, num_qubits, features] 형태의 텐서
        
        Returns:
            시각화된 그리드 문자열
        """
        batch_size, time_steps, num_qubits, features = grid_tensor.shape
        
        # 첫 번째 배치만 시각화
        tensor_2d = grid_tensor[0, :, :, 1]  # occupation 정보만 사용
        
        visualization = "\n그리드 시각화 (x=시간, y=큐빗):\n"
        visualization += "   " + "".join([f"{t:3d}" for t in range(time_steps)]) + "\n"
        
        for qubit_idx in range(num_qubits):
            row = f"q{qubit_idx}: "
            for time_idx in range(time_steps):
                if tensor_2d[time_idx, qubit_idx] > 0:
                    gate_type = int(grid_tensor[0, time_idx, qubit_idx, 0].item())
                    row += f"{gate_type:3d}"
                else:
                    row += "  ."
            visualization += row + "\n"
        
        return visualization

    def process_grid_matrix_data_simple(self, grid_matrix_data: Dict[str, Any], actual_gate_count: int, max_seq_len: int = None) -> Dict[str, torch.Tensor]:
        """
        순수 게이트 시퀀스 기반 Decision Transformer 입력 생성 (분리하지 않음)
        
        Args:
            grid_matrix_data: to_grid_matrix()의 출력
            actual_gate_count: 실제 게이트 수
            max_seq_len: 배치 내 최대 시퀀스 길이 (패딩용)
        
        Returns:
            Dict containing all Decision Transformer inputs (padded if max_seq_len provided)
        """
        # 순수 게이트 시퀀스로 변환 (분리하지 않음)
        tensor_data = self.convert_grid_matrix_to_tensor(grid_matrix_data)
        gate_sequence = tensor_data['grid_tensor']  # [1, num_gates, 4]
        
        # 단일 시퀀스 추출
        single_sequence = gate_sequence[0]  # [num_gates, 4]
        
        # 순수 시퀀스 처리 (패딩 길이 전달)
        results = self._process_single_grid(single_sequence, actual_gate_count, grid_matrix_data, max_seq_len)
        
        return results
