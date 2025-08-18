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
from src.models.decision_transformer import DebugMode
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))

from gates import QuantumGateRegistry

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
        # 🔧 NEW: 2큐빗 게이트 전용 형태로 변경 [qubit1, qubit2]
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
            'position_targets': torch.full((num_gates, 2), -1, dtype=torch.long),  # 🔧 2큐빗 형태로 변경
            'parameter_targets': torch.zeros(num_gates, dtype=torch.float)
        }

class QuantumGateSequenceEmbedding(nn.Module):
    def __init__(self, d_model: int, n_gate_types: int, n_qubits: int, 
                 max_seq_len: int = 1024, grid_size: Tuple[int, int] = (8, 8)):
        super().__init__()
        self.d_model = d_model
        self.n_gate_types = n_gate_types
        self.n_qubits = n_qubits
        self.max_seq_len = max_seq_len
        self.grid_width, self.grid_height = grid_size
        
        # 임베딩 레이어들 (중요도별 차원 배분)
        # 중요도: gate_type > role > occupation > parameter
        gate_dim = d_model // 2      # 50% - 가장 중요 (H, X, CNOT, RZ 등)
        param_dim = d_model // 4      # 25% - 두번째 중요 (파라미터터 구분)
        occupation_dim = d_model // 16  # 6.25% - 세번째 중요 (점유 여부)
        role_dim = d_model - gate_dim - param_dim - occupation_dim  # 나머지 18.75%
        
        self.gate_type_embed = nn.Embedding(n_gate_types, gate_dim)   # 게이트 타입 ID (가장 중요)
        self.role_embed = nn.Embedding(4, role_dim)                   # 역할 (control/target)
        self.occupation_embed = nn.Embedding(2, occupation_dim)       # 점유 상태
        self.param_embed = nn.Linear(1, param_dim)                    # 게이트 파라미터
        
        # 위치 인코딩 (학습 가능한 임베딩)
        self.positional_encoding = nn.Embedding(max_seq_len, d_model)
        
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
        pe = torch.zeros(self.max_seq_len, self.d_model)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                           (-math.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, max_seq_len, d_model]
    
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

    def state(self, x: torch.Tensor, gate_type_indices: torch.Tensor = None, role_indices: torch.Tensor = None) -> torch.Tensor:
        """
        에피소드 시퀀스에 대한 상태 임베딩
        
        Args:
            x: 에피소드 시퀀스 [episode_time_len, features]
            gate_type_indices: Long 타입 게이트 타입 인덱스 [episode_time_len]
            role_indices: Long 타입 역할 인덱스 [episode_time_len]
        
        Returns:
            state: 인코딩된 상태 [episode_time_len, d_model]
        """
        # 🔍 CRITICAL FIX: x 차원 동적 처리
        device = x.device
        
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
        
        # 위치 인덱스 생성
        position_indices = torch.arange(episode_time_len, device=device).long()
        position_emb = self.positional_encoding(position_indices)  # [episode_time_len, d_model]
        
        # 상태 인코딩 (각 피처를 개별로 임베딩)
        if features == 4:  # [gate_type_id, role_id, occupation, parameter_value]
            # 이미 정수 타입인 값들을 embedding 인덱스로 사용
            gate_type_ids = x[:, 0].long()      # [episode_time_len]
            role_ids = x[:, 1].long()           # [episode_time_len]
            occupation = x[:, 2].long()         # [episode_time_len]
            parameters = x[:, 3:4]              # [episode_time_len, 1]
            
            # 각각 임베딩
            gate_embedded = self.gate_type_embed(gate_type_ids)        # [episode_time_len, embed_dim]
            role_embedded = self.role_embed(role_ids)                  # [episode_time_len, embed_dim]
            occupation_embedded = self.occupation_embed(occupation)    # [episode_time_len, embed_dim]
            param_embedded = self.param_embed(parameters)              # [episode_time_len, embed_dim]
            
            # 임베딩 결합 (concatenation)
            state_encoded = torch.cat([
                gate_embedded, 
                role_embedded, 
                occupation_embedded, 
                param_embedded
            ], dim=-1)  # [episode_time_len, d_model]
        
        # 위치 임베딩과 결합
        state = state_encoded + position_emb  # [episode_time_len, d_model]
        
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
            action = torch.zeros(self.d_model, device=device)
        
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
        
        if features == 4:  # [gate_type_id, role_id, occupation, parameter_value]
            # 각 피처 추출
            gate_type_ids = episode_features[:, :, 0].long()      # 게이트 타입 ID
            role_ids = episode_features[:, :, 1].long()           # 역할 ID
            occupation = episode_features[:, :, 2].long()         # 점유 상태
            parameters = episode_features[:, :, 3:4]              # 파라미터 값
            
            # 각각 임베딩 (차원 분할)
            gate_embedded = self.gate_type_embed(gate_type_ids)        # [batch, episode_time, gate_dim]
            role_embedded = self.role_embed(role_ids)                  # [batch, episode_time, role_dim]
            occupation_embedded = self.occupation_embed(occupation)    # [batch, episode_time, occupation_dim]
            param_embedded = self.param_embed(parameters)              # [batch, episode_time, param_dim]
            
            # 임베딩 결합 (concatenation)
            embedded = torch.cat([
                gate_embedded, 
                role_embedded, 
                occupation_embedded, 
                param_embedded
            ], dim=-1)  # [batch, episode_time, d_model]
        else:
            # 다른 피처 차원의 경우 선형 변환
            linear_layer = nn.Linear(features, self.d_model).to(episode_features.device)
            embedded = linear_layer(episode_features)
        
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
        
        # 미래 리워드 마스킹: 현재 시점에서만 리워드 계산 가능
        if current_episode_time < episode_time_len:
            # 상태-액션 쌍에서 리워드 계산
            state_action = torch.cat([masked_state, action], dim=0)  # [2*d_model]
            
            # 리워드 계산 (실제로는 회로 품질 메트릭 기반)
            reward_scalar = torch.norm(state_action, dim=0, keepdim=True)  # [1]
            reward_normalized = torch.sigmoid(reward_scalar)  # [0, 1] 범위로 정규화
            
            # 리워드를 d_model 차원으로 임베딩
            reward_embed_layer = nn.Linear(1, self.d_model).to(device)
            reward = reward_embed_layer(reward_normalized.unsqueeze(0)).squeeze(0)  # [d_model]
        else:
            # 미래 리워드 마스킹: 비어있는 리워드
            reward = torch.zeros(self.d_model, device=device)
        
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
        
        # 시퀀스를 3개씩 그룹으로 나누어 state, action, reward 추출
        sequence_reshaped = sequence.view(batch_size, seq_len, 3, d_model)
        
        # 각 타임스텝에서 state, action, reward 추출
        state = sequence_reshaped[:, :, 0, :]   # [batch_size, seq_len, d_model]
        action = sequence_reshaped[:, :, 1, :]  # [batch_size, seq_len, d_model]
        reward = sequence_reshaped[:, :, 2, :]  # [batch_size, seq_len, d_model]
        
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
        
        # 실제 게이트 수 기반 SAR 시퀀스 길이 계산
        sar_sequence_len = actual_gate_count * 3
        actual_sequence_len = sar_sequence_len + 1  # EOS 토큰 포함
        
        # 패딩 길이 결정 (배치 레벨 최대 길이 사용)
        if max_seq_len is not None:
            sequence_len = max_seq_len
        else:
            sequence_len = actual_sequence_len
        
        # 간단한 더미 임베딩 생성 (실제 게이트 수만큼)
        state_emb = torch.zeros(actual_gate_count, self.d_model, device=device)
        action_emb = torch.zeros(actual_gate_count, self.d_model, device=device)
        reward_emb = torch.zeros(actual_gate_count, self.d_model, device=device)
        
        # SAR 시퀀스 생성
        sar_sequence = torch.zeros(sar_sequence_len, self.d_model, device=device)
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
            padding = torch.zeros(padding_len, self.d_model, device=device)
            input_sequence = torch.cat([actual_input_sequence, padding], dim=0)
        else:
            input_sequence = actual_input_sequence
        
        # 어텐션 마스크 생성 (causal mask)
        attention_mask = torch.tril(torch.ones(sequence_len, sequence_len, device=device, dtype=torch.bool))
        
        # 액션 예측 마스크 생성 (1::3 패턴, 실제 길이만)
        action_prediction_mask = torch.zeros(sequence_len, dtype=torch.bool, device=device)
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
        
        # 타겟 텐서가 없는 경우 더미 생성
        if 'target_actions' not in target_tensors:
            target_tensors['target_actions'] = torch.zeros(actual_gate_count, dtype=torch.long, device=device)
        if 'target_qubits' not in target_tensors:
            target_tensors['target_qubits'] = torch.full((actual_gate_count, 2), -1, dtype=torch.long, device=device)
        if 'target_params' not in target_tensors:
            target_tensors['target_params'] = torch.zeros(actual_gate_count, dtype=torch.float, device=device)
        
        result = {
            'input_sequence': input_sequence,           # [sequence_len, d_model]
            'attention_mask': attention_mask,           # [sequence_len, sequence_len]
            'action_prediction_mask': action_prediction_mask,  # [sequence_len]
            'state_embedded': state_emb,                # [actual_gate_count, d_model]
            'action_embedded': action_emb,              # [actual_gate_count, d_model]
            'reward_embedded': reward_emb,              # [actual_gate_count, d_model]
            'episode_time_len': torch.tensor(actual_gate_count, device=device),
            'sar_sequence_len': torch.tensor(sar_sequence_len, device=device),
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

    def convert_grid_matrix_to_tensor(self, grid_matrix_data: Dict[str, Any]) -> torch.Tensor:
        """
        grid_graph_encoder.to_grid_matrix()의 출력을 텐서로 변환 (x,y 구조)
        gates 모듈의 gate_vocab을 사용하여 게이트 종류와 역할 정보 포함
        
        Args:
            grid_matrix_data: to_grid_matrix()의 출력
                - grid_matrix: [num_qubits][max_parallel_order] 형태의 매트릭스
                - grid_shape: [max_parallel_order, num_qubits]
                - node_lookup: 노드 정보 딕셔너리
        
        Returns:
            grid_tensor: [1, time_steps, num_qubits, features] 형태의 텐서
            features: [gate_type_id, role_id, occupation, parameter_value]
        """
        from gates import QuantumGateRegistry
        
        grid_matrix = grid_matrix_data['grid_matrix']
        grid_shape = grid_matrix_data['grid_shape']
        node_lookup = grid_matrix_data['node_lookup']
        
        max_parallel_order, num_qubits = grid_shape[0], grid_shape[1]
        
        # gates 모듈에서 gate_vocab 가져오기
        gate_registry = QuantumGateRegistry()
        gate_vocab = gate_registry.get_gate_vocab()
        
        # 그리드 텐서 초기화 (features: [gate_type_id, role_id, occupation, parameter_value])
        # gate_type_id와 role_id는 embedding 인덱스로 사용되므로 Long 타입이어야 함
        grid_tensor = torch.zeros(1, max_parallel_order, num_qubits, 4, dtype=torch.float32)
        gate_type_tensor = torch.zeros(1, max_parallel_order, num_qubits, dtype=torch.long)
        role_tensor = torch.zeros(1, max_parallel_order, num_qubits, dtype=torch.long)
        
        # 역할 매핑 (0: empty, 1: single_qubit, 2: control, 3: target)
        role_mapping = {
            'empty': 0,
            'single': 1, 
            'control': 2,
            'target': 3
        }
        
        # grid_matrix는 [qubit][time] 구조이므로 이를 tensor[batch, time, qubit, features]로 변환
        for qubit_idx in range(num_qubits):
            for time_idx in range(max_parallel_order):
                node_id = grid_matrix[qubit_idx][time_idx]
                
                if node_id is not None:
                    # 노드 정보 조회
                    node = node_lookup.get(node_id, {})
                    gate_name = node.get('gate_name', 'unknown')
                    
                    # gate_vocab에서 게이트 ID 가져오기
                    gate_type_id = gate_vocab.get(gate_name, gate_vocab.get('[EMPTY]', 0))
                    
                    # 노드 ID에서 역할 정보 추출
                    role_id = role_mapping['empty']  # 기본값
                    parameter_value = 0.0
                    
                    # 노드 ID 패턴 분석
                    if '_target_' in node_id:
                        role_id = role_mapping['target']
                    elif '_control_' in node_id:
                        role_id = role_mapping['control']
                    elif f'{gate_name}_q' in node_id:
                        role_id = role_mapping['single']
                    
                    # 파라미터 정보 추출 (노드에 있는 경우)
                    if 'parameter_value' in node:
                        parameter_value = float(node['parameter_value'])
                    elif 'parameters' in node and node['parameters']:
                        parameter_value = float(node['parameters'][0])
                    
                    # x,y 구조로 배치: tensor[batch, time, qubit, features]
                    # embedding 인덱스로 사용될 값들을 직접 Long 타입으로 저장
                    gate_type_tensor[0, time_idx, qubit_idx] = gate_type_id    # 게이트 타입 ID (Long)
                    role_tensor[0, time_idx, qubit_idx] = role_id              # 역할 ID (Long)
                    grid_tensor[0, time_idx, qubit_idx, 0] = gate_type_id      # 게이트 타입 ID 
                    grid_tensor[0, time_idx, qubit_idx, 1] = role_id           # 역할 ID 
                    grid_tensor[0, time_idx, qubit_idx, 2] = 1.0               # 점유됨
                    grid_tensor[0, time_idx, qubit_idx, 3] = parameter_value   # 파라미터 값
        
        # embedding 인덱스용 Long 텐서들을 grid_tensor에 추가
        return {
            'grid_tensor': grid_tensor,
            'gate_type_indices': gate_type_tensor,
            'role_indices': role_tensor
        }

    def process_grid_matrix_data_simple(self, grid_matrix_data: Dict[str, Any], actual_gate_count: int, max_seq_len: int = None) -> Dict[str, torch.Tensor]:
        """
        🚀 NEW: 순수 게이트 수 기반 간단한 Decision Transformer 입력 생성 (패딩 지원)
        
        Args:
            grid_matrix_data: to_grid_matrix()의 출력
            actual_gate_count: 실제 게이트 수
            max_seq_len: 배치 내 최대 시퀀스 길이 (패딩용)
        
        Returns:
            Dict containing all Decision Transformer inputs (padded if max_seq_len provided)
        """
        # 🚀 NEW: 순수 게이트 수 기반 간단한 처리
        # 그리드 매트릭스를 텐서로 변환 (더미 데이터로 대체 가능)
        tensor_data = self.convert_grid_matrix_to_tensor(grid_matrix_data)
        grid_states = tensor_data['grid_tensor']
        
        # 단일 그리드에서 첫 번째 샘플 추출
        single_grid = grid_states[0]  # [time_steps, num_qubits, features]
        
        # 새로운 간단한 처리 함수 호출 (패딩 길이 전달)
        # 타겟 텐서 생성을 위해 grid_matrix_data 넘기기
        results = self._process_single_grid(single_grid, actual_gate_count, grid_matrix_data, max_seq_len)
        
        return results
        

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
