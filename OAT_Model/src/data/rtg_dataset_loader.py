"""
RTG Dataset Loader
사전 계산된 RTG 값이 포함된 데이터셋을 로드하는 모듈
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class RTGQuantumDataset(Dataset):
    """사전 계산된 RTG 값이 포함된 양자 회로 데이터셋"""
    
    def __init__(self, dataset_path: str, max_seq_len: int = 512, d_model: int = 512):
        """
        Args:
            dataset_path: RTG 값이 포함된 데이터셋 경로
            max_seq_len: 최대 시퀀스 길이
            d_model: 모델 차원
        """
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # 데이터셋 로드
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"📊 RTG 데이터셋 로드 완료: {len(self.data)}개 시퀀스")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        단일 데이터 아이템 반환
        
        Returns:
            Dict containing:
            - states: [seq_len, d_model] 상태 시퀀스
            - actions: [seq_len, d_model] 액션 시퀀스  
            - rtg_rewards: [seq_len] RTG 리워드 시퀀스
            - attention_mask: [seq_len] 어텐션 마스크
            - targets: 정답 레이블들
        """
        item = self.data[idx]
        
        # RTG 리워드 시퀀스 추출
        rtg_rewards = item.get('rtg_rewards', [])
        seq_len = len(rtg_rewards)
        
        # 패딩 처리
        if seq_len > self.max_seq_len:
            # 시퀀스가 너무 긴 경우 자르기
            seq_len = self.max_seq_len
            rtg_rewards = rtg_rewards[:self.max_seq_len]
        
        # 텐서 생성 및 패딩
        padded_rtg = torch.zeros(self.max_seq_len)
        padded_rtg[:seq_len] = torch.tensor(rtg_rewards[:seq_len], dtype=torch.float32)
        
        # 어텐션 마스크 생성
        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        attention_mask[:seq_len] = True
        
        # 상태와 액션 시퀀스 생성 (플레이스홀더)
        # TODO: 실제 데이터에서 상태/액션 추출 로직 구현
        states = torch.randn(self.max_seq_len, self.d_model)
        actions = torch.randn(self.max_seq_len, self.d_model)
        
        # 정답 속성값들
        predicted_properties = item.get('predicted_properties', {})
        targets = {}
        for prop_name in ['entanglement', 'fidelity', 'expressibility']:
            if prop_name in predicted_properties:
                prop_values = predicted_properties[prop_name]
                padded_prop = torch.zeros(self.max_seq_len)
                padded_prop[:len(prop_values)] = torch.tensor(prop_values[:seq_len], dtype=torch.float32)
                targets[prop_name] = padded_prop
        
        return {
            'states': states,
            'actions': actions,
            'rtg_rewards': padded_rtg,
            'attention_mask': attention_mask,
            'targets': targets,
            'seq_len': seq_len
        }


class RTGDataLoader:
    """RTG 데이터셋용 데이터 로더 래퍼"""
    
    def __init__(self, dataset_path: str, batch_size: int = 32, 
                 max_seq_len: int = 512, d_model: int = 512,
                 shuffle: bool = True, num_workers: int = 0):
        """
        Args:
            dataset_path: RTG 데이터셋 경로
            batch_size: 배치 크기
            max_seq_len: 최대 시퀀스 길이
            d_model: 모델 차원
            shuffle: 데이터 셔플 여부
            num_workers: 워커 프로세스 수
        """
        self.dataset = RTGQuantumDataset(dataset_path, max_seq_len, d_model)
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """배치 데이터 콜레이트 함수"""
        batch_size = len(batch)
        max_seq_len = batch[0]['states'].shape[0]
        d_model = batch[0]['states'].shape[1]
        
        # 배치 텐서 초기화
        batch_states = torch.zeros(batch_size, max_seq_len, d_model)
        batch_actions = torch.zeros(batch_size, max_seq_len, d_model)
        batch_rtg_rewards = torch.zeros(batch_size, max_seq_len)
        batch_attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
        
        # 정답 속성값들
        batch_targets = {}
        prop_names = ['entanglement', 'fidelity', 'expressibility']
        for prop_name in prop_names:
            batch_targets[prop_name] = torch.zeros(batch_size, max_seq_len)
        
        # 배치 데이터 채우기
        for i, item in enumerate(batch):
            batch_states[i] = item['states']
            batch_actions[i] = item['actions']
            batch_rtg_rewards[i] = item['rtg_rewards']
            batch_attention_mask[i] = item['attention_mask']
            
            for prop_name in prop_names:
                if prop_name in item['targets']:
                    batch_targets[prop_name][i] = item['targets'][prop_name]
        
        return {
            'states': batch_states,
            'actions': batch_actions,
            'rtg_rewards': batch_rtg_rewards,
            'attention_mask': batch_attention_mask,
            'targets': batch_targets
        }
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


def create_rtg_dataloaders(train_path: str, val_path: str, test_path: str,
                          batch_size: int = 32, max_seq_len: int = 512, 
                          d_model: int = 512, num_workers: int = 0) -> Tuple[RTGDataLoader, RTGDataLoader, RTGDataLoader]:
    """
    RTG 데이터 로더들 생성
    
    Args:
        train_path: 훈련 데이터셋 경로
        val_path: 검증 데이터셋 경로
        test_path: 테스트 데이터셋 경로
        batch_size: 배치 크기
        max_seq_len: 최대 시퀀스 길이
        d_model: 모델 차원
        num_workers: 워커 프로세스 수
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_loader = RTGDataLoader(
        train_path, batch_size=batch_size, max_seq_len=max_seq_len,
        d_model=d_model, shuffle=True, num_workers=num_workers
    )
    
    val_loader = RTGDataLoader(
        val_path, batch_size=batch_size, max_seq_len=max_seq_len,
        d_model=d_model, shuffle=False, num_workers=num_workers
    )
    
    test_loader = RTGDataLoader(
        test_path, batch_size=batch_size, max_seq_len=max_seq_len,
        d_model=d_model, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 사용 예시
    train_loader, val_loader, test_loader = create_rtg_dataloaders(
        train_path="processed_data/rtg_train.json",
        val_path="processed_data/rtg_val.json", 
        test_path="processed_data/rtg_test.json",
        batch_size=16,
        max_seq_len=256,
        d_model=512
    )
    
    # 데이터 로더 테스트
    for batch in train_loader:
        print("배치 형태:")
        print(f"  States: {batch['states'].shape}")
        print(f"  Actions: {batch['actions'].shape}")
        print(f"  RTG Rewards: {batch['rtg_rewards'].shape}")
        print(f"  Attention Mask: {batch['attention_mask'].shape}")
        print(f"  Targets: {[f'{k}: {v.shape}' for k, v in batch['targets'].items()]}")
        break
