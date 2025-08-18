"""RTG (Return-to-Go) Calculator
Property 모델을 사용하여 Decision Transformer용 RTG 계산

하이브리드 리워드 설계:
- 큰 차이(>60%): 선형 리워드로 빠른 학습 신호
- 정밀한 차이(<60%): 가우시안 리워드로 세밀한 조정
- 적응적 가중치: 속성별 중요도 반영 (fidelity > expressibility > entanglement)
- 연속적 전환: 60% 경계에서 부드러운 전환
- 범위: [0, 1], 차이 크기에 따른 최적 함수 선택

RTG 특성:
- 시퀀스 끝에서 0으로 수렴 (Decision Transformer 표준)
- 인퍼런스 시 명시적 성능 레벨 조건 제공 가능
- RTG[t] = Σ(k=t to T) γ^(k-t) * r[k], RTG[T] = 0
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import json
import math
from pathlib import Path

class RTGCalculator:
    """Property 모델 기반 RTG 계산기"""
    
    def __init__(self, property_model, property_config, device='cpu'):
        """
        Args:
            property_model: 사전 훈련된 Property Prediction 모델
            property_config: Property 모델 설정
            device: 계산 디바이스
        """
        self.property_model = property_model
        self.property_config = property_config
        self.device = device
        
        # Property 모델을 평가 모드로 설정
        self.property_model.eval()
        self.property_model.to(device)
        
    def calculate_sequence_properties(self, state_sequence: torch.Tensor, 
                                    attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        전체 시퀀스에 대해 각 스텝의 속성값 계산
        
        Args:
            state_sequence: [batch_size, seq_len, d_model] 상태 시퀀스
            attention_mask: [batch_size, seq_len] 어텐션 마스크
            
        Returns:
            step_properties: 각 스텝별 속성값 딕셔너리
        """
        batch_size, seq_len, d_model = state_sequence.shape
        
        # 각 스텝별 속성값 저장
        step_properties = {
            'entanglement': torch.zeros(batch_size, seq_len, device=self.device),
            'fidelity': torch.zeros(batch_size, seq_len, device=self.device),
            'expressibility': torch.zeros(batch_size, seq_len, device=self.device)
        }
        
        with torch.no_grad():
            for batch_idx in range(batch_size):
                for step_idx in range(seq_len):
                    # 어텐션 마스크 확인
                    if not attention_mask[batch_idx, step_idx]:
                        continue
                    
                    # 현재 스텝까지의 누적 시퀀스 사용
                    current_seq = state_sequence[batch_idx, :step_idx+1, :].unsqueeze(0)  # [1, step+1, d_model]
                    current_mask = attention_mask[batch_idx, :step_idx+1].unsqueeze(0)   # [1, step+1]
                    
                    # Property 모델로 속성 예측
                    predictions = self.property_model.predict(
                        input_sequence=current_seq,
                        attention_mask=current_mask,
                        return_hidden=False
                    )
                    
                    # 예측된 속성값 저장
                    for prop_name in ['entanglement', 'fidelity', 'expressibility']:
                        if prop_name in predictions:
                            step_properties[prop_name][batch_idx, step_idx] = predictions[prop_name].item()
        
        return step_properties
    
    def calculate_property_distance(self, predicted_properties: Dict[str, torch.Tensor],
                                  target_properties: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        예측된 속성값과 정답 레이블 간의 거리 계산
        
        Args:
            predicted_properties: 예측된 속성값들
            target_properties: 정답 속성값들
            
        Returns:
            distances: [batch_size, seq_len] 각 스텝별 거리
        """
        batch_size, seq_len = predicted_properties['entanglement'].shape
        distances = torch.zeros(batch_size, seq_len, device=self.device)
        
        for batch_idx in range(batch_size):
            for step_idx in range(seq_len):
                step_distance = 0.0
                valid_properties = 0
                
                for prop_name in ['entanglement', 'fidelity', 'expressibility']:
                    if prop_name in predicted_properties and prop_name in target_properties:
                        pred_val = predicted_properties[prop_name][batch_idx, step_idx]
                        target_val = target_properties[prop_name][batch_idx, step_idx]
                        
                        # L2 거리 계산
                        step_distance += (pred_val - target_val) ** 2
                        valid_properties += 1
                
                if valid_properties > 0:
                    # RMSE로 정규화
                    distances[batch_idx, step_idx] = torch.sqrt(step_distance / valid_properties)
        
        return distances
    
    def calculate_step_rewards(self, predicted_properties: Dict[str, torch.Tensor],
                             target_properties: Dict[str, torch.Tensor],
                             attention_mask: torch.Tensor) -> torch.Tensor:
        """
        하이브리드 리워드 함수: 차이 크기에 따른 적응적 계산
        
        핵심 설계:
        1. 큰 차이(>60%): 선형 함수로 빠른 학습 신호 제공
        2. 정밀한 차이(<60%): 가우시안 함수로 세밀한 조정
        3. 부드러운 전환: 60% 경계에서 연속성 보장
        4. 속성별 가중치: fidelity > expressibility > entanglement
        
        수학적 정의:
        - distance > 0.6: r = 1 - distance (선형)
        - distance ≤ 0.6: r = exp(-0.5 * distance² / σ²) (가우시안, σ=0.25)
        
        Args:
            predicted_properties: 예측된 속성값들
            target_properties: 목표 속성값들
            attention_mask: [batch_size, seq_len] 어텐션 마스크
            
        Returns:
            step_rewards: [batch_size, seq_len] 각 스텝의 즉시 리워드 (0~1 범위)
        """
        batch_size, seq_len = attention_mask.shape
        step_rewards = torch.zeros(batch_size, seq_len, device=self.device)
        
        for batch_idx in range(batch_size):
            for step_idx in range(seq_len):
                if not attention_mask[batch_idx, step_idx]:
                    continue
                
                property_rewards = []
                
                # 하이브리드 리워드: 차이 크기에 따른 적응적 계산
                property_weights = {
                    'fidelity': 0.4,        # 가장 중요: 회로 정확도
                    'expressibility': 0.3,  # 중간: 탐색 능력
                    'entanglement': 0.3     # 보조: 양자 특성
                }
                
                # 하이브리드 전환 임계값
                hybrid_threshold = 0.6  # 60% 차이 기준
                
                for prop_name in ['entanglement', 'fidelity', 'expressibility']:
                    if prop_name in predicted_properties and prop_name in target_properties:
                        pred_val = predicted_properties[prop_name][batch_idx, step_idx]
                        target_val = target_properties[prop_name][batch_idx, step_idx]
                        
                        distance = torch.abs(pred_val - target_val)
                        weight = property_weights[prop_name]
                        
                        if distance > hybrid_threshold:
                            # 큰 차이(>60%): 선형 리워드로 빠른 학습
                            linear_reward = 1.0 - torch.clamp(distance, 0.0, 1.0)
                            reward = linear_reward
                        else:
                            # 정밀한 차이(≤60%): 가우시안 리워드로 세밀한 조정
                            sigma = 0.25  # 가우시안 표준편차
                            gaussian_reward = torch.exp(-0.5 * (distance ** 2) / (sigma ** 2))
                            
                            # 60% 지점에서 연속성 보장을 위한 스케일링
                            # 선형 함수의 60% 지점 값: 1 - 0.6 = 0.4
                            # 가우시안 함수의 60% 지점 값: exp(-0.5 * 0.6² / 0.25²)
                            gaussian_at_threshold = torch.exp(-0.5 * (hybrid_threshold ** 2) / (sigma ** 2))
                            linear_at_threshold = 1.0 - hybrid_threshold
                            
                            # 연속성을 위한 스케일링
                            scale_factor = linear_at_threshold / gaussian_at_threshold
                            reward = gaussian_reward * scale_factor
                        
                        # 가중 리워드 적용
                        weighted_reward = weight * reward
                        property_rewards.append(weighted_reward)
                
                # 가중 합계 (이미 가중치 적용됨)
                if property_rewards:
                    # 가중 합계 (총합 = 1.0 보장)
                    total_reward = torch.stack(property_rewards).sum()
                    step_rewards[batch_idx, step_idx] = torch.clamp(total_reward, 0.0, 1.0)
                else:
                    # 기본값
                    step_rewards[batch_idx, step_idx] = 0.0
        
        return step_rewards
    
    def calculate_rtg_rewards(self, predicted_properties: Dict[str, torch.Tensor],
                            target_properties: Dict[str, torch.Tensor],
                            attention_mask: torch.Tensor,
                            gamma: float = 0.99,
                            normalize_rtg: bool = True) -> torch.Tensor:
        """
        표준 Decision Transformer RTG (Return-to-Go) 계산
        RTG[t] = Σ(k=t to T) γ^(k-t) * r[k]
        
        중요: RTG는 시퀀스 끝에서 0으로 수렴해야 함 (Decision Transformer 표준)
        이를 통해 인퍼런스 시 명시적 성능 레벨 조건 제공 가능
        
        Args:
            predicted_properties: 예측된 속성값들
            target_properties: 목표 속성값들  
            attention_mask: [batch_size, seq_len] 어텐션 마스크
            gamma: 할인 팩터
            normalize_rtg: RTG 정규화 여부
            
        Returns:
            rtg_rewards: [batch_size, seq_len] RTG 시퀀스 (끝에서 0으로 수렴)
        """
        # 각 스텝의 즉시 리워드 계산 (정규화된 거리 기반)
        step_rewards = self.calculate_step_rewards(
            predicted_properties, target_properties, attention_mask
        )
        
        batch_size, seq_len = step_rewards.shape
        rtg_rewards = torch.zeros(batch_size, seq_len, device=self.device)
        
        for batch_idx in range(batch_size):
            # 유효한 스텝들 찾기
            valid_steps = attention_mask[batch_idx].nonzero(as_tuple=True)[0]
            
            if len(valid_steps) == 0:
                continue
            
            # 뒤에서부터 RTG 계산 (동적 프로그래밍)
            # 마지막 스텝에서 RTG = 0 (Decision Transformer 표준)
            for i in range(len(valid_steps) - 1, -1, -1):
                step_idx = valid_steps[i]
                
                if i == len(valid_steps) - 1:
                    # 마지막 스텝: RTG = 0 (표준 Decision Transformer)
                    rtg_rewards[batch_idx, step_idx] = 0.0
                else:
                    # 이전 스텝들: RTG = r[t] + γ * RTG[t+1]
                    current_reward = step_rewards[batch_idx, step_idx]
                    next_step_idx = valid_steps[i + 1]
                    future_rtg = gamma * rtg_rewards[batch_idx, next_step_idx]
                    
                    rtg_rewards[batch_idx, step_idx] = current_reward + future_rtg
        
        # RTG 정규화 (선택적)
        if normalize_rtg:
            for batch_idx in range(batch_size):
                valid_steps = attention_mask[batch_idx].nonzero(as_tuple=True)[0]
                if len(valid_steps) > 0:
                    # 각 시퀀스별로 독립적으로 정규화
                    valid_rtg = rtg_rewards[batch_idx, valid_steps]
                    if valid_rtg.max() > 0:
                        # [0, 1] 범위로 정규화
                        rtg_rewards[batch_idx, valid_steps] = valid_rtg / valid_rtg.max()
        
        return rtg_rewards
    
    def precompute_rtg_for_dataset(self, dataset_path: str, 
                                 output_path: str,
                                 batch_size: int = 32) -> None:
        """
        전체 데이터셋에 대해 RTG 값을 사전 계산하여 저장
        
        Args:
            dataset_path: 입력 데이터셋 경로
            output_path: RTG 계산 결과 저장 경로
            batch_size: 배치 크기
        """
        print(f"🔄 RTG 사전 계산 시작: {dataset_path}")
        
        # 데이터셋 로드
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        rtg_results = []
        
        # 배치별로 처리
        for i in range(0, len(dataset), batch_size):
            batch_data = dataset[i:i+batch_size]
            print(f"📊 배치 {i//batch_size + 1}/{(len(dataset) + batch_size - 1)//batch_size} 처리 중...")
            
            # 배치 데이터를 텐서로 변환
            batch_states, batch_targets, batch_masks = self._prepare_batch(batch_data)
            
            # 속성값 계산
            predicted_properties = self.calculate_sequence_properties(batch_states, batch_masks)
            
            # RTG 리워드 계산 (새로운 표준 RL 방식)
            rtg_rewards = self.calculate_rtg_rewards(predicted_properties, batch_targets, batch_masks)
            
            # 거리 계산 (디버깅용)
            property_distances = self.calculate_property_distance(predicted_properties, batch_targets)
            
            # 결과 저장
            for j, data_item in enumerate(batch_data):
                rtg_sequence = rtg_rewards[j].cpu().numpy().tolist()
                result_item = {
                    **data_item,  # 원본 데이터 유지
                    'rtg_rewards': rtg_sequence,
                    'predicted_properties': {
                        key: predicted_properties[key][j].cpu().numpy().tolist()
                        for key in predicted_properties
                    },
                    'property_distances': property_distances[j].cpu().numpy().tolist()
                }
                rtg_results.append(result_item)
        
        # 결과 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rtg_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ RTG 사전 계산 완료: {output_path}")
        print(f"📈 총 {len(rtg_results)}개 시퀀스 처리됨")
    
    def calculate_single_step_rtg(self, state_action: torch.Tensor, 
                                attention_mask: torch.Tensor,
                                target_properties: Dict[str, float],
                                current_step: int, total_steps: int,
                                desired_rtg: float = None,
                                gamma: float = 0.99) -> float:
        """
        단일 스텝에 대한 RTG 값 계산 (실시간 사용)
        Decision Transformer 표준: RTG는 0으로 수렴, 인퍼런스 시 명시적 조건 제공
        
        Args:
            state_action: [1, 1, 2*d_model] 상태-액션 텐서
            attention_mask: [1, 1] 어텐션 마스크
            target_properties: 목표 속성값들 {'entanglement': 0.8, 'fidelity': 0.9, ...}
            current_step: 현재 스텝
            total_steps: 전체 스텝 수
            desired_rtg: 인퍼런스 시 원하는 성능 레벨 (None이면 자동 계산)
            gamma: 할인 팩터
            
        Returns:
            rtg_value: RTG 값 (0으로 수렴)
        """
        with torch.no_grad():
            # 인퍼런스 모드: 명시적 RTG 조건 사용
            if desired_rtg is not None:
                # 선형 감소로 0에 수렴
                progress = current_step / max(1, total_steps - 1)
                return desired_rtg * (1.0 - progress)
            
            # 훈련 모드: Property 모델 기반 RTG 계산
            predictions = self.property_model.predict(
                input_sequence=state_action,
                attention_mask=attention_mask,
                return_hidden=False
            )
            
            # 현재 스텝의 즉시 리워드 계산 (정규화된 거리 기반)
            current_reward = 0.0
            valid_properties = 0
            
            for prop_name in ['entanglement', 'fidelity', 'expressibility']:
                if prop_name in predictions and prop_name in target_properties:
                    pred_val = predictions[prop_name].item()
                    target_val = target_properties[prop_name]
                    
                    # 정규화된 거리 기반 리워드: r = 1 - |pred - target|
                    distance = abs(pred_val - target_val)
                    reward = 1.0 - min(distance, 1.0)  # [0, 1] 클램핑
                    
                    current_reward += reward
                    valid_properties += 1
            
            if valid_properties > 0:
                current_reward /= valid_properties
            else:
                current_reward = 0.0  # 기본값
            
            # RTG 계산: 남은 스텝에 따라 0으로 수렴
            remaining_steps = max(0, total_steps - current_step - 1)
            
            if remaining_steps == 0:
                # 마지막 스텝: RTG = 0
                return 0.0
            else:
                # 이전 스텝들: 기하급수적 감소로 0에 수렴
                if gamma < 1.0:
                    rtg_value = current_reward * (1 - gamma ** remaining_steps) / (1 - gamma)
                else:
                    rtg_value = current_reward * remaining_steps
                
                # 정규화: 최대 가능한 RTG로 나누기
                max_possible_rtg = remaining_steps if gamma >= 1.0 else 1.0 / (1 - gamma)
                normalized_rtg = rtg_value / max_possible_rtg
                
                return float(normalized_rtg)
    
    def _prepare_batch(self, batch_data: List[Dict]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        배치 데이터를 텐서로 변환
        
        Args:
            batch_data: 배치 데이터 리스트
            
        Returns:
            batch_states: [batch_size, seq_len, d_model]
            batch_targets: 정답 속성값들
            batch_masks: [batch_size, seq_len]
        """
        # 구현 필요: 데이터셋 형식에 맞게 조정
        # 현재는 플레이스홀더
        batch_size = len(batch_data)
        max_seq_len = max(len(item.get('sequence', [])) for item in batch_data)
        d_model = self.property_config.d_model
        
        batch_states = torch.zeros(batch_size, max_seq_len, d_model, device=self.device)
        batch_masks = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=self.device)
        
        batch_targets = {
            'entanglement': torch.zeros(batch_size, max_seq_len, device=self.device),
            'fidelity': torch.zeros(batch_size, max_seq_len, device=self.device),
            'expressibility': torch.zeros(batch_size, max_seq_len, device=self.device)
        }
        
        # TODO: 실제 데이터 변환 로직 구현
        
        return batch_states, batch_targets, batch_masks


def create_rtg_calculator(property_checkpoint_path: str, 
                         property_config_path: str,
                         device: str = 'cpu') -> RTGCalculator:
    """
    RTG Calculator 생성 함수
    
    Args:
        property_checkpoint_path: Property 모델 체크포인트 경로
        property_config_path: Property 모델 설정 경로
        device: 계산 디바이스
        
    Returns:
        RTGCalculator 인스턴스
    """
    from models.property_prediction_transformer import create_property_prediction_model, PropertyPredictionConfig
    
    # 설정 로드
    with open(property_config_path, 'r') as f:
        config_dict = json.load(f)
    
    property_config = PropertyPredictionConfig(**config_dict)
    
    # 모델 생성 및 가중치 로드
    property_model = create_property_prediction_model(property_config)
    checkpoint = torch.load(property_checkpoint_path, map_location=device)
    property_model.load_state_dict(checkpoint['model_state_dict'])
    
    return RTGCalculator(property_model, property_config, device)


if __name__ == "__main__":
    # 사용 예시
    calculator = create_rtg_calculator(
        property_checkpoint_path="property_prediction_checkpoints/best_model.pt",
        property_config_path="configs/property_config.json",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 데이터셋에 대해 RTG 사전 계산
    calculator.precompute_rtg_for_dataset(
        dataset_path="raw_data/merged_data.json",
        output_path="processed_data/rtg_dataset.json",
        batch_size=32
    )
