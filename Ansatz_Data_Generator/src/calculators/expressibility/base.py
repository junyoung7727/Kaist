#!/usr/bin/env python3
"""
표현력(Expressibility) 계산기 기본 클래스 및 유틸리티 정의

이 모듈은 양자 회로의 표현력 계산에 필요한 기본 클래스와 유틸리티를 제공합니다.
"""

import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple, Union


class ExpressibilityCalculatorBase:
    """
    표현력 계산기 기본 클래스
    
    모든 표현력 계산기는 이 클래스를 상속받아야 합니다.
    """
    def __init__(self, seed=None):
        """
        표현력 계산기 초기화 (seed 설정으로 실험 재현성 보장)
        
        Args:
            seed (Optional[int]): 랜덤 시드 (None이면 랜덤 생성)
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def calculate_expressibility(self, circuit_info: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        """
        표현력 계산 메서드 (하위 클래스에서 구현 필요)
        
        Args:
            circuit_info (Dict[str, Any]): 회로 정보
            *args: 추가 위치 인자
            **kwargs: 추가 키워드 인자
            
        Returns:
            Dict[str, Any]: 표현력 계산 결과
            
        Raises:
            NotImplementedError: 하위 클래스에서 구현해야 함
        """
        raise NotImplementedError("Subclasses must implement calculate_expressibility")


def calculate_measurement_entropy(measurement_data: Union[Dict[str, int], List[Dict[str, Any]]], 
                                  weights=None, n_bins=None) -> float:
    """
    측정 결과 분포의 엔트로피 계산
    
    이 함수는 양자 회로의 측정 결과로부터 샤논 엔트로피를 계산합니다.
    측정 결과의 확률 분포를 기반으로 계산되며, 출력 분포의 불확실성을 측정합니다.
    
    Args:
        measurement_data: 측정 결과 (다양한 형식 지원)
            - dict: {'00': 100, '01': 50, ...} 형태의 측정 카운트
            - list: [{'state': '00', 'count': 100}, ...] 형태의 측정 리스트
        weights: 가중치 (호환성을 위해 유지, 사용하지 않음)
        n_bins: 사용하지 않음 (호환성을 위해 유지)
        
    Returns:
        float: 측정 결과 분포의 샤논 엔트로피 (비트 단위). 
              값이 클수록 출력 분포가 균일하고, 작을수록 특정 상태에 집중된 분포를 나타냅니다.
    """
    import math
    from scipy.stats import entropy
    
    # 측정 데이터 형식 변환
    counts = {}
    
    if isinstance(measurement_data, dict):
        # 딕셔너리 형식 {'00': 100, '01': 50, ...}
        counts = measurement_data
    elif isinstance(measurement_data, list):
        # 리스트 형식 [{'state': '00', 'count': 100}, ...]
        for item in measurement_data:
            if isinstance(item, dict) and 'state' in item and 'count' in item:
                counts[item['state']] = item['count']
    else:
        raise ValueError(f"지원되지 않는 측정 데이터 형식: {type(measurement_data)}")
    
    # 전체 측정 횟수 계산
    total_counts = sum(counts.values())
    
    if total_counts == 0:
        return 0.0
    
    # 확률 분포 계산
    probabilities = [count / total_counts for count in counts.values()]
    
    # 샤논 엔트로피 계산
    shannon_entropy = entropy(probabilities, base=2)
    
    return shannon_entropy
