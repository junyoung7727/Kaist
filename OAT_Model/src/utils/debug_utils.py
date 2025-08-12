"""
Debug Utilities Module

공통 디버그 유틸리티 함수들을 제공합니다.
중복 코드를 제거하고 일관된 디버깅 환경을 제공합니다.
"""

import os
from typing import Any

# 환경 변수로 디버그 모드 제어
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() in ['true', '1', 'yes']

def debug_print(*args, **kwargs) -> None:
    """디버그 모드일 때만 출력
    
    Args:
        *args: 출력할 인자들
        **kwargs: print 함수의 키워드 인자들
    """
    if DEBUG_MODE:
        print("[DEBUG]", *args, **kwargs)

def set_debug_mode(enabled: bool) -> None:
    """디버그 모드 설정
    
    Args:
        enabled: 디버그 모드 활성화 여부
    """
    global DEBUG_MODE
    DEBUG_MODE = enabled

def is_debug_enabled() -> bool:
    """디버그 모드 활성화 여부 반환
    
    Returns:
        bool: 디버그 모드 활성화 여부
    """
    return DEBUG_MODE

def debug_tensor_info(tensor: Any, name: str = "tensor") -> None:
    """텐서 정보 디버그 출력
    
    Args:
        tensor: 디버그할 텐서
        name: 텐서 이름
    """
    if not DEBUG_MODE:
        return
        
    try:
        import torch
        if isinstance(tensor, torch.Tensor):
            debug_print(f"{name} - shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")
            debug_print(f"{name} - contains NaN: {torch.isnan(tensor).any()}")
            debug_print(f"{name} - min/max: {tensor.min().item():.4f}/{tensor.max().item():.4f}")
        else:
            debug_print(f"{name} - type: {type(tensor)}, value: {tensor}")
    except Exception as e:
        debug_print(f"Error debugging {name}: {e}")
