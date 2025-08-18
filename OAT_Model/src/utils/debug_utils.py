"""
Unified Debug Utilities Module

통합된 디버그 유틸리티 함수들을 제공합니다.
모든 모듈에서 일관된 디버깅 환경을 제공합니다.
"""

import os
import time
import torch
from typing import Any, Dict, List, Union
from dotenv import load_dotenv

load_dotenv()

# 디버그 설정
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() in ["true", "1", "t"]  # Set to True to enable debug logging
DEBUG_DETAILED = os.getenv("DEBUG_DETAILED", "False").lower() in ["true", "1", "t"]  # Set to True for detailed tensor information

def set_debug_mode(enabled: bool, detailed: bool = True) -> None:
    """디버그 모드 설정
    
    Args:
        enabled: 디버그 모드 활성화 여부
        detailed: 상세 정보 포함 여부
    """
    global DEBUG_MODE, DEBUG_DETAILED
    DEBUG_MODE = enabled
    DEBUG_DETAILED = detailed

def is_debug_enabled() -> bool:
    """디버그 모드 활성화 여부 반환"""
    return DEBUG_MODE

def debug_log(message: str, level: str = "INFO", module: str = "DEBUG") -> None:
    """통합 디버그 로깅 유틸리티
    
    Args:
        message: 로그 메시지
        level: 로그 레벨 (INFO, WARN, ERROR)
        module: 모듈 이름
    """
    if DEBUG_MODE:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{module}] [{level}] {message}")

def debug_tensor_info(tensor_name: str, tensor_data: Any, detailed: bool = False, module: str = "DEBUG") -> None:
    """통합 텐서 정보 디버깅
    
    Args:
        tensor_name: 텐서 이름
        tensor_data: 텐서 데이터
        detailed: 상세 정보 포함 여부
        module: 모듈 이름
    """
    if not DEBUG_MODE:
        return
        
    if tensor_data is None:
        debug_log(f"{tensor_name}: None", "WARN", module)
        return
        
    if isinstance(tensor_data, torch.Tensor):
        shape = tuple(tensor_data.shape)
        dtype = tensor_data.dtype
        device = tensor_data.device
        
        # Basic info
        info = f"{tensor_name}: shape={shape}, dtype={dtype}, device={device}"
        
        if detailed and DEBUG_DETAILED:
            # Additional statistics
            if tensor_data.numel() > 0:
                min_val = tensor_data.min().item()
                max_val = tensor_data.max().item()
                
                # Handle different dtypes for mean/std calculation
                if tensor_data.dtype in [torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128]:
                    mean_val = tensor_data.mean().item()
                    std_val = tensor_data.std().item()
                    info += f", min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}, std={std_val:.4f}"
                else:
                    # For integer types, convert to float for mean/std
                    float_tensor = tensor_data.float()
                    mean_val = float_tensor.mean().item()
                    std_val = float_tensor.std().item()
                    info += f", min={min_val}, max={max_val}, mean={mean_val:.4f}, std={std_val:.4f}"
                
                # Check for NaN/Inf (only for floating point types)
                if tensor_data.dtype in [torch.float16, torch.float32, torch.float64]:
                    has_nan = torch.isnan(tensor_data).any().item()
                    has_inf = torch.isinf(tensor_data).any().item()
                    if has_nan or has_inf:
                        info += f", NaN={has_nan}, Inf={has_inf}"
        
        debug_log(info, "INFO", module)
    elif isinstance(tensor_data, dict):
        debug_log(f"{tensor_name}: dict with keys={list(tensor_data.keys())}", "INFO", module)
        if detailed and DEBUG_DETAILED:
            for key, value in tensor_data.items():
                debug_tensor_info(f"{tensor_name}.{key}", value, detailed=False, module=module)
    elif isinstance(tensor_data, list):
        debug_log(f"{tensor_name}: list with length={len(tensor_data)}", "INFO", module)
    else:
        debug_log(f"{tensor_name}: {type(tensor_data)} = {tensor_data}", "INFO", module)

def debug_print(*args, **kwargs) -> None:
    """기존 호환성을 위한 디버그 출력"""
    if DEBUG_MODE:
        print("[DEBUG]", *args, **kwargs)

# 모듈별 디버그 함수들
def embed_debug_log(message: str, level: str = "INFO") -> None:
    """임베딩 파이프라인 디버그 로깅"""
    debug_log(message, level, "EMBED_DEBUG")

def dt_debug_log(message: str, level: str = "INFO") -> None:
    """Decision Transformer 임베딩 디버그 로깅"""
    debug_log(message, level, "DT_EMBED_DEBUG")

def pred_debug_log(message: str, level: str = "INFO") -> None:
    """Predictor 임베딩 디버그 로깅"""
    debug_log(message, level, "PRED_EMBED_DEBUG")

def embed_debug_tensor(tensor_name: str, tensor_data: Any, detailed: bool = False) -> None:
    """임베딩 파이프라인 텐서 디버깅"""
    debug_tensor_info(tensor_name, tensor_data, detailed, "EMBED_DEBUG")

def dt_debug_tensor(tensor_name: str, tensor_data: Any, detailed: bool = False) -> None:
    """Decision Transformer 임베딩 텐서 디버깅"""
    debug_tensor_info(tensor_name, tensor_data, detailed, "DT_EMBED_DEBUG")

def pred_debug_tensor(tensor_name: str, tensor_data: Any, detailed: bool = False) -> None:
    """Predictor 임베딩 텐서 디버깅"""
    debug_tensor_info(tensor_name, tensor_data, detailed, "PRED_EMBED_DEBUG")
