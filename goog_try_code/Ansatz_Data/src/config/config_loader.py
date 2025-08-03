#!/usr/bin/env python3
"""
설정 로더 모듈

이 모듈은 다양한 소스(기본 설정, 로컬 설정 파일, 환경 변수)로부터
설정을 로드하고 병합하는 기능을 제공합니다.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional, Union
from copy import deepcopy

# 기본 설정 가져오기
from src.config.default_config import get_default_config
# ConfigBox 유틸리티 클래스 가져오기
from src.utils.config_utils import ConfigBox

# 로깅 설정
logger = logging.getLogger(__name__)


def load_local_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    로컬 설정 파일에서 설정 로드
    
    Args:
        config_path (Optional[str]): 설정 파일 경로. None인 경우 기본 위치 사용
        
    Returns:
        Dict[str, Any]: 로컬 설정 딕셔너리. 파일이 없는 경우 빈 딕셔너리 반환
    """
    # 기본 경로 설정
    if config_path is None:
        # 프로젝트 루트 디렉토리에서 local_config.json 찾기
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(root_dir, "local_config.json")
    
    # 파일 존재 여부 확인
    if not os.path.exists(config_path):
        logger.debug(f"로컬 설정 파일을 찾을 수 없습니다: {config_path}")
        return {}
    
    # JSON 파일 로드
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"로컬 설정 파일을 로드했습니다: {config_path}")
        return config
    except Exception as e:
        logger.error(f"로컬 설정 파일 로드 오류: {e}")
        return {}


def load_env_config() -> Dict[str, Any]:
    """
    환경 변수에서 설정 로드
    
    환경 변수는 'QEXPRESS_' 접두사로 시작해야 함
    예: QEXPRESS_DEBUG=True, QEXPRESS_IBM_QUANTUM__API_TOKEN=xxxx
    
    Returns:
        Dict[str, Any]: 환경 변수에서 로드된 설정 딕셔너리
    """
    env_config = {}
    prefix = "QEXPRESS_"
    
    for key, value in os.environ.items():
        if key.startswith(prefix):
            # 접두사 제거
            config_key = key[len(prefix):].lower()
            
            # 중첩 키 처리 (예: IBM_QUANTUM__API_TOKEN -> ibm_quantum.api_token)
            if "__" in config_key:
                main_key, sub_key = config_key.split("__", 1)
                if main_key not in env_config:
                    env_config[main_key] = {}
                env_config[main_key][sub_key] = parse_env_value(value)
            else:
                env_config[config_key] = parse_env_value(value)
    
    return env_config


def parse_env_value(value: str) -> Any:
    """
    환경 변수 값을 적절한 타입으로 변환
    
    Args:
        value (str): 환경 변수 값
        
    Returns:
        Any: 변환된 값
    """
    # 불리언 값
    if value.lower() in ('true', 'yes', '1'):
        return True
    if value.lower() in ('false', 'no', '0'):
        return False
    
    # 숫자 값
    try:
        # 정수
        if value.isdigit():
            return int(value)
        # 실수
        return float(value)
    except ValueError:
        pass
    
    # 리스트 (콤마로 구분된 값)
    if ',' in value:
        return [parse_env_value(item.strip()) for item in value.split(',')]
    
    # 기본: 문자열 반환
    return value


def merge_configs(*configs) -> Dict[str, Any]:
    """
    여러 설정 딕셔너리를 병합
    
    우선 순위는 인자 순서대로 (마지막 인자가 가장 높음)
    
    Returns:
        Dict[str, Any]: 병합된 설정 딕셔너리
    """
    result = {}
    
    for config in configs:
        deep_update(result, config)
    
    return result


def deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """
    중첩된 딕셔너리의 재귀적 업데이트
    
    Args:
        target (Dict[str, Any]): 업데이트할 대상 딕셔너리
        source (Dict[str, Any]): 소스 딕셔너리
        
    Returns:
        Dict[str, Any]: 업데이트된 딕셔너리
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            # 중첩된 딕셔너리인 경우 재귀 호출
            deep_update(target[key], value)
        else:
            # 그 외 경우 값 덮어쓰기
            target[key] = deepcopy(value)
    
    return target


def setup_directories(config: Dict[str, Any]) -> None:
    """
    설정에 지정된 디렉토리 생성
    
    Args:
        config (Dict[str, Any]): 설정 딕셔너리
    """
    paths = config.get('paths', {})
    for key, path in paths.items():
        if key.endswith('_dir') and path:
            os.makedirs(path, exist_ok=True)
            logger.debug(f"디렉토리 생성 (없는 경우): {path}")


def load_config(local_config_path: Optional[str] = None) -> ConfigBox:
    """
    모든 소스에서 설정 로드 및 병합
    
    Args:
        local_config_path (Optional[str]): 로컬 설정 파일 경로. None인 경우 기본 위치 사용
        
    Returns:
        ConfigBox: 병합된 최종 설정 (.을 사용해 속성처럼 접근 가능)
    """
    # 1. 기본 설정 가져오기
    default_config = get_default_config()
    
    # 2. 로컬 설정 로드
    local_config = load_local_config(local_config_path)
    
    # 3. 환경 변수에서 설정 로드
    env_config = load_env_config()
    
    # 4. 모든 설정 병합 (우선순위: 기본 < 로컬 < 환경변수)
    merged_config = merge_configs(default_config, local_config, env_config)
    
    # ConfigBox를 사용하여 속성 접근 방식(.) 지원
    config_box = ConfigBox(merged_config)
    
    # 필요한 디렉토리 생성
    setup_directories(config_box)
    
    return config_box


def get_shadow_params(config: ConfigBox):
    """
    Classical Shadow 관련 파라미터를 반환하는 포팅된 함수
    
    리팩토링된 구조에 맞게 config 객체에서 값을 참조
    
    Returns:
        Dict[str, Any]: Shadow 파라미터 맞
    """
    return {
        "S": config.get("expressibility", {}).get("n_samples", 1000),  # 파라미터 샘플링 수 
        "M": config.get("expressibility", {}).get("shadow_measurements", 100),  # Shadow 측정 수
        "confidence_level": config.get("expressibility", {}).get("confidence_level", 0.95),  # 신뢰 수준
    }

def apply_preset(config_obj, preset_name: str) -> ConfigBox:
    """
    설정에 지정된 프리셋 적용
    
    Args:
        config_obj: 현재 설정 객체 (ConfigBox 또는 dict)
        preset_name: 적용할 프리셋 이름
    
    Returns:
        ConfigBox: 프리셋이 적용된 설정 객체 (속성 접근 방식 사용 가능)
    """
    # 사용 가능한 프리셋 정의
    presets = {
        "expressibility": {
            "simulator": {
                "fidelity_shots": 512,
                "entropy_shots": 2048
            },
            "classical_shadow": {
                "default_samples": 2000,
                "shadow_shots": 4096
            }
        },
        "scaling": {
            "data_generation": {
                "batch_size": 50,
                "max_batches": 5
            },
            "ibm_backend": {
                "target_total_shots": 4000000,
                "max_batch_size": 150,
                "default_shots": 8192,
                "max_executions_per_job": 300
            }
        },
        "noise": {
            "ibm_backend": {
                "noise_model": "ibmq_jakarta"
            },
            "simulator": {
                "default_shots": 512
            }
        },
        "test": {  # 간이 테스트용 프리셋
            "data_generation": {
                "batch_size": 2,
                "max_batches": 2,
                "qubit_presets": [3, 5],  # 간이 테스트용 쿠빗 수 설정
                "min_depth": 5,
                "max_depth": 10
            },
            "circuit_generation_params": {
                "two_qubit_ratios": [0.3, 0.5],  # 테스트용 2쿠빗 게이트 비율
                "circuits_per_config": 1
            },
            "ibm_backend": {
                "target_total_shots": 2000,
                "max_batch_size": 2,
                "default_shots": 512,
                "max_executions_per_job": 5
            },
            "simulator": {
                "default_shots": 128,
                "fidelity_shots": 64
            }
        }
    }
    
    # 프리셋 없음
    if preset_name not in presets:
        logger.warning(f"정의되지 않은 프리셋: {preset_name}. 기본 설정을 사용합니다.")
        # 입력 객체가 ConfigBox가 아니라면 변환
        if not isinstance(config_obj, ConfigBox):
            return ConfigBox(config_obj)
        return config_obj
    
    # 프리셋 적용
    logger.info(f"프리셋 적용: {preset_name}")
    
    # 입력 객체가 ConfigBox 인스턴스인 경우
    if isinstance(config_obj, ConfigBox):
        # 디셔너리로 변환하여 복사
        config_dict = config_obj.to_dict()
        config_copy = deepcopy(config_dict)
    else:
        # 일반 디셔너리로 간주하고 복사
        config_copy = deepcopy(config_obj)
    
    # 프리셋 설정 적용
    preset = presets[preset_name]
    
    for section, values in preset.items():
        # 섹션이 존재하지 않으면 생성
        if section not in config_copy:
            config_copy[section] = {}
            
        # 값 적용
        for key, value in values.items():
            config_copy[section][key] = value
    
    # ConfigBox로 변환하여 반환
    return ConfigBox(config_copy)
