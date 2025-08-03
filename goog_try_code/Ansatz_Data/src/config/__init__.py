"""
설정 관리 모듈

이 모듈은 애플리케이션 전체에서 사용되는 설정을 관리합니다.
기본 설정, 로컬 설정 파일, 환경 변수에서 설정을 로드하고 병합하는 기능을 제공합니다.

사용 예시:
    from src.config import config
    
    # 딕셔너리 방식 접근
    print(config["experiment_mode"])
    
    # 또는 속성 방식 접근 (추천)
    print(config.experiment_mode)
    print(config.ibm_backend.default_shots)
    
    # 직접 로드
    from src.config import load_config
    my_config = load_config("custom_config.json")
"""

import os
import logging
from typing import Dict, Any

# 설정 로더 가져오기
from src.config.config_loader import load_config, load_local_config, load_env_config, setup_directories, apply_preset
from src.config.default_config import get_default_config

# 기본 설정 로드
config = load_config()


# 설정 관련 유틸리티 함수 노출
__all__ = [
    "config",           # 기본 설정 인스턴스
    "load_config",      # 설정 로더
    "get_default_config",  # 기본 설정 가져오기
    "load_local_config",  # 로컬 설정 로드
    "load_env_config",   # 환경 변수 설정 로드
    "setup_directories",   # 디렉토리 생성
    "apply_preset"       # 프리셋 적용
]
