#!/usr/bin/env python3
"""
파일 및 디렉토리 유틸리티 모듈 - 파일 및 디렉토리 관리 관련 함수를 제공합니다.
"""

import os
import sys
from typing import Dict, List, Any, Optional

# 현재 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def setup_directories():
    """
    프로그램 실행에 필요한 모든 디렉토리 생성
    
    Returns:
        Dict[str, str]: 생성된 디렉토리 경로 정보
    """
    # 기본 디렉토리 목록
    directories = {
        "experiments": "experiments",
        "results": "experiments/results",
        "checkpoints": "experiments/checkpoints",
        "models": "models",
        "reports": "reports",
        "plots": "plots",
        "plots_training": "plots/training",
        "plots_results": "plots/results",
        "plots_analysis": "plots/analysis"
    }
    
    print("📁 프로그램 디렉토리 구조 설정 중...")
    
    # 디렉토리 생성
    for name, path in directories.items():
        try:
            os.makedirs(path, exist_ok=True)
            print(f"   ✓ {name} 디렉토리 확인: {path}")
        except Exception as e:
            print(f"   ⚠️ {name} 디렉토리 생성 실패: {str(e)}")
    
    # 사용자 홈 디렉토리에 .ansatz_data 디렉토리 생성 (설정 및 캐시용)
    try:
        home_dir = os.path.expanduser("~")
        config_dir = os.path.join(home_dir, ".ansatz_data")
        os.makedirs(config_dir, exist_ok=True)
        directories["config"] = config_dir
        print(f"   ✓ 사용자 설정 디렉토리 확인: {config_dir}")
    except Exception as e:
        print(f"   ⚠️ 사용자 설정 디렉토리 생성 실패: {str(e)}")
    
    # README 파일 생성
    try:
        readme_path = os.path.join("experiments", "README.md")
        if not os.path.exists(readme_path):
            with open(readme_path, "w") as f:
                f.write("# Ansatz 데이터 생성기 실험 디렉토리\n\n")
                f.write("이 디렉토리에는 양자 회로 실험 결과가 저장됩니다.\n\n")
                f.write("- `results`: 개별 실험 결과 (JSON, CSV, HDF5)\n")
                f.write("- `checkpoints`: 학습 체크포인트\n")
            print(f"   ✓ README 파일 생성: {readme_path}")
    except Exception as e:
        print(f"   ⚠️ README 파일 생성 실패: {str(e)}")
    
    print("✅ 디렉토리 설정 완료!")
    
    return directories


def save_json_data(data: Any, file_path: str, indent: int = 2) -> bool:
    """
    데이터를 JSON 파일로 저장
    
    Args:
        data: 저장할 데이터
        file_path (str): 저장할 파일 경로
        indent (int): JSON 들여쓰기 수준 (기본값: 2)
        
    Returns:
        bool: 성공 여부
    """
    import json
    
    try:
        # 파일 디렉토리 확인 및 생성
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)
        return True
    except Exception as e:
        print(f"⚠️ JSON 저장 실패 ({file_path}): {str(e)}")
        return False


def load_json_data(file_path: str) -> Optional[Any]:
    """
    JSON 파일에서 데이터 로드
    
    Args:
        file_path (str): 로드할 파일 경로
        
    Returns:
        Optional[Any]: 로드된 데이터 또는 None (실패시)
    """
    import json
    
    try:
        if not os.path.exists(file_path):
            print(f"⚠️ 파일이 존재하지 않음: {file_path}")
            return None
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"⚠️ JSON 로드 실패 ({file_path}): {str(e)}")
        return None


def ensure_path_exists(path: str, is_file: bool = False) -> bool:
    """
    경로가 존재하는지 확인하고, 필요하면 생성
    
    Args:
        path (str): 확인할 경로
        is_file (bool): 파일 경로인지 여부 (기본값: False)
        
    Returns:
        bool: 성공 여부
    """
    try:
        if is_file:
            dir_path = os.path.dirname(path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        print(f"⚠️ 경로 생성 실패 ({path}): {str(e)}")
        return False
