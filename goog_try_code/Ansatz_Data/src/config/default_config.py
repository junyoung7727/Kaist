#!/usr/bin/env python3
"""
기본 설정 값 모듈

이 모듈은 프로젝트 전체에서 사용되는 기본 설정 값을 정의합니다.
환경별 설정을 위해서는 local_config.py를 사용하세요.
"""

import os
from typing import Dict, Any, List, Optional

# 시드 설정 (재현성)
DEFAULT_SEED = 42

# IBM Quantum 관련 설정
IBM_QUANTUM = {
    "hub": "ibm-q",
    "group": "open",
    "project": "main",
    "provider": None,  # 런타임에 설정됨
    "max_jobs": 5,     # 동시 실행 최대 작업 수
}

# IBM 백엔드 설정
IBM_BACKEND = {
    "default_shots": 256,             # 기본 샷 수
    "target_total_shots": 100000,      # 목표 총 샷 수
    "max_batch_size": 1000,             # 배치당 최대 회로 수
    "max_executions_per_job": 1000      # job당 최대 실행 수
}

# 데이터 생성 설정 - 중앙집중식 관리를 위해 추가
DATA_GENERATION = {
    "qubit_presets": [5,7,10,15],  # 쿠빗 수 리스트
    "batch_size": 50,                 # 배치 크기
    "max_batches": 10,                # 최대 배치 수
    "depth_presets": [1,2,3,5]   # 깊이 리스트
}

# 회로 생성 파라미터
CIRCUIT_GENERATION_PARAMS = {
    "two_qubit_ratios": [0.2,0.5,0.8],  # 2쿠빗 게이트 비율 리스트
    "circuits_per_config": 10,                     # 각 회로 설정당 생성할 회로 수
    "generation_strategy": "hardware_efficient"     # 회로 생성 전략
}

# 실험 모드 설정
EXPERIMENT_MODES = ["SIMULATOR", "IBM_QUANTUM", "LOCAL_QUANTUM"]
DEFAULT_EXPERIMENT_MODE = "SIMULATOR"

# 기본 양자 회로 설정
DEFAULT_CIRCUIT = {
    "n_qubits": 5,
    "depth": 10,
    "two_qubit_ratio": 0.5,  # 2큐빗 게이트 비율
    "random_seed": DEFAULT_SEED,
    "parameter_count": 10,
    "max_shots": 1024,
    "optimization_level": 1,   # 최적화 레벨 (0-3)
}

# 표현력 계산 설정
EXPRESSIBILITY = {
    "n_samples": 50,  # 파라미터 샘플링 수
    "metric": "fidelity",  # 기본 메트릭: "fidelity", "classical_shadow", "entropy" 중 하나
    "simulator_shots": 256,  # 시뮬레이터 측정 횟수
    "backend_shots": 256,   # IBM 백엔드 측정 횟수
    "confidence_level": 0.95,  # 신뢰 수준
    "shadow_measurements": 50,  # Classical shadow 측정 수
    "distance_metric": "kl"
}

# 시뮬레이터 설정
SIMULATOR = {
    "max_fidelity_qubits": 20,  # 피델리티 계산을 위한 최대 큐빗 수
    "fidelity_shots": 256,      # 피델리티 계산용 샷 수
    "fidelity_kl_num_bins": 100,  # KL 발산 계산용 히스토그램 빈 수
    "max_expressibility_qubits": 15,  # 표현력 계산을 위한 최대 큐빗 수
    "enable_noise": False,      # 노이즈 모델 활성화 여부
    "seed": DEFAULT_SEED        # 시뮬레이터 시드
}

# 메타데이터 설정
METADATA = {
    "version": "1.0.0",
    "project": "Quantum Expressibility",
    "authors": ["Junyoung Jung"],
}

# 파일 경로 설정
PATHS = {
    "output_dir": os.path.join(os.getcwd(), "outputs"),
    "plots_dir": os.path.join(os.getcwd(), "outputs", "plots"),
    "data_dir": os.path.join(os.getcwd(), "outputs", "data"),
    "logs_dir": os.path.join(os.getcwd(), "outputs", "logs"),
}

# 로깅 설정
LOGGING = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": os.path.join(PATHS["logs_dir"], "app.log"),
}

# 디버그 모드
DEBUG = False

def get_default_config() -> Dict[str, Any]:
    """
    기본 설정 값들을 딕셔너리 형태로 반환

    Returns:
        Dict[str, Any]: 기본 설정 딕셔너리
    """
    return {
        "seed": DEFAULT_SEED,
        "ibm_quantum": IBM_QUANTUM.copy(),
        "ibm_backend": IBM_BACKEND.copy(),  # IBM 백엔드 설정 추가
        "data_generation": DATA_GENERATION.copy(),  # 데이터 생성 설정 추가
        "circuit_generation_params": CIRCUIT_GENERATION_PARAMS.copy(),  # 회로 생성 파라미터 추가
        "experiment_mode": DEFAULT_EXPERIMENT_MODE,
        "circuit": DEFAULT_CIRCUIT.copy(),
        "expressibility": EXPRESSIBILITY.copy(),
        "simulator": SIMULATOR.copy(),  # 시뮬레이터 설정 추가
        "metadata": METADATA.copy(),
        "paths": PATHS.copy(),
        "logging": LOGGING.copy(),
        "debug": DEBUG
    }
