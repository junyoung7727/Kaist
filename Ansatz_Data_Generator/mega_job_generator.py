#!/usr/bin/env python3
"""
Mega Job 600개 양자 회로 실행 스크립트
- 모든 회로를 한 번의 거대한 job으로 제출
- 최대 효율성과 최소 대기 시간
"""

import os
import sys
import time
import gc
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

# 환경 변수 로드
load_dotenv()

# 현재 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 리팩토링된 모듈 임포트
from src.core.circuit_generator import generate_all_circuits
from src.core.qiskit_helper import convert_to_qiskit_circuits
from src.core.job_runner import run_mega_job, calculate_optimal_shots_and_batching
from src.core.result_processor import process_mega_results, save_mega_results, analyze_two_qubit_ratio_results
from src.utils.file_utils import setup_directories

# 새로운 expressibility 모듈 임포트
from src.calculators.expressibility import ExpressibilityCalculator
from src.calculators.expressibility.simulator import SimulatorExpressibilityCalculator
from src.calculators.expressibility.ibm import IBMExpressibilityCalculator
from src.calculators.expressibility.entropy import (
    calculate_entropy_expressibility,
    entropy_based_expressibility,
    calculate_angle_entropy,
    calculate_entropy_expressibility_from_ibm_results
)

# 메트릭 계산 모듈 임포트
from src.calculators.metrics.circuit_metrics import (
    calculate_circuit_metrics,
    calculate_gate_counts,
    calculate_circuit_depth,
    calculate_two_qubit_gate_ratio
)

# 설정 모듈 임포트
from src.config import config, setup_directories, apply_preset


def run_mega_job_generator(preset_name: Optional[str] = None):
    """
    메인 2큐빗 게이트 비율 테스트 실행 함수 (1800개 회로)
    
    Args:
        preset_name: 실행할 프리셋 설정 이름 (선택 사항: "expressibility", "scaling", "noise")
    """
    print("\n🚀 Mega Job Generator 시작!")
    print("=" * 80)
    print("📌 기능: 회로 생성 → 변환 → 배치 실행 → 결과 분석")
    print("=" * 80)
    
    # 프리셋 설정 적용 (지정된 경우)
    from src.config import config as global_config
    if preset_name:
        # config_obj를 첫 번째 인자로 전달하고 결과를 전역 설정으로 적용
        global_config = apply_preset(global_config, preset_name)
        # config 변수를 전역 설정으로 업데이트
        globals()['config'] = global_config
        print(f"🔧 '{preset_name}' 프리셋 설정이 적용되었습니다.")
    
    # 디렉토리 설정
    setup_directories(config)
    
    try:
        # IBM 백엔드 설정
        from src.backends import IBMBackendManager
        
        # 백엔드 연결 시도 및 실패 시 시뮬레이터로 대체
        try:
            if config.experiment_mode == "IBM_QUANTUM":
                ibm_backend = IBMBackendManager()
                backend_name = ibm_backend.name
                print(f"💻  IBM 백엔드 초기화 완료: {backend_name}")
            elif config.experiment_mode == "SIMULATOR":
                ibm_backend = None
                backend_name = "AerSimulator"
                ibm_backend = IBMBackendManager(use_simulator=True)
                print(f"💻  시뮬레이터 백엔드 초기화 완료: {backend_name}")
            else:
                raise ValueError(f"Invalid experiment mode: {config.experiment_mode}")
        except RuntimeError as e:
            print(f"\n⚠️ IBM Quantum 백엔드 연결 오류: {str(e)}")
            print("⚠️ AerSimulator를 사용하여 계속 진행합니다.")
            ibm_backend = IBMBackendManager(use_simulator=True)
            backend_name = ibm_backend.name
            print(f"💻  시뮬레이터 백엔드 초기화 완료: {backend_name}")
        
        # 테스트용 회로 생성
        all_circuits = generate_all_circuits()
        
        if not all_circuits:
            print("❌ 회로 생성 실패!")
            return
        
        # Qiskit 회로로 변환
        qiskit_circuits, circuit_metadata = convert_to_qiskit_circuits(all_circuits, ibm_backend)
        
        if not qiskit_circuits:
            print("❌ Qiskit 회로 변환 실패!")
            return
        
        # 배치 최적화 계산 - config 속성 접근 사용
        batch_info = calculate_optimal_shots_and_batching(
            total_circuits=len(qiskit_circuits),
            target_total_shots=config.ibm_backend.target_total_shots,  # 속성 접근 방식
            max_executions=config.ibm_backend.max_executions_per_job  # 속성 접근 방식
        )
        
        # IBM 백엔드에서 배치 실행
        optimal_shots = batch_info["optimal_shots"]
        result, execution_time, updated_metadata = run_mega_job(
            qiskit_circuits,
            circuit_metadata,
            ibm_backend,
            shots=optimal_shots
        )
        
        if result is None:
            print("❌ 회로 실행 실패!")
            return
        
        # 결과 처리
        all_results = process_mega_results(result, updated_metadata, execution_time, ibm_backend)
        
        # 결과 저장
        save_info = save_mega_results(all_results, None)
        
        # 2큐빗 게이트 비율별 분석
        analysis_result = analyze_two_qubit_ratio_results(all_results)
        
        print("\n✅ Mega Job Generator 모든 작업 완료!")
        print(f"   저장된 결과: {save_info['json_file']}")
        print(f"   CSV 요약: {save_info['csv_file']}")
        print(f"   분석 데이터: {len(analysis_result)} 회로")
        
        # 메모리 정리
        all_circuits = None
        qiskit_circuits = None
        circuit_metadata = None
        result = None
        all_results = None
        gc.collect()
        
    except Exception as e:
        print(f"\n❌ 실행 중 오류 발생: {str(e)}")


if __name__ == "__main__":
    start_time = time.time()
    
    run_mega_job_generator('test')
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  총 실행 시간: {elapsed_time:.2f}초")
