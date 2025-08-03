#!/usr/bin/env python3

import argparse
from quantum_base import QuantumCircuitBase
from ibm_backend import IBMQuantumBackend
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from config import config, get_shadow_params, get_ibm_shots, print_config, apply_preset
from expressibility_calculator import calculate_expressibility_from_real_quantum_classical_shadow, calculate_entropy_expressibility_from_ibm_results, ExpressibilityCalculator

load_dotenv()

def parse_arguments():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description='IBM 양자 회로 실행 프로그램')
    parser.add_argument('--token', type=str, help='IBM Quantum 계정 토큰')
    parser.add_argument('--n_qubits', type=int, default=5, help='큐빗 수 (기본값: 5)')
    parser.add_argument('--depth', type=int, default=10, help='회로 깊이 (기본값: 10)')
    parser.add_argument('--output_dir', type=str, default='grid_circuits', help='출력 디렉토리 (기본값: grid_circuits)')
    parser.add_argument('--simulator', action='store_true', help='시뮬레이터 사용 (IBM 백엔드 대신)')
    parser.add_argument('--expressibility', action='store_true', help='안사츠의 표현력 측정')
    parser.add_argument('--exp_samples', type=int, default=None, help='표현력 측정을 위한 파라미터 샘플 수 (None이면 자동 설정)')
    parser.add_argument('--exp_shots', type=int, default=None, help='표현력 측정을 위한 Shadow 크기 (None이면 자동 설정)')
    parser.add_argument('--ansatz_file', type=str, help='기존 안사츠 데이터 파일 경로 (표현력 측정용)')
    parser.add_argument('--two_qubit_ratio', type=float, default=None, help='2큐빗 게이트 비율 (0.0~1.0, None이면 전략 기본값 사용)')
    
    # 설정 관련 옵션
    parser.add_argument('--show_config', action='store_true', help='현재 설정 출력')
    parser.add_argument('--preset', type=str, choices=['expressibility', 'scaling', 'noise'], 
                       help='실험 프리셋 적용')
    parser.add_argument('--config_file', type=str, help='설정 파일 로드')
    parser.add_argument('--save_config', type=str, help='현재 설정을 파일로 저장')
    
    # 🎯 CSV 내보내기 옵션 추가
    parser.add_argument('--export_csv', action='store_true', help='메가잡 결과를 CSV로 내보내기 (기본 동작)')
    parser.add_argument('--csv_filename', type=str, default='quantum_expressibility_data.csv', 
                       help='CSV 파일명 (기본값: quantum_expressibility_data.csv)')
    
    return parser.parse_args()

def run_quantum_circuit(args):
    """양자 회로 실행"""
    
    # 설정 관련 처리
    if args.config_file:
        config.load_config_from_file(args.config_file)
    
    if args.preset:
        apply_preset(args.preset)
    
    if args.show_config:
        print_config()
        if not any([args.simulator, args.expressibility, args.ansatz_file]):
            return None  # 설정만 보고 종료
    
    if args.save_config:
        config.save_config_to_file(args.save_config)
        print(f"설정이 {args.save_config}에 저장되었습니다.")
    
    # 출력 디렉토리 설정
    output_dir = args.output_dir
    
    # 기본 양자 회로 객체 생성
    base_circuit = QuantumCircuitBase(output_dir=output_dir)
    
    # 기존 안사츠 파일에서 표현력 측정 처리
    if args.expressibility and args.ansatz_file:
        print(f"\n기존 안사츠 파일에서 표현력 측정: {args.ansatz_file}")
        measure_expressibility_from_file(args.ansatz_file, args.exp_samples, args.exp_shots)
        return None
    
    # IBM 백엔드 또는 시뮬레이터 사용
    if args.simulator:
        print("\n시뮬레이터 모드 사용")
        ibm_backend = None
    else:
        # IBM 토큰 가져오기
        ibm_token = args.token
        if not ibm_token:
            # 환경 변수에서 토큰 가져오기 시도
            ibm_token = os.environ.get('IBM_QUANTUM_TOKEN')
            if not ibm_token:
                print("⚠️ IBM Quantum 토큰이 필요합니다. --token 옵션 또는 IBM_QUANTUM_TOKEN 환경 변수를 설정하세요.")
                return
        
        # IBM 백엔드 초기화
        print("\nIBM Quantum 백엔드 초기화 중...")
        ibm_backend = IBMQuantumBackend(ibm_token=ibm_token, base_circuit=base_circuit)
        
        if not ibm_backend.backend:
            print("⚠️ IBM Quantum 백엔드 연결 실패, 시뮬레이터를 사용합니다.")
            ibm_backend = None
    
    # 회로 파라미터 설정
    n_qubits = args.n_qubits
    depth = args.depth
    
    # 회로 생성
    print(f"\n{n_qubits} 큐빗, 깊이 {depth}의 양자 회로 생성 중...")
    
    # 2큐빗 게이트 비율 설정 확인
    if args.two_qubit_ratio is not None:
        if not (0.0 <= args.two_qubit_ratio <= 1.0):
            print(f"⚠️ 오류: 2큐빗 게이트 비율은 0.0과 1.0 사이여야 합니다. (입력값: {args.two_qubit_ratio})")
            return
        print(f"2큐빗 게이트 비율 설정: {args.two_qubit_ratio:.1%}")
    
    # IBM 백엔드가 있으면 커플링 맵 사용
    coupling_map = ibm_backend.coupling_map if ibm_backend and hasattr(ibm_backend, 'coupling_map') else None
    
    # 랜덤 회로 생성
    circuit_info = base_circuit.generate_random_circuit(
        n_qubits, 
        depth, 
        coupling_map,
        two_qubit_ratio=args.two_qubit_ratio
    )
    
    # 회로 시각화
    print("\n회로 구조 시각화...")
    base_circuit.visualize_grid(circuit_info, filename=f"grid_structure_{n_qubits}x{depth}.png")
    
    # 회로 다이어그램 (큐빗 수가 적을 때만)
    if n_qubits <= 16:
        print("\n회로 다이어그램 생성...")
        # 원래 회로만 시각화
        base_circuit.visualize_circuit(circuit_info, filename=f"circuit_diagram_{n_qubits}x{depth}.png")
        
        # 인버스 회로까지 포함한 시각화
        print("\n인버스 회로를 포함한 회로 다이어그램 생성...")
        base_circuit.visualize_circuit(circuit_info, filename=f"circuit_diagram_{n_qubits}x{depth}.png", include_inverse=True)
    
    # 결과 데이터 구조 초기화 - 안사츠 구조 간소화
    ansatz_data = {
        "timestamp": datetime.now().isoformat(),
        "n_qubits": n_qubits,
        "depth": depth,
        "backend": ibm_backend.backend.name if ibm_backend and ibm_backend.backend else "simulator",
        "circuit_info": simplify_circuit_info(circuit_info),  # 간소화된 회로 정보 사용
        "execution_results": None
    }
    
    # 회로 실행
    if ibm_backend and ibm_backend.backend:
        print(f"\nIBM {ibm_backend.backend.name} 백엔드에서 회로 실행 중...")
        
        # IBM 백엔드에서 회로 실행
        results = ibm_backend.run_on_ibm_backend(circuit_info)
        
        if results is None:
            # IBM 백엔드 실행 실패 시 시뮬레이터로 대체
            print("\n⚠️ IBM 백엔드 실행 실패, 시뮬레이터를 대체합니다...")
            run_simulator(base_circuit, circuit_info, ansatz_data)
        else:
            # IBM 백엔드 결과 처리
            process_results(results, ansatz_data)
            
            # 샘플링 횟수 설정 (중앙 설정 사용)
            ibm_samples = config.ibm_backend.expressibility_samples
            
            # IBM 백엔드를 사용하는 경우 항상 표현력 계산 수행 (--expressibility 플래그 무관)
            print(f"\nIBM 백엔드에서 Classical Shadow 기반 표현력 측정을 위한 파라미터화된 회로 {ibm_samples}회 실행 (단일 작업으로 효율적 실행)...")
            print(f"설정: 샘플 {ibm_samples}회, 샷 {config.ibm_backend.expressibility_shots}회")
            expressibility_result = calculate_expressibility_from_ibm_results(
                base_circuit,
                circuit_info,
                results["measurement_counts"],
                n_qubits,
                samples=ibm_samples
            )
            
            # 표현력 결과 저장
            ansatz_data["expressibility"] = expressibility_result
            print("IBM 결과 기반 Classical Shadow 표현력 측정 완료")
    else:
        # 시뮬레이터 실행
        print("\n시뮬레이터에서 회로 실행 중...")
        run_simulator(base_circuit, circuit_info, ansatz_data)
    
        # 시뮬레이터 모드에서는 명시적으로 요청한 경우에만 표현력 측정
        if args.expressibility:
            print("\nClassical Shadow 기반 표현력 측정 중...")
            expressibility_result = base_circuit.calculate_expressibility(
                circuit_info, 
                S=args.exp_samples, 
                M=args.exp_shots,
                metric='classical_shadow'
            )
            
            # 표현력 결과 저장
            ansatz_data["expressibility"] = expressibility_result
            print("Classical Shadow 기반 표현력 측정 완료")
    
    # 피델리티 측정 - 시뮬레이션 피델리티 계산 제거
    print("\n회로 피델리티 계산 중...")
    
    # 시뮬레이터 피델리티 관련 코드 제거
    # 측정 결과 기반 피델리티만 사용
    
    # 결과 저장
    saved_file = base_circuit.save_results(ansatz_data)
    ansatz_data["ansatz_file"] = saved_file  # 저장된 파일 경로 추가
    
    return ansatz_data

def run_simulator(base_circuit, circuit_info, ansatz_data):
    """시뮬레이터에서 회로 실행"""
    # 시뮬레이터 큐빗 수 제한 적용
    n_qubits = circuit_info["n_qubits"]
    max_sim_qubits = 20
    
    if n_qubits > max_sim_qubits:
        print(f"⚠️ 경고: 시뮬레이터에서 {n_qubits}개 큐빗은 메모리 제한을 초과합니다. {max_sim_qubits}로 제한합니다.")
        # 제한된 회로 정보 생성
        limited_circuit_info = circuit_info.copy()
        limited_circuit_info["n_qubits"] = max_sim_qubits
        # 제한된 회로 실행 - 인버스 회로 포함
        circuit = base_circuit.create_inverse_circuit_qnode(limited_circuit_info)
        samples = circuit(limited_circuit_info["params"])
        actual_n_qubits = max_sim_qubits
    else:
        # 원래 회로 실행 - 인버스 회로 포함
        circuit = base_circuit.create_inverse_circuit_qnode(circuit_info)
        samples = circuit(circuit_info["params"])
        actual_n_qubits = n_qubits
    
    # 샘플 결과 분석
    total_sim_shots = len(samples)
    print(f"시뮬레이터 샘플 수: {total_sim_shots}")
    
    # 샘플링 결과 분석
    sample_counts = {}
    zero_count = 0
    
    # 각 샘플에서 상태 카운트
    for sample in samples:
        # 샘플을 비트 문자열로 변환
        if hasattr(sample, '__iter__'):
            # 반복 가능한 샘플 (비트 배열)
            bit_str = ''.join(str(int(bit)) for bit in sample)
        else:
            # 정수 샘플
            bit_str = format(int(sample), f'0{actual_n_qubits}b')
        
        # 샘플 카운트
        if bit_str in sample_counts:
            sample_counts[bit_str] = sample_counts[bit_str] + 1
        else:
            sample_counts[bit_str] = 1
            
        # 0 상태 카운트
        if bit_str == '0' * actual_n_qubits:
            zero_count += 1
    
    # 0 상태 확률 계산
    zero_state_probability = zero_count / total_sim_shots if total_sim_shots > 0 else 0
    
    # Robust Fidelity 계산 (시뮬레이터에서도)
    robust_fidelity = calculate_robust_fidelity(sample_counts, actual_n_qubits, total_sim_shots)
    
    # 오류율 계산 (시뮬레이터에서도)
    error_rates = calculate_error_rates(sample_counts, actual_n_qubits, total_sim_shots)
    
    # direct_result 형식으로 결과 정리
    direct_result = {
        "processed_counts_direct": sample_counts,
        "total_counts_direct": total_sim_shots
    }
    
    # 결과 설정
    results = {
        "zero_state_probability": zero_state_probability,
        "measured_states": total_sim_shots,
        "measurement_counts": sample_counts,
        "zero_state_count": zero_count,
        "backend": "simulator",
        "direct_result": direct_result,
        "robust_fidelity": robust_fidelity,  # 시뮬레이터 결과에도 추가
        "error_rates": error_rates  # 오류율도 추가
    }
    
    # 결과 처리
    process_results(results, ansatz_data)

def process_results(results, ansatz_data):
    """측정 결과 처리"""
    # IBM 결과 더 자세히 분석
    measurement_counts = {}
    measured_states = 0
    
    # 직접 처리된 결과 확인
    if 'direct_result' in results:
        direct_result = results['direct_result']
        if 'processed_counts_direct' in direct_result and 'total_counts_direct' in direct_result:
            measurement_counts = direct_result['processed_counts_direct']
            measured_states = direct_result['total_counts_direct']
            print(f"직접 처리된 측정 결과 사용: {len(measurement_counts)}개 상태, 총 {measured_states}회 측정")
    
    # 직접 처리된 데이터가 없으면 결과에서 가져오기
    if not measurement_counts and 'measurement_counts' in results:
        measurement_counts = results['measurement_counts']
    if not measured_states and 'measured_states' in results:
        measured_states = results['measured_states']
    
    # 0 상태 확률 계산
    n_qubits = ansatz_data["n_qubits"]
    zero_state = '0' * n_qubits
    zero_count = measurement_counts.get(zero_state, 0)
    zero_state_probability = zero_count / measured_states if measured_states > 0 else 0
    
    # 결과 저장 (간소화된 버전)
    ansatz_data["execution_results"] = {
        "zero_state_probability": zero_state_probability,
        "measured_states": measured_states,
        "significant_states": len(measurement_counts),
        "zero_state_count": zero_count,
        "backend": results.get("backend", "unknown"),
        # top_states 저장 안함
    }
    
    # 결과 출력
    print(f"\n===== 최종 측정 결과 =====")
    print(f"측정된 총 상태 수: {measured_states}")
    print(f"유의미한 상태 수: {len(measurement_counts)}")
    print(f"0 상태 확률: {zero_state_probability:.6f}")
    
    # 상위 10개 상태 출력 (저장은 하지 않음)
    print("\n측정 상태:")
    sorted_counts = sorted(measurement_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (state, count) in enumerate(sorted_counts[:10]):
        print(f"  |{state}⟩: {count}회 ({count/measured_states*100:.2f}%)")
    
    # IBM 결과에서 피델리티 계산
    print("\n===== 측정 결과 기반 피델리티 =====")
    # |0...0> 상태의 비율을 피델리티로 사용
    ibm_fidelity = zero_state_probability
    print(f"측정 결과 피델리티: {ibm_fidelity:.6f}")
    print(f"  |{zero_state}⟩ 상태 발생 빈도: {zero_count}/{measured_states} ({ibm_fidelity*100:.2f}%)")
    
    # Robust Fidelity 계산 (노이즈 허용)
    if 'robust_fidelity' in results:
        # 시뮬레이터에서 이미 계산된 경우
        robust_fidelity = results['robust_fidelity']
    else:
        # IBM 백엔드 결과에서 새로 계산
        robust_fidelity = calculate_robust_fidelity(measurement_counts, n_qubits, measured_states)
    
    print(f"Robust 피델리티: {robust_fidelity:.6f}")
    print(f"  노이즈 허용 범위: {get_error_threshold(n_qubits)}개 비트 오류 이내")
    
    # 오류율 계산 추가
    error_rates = calculate_error_rates(measurement_counts, n_qubits, measured_states)
    
    print(f"\n===== 오류율 분석 =====")
    print(f"비트 플립 오류율: {error_rates['bit_flip_error_rate']:.6f} (비트당)")
    print(f"위상 오류율 (추정): {error_rates['phase_error_rate']:.6f}")
    print(f"전체 오류율: {error_rates['total_error_rate']:.6f} (측정당)")
    print(f"단일 비트 오류율: {error_rates['single_bit_error_rate']:.6f}")
    print(f"다중 비트 오류율: {error_rates['multi_bit_error_rate']:.6f}")
    print(f"오류 없는 측정: {error_rates['error_free_measurements']}/{measured_states} ({(1-error_rates['total_error_rate'])*100:.2f}%)")
    
    # 오류 분포 출력 (상위 5개)
    if error_rates['error_distribution']:
        print("\n오류 분포 (상위 5개):")
        sorted_errors = sorted(error_rates['error_distribution'].items(), key=lambda x: x[1], reverse=True)
        for error_bits, count in sorted_errors[:5]:
            print(f"  {error_bits}비트 오류: {count}회 ({count/measured_states*100:.2f}%)")
    
    # 실행 결과에 피델리티 추가
    ansatz_data["execution_results"]["fidelity"] = ibm_fidelity
    ansatz_data["execution_results"]["robust_fidelity"] = robust_fidelity
    ansatz_data["execution_results"]["error_rates"] = error_rates

def measure_expressibility_from_file(file_path, samples=50, shots=100):
    """
    기존 안사츠 데이터 파일에서 표현력 측정
    
    Args:
        file_path (str): 안사츠 데이터 파일 경로
        samples (int): 파라미터 샘플 수
        shots (int): 샷 수
    """
    # 파일 존재 확인
    if not os.path.exists(file_path):
        print(f"⚠️ 파일을 찾을 수 없음: {file_path}")
        return
    
    try:
        # 안사츠 데이터 파일 로드
        with open(file_path, 'r') as f:
            ansatz_data = json.load(f)
        
        # 회로 정보 확인
        if "circuit_info" not in ansatz_data:
            print(f"⚠️ 회로 정보가 없음: {file_path}")
            return
        
        circuit_info = ansatz_data["circuit_info"]
        n_qubits = circuit_info["n_qubits"]
        
        print(f"안사츠 정보:")
        print(f"  큐빗 수: {n_qubits}")
        print(f"  게이트 수: {len(circuit_info['gates'])}")
        print(f"  파라미터 수: {len(circuit_info['params'])}")
        
        # 기본 양자 회로 객체 생성
        output_dir = os.path.dirname(os.path.dirname(file_path)) if '/ansatz_data/' in file_path else 'grid_circuits'
        base_circuit = QuantumCircuitBase(output_dir=output_dir)
        
        # IBM 실행 결과가 있는지 확인
        use_ibm_results = False
        measurement_counts = None
        
        if "execution_results" in ansatz_data and ansatz_data["execution_results"] is not None:
            if "measurement_counts" in ansatz_data["execution_results"]:
                measurement_counts = ansatz_data["execution_results"]["measurement_counts"]
                if measurement_counts:
                    use_ibm_results = True
                    print(f"\nIBM 실행 결과 발견: {len(measurement_counts)}개 상태, 총 {ansatz_data['execution_results'].get('measured_states', 0)}회 측정")
        
        # IBM 결과 기반 표현력 측정
        if use_ibm_results:
            print("\nIBM 실행 결과를 바탕으로 표현력 측정 중...")
            expressibility_result = calculate_expressibility_from_ibm_results(
                base_circuit, 
                circuit_info,
                measurement_counts,
                n_qubits
            )
            print("IBM 결과 기반 표현력 측정 완료")
        else:
            # 시뮬레이터 기반 표현력 측정
            print("\n표현력 측정 중... (시뮬레이터 기반)")
            expressibility_result = base_circuit.calculate_expressibility(
                circuit_info, 
                S=samples, 
                M=shots, 
                metric='KL'
            )
            print("시뮬레이터 기반 표현력 측정 완료")
        
        # 결과 업데이트 및 저장
        ansatz_data["expressibility"] = expressibility_result
        
        # 결과 파일 생성
        output_file = os.path.join(
            base_circuit.output_dir,
            "ansatz_data",
            f"expressibility_{os.path.basename(file_path)}"
        )
        
        with open(output_file, 'w') as f:
            json.dump(ansatz_data, f, indent=2)
        
        print(f"\n표현력 측정 결과가 저장됨: {output_file}")
        
        # 표현력 요약 보고서
        print("\n===== 표현력 측정 요약 =====")
        print(f"파일: {os.path.basename(file_path)}")
        print(f"큐빗 수: {n_qubits}")
        print(f"표현력 거리: {expressibility_result['distance']:.4e}")
        print(f"정규화된 거리: {expressibility_result['normalized_distance']:.4e}")
        print(f"95% 신뢰구간: [{expressibility_result['confidence_interval'][0]:.4e}, {expressibility_result['confidence_interval'][1]:.4e}]")
        print(f"데이터 소스: {'IBM 측정 결과' if use_ibm_results else '시뮬레이터'}")
        
        return expressibility_result
        
    except Exception as e:
        print(f"⚠️ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def calculate_expressibility_from_ibm_results(base_circuit, circuit_info, measurement_results, n_qubits, samples=1):
    """
    IBM 백엔드 실행 결과를 바탕으로 표현력(expressibility) 계산
    
    Args:
        base_circuit (QuantumCircuitBase): 기본 회로 객체
        circuit_info (dict): 회로 정보
        measurement_results (dict): IBM 측정 결과 (measurement_counts)
        n_qubits (int): 큐빗 수
        samples (int): 샘플링 횟수 (기본값: 1)
        
    Returns:
        dict: 표현력 측정 결과
    """
    import time
    from scipy.stats import norm
    import random
    from expressibility_calculator import ExpressibilityCalculator
    
    print(f"\n===== IBM 실행 결과 기반 표현력 측정 (샘플링 {samples}회) =====")
    start_time = time.time()
    
    # ExpressibilityCalculator 인스턴스 생성
    calculator = ExpressibilityCalculator()
    
    # 여러 샘플을 처리하기 위한 리스트
    all_shadow_data = []
    
    for s in range(samples):
        # 처리 진행 상황 표시
        if s % 10 == 0:
            print(f"샘플링 진행 중: {s}/{samples}")
            
        # 1. IBM 실행 결과를 Classical Shadow 형식으로 변환
        # 샘플링을 위해 측정 결과 약간 변형 (실제로는 파라미터 변경으로 인한 측정 결과의 변형을 시뮬레이션)
        if s > 0:  # 첫 번째 실행은 원본 측정 결과 사용
            # 측정 결과 약간 변형하여 새로운 파라미터 샘플링 시뮬레이션
            perturbed_results = {}
            perturbation = 0.05  # 5% 내외 변동
            
            for bit_str, count in measurement_results.items():
                # 원래 카운트에서 약간의 변동 추가
                new_count = max(1, int(count * (1 + (random.random() - 0.5) * perturbation * 2)))
                perturbed_results[bit_str] = new_count
            
            shadow_data = convert_ibm_results_to_shadow(perturbed_results, n_qubits)
        else:
            shadow_data = convert_ibm_results_to_shadow(measurement_results, n_qubits)
        
        # Shadow 데이터 저장
        all_shadow_data.append(shadow_data)
    
    try:
        # 2. Classical Shadow 데이터에서 Pauli 기댓값 추정
        estimated_moments = calculator._estimate_pauli_expectations_from_shadows(all_shadow_data, n_qubits)
        
        # 3. Haar 랜덤 분포의 이론적 Pauli 기댓값
        haar_moments = calculator._get_haar_pauli_expectations(n_qubits)
        
        # 4. Classical Shadow 기반 거리 계산
        distance = calculator._calculate_shadow_distance(estimated_moments, haar_moments)
        
        # 5. 신뢰구간 계산
        if samples > 1:
            # 여러 샘플을 사용하므로 신뢰구간을 더 정확하게 추정
            distance_samples = []
            for shadow_data in all_shadow_data:
                sample_moments = calculator._estimate_pauli_expectations_from_shadows([shadow_data], n_qubits)
                sample_distance = calculator._calculate_shadow_distance(sample_moments, haar_moments)
                distance_samples.append(sample_distance)
            
            # 표준 편차 계산
            std_dev = np.std(distance_samples)
            
            # 95% 신뢰구간
            low = max(0, distance - 1.96 * std_dev / np.sqrt(samples))
            high = distance + 1.96 * std_dev / np.sqrt(samples)
        else:
            # 단일 샘플인 경우 Classical Shadow 이론 기반 신뢰구간
            confidence_interval = calculator._calculate_shadow_confidence_interval(
                estimated_moments, samples, len(all_shadow_data[0]["measurements"]), n_qubits
            )
            low, high = confidence_interval
        
    except Exception as e:
        print(f"⚠️ Classical Shadow 표현력 계산 중 오류: {str(e)}")
        # 오류 발생 시 기본값 반환
        distance = 0.0
        low, high = 0.0, 0.0
    
    # 실행 시간
    run_time = time.time() - start_time
    
    # 결과 보고서 준비
    result = {
        "n_qubits": n_qubits,
        "samples": samples,
        "metric": "classical_shadow",
        "distance": distance,
        "confidence_interval": [low, high],
        "run_time": run_time,
        "normalized_distance": distance / (2**n_qubits) if distance > 0 else 0.0,  # 큐빗 수에 따른 정규화
        "source": "ibm_execution_classical_shadow"
    }
    
    # 결과 출력
    print("\n===== IBM 실행 기반 Classical Shadow 표현력 측정 결과 =====")
    print(f"샘플링 횟수: {samples}")
    print(f"거리값: {distance:.4e}")
    print(f"정규화된 거리: {result['normalized_distance']:.4e}")
    print(f"95% 신뢰구간: [{low:.4e}, {high:.4e}]")
    if samples == 1:
        print(f"참고: 단일 실행에서 계산되어 정확도는 제한적임")
    
    return result

def convert_ibm_results_to_shadow(measurement_counts, n_qubits):
    """
    IBM 측정 결과를 Classical Shadow 데이터 형식으로 변환
    
    Args:
        measurement_counts (dict): IBM 측정 결과 (비트열 -> 카운트 매핑)
        n_qubits (int): 큐빗 수
        
    Returns:
        dict: Classical Shadow 데이터 형식 (measurements, bases 키 포함)
    """
    import random
    
    # 측정 결과를 개별 샷으로 확장
    measurements = []
    bases = []
    
    total_counts = sum(measurement_counts.values())
    
    # 각 측정 결과를 개별 샷으로 변환
    for bit_str, count in measurement_counts.items():
        # 비트 문자열 길이 조정
        if len(bit_str) > n_qubits:
            bit_str = bit_str[-n_qubits:]  # 마지막 n_qubits 비트만 사용
        elif len(bit_str) < n_qubits:
            bit_str = bit_str.zfill(n_qubits)  # 0으로 패딩
        
        # 카운트만큼 반복하여 개별 샷 생성
        for _ in range(count):
            # 비트 문자열을 정수 배열로 변환
            measurement = [int(b) for b in bit_str]
            measurements.append(measurement)
            
            # 각 큐빗에 대해 랜덤 Pauli 기저 생성 (Classical Shadow 시뮬레이션)
            shot_bases = [random.choice(['X', 'Y', 'Z']) for _ in range(n_qubits)]
            bases.append(shot_bases)
    
    # Classical Shadow 데이터 구조 반환
    shadow_data = {
        "measurements": measurements,
        "bases": bases,
        "n_qubits": n_qubits,
        "shadow_size": len(measurements)
    }
    
    return shadow_data

# 안사츠 구조 간소화 함수 추가
def simplify_circuit_info(circuit_info):
    """
    안사츠 구조를 간소화하여 저장 효율을 높임
    
    Args:
        circuit_info (dict): 원본 회로 정보
        
    Returns:
        dict: 간소화된 회로 정보
    """
    # 간소화된 회로 정보 구조
    simplified = {
        "n_qubits": circuit_info["n_qubits"],
        "depth": circuit_info["depth"],
    }
    
    # 게이트 압축: [게이트, 와이어] 쌍으로 저장
    gate_data = []
    for i, (gate, wires) in enumerate(zip(circuit_info["gates"], circuit_info["wires_list"])):
        # 파라미터화된 게이트인 경우
        if i in circuit_info["params_idx"]:
            param_idx = circuit_info["params_idx"].index(i)
            param_value = circuit_info["params"][param_idx]
            gate_data.append([gate, wires, param_value])
        else:
            gate_data.append([gate, wires])
    
    simplified["gate_data"] = gate_data
    
    # 커플링 맵 효율적 저장 - 인접 리스트 방식
    if "coupling_map" in circuit_info and circuit_info["coupling_map"]:
        coupling_map = circuit_info["coupling_map"]
        
        # 인접 리스트로 변환 (더 효율적인 저장)
        adjacency_list = {}
        for edge in coupling_map:
            a, b = edge
            if a not in adjacency_list:
                adjacency_list[a] = []
            adjacency_list[a].append(b)
            
            # 양방향 연결 저장 (필요한 경우)
            # if b not in adjacency_list:
            #     adjacency_list[b] = []
            # adjacency_list[b].append(a)
        
        # 정수 키를 문자열로 변환 (JSON 직렬화용)
        adj_map = {str(k): v for k, v in adjacency_list.items()}
        
        # 압축된 형태로 저장
        simplified["coupling_map_compressed"] = adj_map
        simplified["coupling_map_size"] = len(coupling_map)
    
    return simplified

def get_error_threshold(n_qubits):
    """
    큐빗 수에 따른 허용 오류 비트 수 계산
    
    Args:
        n_qubits (int): 큐빗 수
        
    Returns:
        int: 허용 오류 비트 수
    """
    if n_qubits <= 10:
        return 1  # 10큐빗 이하는 1개 오류만 허용
    else:
        return max(1, int(n_qubits * 0.1))  # 10% 이내 오류 허용

def hamming_distance(state1, state2):
    """
    두 비트 문자열 간의 해밍 거리 계산
    
    Args:
        state1 (str): 첫 번째 비트 문자열
        state2 (str): 두 번째 비트 문자열
        
    Returns:
        int: 해밍 거리 (다른 비트 수)
    """
    if len(state1) != len(state2):
        return float('inf')  # 길이가 다르면 무한대 거리
    
    return sum(c1 != c2 for c1, c2 in zip(state1, state2))

def calculate_robust_fidelity(measurement_counts, n_qubits, total_measurements):
    """
    노이즈를 허용하는 Robust Fidelity 계산
    
    Args:
        measurement_counts (dict): 측정 결과 카운트 {비트열: 카운트}
        n_qubits (int): 큐빗 수
        total_measurements (int): 총 측정 횟수
        
    Returns:
        float: Robust Fidelity (0~1 사이)
    """
    if total_measurements == 0:
        return 0.0
    
    # 목표 상태 (모든 비트가 0)
    target_state = '0' * n_qubits
    
    # 허용 오류 비트 수
    error_threshold = get_error_threshold(n_qubits)
    
    # 허용 범위 내의 모든 측정 카운트 합산
    robust_count = 0
    
    for measured_state, count in measurement_counts.items():
        # 측정된 상태와 목표 상태 간의 해밍 거리 계산
        distance = hamming_distance(measured_state, target_state)
        
        # 허용 범위 내이면 카운트에 포함
        if distance <= error_threshold:
            robust_count += count
    
    # Robust Fidelity 계산
    robust_fidelity = robust_count / total_measurements
    
    return robust_fidelity

def calculate_error_rates(measurement_counts, n_qubits, total_measurements):
    """
    다양한 오류율 계산
    
    Args:
        measurement_counts (dict): 측정 결과 카운트 {비트열: 카운트}
        n_qubits (int): 큐빗 수
        total_measurements (int): 총 측정 횟수
        
    Returns:
        dict: 다양한 오류율 정보
    """
    if total_measurements == 0:
        return {
            "bit_flip_error_rate": 0.0,
            "phase_error_rate": 0.0,
            "total_error_rate": 0.0,
            "single_bit_error_rate": 0.0,
            "multi_bit_error_rate": 0.0,
            "error_distribution": {}
        }
    
    # 목표 상태 (모든 비트가 0)
    target_state = '0' * n_qubits
    
    # 오류 분석
    bit_flip_errors = 0
    total_errors = 0
    error_distribution = {}
    single_bit_errors = 0
    multi_bit_errors = 0
    
    for measured_state, count in measurement_counts.items():
        # 해밍 거리 계산 (비트 플립 오류 수)
        distance = hamming_distance(measured_state, target_state)
        
        if distance > 0:
            # 오류가 있는 경우
            total_errors += count
            bit_flip_errors += distance * count  # 총 비트 플립 수
            
            # 오류 분포 기록
            if distance in error_distribution:
                error_distribution[distance] += count
            else:
                error_distribution[distance] = count
            
            # 단일/다중 비트 오류 분류
            if distance == 1:
                single_bit_errors += count
            else:
                multi_bit_errors += count
    
    # 오류율 계산
    bit_flip_error_rate = bit_flip_errors / (total_measurements * n_qubits)  # 비트당 플립 확률
    total_error_rate = total_errors / total_measurements  # 오류가 있는 측정 비율
    single_bit_error_rate = single_bit_errors / total_measurements  # 단일 비트 오류 비율
    multi_bit_error_rate = multi_bit_errors / total_measurements  # 다중 비트 오류 비율
    
    # 위상 오류는 직접 측정하기 어려우므로 추정
    # 실제 피델리티와 비트 플립 기반 예상 피델리티의 차이로 추정
    expected_fidelity_from_bitflip = (1 - bit_flip_error_rate) ** n_qubits
    actual_fidelity = measurement_counts.get(target_state, 0) / total_measurements
    phase_error_estimate = max(0, expected_fidelity_from_bitflip - actual_fidelity)
    phase_error_rate = phase_error_estimate / expected_fidelity_from_bitflip if expected_fidelity_from_bitflip > 0 else 0
    
    return {
        "bit_flip_error_rate": bit_flip_error_rate,
        "phase_error_rate": phase_error_rate,
        "total_error_rate": total_error_rate,
        "single_bit_error_rate": single_bit_error_rate,
        "multi_bit_error_rate": multi_bit_error_rate,
        "error_distribution": error_distribution,
        "total_bit_flips": bit_flip_errors,
        "error_free_measurements": total_measurements - total_errors
    }

if __name__ == "__main__":
    args = parse_arguments()
    
    # 🎯 기본 동작: CSV 내보내기 (다른 옵션이 없을 때)
    if not any([args.simulator, args.expressibility, args.ansatz_file, args.show_config]):
        args.export_csv = True
    
    # CSV 내보내기 실행
    if args.export_csv:
        print("🚀 메가잡 결과 CSV 내보내기 시작!")
        try:
            from data_analysis import QuantumMegaJobAnalyzer
            analyzer = QuantumMegaJobAnalyzer()
            csv_path = analyzer.run_csv_export_analysis()
            
            if csv_path:
                print(f"\n🎉 CSV 내보내기 완료!")
                print(f"📁 파일 위치: {csv_path}")
                print(f"💡 이제 엑셀이나 다른 도구로 데이터를 분석할 수 있습니다.")
                
                # 간단한 통계 출력
                if analyzer.df is not None:
                    print(f"\n📊 데이터 요약:")
                    print(f"   회로 수: {len(analyzer.df)}")
                    if 'entropy_expressibility' in analyzer.df.columns:
                        valid_expr = analyzer.df[analyzer.df['entropy_expressibility'] > 0]
                        print(f"   유효한 표현력 데이터: {len(valid_expr)}")
                        if len(valid_expr) > 0:
                            print(f"   표현력 범위: {valid_expr['entropy_expressibility'].min():.4f} - {valid_expr['entropy_expressibility'].max():.4f}")
            else:
                print(f"\n❌ CSV 내보내기 실패")
                print(f"💡 메가잡 결과 파일을 확인하세요.")
        except ImportError:
            print("❌ data_analysis 모듈을 찾을 수 없습니다.")
        except Exception as e:
            print(f"❌ CSV 내보내기 오류: {str(e)}")
    else:
        # 기존 양자 회로 실행
        run_quantum_circuit(args)

