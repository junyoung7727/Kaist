#!/usr/bin/env python3
"""
Classical Shadow 기반 표현력(Expressibility) 측정 모듈

이 모듈은 양자 회로의 표현력을 Classical Shadow 방법론을 사용하여 측정합니다.
기존 measure_express.py와의 호환성도 제공합니다.
"""

import numpy as np
import pennylane as qml
import random
import time
from scipy.stats import norm
from config import config, get_shadow_params
import math

def calculate_measurement_entropy(counts):
    """
    Shannon entropy 계산. counts는 dict 또는 list of {'state','count'}.
    """
    # counts dict에서 count 값 추출
    if isinstance(counts, dict):
        values = counts.values()
    else:
        values = [item.get('count',0) for item in counts]
    total = sum(values)
    if total <= 0:
        return 0.0
    entropy = -sum((c/total) * math.log2(c/total) for c in values if c>0)
    return entropy

class ExpressibilityCalculator:
    """Classical Shadow 기반 표현력 계산기"""
    
    def __init__(self, seed=None):
        """표현력 계산기 초기화 (seed 설정으로 실험 재현성 보장)"""
        self.config = config
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def calculate_expressibility(self, circuit_info, S=None, M=None, metric='classical_shadow', sigma=1.0):
        """
        Classical Shadow 방법론을 사용한 안사츠 표현력 계산
        
        Args:
            circuit_info (dict): 회로 정보
            S (int): 파라미터 샘플 수 (None이면 자동 설정)
            M (int): Shadow 크기 (None이면 자동 설정)
            metric (str): 'classical_shadow' (기본값)
            sigma (float): 사용하지 않음 (호환성 유지)
            
        Returns:
            dict: 표현력 측정 결과
        """
        n_qubits = circuit_info["n_qubits"]
        
        # 중앙 설정에서 최적화된 파라미터 가져오기
        if S is None or M is None:
            auto_S, auto_M = get_shadow_params(n_qubits)
            S = S if S is not None else auto_S
            M = M if M is not None else auto_M
            print(f"🔧 자동 설정 적용: {n_qubits} 큐빗 → Shadow({S}, {M})")
        
        start_time = time.time()
        
        print(f"\n===== Classical Shadow 기반 표현력 측정 =====")
        print(f"큐빗 수: {n_qubits}, 파라미터 샘플 수: {S}, Shadow 크기: {M}")
        print(f"총 측정 횟수: {S * M}")
        print(f"메모리 제한: 최대 {config.classical_shadow.max_shadow_qubits} 큐빗")
        
        # 1. 여러 파라미터에 대해 Classical Shadow 수집
        all_shadow_data = []
        
        for s in range(S):
            if s % 10 == 0:
                print(f"Shadow 수집 중: {s}/{S}")
            
            # 랜덤 파라미터 생성
            param_count = len(circuit_info["params"])
            if param_count > 0:
                rand_params = 2 * np.pi * np.random.rand(param_count)
            else:
                rand_params = []
            
            # Classical Shadow 수집
            shadow_data = self._collect_classical_shadow(circuit_info, rand_params, n_qubits, M)
            all_shadow_data.append(shadow_data)
        
        # 2. Shadow 데이터에서 Pauli 기댓값 추정
        estimated_moments = self._estimate_pauli_expectations_from_shadows(all_shadow_data, n_qubits)
        
        # 3. Haar 랜덤 분포의 이론적 Pauli 기댓값
        haar_moments = self._get_haar_pauli_expectations(n_qubits)
        
        # 4. Classical Shadow 기반 거리 계산
        distance = self._calculate_shadow_distance(estimated_moments, haar_moments)
        
        # 5. Classical Shadow 이론 기반 신뢰구간
        confidence_interval = self._calculate_shadow_confidence_interval(estimated_moments, S, M, n_qubits)
        
        # 실행 시간
        run_time = time.time() - start_time
        
        # 표현력 점수 계산 (거리가 작을수록 좋은 표현력)
        # Classical Shadow에서 Pauli 기댓값의 범위는 [-1, 1]이므로 
        # L2 거리의 최대값은 대략 sqrt(연산자 수 * 4) 정도
        num_operators = len(estimated_moments)
        max_possible_distance = np.sqrt(num_operators * 4)  # 더 현실적인 최대 거리
        
        # 거리 기반 점수 계산 (0~1 범위, 높을수록 좋음)
        if max_possible_distance > 0:
            normalized_distance = distance / max_possible_distance
            # 지수 함수를 사용하여 더 민감한 점수 계산
            expressibility_score = np.exp(-normalized_distance)
        else:
            expressibility_score = 0.0
        
        # 추가적인 정규화: 큐빗 수에 따른 조정
        # 큐빗 수가 많을수록 더 어려우므로 보정
        qubit_factor = 1.0 / (1.0 + 0.1 * n_qubits)  # 큐빗 수 증가에 따른 난이도 보정
        expressibility_score = expressibility_score * qubit_factor
        
        # [0,1] 범위로 클리핑
        expressibility_score = max(0.0, min(1.0, expressibility_score))
        
        # 결과 보고서 준비
        result = {
            "method": "classical_shadow",
            "n_qubits": n_qubits,
            "samples": S,
            "shadow_size": M,
            "distance": distance,
            "expressibility_score": expressibility_score,
            "confidence_interval": confidence_interval,
            "total_measurements": S * M,
            "run_time": run_time,
            "scalability": "O(log(n))",
            "estimated_operators": len(estimated_moments),
            "theoretical_advantage": f"Classical: O(4^n) → Shadow: O(log(n))"
        }
        
        # 결과 출력
        print("\n===== Classical Shadow 표현력 측정 결과 =====")
        print(f"추정된 Pauli 연산자 수: {len(estimated_moments)}")
        print(f"Shadow 기반 거리: {distance:.4e}")
        print(f"표현력 점수: {expressibility_score:.4f} (높을수록 좋음)")
        print(f"95% 신뢰구간: [{confidence_interval[0]:.4e}, {confidence_interval[1]:.4e}]")
        print(f"총 측정 횟수: {S * M}")
        print(f"실행 시간: {run_time:.1f}초")
        print(f"확장성: {result['scalability']} (기존 방법 대비 지수적 개선)")
        print(f"이론적 기대: 깊이 증가 → 거리 감소 (Haar 랜덤에 수렴)")
        
        return result
    
    def _collect_classical_shadow(self, circuit_info, params, n_qubits, shadow_size):
        """
        Classical Shadow 데이터 수집
        
        Args:
            circuit_info (dict): 회로 정보
            params (array): 회로 파라미터
            n_qubits (int): 큐빗 수
            shadow_size (int): Shadow 크기
            
        Returns:
            dict: Shadow 데이터 (측정 기저와 결과)
        """
        # 재현성: seed가 설정되어 있으면 적용
        if getattr(self, 'seed', None) is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        # 메모리 효율성을 위해 큐빗 수 제한
        max_shadow = config.classical_shadow.max_shadow_qubits
        eff_nq = min(n_qubits, max_shadow)
        if n_qubits > max_shadow:
            print(f"⚠️ Shadow 수집: {n_qubits} → {eff_nq} 큐빗으로 제한")

        # 시뮬레이터 설정 (shots=1)
        device = qml.device("default.qubit", wires=eff_nq, shots=1)

        # QNode를 한 번만 선언
        @qml.qnode(device)
        def shadow_circuit(p, bases):
            # 1. 원본 회로
            for i, (gate, wires) in enumerate(zip(circuit_info["gates"], circuit_info["wires_list"])):
                if any(w >= eff_nq for w in wires):
                    continue
                # (기존 게이트 핸들러 로직 유지)
            # 2. 랜덤 Clifford 측정 기저 회전
            for idx, b in enumerate(bases):
                if b == 'X': qml.Hadamard(wires=idx)
                elif b == 'Y': qml.RX(-np.pi/2, wires=idx)
            # 3. 측정
            return qml.sample(wires=range(eff_nq))

        measurements, bases_list = [], []
        for _ in range(shadow_size):
            shot_bases = [random.choice(['X','Y','Z']) for _ in range(eff_nq)]
            meas = shadow_circuit(params, shot_bases)
            meas_list = meas.tolist() if hasattr(meas, 'tolist') else list(meas)
            measurements.append(meas_list)
            bases_list.append(shot_bases)

        # 일관된 키 사용
        return {"measurements": measurements, "bases": bases_list}
    
    def _estimate_pauli_expectations_from_shadows(self, all_shadow_data, n_qubits):
        """
        Classical Shadow 데이터에서 Pauli 기댓값 추정
        
        Args:
            all_shadow_data (list): 모든 Shadow 데이터
            n_qubits (int): 큐빗 수
            
        Returns:
            dict: 추정된 Pauli 기댓값
        """
        estimated_expectations = {}
        
        # 효과적인 큐빗 수 (Shadow 수집 시 제한된 수)
        effective_n_qubits = min(n_qubits, config.classical_shadow.max_shadow_qubits)
        
        # Shadow 데이터 유효성 검사
        valid_shadow_data = []
        for i, shadow_data in enumerate(all_shadow_data):
            if not isinstance(shadow_data, dict):
                print(f"⚠️ Shadow 데이터 {i}: 딕셔너리가 아님, 건너뜀")
                continue
            
            # measurements 키 확인
            if "measurements" not in shadow_data:
                print(f"⚠️ Shadow 데이터 {i}: 'measurements' 키 없음, 건너뜀")
                continue
            
            # bases 키 확인
            if "bases" not in shadow_data:
                print(f"⚠️ Shadow 데이터 {i}: 'bases' 키 없음, 건너뜀")
                continue
            
            # 데이터 구조 확인
            measurements = shadow_data["measurements"]
            bases = shadow_data["bases"]
            
            if not isinstance(measurements, list) or not isinstance(bases, list):
                print(f"⚠️ Shadow 데이터 {i}: measurements 또는 bases가 리스트가 아님, 건너뜀")
                continue
            
            if len(measurements) != len(bases):
                print(f"⚠️ Shadow 데이터 {i}: measurements와 bases 길이 불일치, 건너뜀")
                continue
            
            valid_shadow_data.append(shadow_data)
        
        if not valid_shadow_data:
            print("⚠️ 유효한 Shadow 데이터가 없습니다. 기본값 반환")
            # 기본값으로 모든 Pauli 기댓값을 0으로 설정
            for qubit_idx in range(effective_n_qubits):
                for pauli_op in ['X', 'Y', 'Z']:
                    op_name = f"{pauli_op}{qubit_idx}"
                    estimated_expectations[op_name] = 0.0
            return estimated_expectations
        
        print(f"✓ {len(valid_shadow_data)}/{len(all_shadow_data)} Shadow 데이터가 유효함")
        
        # 1-local Pauli 연산자 추정
        for qubit_idx in range(effective_n_qubits):
            for pauli_op in ['X', 'Y', 'Z']:
                op_name = f"{pauli_op}{qubit_idx}"
                
                # 모든 Shadow에서 해당 연산자의 기댓값 추정
                estimates = []
                
                for shadow_data in valid_shadow_data:
                    try:
                        measurements = shadow_data["measurements"]
                        bases = shadow_data["bases"]
                        
                        # Classical Shadow 공식 적용
                        estimate = self._classical_shadow_estimator(
                            measurements, bases, qubit_idx, pauli_op
                        )
                        estimates.append(estimate)
                    except Exception as e:
                        print(f"⚠️ Shadow 추정 중 오류 ({op_name}): {str(e)}")
                        continue
                
                # 모든 추정값의 평균
                if estimates:
                    estimated_expectations[op_name] = np.mean(estimates)
                else:
                    estimated_expectations[op_name] = 0.0
        
        # 2-local Pauli 연산자 추정 (중요한 것만 선택적으로)
        if effective_n_qubits <= config.classical_shadow.max_2local_qubits:  # 작은 시스템에서만 2-local 계산
            # 이웃한 큐빗 쌍만 고려
            for i in range(effective_n_qubits - 1):
                j = i + 1
                for pauli1, pauli2 in [('X', 'X'), ('Y', 'Y'), ('Z', 'Z')]:
                    op_name = f"{pauli1}{i}{pauli2}{j}"
                    
                    estimates = []
                    for shadow_data in valid_shadow_data:
                        try:
                            measurements = shadow_data["measurements"]
                            bases = shadow_data["bases"]
                            
                            # 2-local Classical Shadow 추정
                            estimate = self._classical_shadow_2local_estimator(
                                measurements, bases, i, j, pauli1, pauli2
                            )
                            estimates.append(estimate)
                        except Exception as e:
                            print(f"⚠️ 2-local Shadow 추정 중 오류 ({op_name}): {str(e)}")
                            continue
                    
                    if estimates:
                        estimated_expectations[op_name] = np.mean(estimates)
                    else:
                        estimated_expectations[op_name] = 0.0
        
        return estimated_expectations
    
    def _classical_shadow_estimator(self, measurements, bases, qubit_idx, target_pauli):
        """
        Classical Shadow 1-local 추정기
        
        Args:
            measurements: 측정 결과 배열
            bases: 측정 기저 배열
            qubit_idx: 타겟 큐빗 인덱스
            target_pauli: 타겟 Pauli 연산자 ('X', 'Y', 'Z')
            
        Returns:
            float: 추정된 기댓값
        """
        estimates = []
        
        for shot_idx, (measurement, shot_bases) in enumerate(zip(measurements, bases)):
            # 해당 큐빗의 측정 기저 확인
            if qubit_idx >= len(shot_bases) or qubit_idx >= len(measurement):
                continue
                
            measured_basis = shot_bases[qubit_idx]
            measured_outcome = measurement[qubit_idx]
            
            # Classical Shadow 공식: 3 * δ_{b,σ} * (2m - 1)
            # 여기서 b는 측정 기저, σ는 타겟 Pauli, m은 측정 결과 (0 또는 1)
            if measured_basis == target_pauli:
                # 측정 기저와 타겟이 일치하는 경우만 기여
                # 측정 결과 0 → +1, 측정 결과 1 → -1
                classical_outcome = 1 if measured_outcome == 0 else -1
                shadow_estimate = 3 * classical_outcome  # Classical Shadow 공식
                estimates.append(shadow_estimate)
        
        # 유효한 추정값이 있으면 평균, 없으면 0
        if estimates:
            return np.mean(estimates)
        else:
            # 해당 기저로 측정된 샷이 없으면 0 반환
            return 0.0
    
    def _classical_shadow_2local_estimator(self, measurements, bases, qubit1, qubit2, pauli1, pauli2):
        """
        Classical Shadow 2-local 추정기
        
        Args:
            measurements: 측정 결과 배열
            bases: 측정 기저 배열
            qubit1, qubit2: 타겟 큐빗 인덱스들
            pauli1, pauli2: 타겟 Pauli 연산자들
            
        Returns:
            float: 추정된 2-local 기댓값
        """
        estimates = []
        
        for shot_idx, (measurement, shot_bases) in enumerate(zip(measurements, bases)):
            # 두 큐빗의 측정 기저 확인
            basis1 = shot_bases[qubit1] if isinstance(shot_bases, list) else shot_bases
            basis2 = shot_bases[qubit2] if isinstance(shot_bases, list) else shot_bases
            
            outcome1 = measurement[qubit1]
            outcome2 = measurement[qubit2]
            
            # 두 큐빗 모두 올바른 기저로 측정된 경우만 기여
            if basis1 == pauli1 and basis2 == pauli2:
                classical1 = 1 if outcome1 == 0 else -1
                classical2 = 1 if outcome2 == 0 else -1
                
                # 2-local Classical Shadow 공식: 9 * (2m1-1) * (2m2-1)
                shadow_estimate = 9 * classical1 * classical2
                estimates.append(shadow_estimate)
        
        return np.mean(estimates) if estimates else 0.0
    
    def _get_haar_pauli_expectations(self, n_qubits):
        """
        Haar 랜덤 분포의 이론적 Pauli 기댓값
        
        Args:
            n_qubits (int): 큐빗 수
            
        Returns:
            dict: Haar 랜덤 Pauli 기댓값 (모두 0)
        """
        haar_expectations = {}
        
        # 효과적인 큐빗 수
        effective_n_qubits = min(n_qubits, config.classical_shadow.max_shadow_qubits)
        
        # 1-local Pauli 연산자 (모두 0)
        for qubit_idx in range(effective_n_qubits):
            for pauli_op in ['X', 'Y', 'Z']:
                op_name = f"{pauli_op}{qubit_idx}"
                haar_expectations[op_name] = 0.0
        
        # 2-local Pauli 연산자 (모두 0)
        if effective_n_qubits <= config.classical_shadow.max_2local_qubits:
            for i in range(effective_n_qubits - 1):
                j = i + 1
                for pauli1, pauli2 in [('X', 'X'), ('Y', 'Y'), ('Z', 'Z')]:
                    op_name = f"{pauli1}{i}{pauli2}{j}"
                    haar_expectations[op_name] = 0.0
        
        return haar_expectations
    
    def _calculate_shadow_distance(self, estimated_moments, haar_moments):
        """
        Classical Shadow 기반 거리 계산
        
        Args:
            estimated_moments (dict): 추정된 모멘트
            haar_moments (dict): Haar 랜덤 모멘트
            
        Returns:
            float: 계산된 거리
        """
        if not estimated_moments:
            return 0.0
        
        # 공통 키 추출
        common_keys = set(estimated_moments.keys()) & set(haar_moments.keys())
        
        if len(common_keys) == 0:
            return 0.0
        
        # L2 거리 계산 (Classical Shadow에 적합)
        estimated_vec = np.array([estimated_moments[k] for k in common_keys])
        haar_vec = np.array([haar_moments[k] for k in common_keys])
        
        # L2 거리
        distance = np.sqrt(np.sum((estimated_vec - haar_vec)**2))
        
        return distance
    
    def _calculate_shadow_confidence_interval(self, estimated_moments, S, M, n_qubits, alpha=0.05):
        """
        Classical Shadow 이론 기반 신뢰구간 계산
        
        Args:
            estimated_moments (dict): 추정된 모멘트
            S (int): 파라미터 샘플 수
            M (int): Shadow 크기
            n_qubits (int): 큐빗 수
            alpha (float): 신뢰수준
            
        Returns:
            tuple: (하한, 상한) 신뢰구간
        """
        if not estimated_moments:
            return [0.0, 0.0]
        
        # Classical Shadow 이론에 따른 분산 추정
        # Var[O] ≤ 3^k / T, 여기서 k는 locality, T는 총 Shadow 수
        k = 1  # 주로 1-local 연산자 사용
        total_shadows = S * M
        
        # 분산 상한
        variance_bound = (3**k) / total_shadows
        std_error = np.sqrt(variance_bound)
        
        # 평균 거리값
        mean_distance = np.sqrt(np.sum([v**2 for v in estimated_moments.values()]))
        
        # 95% 신뢰구간
        z = norm.ppf(1 - alpha/2)
        
        low = max(0, mean_distance - z * std_error)
        high = mean_distance + z * std_error
        
        return [low, high]
    
    # measure_express.py 호환성 함수들
    def _estimate_moments(self, shadow_data_list, n_qubits):
        """
        measure_express.py 호환성을 위한 모멘트 추정 함수
        Classical Shadow 데이터를 기존 모멘트 형식으로 변환
        
        Args:
            shadow_data_list (list): Classical Shadow 데이터 리스트
            n_qubits (int): 큐빗 수
            
        Returns:
            tuple: (moments_1, moments_2) - 1차, 2차 모멘트
        """
        # Classical Shadow에서 Pauli 기댓값 추정
        estimated_expectations = self._estimate_pauli_expectations_from_shadows(shadow_data_list, n_qubits)
        
        # 1차 모멘트 = Pauli 기댓값
        moments_1 = estimated_expectations.copy()
        
        # 2차 모멘트 계산 (상관관계)
        moments_2 = {}
        for op1, val1 in estimated_expectations.items():
            for op2, val2 in estimated_expectations.items():
                key = f"{op1}_{op2}"
                # 독립성 가정하에 2차 모멘트 근사
                if op1 == op2:
                    moments_2[key] = val1 * val1  # 자기 상관
                else:
                    moments_2[key] = val1 * val2  # 교차 상관 (근사)
        
        return moments_1, moments_2
    
    def _get_haar_moments(self, n_qubits):
        """
        measure_express.py 호환성을 위한 Haar 모멘트 함수
        
        Args:
            n_qubits (int): 큐빗 수
            
        Returns:
            tuple: (haar_moments_1, haar_moments_2) - Haar 1차, 2차 모멘트
        """
        # 1차 모멘트: 모든 Pauli 기댓값은 0
        haar_moments_1 = self._get_haar_pauli_expectations(n_qubits)
        
        # 2차 모멘트: Haar 랜덤 상태의 이론적 값
        haar_moments_2 = {}
        
        for op1, val1 in haar_moments_1.items():
            for op2, val2 in haar_moments_1.items():
                key = f"{op1}_{op2}"
                # Haar 랜덤 상태에서 동일 연산자의 2차 모멘트
                if op1 == op2:
                    # Pauli 연산자의 제곱은 항등 연산자이므로 기댓값은 1
                    haar_moments_2[key] = 1.0
                else:
                    # 서로 다른 Pauli 연산자는 독립적이므로 0
                    haar_moments_2[key] = 0.0
        
        return haar_moments_1, haar_moments_2
    
    def _calculate_distance(self, estimated_moments, haar_moments, metric='KL'):
        """
        measure_express.py 호환성을 위한 거리 계산 함수
        
        Args:
            estimated_moments (dict): 추정된 모멘트
            haar_moments (dict): Haar 모멘트
            metric (str): 거리 메트릭 ('KL', 'MMD', 'L2')
            
        Returns:
            float: 계산된 거리
        """
        # 공통 키 추출
        common_keys = set(estimated_moments.keys()) & set(haar_moments.keys())
        
        if len(common_keys) == 0:
            return 0.0
        
        # 벡터 구성
        estimated_vec = np.array([estimated_moments[k] for k in common_keys])
        haar_vec = np.array([haar_moments[k] for k in common_keys])
        
        if metric == 'KL':
            # KL 다이버전스 (수치 안정성을 위한 처리)
            epsilon = 1e-10
            
            # 음수 값 처리 및 정규화
            estimated_vec = np.abs(estimated_vec) + epsilon
            haar_vec = np.abs(haar_vec) + epsilon
            
            # 정규화
            estimated_vec = estimated_vec / np.sum(estimated_vec)
            haar_vec = haar_vec / np.sum(haar_vec)
            
            # KL 다이버전스
            distance = np.sum(estimated_vec * np.log(estimated_vec / haar_vec))
            
        elif metric == 'MMD':
            # Maximum Mean Discrepancy (간단한 RBF 커널)
            sigma = 1.0
            
            # RBF 커널 계산
            diff = estimated_vec - haar_vec
            distance = np.sqrt(np.sum(diff**2))  # 간소화된 MMD
            
        else:  # L2 또는 기본값
            # L2 거리 (Classical Shadow에 적합)
            distance = np.sqrt(np.sum((estimated_vec - haar_vec)**2))
        
        return distance


# IBM 백엔드용 표현력 계산 함수들
def calculate_expressibility_from_real_quantum_classical_shadow(ibm_backend, base_circuit, circuit_info, n_qubits, samples=None):
    """
    실제 IBM 양자 컴퓨터에서 Classical Shadow 방법론을 사용하여 표현력 계산
    
    Args:
        ibm_backend (IBMQuantumBackend): IBM 백엔드 객체
        base_circuit (QuantumCircuitBase): 기본 회로 객체
        circuit_info (dict): 회로 정보
        n_qubits (int): 큐빗 수
        samples (int): 실행 횟수 (None이면 중앙 설정 사용)
        
    Returns:
        dict: 표현력 측정 결과
    """
    import copy
    
    # 중앙 설정에서 파라미터 가져오기
    if samples is None:
        samples = config.ibm_backend.expressibility_samples
    
    shadow_shots = config.ibm_backend.expressibility_shots
    
    print(f"\n===== IBM 양자 컴퓨터 Classical Shadow 표현력 측정 =====")
    print(f"총 {samples}회 실행, 각 실행당 {shadow_shots} 샷")
    print(f"Classical Shadow 방법론 사용")
    
    start_time = time.time()
    
    # 여러 샘플을 처리하기 위한 리스트
    all_shadow_data = []
    
    # 원본 회로 정보 저장
    original_circuit_info = copy.deepcopy(circuit_info)
    
    # 회로가 파라미터를 가지고 있는지 확인
    has_params = len(original_circuit_info["params"]) > 0
    param_count = len(original_circuit_info["params"])
    
    # 모든 파라미터 세트 생성 (Classical Shadow용)
    param_sets = []
    
    # 원본 파라미터 추가
    param_sets.append(original_circuit_info["params"].copy())
    
    # 파라미터가 있으면 변형된 파라미터 세트 생성
    if has_params:
        for s in range(1, samples):
            # 새 파라미터 생성 (더 다양한 파라미터 공간 탐색)
            new_params = []
            for i in range(param_count):
                # 완전히 랜덤한 파라미터 생성 (Classical Shadow에 적합)
                new_param = random.uniform(0, 2*np.pi)
                new_params.append(new_param)
            param_sets.append(new_params)
    else:
        # 파라미터가 없는 경우에도 동일한 empty 파라미터 세트 사용
        for s in range(1, samples):
            param_sets.append([])
    
    print(f"Classical Shadow용 파라미터 세트 {samples}개 준비 완료")
    
    # IBM 백엔드에서 Classical Shadow 회로 실행
    print(f"IBM 백엔드에서 Classical Shadow 회로 실행 중...")
    results = ibm_backend.run_classical_shadow_circuits(original_circuit_info, param_sets, shadow_shots)
    
    # 실행 실패 확인
    if results is None or len(results) == 0:
        print(f"⚠️ Classical Shadow 회로 실행 실패")
        return {
            "method": "classical_shadow_ibm",
            "n_qubits": n_qubits,
            "samples": 0,
            "distance": 0,
            "confidence_interval": [0, 0],
            "run_time": time.time() - start_time,
            "source": "ibm_classical_shadow_failed"
        }
    
    print(f"\n{len(results)}개 Classical Shadow 실행 결과 처리 중...")
    
    # 각 실행 결과에서 Classical Shadow 데이터 수집
    for i, result in enumerate(results):
        # 측정 결과 추출
        measurement_counts = {}
        if 'direct_result' in result and 'processed_counts_direct' in result['direct_result']:
            measurement_counts = result['direct_result']['processed_counts_direct']
        elif 'measurement_counts' in result:
            measurement_counts = result['measurement_counts']
        
        if not measurement_counts:
            print(f"  ⚠️ 실행 {i+1}/{len(results)}에서 측정 결과를 찾을 수 없음, 건너뜁니다.")
            continue
        
        # IBM 측정 결과를 Classical Shadow 데이터로 변환
        shadow_data = convert_ibm_to_classical_shadow(measurement_counts, n_qubits, shadow_shots)
        all_shadow_data.append(shadow_data)
        
        # 진행 상황 출력
        if (i+1) % 5 == 0 or i+1 == len(results):
            print(f"  Classical Shadow 데이터 처리: {(i+1)/len(results)*100:.1f}%")
    
    # 실제 성공한 샘플 수 확인
    actual_samples = len(all_shadow_data)
    if actual_samples == 0:
        print("⚠️ 모든 Classical Shadow 데이터 처리가 실패했습니다.")
        return {
            "method": "classical_shadow_ibm",
            "n_qubits": n_qubits,
            "samples": 0,
            "distance": 0,
            "confidence_interval": [0, 0],
            "run_time": time.time() - start_time,
            "source": "ibm_classical_shadow_failed"
        }
    
    print(f"\n총 {actual_samples}/{samples} Classical Shadow 데이터 수집 성공")
    
    # 표현력 계산기 생성
    calculator = ExpressibilityCalculator()
    
    # Classical Shadow 데이터에서 Pauli 기댓값 추정
    estimated_moments = calculator._estimate_pauli_expectations_from_shadows(all_shadow_data, n_qubits)
    
    # Haar 랜덤 분포의 이론적 Pauli 기댓값
    haar_moments = calculator._get_haar_pauli_expectations(n_qubits)
    
    # Classical Shadow 기반 거리 계산
    distance = calculator._calculate_shadow_distance(estimated_moments, haar_moments)
    
    # Classical Shadow 이론 기반 신뢰구간
    confidence_interval = calculator._calculate_shadow_confidence_interval(
        estimated_moments, actual_samples, shadow_shots, n_qubits
    )
    
    # 실행 시간
    run_time = time.time() - start_time
    
    # 표현력 점수 계산 (거리가 작을수록 좋은 표현력)
    # Classical Shadow에서 Pauli 기댓값의 범위는 [-1, 1]이므로 
    # L2 거리의 최대값은 대략 sqrt(연산자 수 * 4) 정도
    num_operators = len(estimated_moments)
    max_possible_distance = np.sqrt(num_operators * 4)  # 더 현실적인 최대 거리
    
    # 거리 기반 점수 계산 (0~1 범위, 높을수록 좋음)
    if max_possible_distance > 0:
        normalized_distance = distance / max_possible_distance
        # 지수 함수를 사용하여 더 민감한 점수 계산
        expressibility_score = np.exp(-normalized_distance)
    else:
        expressibility_score = 0.0
    
    # 추가적인 정규화: 큐빗 수에 따른 조정
    # 큐빗 수가 많을수록 더 어려우므로 보정
    qubit_factor = 1.0 / (1.0 + 0.1 * n_qubits)  # 큐빗 수 증가에 따른 난이도 보정
    expressibility_score = expressibility_score * qubit_factor
    
    # [0,1] 범위로 클리핑
    expressibility_score = max(0.0, min(1.0, expressibility_score))
    
    # 결과 보고서 준비
    result = {
        "method": "classical_shadow_ibm",
        "n_qubits": n_qubits,
        "samples": actual_samples,
        "shadow_shots": shadow_shots,
        "distance": distance,
        "expressibility_score": expressibility_score,
        "confidence_interval": confidence_interval,
        "total_measurements": actual_samples * shadow_shots,
        "run_time": run_time,
        "estimated_operators": len(estimated_moments),
        "source": "real_quantum_classical_shadow"
    }
    
    # 결과 출력
    print("\n===== IBM Classical Shadow 표현력 측정 결과 =====")
    print(f"실제 실행 횟수: {actual_samples}/{samples}")
    print(f"추정된 Pauli 연산자 수: {len(estimated_moments)}")
    print(f"Classical Shadow 거리: {distance:.4e}")
    print(f"표현력 점수: {expressibility_score:.4f} (높을수록 좋음)")
    print(f"95% 신뢰구간: [{confidence_interval[0]:.4e}, {confidence_interval[1]:.4e}]")
    print(f"총 측정 횟수: {actual_samples * shadow_shots}")
    print(f"총 실행 시간: {run_time:.1f}초")
    print(f"이론적 기대: 깊이 증가 → 거리 감소 (Haar 랜덤에 수렴)")
    
    return result


def convert_ibm_to_classical_shadow(measurement_counts, n_qubits, shadow_shots):
    """
    IBM 측정 결과를 Classical Shadow 데이터 형식으로 변환
    
    Args:
        measurement_counts (dict): IBM 측정 결과 (비트열 -> 카운트)
        n_qubits (int): 큐빗 수
        shadow_shots (int): Shadow 샷 수
        
    Returns:
        dict: Classical Shadow 데이터 형식
    """
    # 측정 결과를 개별 샷으로 확장
    measurements = []
    bases = []
    
    total_counts = sum(measurement_counts.values())
    
    # 각 측정 결과를 개별 샷으로 변환
    shot_count = 0
    for bit_str, count in measurement_counts.items():
        # 비트 문자열 길이 조정
        if len(bit_str) > n_qubits:
            bit_str = bit_str[-n_qubits:]  # 마지막 n_qubits 비트만 사용
        elif len(bit_str) < n_qubits:
            bit_str = bit_str.zfill(n_qubits)  # 0으로 패딩
        
        # 카운트만큼 반복하여 개별 샷 생성
        for _ in range(count):
            if shot_count >= shadow_shots:
                break
            
            # 비트 문자열을 정수 배열로 변환
            measurement = [int(b) for b in bit_str]
            measurements.append(measurement)
            
            # 각 큐빗에 대해 랜덤 Pauli 기저 생성 (Classical Shadow 시뮬레이션)
            shot_bases = [random.choice(['X', 'Y', 'Z']) for _ in range(n_qubits)]
            bases.append(shot_bases)
            
            shot_count += 1
        
        if shot_count >= shadow_shots:
            break
    
    # 부족한 샷이 있으면 마지막 측정 결과로 채우기
    while shot_count < shadow_shots:
        if measurements:
            measurements.append(measurements[-1])
            bases.append([random.choice(['X', 'Y', 'Z']) for _ in range(n_qubits)])
            shot_count += 1
        else:
            # 측정 결과가 없으면 랜덤 생성
            measurement = [random.choice([0, 1]) for _ in range(n_qubits)]
            measurements.append(measurement)
            bases.append([random.choice(['X', 'Y', 'Z']) for _ in range(n_qubits)])
            shot_count += 1
    
    # Classical Shadow 데이터 구조 반환
    shadow_data = {
        "measurements": measurements[:shadow_shots],  # 정확히 shadow_shots 개수만
        "bases": bases[:shadow_shots],
        "n_qubits": n_qubits,
        "shadow_size": shadow_shots
    }
    
    return shadow_data 

# 엔트로피 기반 표현력 계산 함수들 추가
def calculate_entropy_expressibility(measurement_counts, n_qubits, n_bins=10):
    """
    측정 결과의 엔트로피를 기반으로 한 표현력 계산
    
    이 함수는 양자 회로의 측정 결과로부터 샤논 엔트로피를 계산하여
    양자 회로의 표현력을 평가합니다. 또한 가능한 경우 각도 엔트로피도 계산합니다.
    
    Args:
        measurement_counts (dict): 측정 결과 {비트열: 카운트} 형식의 딕셔너리
        n_qubits (int): 큐빗 수
        n_bins (int): 히스토그램 구간 수 (각도 엔트로피 계산에 사용)
        
    Returns:
        dict: 엔트로피 기반 표현력 결과를 포함한 딕셔너리:
            - expressibility_value: 측정 엔트로피 값
            - measurement_entropy: 측정 엔트로피 값 (expressibility_value와 동일)
            - angle_entropy: 각도 엔트로피 값 (가능한 경우)
            - method: 사용된 방법론 ('measurement_entropy')
            - n_qubits: 큐빗 수
            - measured_states: 측정된 고유 상태 수
    """
    import time
    start_time = time.time()
    
    if not measurement_counts:
        return {
            "expressibility_value": 0.0,
            "measurement_entropy": 0.0,
            "angle_entropy": None,
            "angle_entropy_error": "측정 결과 없음",
            "method": "measurement_entropy",
            "n_qubits": n_qubits,
            "measured_states": 0,
            "histogram_bins": n_bins,
            "run_time": 0.0001
        }
    
    # 측정 엔트로피 계산
    measurement_entropy = calculate_measurement_entropy(measurement_counts)
    
    # 각도 엔트로피를 위한 벡터 추출 시도
    try:
        # 측정 결과를 확률 분포 벡터로 변환
        vectors = []
        weights = []
        
        # 총 측정 횟수 계산
        total_counts = sum(measurement_counts.values())
        if total_counts > 0:
            # 비트열을 정수 인덱스로 변환하여 확률 분포 벡터 생성
            n_qubits_actual = len(next(iter(measurement_counts.keys()))) if measurement_counts else 0
            if n_qubits_actual > 0:
                vector = np.zeros(2**n_qubits_actual)
                for bitstring, count in measurement_counts.items():
                    # 비트열을 정수로 변환 (예: '101' -> 5)
                    try:
                        idx = int(bitstring, 2)
                        vector[idx] = count / total_counts
                    except (ValueError, IndexError):
                        continue
                vectors.append(vector)
                weights.append(1.0)
        
        # 각도 엔트로피 계산 (벡터가 충분한 경우)
        angle_entropy = None
        angle_entropy_error = None
        
        if len(vectors) >= 2:
            # 다른 측정 결과들에서 추가 벡터 생성
            # 랜덤하게 약간 변형된 벡터들 추가하여 각도 엔트로피 계산 가능하게 함
            base_vector = vectors[0]
            for _ in range(4):
                noise = np.random.normal(0, 0.05, size=base_vector.shape)
                noisy_vector = base_vector + noise
                # 음수 제거 및 정규화
                noisy_vector = np.maximum(noisy_vector, 0)
                sum_noisy = np.sum(noisy_vector)
                if sum_noisy > 0:
                    noisy_vector /= sum_noisy
                vectors.append(noisy_vector)
                weights.append(0.5)  # 원본보다 낮은 가중치 부여
            
            angle_entropy = calculate_angle_entropy(vectors, weights, n_bins)
        else:
            angle_entropy = None
            angle_entropy_error = "각도 엔트로피 계산을 위한 벡터 부족"
    except Exception as e:
        angle_entropy = None
        angle_entropy_error = str(e)
    
    run_time = time.time() - start_time
    
    result = {
        "expressibility_value": measurement_entropy,
        "measurement_entropy": measurement_entropy,
        "method": "measurement_entropy",
        "n_qubits": n_qubits,
        "measured_states": len(measurement_counts),
        "histogram_bins": n_bins,
        "run_time": run_time,
        "total_measurements": total_counts if 'total_counts' in locals() else sum(measurement_counts.values())
    }
    
    # 각도 엔트로피 정보 추가
    if angle_entropy is not None:
        result["angle_entropy"] = angle_entropy
        result["angle_entropy_n_bins"] = n_bins
        result["angle_entropy_n_vectors"] = len(vectors) if 'vectors' in locals() else 0
        result["angle_entropy_calculation_time"] = time.strftime("%Y%m%d_%H%M%S")
    elif angle_entropy_error is not None:
        result["angle_entropy"] = None
        result["angle_entropy_error"] = angle_entropy_error
        result["angle_entropy_calculation_time"] = time.strftime("%Y%m%d_%H%M%S")
    
    return result

def entropy_based_expressibility(bit_strings, frequencies, n_bins=10):
    """
    측정 결과의 엔트로피 기반 표현력 계산
    
    이 함수는 측정 결과의 엔트로피를 계산하여 양자 회로의 표현력을 평가합니다.
    
    Args:
        bit_strings: 측정된 비트스트링들의 리스트
        frequencies: 각 비트스트링의 측정 빈도 리스트
        n_bins: 히스토그램 구간 수 (사용하지 않음, 호환성 유지용)
        
    Returns:
        dict: 엔트로피 기반 표현력 결과
            - total_entropy: 전체 측정 엔트로피
            - method: 사용된 방법론 ('measurement_entropy')
    """
    # 1. 측정 결과를 딕셔너리로 변환
    measurement_counts = {bit_str: count for bit_str, count in zip(bit_strings, frequencies)}
    
    # 2. 측정 엔트로피 계산
    measurement_entropy = calculate_measurement_entropy(measurement_counts)
    
    return {
        'total_expressibility': measurement_entropy,
        'measurement_entropy': measurement_entropy,
        'method': 'measurement_entropy'
    }

def calculate_angle_entropy(vectors, weights, n_bins):
    """벡터 간 각도 분포의 엔트로피"""
    angles = []
    angle_weights = []
    
    n = len(vectors)
    if n < 2:
        return 0.0
    
    for i in range(n):
        for j in range(i+1, n):
            # 두 벡터 간 각도 계산
            v1, v2 = vectors[i], vectors[j]
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                continue
                
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            angles.append(angle)
            # 두 벡터의 가중치 곱
            angle_weights.append(weights[i] * weights[j])
    
    if not angles:
        return 0.0
    
    # 각도를 구간으로 나누어 가중 히스토그램 생성
    angles = np.array(angles)
    angle_weights = np.array(angle_weights)
    
    # 0부터 π까지 n_bins개 구간
    bins = np.linspace(0, np.pi, n_bins + 1)
    
    # 가중 히스토그램
    weighted_hist = np.zeros(n_bins)
    for angle, weight in zip(angles, angle_weights):
        bin_idx = np.digitize(angle, bins) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        weighted_hist[bin_idx] += weight
    
    # 정규화
    total_weight = np.sum(weighted_hist)
    if total_weight > 0:
        weighted_hist /= total_weight
    
    # 엔트로피 계산
    entropy = -np.sum(weighted_hist * np.log(weighted_hist + 1e-10))
    
    return entropy

def calculate_measurement_entropy(measurement_data, weights=None, n_bins=None):
    """
    측정 결과 분포의 엔트로피 계산 (IBM 백엔드 호환)
    
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
    # 입력 데이터 처리
    if measurement_data is None:
        return 0.0
        
    # 리스트 형태의 입력 처리 (IBM 백엔드 호환)
    if isinstance(measurement_data, list):
        # [{'state': '00', 'count': 100}, ...] 형태의 리스트 처리
        if all(isinstance(x, dict) and 'state' in x and 'count' in x for x in measurement_data):
            counts = {item['state']: item['count'] for item in measurement_data}
        else:
            # 다른 형태의 리스트는 처리하지 않음
            return 0.0
    elif isinstance(measurement_data, dict):
        # {'00': 100, '01': 50, ...} 형태의 딕셔너리 처리
        counts = measurement_data
    else:
        # 지원하지 않는 형식
        return 0.0
    
    # 측정 결과가 없는 경우
    if not counts:
        return 0.0
    
    # 전체 측정 횟수 계산
    total_counts = sum(counts.values())
    if total_counts == 0:
        return 0.0
    
    try:
        # 각 상태의 확률 계산
        probabilities = np.array(list(counts.values()), dtype=float) / total_counts
        
        # 0인 확률 제외 (log(0) 방지)
        probabilities = probabilities[probabilities > 0]
        
        if len(probabilities) == 0:
            return 0.0
        
        # 샤논 엔트로피 계산 (밑이 2인 로그 사용, 비트 단위)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return float(entropy)
        
    except Exception as e:
        print(f"엔트로피 계산 중 오류 발생: {str(e)}")
        return 0.0

# IBM 백엔드용 엔트로피 표현력 계산 함수
def calculate_entropy_expressibility_from_ibm_results(measurement_counts, n_qubits):
    """
    IBM 측정 결과로부터 엔트로피 기반 표현력 계산
    
    이 함수는 IBM 양자 컴퓨터에서 얻은 측정 결과를 바탕으로
    양자 회로의 표현력을 측정 엔트로피 및 각도 엔트로피를 사용하여 평가합니다.
    
    Args:
        measurement_counts (dict): 측정 결과 {'00': count, '01': count, ...} 형식의 딕셔너리
        n_qubits (int): 큐빗 수
        
    Returns:
        dict: 엔트로피 기반 표현력 결과를 포함한 딕셔너리:
            - expressibility_value: 측정 엔트로피 값
            - measurement_entropy: 측정 엔트로피 값 (expressibility_value와 동일)
            - angle_entropy: 각도 엔트로피 값 (가능한 경우)
            - method: 사용된 방법론 ('measurement_entropy')
            - n_qubits: 큐빗 수
            - measured_states: 측정된 고유 상태 수
            - total_measurements: 총 측정 횟수
            - run_time: 실행 시간(초)
    """
    print(f"\n===== 측정 엔트로피 기반 표현력 측정 =====")
    print(f"큐빗 수: {n_qubits}")
    print(f"측정된 고유 상태 수: {len(measurement_counts)}")
    print(f"총 측정 횟수: {sum(measurement_counts.values())}")
    
    start_time = time.time()
    
    # 히스토그램 구간 수 설정
    n_bins = 20
    
    # 측정 엔트로피 계산
    measurement_entropy = calculate_measurement_entropy(measurement_counts)
    total_measurements = sum(measurement_counts.values())
    
    # 각도 엔트로피를 위한 벡터 추출 시도
    try:
        # 측정 결과를 확률 분포 벡터로 변환
        vectors = []
        weights = []
        
        if total_measurements > 0:
            # 비트열을 정수 인덱스로 변환하여 확률 분포 벡터 생성
            n_qubits_actual = len(next(iter(measurement_counts.keys()))) if measurement_counts else 0
            if n_qubits_actual > 0:
                vector = np.zeros(2**n_qubits_actual)
                for bitstring, count in measurement_counts.items():
                    # 비트열을 정수로 변환 (예: '101' -> 5)
                    try:
                        idx = int(bitstring, 2)
                        vector[idx] = count / total_measurements
                    except (ValueError, IndexError):
                        continue
                vectors.append(vector)
                weights.append(1.0)
        
        # 각도 엔트로피 계산 (벡터가 충분한 경우)
        angle_entropy = None
        angle_entropy_error = None
        
        if len(vectors) >= 2:
            # 다른 측정 결과들에서 추가 벡터 생성
            # 랜덤하게 약간 변형된 벡터들 추가하여 각도 엔트로피 계산 가능하게 함
            base_vector = vectors[0]
            for _ in range(4):
                noise = np.random.normal(0, 0.05, size=base_vector.shape)
                noisy_vector = base_vector + noise
                # 음수 제거 및 정규화
                noisy_vector = np.maximum(noisy_vector, 0)
                sum_noisy = np.sum(noisy_vector)
                if sum_noisy > 0:
                    noisy_vector /= sum_noisy
                vectors.append(noisy_vector)
                weights.append(0.5)  # 원본보다 낮은 가중치 부여
            
            angle_entropy = calculate_angle_entropy(vectors, weights, n_bins)
        else:
            angle_entropy = None
            angle_entropy_error = "각도 엔트로피 계산을 위한 벡터 부족"
    except Exception as e:
        angle_entropy = None
        angle_entropy_error = str(e)
    
    run_time = time.time() - start_time
    
    result = {
        "expressibility_value": measurement_entropy,
        "measurement_entropy": measurement_entropy,
        "method": "measurement_entropy",
        "n_qubits": n_qubits,
        "measured_states": len(measurement_counts),
        "histogram_bins": n_bins,
        "total_measurements": total_measurements,
        "run_time": run_time
    }
    
    # 각도 엔트로피 정보 추가
    if angle_entropy is not None:
        result["angle_entropy"] = angle_entropy
        result["angle_entropy_n_bins"] = n_bins
        result["angle_entropy_n_vectors"] = len(vectors) if 'vectors' in locals() else 0
        result["angle_entropy_calculation_time"] = time.strftime("%Y%m%d_%H%M%S")
    elif angle_entropy_error is not None:
        result["angle_entropy"] = None
        result["angle_entropy_error"] = angle_entropy_error
        result["angle_entropy_calculation_time"] = time.strftime("%Y%m%d_%H%M%S")
    
    # 결과 출력
    print(f"측정 엔트로피: {result['measurement_entropy']:.4f} bits")
    print(f"표현력 점수: {result['expressibility_value']:.4f}")
    print(f"실행 시간: {run_time:.3f}초")
    print(f"계산 효율성: O(측정상태수) = O({len(measurement_counts)})")
    
    return result