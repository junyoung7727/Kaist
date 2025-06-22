#!/usr/bin/env python3
"""
IBM 양자 컴퓨터 특화 표현력(Expressibility) 계산 모듈

이 모듈은 IBM 양자 컴퓨터에서의 표현력 계산에 필요한 클래스와 함수를 제공합니다.
Classical Shadow 방법론 및 측정 결과 처리 함수가 포함됩니다.
"""

import numpy as np
import time
import random
import scipy.stats
from typing import Dict, List, Any, Optional, Tuple, Union

# 내부 모듈 임포트
from src.calculators.expressibility.base import ExpressibilityCalculatorBase
from src.config import config


class IBMExpressibilityCalculator(ExpressibilityCalculatorBase):
    """
    IBM 실제 백엔드 표현력 계산기
    """
    
    def _create_shadow_circuit(self, base_circuit, n_qubits):
        """
        Classical Shadow 방법에 필요한 회로 생성
        
        Args:
            base_circuit: 기본 회로 객체
            n_qubits (int): 큐빗 수
            
        Returns:
            Tuple[QuantumCircuit, List[str]]: Shadow 회로와 사용된 기저 목록
        """
        # 기본 회로 복사
        shadow_circuit = base_circuit.copy()
        
        # 1. 크로노스 상태 초기화
        for q in range(n_qubits):
            shadow_circuit.reset(q)
        
        # 2. 랭덤 로테이션 추가 (Identity 포함)
        pauli_bases = ['I', 'X', 'Y', 'Z']  # Identity 추가
        bases_used = []
        
        for q in range(n_qubits):
            random_basis = random.choice(pauli_bases)
            bases_used.append(random_basis)
            
            if random_basis == 'I':
                # Identity: 아무 게이트도 적용하지 않음 (원래 상태 유지)
                pass
            elif random_basis == 'X':
                shadow_circuit.h(q)
            elif random_basis == 'Y':
                shadow_circuit.sdg(q)  # S† gate
                shadow_circuit.h(q)
            # Z basis: 아무 게이트도 적용하지 않음 (계산 기저와 동일)
        
        # 3. 측정 추가
        shadow_circuit.measure_all()
        
        return shadow_circuit, bases_used

    
    def calculate_expressibility(self, ibm_backend, base_circuit, circuit_info, n_qubits, samples=None) -> Dict[str, Any]:
        """
        IBM 백엔드에서 표현력 계산
        
        Args:
            ibm_backend: IBM 백엔드 객체
            base_circuit: 기본 회로 객체
            circuit_info (Dict[str, Any]): 회로 정보
            n_qubits (int): 큐빗 수
            samples (Optional[int]): 샘플 수
            
        Returns:
            Dict[str, Any]: 표현력 계산 결과
        """
        return self.calculate_expressibility_from_real_quantum_classical_shadow(
            ibm_backend, base_circuit, circuit_info, n_qubits, samples
        )


    def _expand_ibm_to_classical_shadow_data(self, measurement_counts: Dict[str, int], bases_used: List[str], n_qubits: int, shadow_shots: int) -> Tuple[List[List[int]], List[List[str]]]:
        """
        IBM 측정 결과를 Classical Shadow용 measurements, bases로 변환
        
        Args:
            measurement_counts (Dict[str, int]): IBM 측정 결과 (bitstring->count)
            bases_used (List[str]): 측정에 사용된 기저 목록 ("X", "Y", "Z" 등)
            n_qubits (int): 큐빗 수
            shadow_shots (int): Shadow 샷 수
        
        Returns:
            Tuple[List[List[int]], List[List[str]]]: measurements (list of lists of int), bases (list of list of str)
        """
        # 측정 결과와 기저 정보를 결합하여 처리
        base_info = {}
        
        # 기본 기저 설정 (일반적으로 IBM은 Z 기저로 측정)
        default_bases = bases_used if bases_used else ["Z"] * n_qubits
        
        for bitstring in measurement_counts:
            # IBM 백엔드에서 반환된 측정 결과는 단순 비트스트링("00", "01", 등)
            base_info[bitstring] = {
                "bases": default_bases,  # 전달받은 기저 사용
                "bits": bitstring,
                "count": measurement_counts[bitstring]
            }
        
        # Shadow 데이터로 확장
        all_measurements = []
        all_bases = []
        
        # 정해진 샷 수만큼 샘플링
        shots_sampled = 0
        while shots_sampled < shadow_shots:
            for bitstring, info in base_info.items():
                count = info["count"]
                for _ in range(count):
                    if shots_sampled >= shadow_shots:
                        break
                    
                    # 비트열을 이진값 리스트로 변환 (e.g. "01" -> [0, 1])
                    bits_list = []
                    for bit in info["bits"]:
                        bits_list.append(int(bit))
                    
                    # 부족한 비트 0으로 채우기
                    while len(bits_list) < n_qubits:
                        bits_list.append(0)
                        
                    all_measurements.append(bits_list)
                    all_bases.append(info["bases"])
                    shots_sampled += 1
                
                if shots_sampled >= shadow_shots:
                    break
                    
        return all_measurements, all_bases


    def convert_ibm_to_classical_shadow(self, measurement_counts: Dict[str, int], bases_used: List[str], n_qubits: int, shadow_shots: int) -> Dict[str, Any]:
        """
        IBM 측정 결과를 Classical Shadow 데이터 형식으로 변환
        
        Args:
            measurement_counts (Dict[str, int]): IBM 측정 결과 (비트열 -> 카운트)
            bases_used (List[str]): 측정에 사용된 기저 목록 ("X", "Y", "Z" 등)
            n_qubits (int): 큐빗 수
            shadow_shots (int): Shadow 샷 수
            
        Returns:
            Dict[str, Any]: Classical Shadow 데이터 형식
        """
        measurements, bases = self._expand_ibm_to_classical_shadow_data(
            measurement_counts, bases_used, n_qubits, shadow_shots
        )
        
        # Shadow 데이터 형식으로 변환
        shadow_data = {
            "measurements": measurements,
            "bases": bases,
            "n_qubits": n_qubits,
            "shots": len(measurements)
        }
        
        return shadow_data


    def estimate_pauli_expectations_from_shadows(self, shadow_data_list: List[Dict], n_qubits: int) -> Dict[str, float]:
        """
        Classical Shadow 데이터로부터 Pauli 연산자 기댓값 추정
        
        Args:
            shadow_data_list (List[Dict]): Classical Shadow 데이터 목록
            n_qubits (int): 큐빗 수
            
        Returns:
            Dict[str, float]: Pauli 연산자 기댓값 딕셔너리
        """
        # Pauli 연산자 기댓값 초기화 (Identity 포함)
        pauli_expectations = {}
        pauli_ops = ["I", "X", "Y", "Z"]  # Identity 추가
        
        # 1-local Pauli 연산자 (각 큐빗별)
        for q in range(n_qubits):
            for op in pauli_ops:
                pauli_expectations[f"{op}{q}"] = 0.0
        
        # 2-local Pauli 연산자 (큐빗 쌍별)
        for q1 in range(n_qubits):
            for q2 in range(q1 + 1, n_qubits):
                for op1 in pauli_ops:
                    for op2 in pauli_ops:
                        pauli_expectations[f"{op1}{q1}{op2}{q2}"] = 0.0
        
        # 각 Shadow 데이터에서 기댓값 추정
        total_samples = len(shadow_data_list)
        if total_samples == 0:
            return pauli_expectations
        
        # 각 Shadow에서 측정값 추정
        for shadow_data in shadow_data_list:
            bases = shadow_data.get("bases", [])
            measurements = shadow_data.get("measurements", [])
            
            for basis, meas in zip(bases, measurements):
                # 1-local Pauli 연산자 추정
                for q1 in range(n_qubits):
                    for op1 in pauli_ops:
                        # 연산자가 현재 측정 기저와 일치하면 값 추정
                        if basis[q1] == op1:
                            pauli_op_name = f"{op1}{q1}"
                            if op1 == 'I':
                                # Identity: 항상 1
                                pauli_val = 1.0
                            else:
                                # 측정 결과에 따라 +1 또는 -1
                                pauli_val = 1 - 2 * meas[q1]  # 0 -> +1, 1 -> -1
                            pauli_expectations[pauli_op_name] += pauli_val / total_samples
                
                # 2-local Pauli 연산자 추정
                for q1 in range(n_qubits):
                    for q2 in range(q1 + 1, n_qubits):
                        for op1 in pauli_ops:
                            for op2 in pauli_ops:
                                # 두 큐빗 모두 해당 기저에서 측정되었는지 확인
                                if basis[q1] == op1 and basis[q2] == op2:
                                    pauli_op_name = f"{op1}{q1}{op2}{q2}"
                                    if op1 == 'I' and op2 == 'I':
                                        # II: 항상 1
                                        pauli_val = 1.0
                                    elif op1 == 'I':
                                        # IX, IY, IZ: 두 번째 큐빗만 고려
                                        pauli_val = 1 - 2 * meas[q2] if op2 != 'I' else 1.0
                                    elif op2 == 'I':
                                        # XI, YI, ZI: 첫 번째 큐빗만 고려
                                        pauli_val = 1 - 2 * meas[q1] if op1 != 'I' else 1.0
                                    else:
                                        # XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ
                                        pauli_val = (1 - 2 * meas[q1]) * (1 - 2 * meas[q2])
                                    pauli_expectations[pauli_op_name] += pauli_val / total_samples
        
        return pauli_expectations


    def get_haar_pauli_expectations(self, n_qubits: int) -> Dict[str, float]:
        """
        Haar 분포의 이론적 Pauli 기댓값 계산 (Identity 포함)
        
        - I (Identity): 기댓값 = 1 (항상 1)
        - X, Y, Z: 기댓값 = 0 (Haar 분포에서 평균적으로 0)
        - I와 다른 연산자의 텐서곱: I의 기댓값만 고려
        - 비-Identity 연산자들의 텐서곱: 0
        
        Args:
            n_qubits (int): 큐빗 수
            
        Returns:
            Dict[str, float]: Pauli 연산자별 이론적 기댓값
        """
        pauli_expectations = {}
        pauli_ops = ["I", "X", "Y", "Z"]  # Identity 포함
        
        # 1-local Pauli 연산자 (Identity = 1, 나머지 = 0)
        for q in range(n_qubits):
            for op in pauli_ops:
                if op == "I":
                    pauli_expectations[f"{op}{q}"] = 1.0  # Identity는 항상 1
                else:
                    pauli_expectations[f"{op}{q}"] = 0.0  # X, Y, Z는 0
        
        # 2-local Pauli 연산자 
        for q1 in range(n_qubits):
            for q2 in range(q1 + 1, n_qubits):
                for op1 in pauli_ops:
                    for op2 in pauli_ops:
                        if op1 == "I" and op2 == "I":
                            # II = 1
                            pauli_expectations[f"{op1}{q1}{op2}{q2}"] = 1.0
                        elif op1 == "I" or op2 == "I":
                            # I와 다른 연산자의 텐서곱: I는 1, 나머지는 0 -> 전체는 0
                            # 예: IX = I ⊗ X = 1 * 0 = 0
                            pauli_expectations[f"{op1}{q1}{op2}{q2}"] = 0.0
                        else:
                            # 비-Identity 연산자들의 텐서곱: 0
                            pauli_expectations[f"{op1}{q1}{op2}{q2}"] = 0.0
        
        return pauli_expectations


    def calculate_shadow_distance(self, estimated_moments: Dict[str, float], haar_moments: Dict[str, float], 
                                distance_metric: str = 'mse') -> float:
        """
        추정된 기댓값과 Haar 기댓값 사이의 거리 계산
        
        여러 거리 측정 방법을 지원합니다:
        - mse: 평균 제곱 오차 (Mean Squared Error)
        - mmd: 최대 평균 불일치 (Maximum Mean Discrepancy)
        - kl: 쿨백-라이블러 발산 (Kullback-Leibler Divergence)
        - js: 젠슨-섀넌 발산 (Jensen-Shannon Divergence)
        
        Args:
            estimated_moments (Dict[str, float]): 추정된 Pauli 기댓값
            haar_moments (Dict[str, float]): Haar 분포의 이론적 기댓값
            distance_metric (str): 사용할 거리 측정 방법 ('mse', 'mmd', 'kl', 'js')
            
        Returns:
            float: 거리값 (표현력 값)
        """
        if not estimated_moments or not haar_moments:
            return float('nan')
        
        # 모든 키가 동일한지 확인
        if set(estimated_moments.keys()) != set(haar_moments.keys()):
            # 키가 다른 경우, 공통 키만 사용
            common_keys = set(estimated_moments.keys()) & set(haar_moments.keys())
            if not common_keys:
                return float('nan')
            
            # 공통 키에 대한 값만 추출
            est_values = np.array([estimated_moments[key] for key in common_keys])
            haar_values = np.array([haar_moments[key] for key in common_keys])
            n_features = len(common_keys)
        else:
            # 모든 키가 동일한 경우
            est_values = np.array(list(estimated_moments.values()))
            haar_values = np.array(list(haar_moments.values()))
            n_features = len(estimated_moments)
        
        # 선택된 거리 측정 방법에 따라 계산
        if distance_metric.lower() == 'mse':
            # 평균 제곱 오차 (Mean Squared Error)
            distance = np.mean((est_values - haar_values) ** 2)
            
        elif distance_metric.lower() == 'mmd':
            # 최대 평균 불일치 (Maximum Mean Discrepancy)
            # 가우시안 커널 사용
            sigma = np.std(est_values) if np.std(est_values) > 0 else 1.0
            
            def gaussian_kernel(x, y, sigma):
                return np.exp(-np.sum((x - y) ** 2) / (2 * sigma ** 2))
            
            # MMD^2 계산
            n = len(est_values)
            xx_sum = 0
            yy_sum = 0
            xy_sum = 0
            
            for i in range(n):
                for j in range(n):
                    xx_sum += gaussian_kernel(est_values[i], est_values[j], sigma)
                    yy_sum += gaussian_kernel(haar_values[i], haar_values[j], sigma)
                    xy_sum += gaussian_kernel(est_values[i], haar_values[j], sigma)
            
            distance = (xx_sum + yy_sum - 2 * xy_sum) / (n * n)
            
        elif distance_metric.lower() == 'kl':
            # 쿨백-라이블러 발산 (Kullback-Leibler Divergence)
            # 확률 분포로 변환 (음수 값 처리 및 정규화)
            est_prob = np.abs(est_values)
            haar_prob = np.abs(haar_values)
            
            # 0이 아닌 값으로 만들기 (KL 발산 계산을 위해)
            epsilon = 1e-10
            est_prob = est_prob + epsilon
            haar_prob = haar_prob + epsilon
            
            # 정규화
            est_prob = est_prob / np.sum(est_prob)
            haar_prob = haar_prob / np.sum(haar_prob)
            
            # KL 발산 계산
            distance = np.sum(est_prob * np.log(est_prob / haar_prob))
            
        elif distance_metric.lower() == 'js':
            # 젠슨-섀넌 발산 (Jensen-Shannon Divergence)
            # 확률 분포로 변환 (음수 값 처리 및 정규화)
            est_prob = np.abs(est_values)
            haar_prob = np.abs(haar_values)
            
            # 0이 아닌 값으로 만들기
            epsilon = 1e-10
            est_prob = est_prob + epsilon
            haar_prob = haar_prob + epsilon
            
            # 정규화
            est_prob = est_prob / np.sum(est_prob)
            haar_prob = haar_prob / np.sum(haar_prob)
            
            # 평균 분포 계산
            m_prob = 0.5 * (est_prob + haar_prob)
            
            # JS 발산 계산 (KL의 대칭 버전)
            kl_est_m = np.sum(est_prob * np.log(est_prob / m_prob))
            kl_haar_m = np.sum(haar_prob * np.log(haar_prob / m_prob))
            distance = 0.5 * (kl_est_m + kl_haar_m)
            
        else:
            # 기본값: MSE
            distance = np.mean((est_values - haar_values) ** 2)
        
        return float(distance)


    def calculate_shadow_confidence_interval(self, estimated_moments: Dict[str, float], 
                                        S: int, M: int, n_qubits: int) -> Tuple[float, float]:
        """
        표현력 추정값의 신뢰구간 계산
        
        Args:
            estimated_moments (Dict[str, float]): 추정된 Pauli 기댓값
            S (int): 파라미터 샘플 수
            M (int): Shadow 측정 수
            n_qubits (int): 큐빗 수
            
        Returns:
            Tuple[float, float]: 95% 신뢰구간 (lower, upper)
        """
        if not estimated_moments or S <= 1 or M <= 0:
            return (float('nan'), float('nan'))
        
        # 추정된 기댓값의 제곱합 계산
        moment_sum_squares = sum(val**2 for val in estimated_moments.values())
        
        # 표준 편차 추정 (과학적 접근법)
        # 표준 편차는 샘플 크기와 큐빗 수에 의해 영향을 받음
        std_error = np.sqrt(moment_sum_squares / (S * M)) * np.sqrt(3**n_qubits / (3**n_qubits - 1))
        
        # 95% 신뢰구간 계산 (Z-점수 1.96)
        mean = moment_sum_squares / len(estimated_moments) if estimated_moments else 0
        margin = 1.96 * std_error
        
        lower = max(0.0, mean - margin)  # 표현력은 음수가 아니므로 최소 0
        upper = mean + margin
        
        return (float(lower), float(upper))

    def calculate_expressibility_from_real_quantum_classical_shadow(self, ibm_backend, base_circuit, circuit_info, n_qubits, samples=None) -> Dict[str, Any]:
        """
        실제 IBM 양자 컴퓨터에서 Classical Shadow 방법론을 사용하여 표현력 계산
        배치 처리 방식으로 구현
        
        Args:
            ibm_backend: IBM 백엔드 객체
            base_circuit: 기본 회로 객체
            circuit_info (Dict[str, Any]): 회로 정보
            n_qubits (int): 큐빗 수
            samples (Optional[int]): 실행 횟수 (None이면 중앙 설정 사용)
            
        Returns:
            Dict[str, Any]: 표현력 측정 결과
        """
        # Shadow 파라미터 설정 - 리팩토링된 config에서 직접 속성 접근
        if samples is None:
            # ConfigBox를 통한 속성 접근으로 파라미터 가져오기
            S = config.expressibility.n_samples  # 파라미터 샘플 수 (randparam)
        else:
            S = samples
        
        # ConfigBox를 통한 속성 접근으로 파라미터 가져오기
        M = config.expressibility.shadow_measurements  # Shadow 크기 (random measurements)
        
        start_time = time.time()
        shadow_size = M
        all_shadow_data_list = []
        
        print(f"🔍 IBM 백엔드 표현력 측정 시작 ({S} 파라미터 샘플)")
        
        # 실행할 모든 회로 및 기저 정보 생성
        all_circuits = []
        all_bases_info = []
        
        print(f" ⚡️ 샤도우 회로 {S}개 생성 중...")
        for param_idx in range(S):
            # 현재 회로에 대한 샤도우 회로 생성
            shadow_circuit, bases_used = self._create_shadow_circuit(base_circuit, n_qubits)
            
            # 생성된 회로와 기저 정보 저장
            all_circuits.append(shadow_circuit)
            all_bases_info.append(bases_used)
            
        # 모든 회로를 한번에 실행 (배치 처리)
        print(f" 🔌 {S}개 회로 동시 실행 중...")
        try:
            batch_results = ibm_backend.run_circuits(all_circuits, shots=shadow_size)
            if batch_results is None or len(batch_results) == 0:
                raise ValueError("IBM 백엔드에서 동시 실행 결과가 없습니다")
                
            # 각 결과를 처리
            print(f"  📊 결과 분석 및 샤도우 변환 중...")
            for i, (result_dict, bases_used) in enumerate(zip(batch_results, all_bases_info)):
                try:
                    # 회로 결과에서 카운트 추출
                    counts = result_dict.get('counts', {})
                    if not counts:
                        print(f"  ⚠️ 회로 {i+1}/{S} 결과에 카운트 정보가 없습니다")
                        continue
                    
                    # Classical Shadow 데이터로 변환
                    shadow_data = self.convert_ibm_to_classical_shadow(
                        counts, bases_used, n_qubits, shadow_size
                    )
                    all_shadow_data_list.append(shadow_data)
                    
                except Exception as e:
                    print(f"  ⚠️ 회로 {i+1}/{S} 결과 처리 오류: {str(e)}")
                    
        except Exception as e:
            print(f"⚠️ 동시 실행 오류: {str(e)}")
            print(f"    오류 유형: {type(e).__name__}")
        
        # Classical Shadow 데이터에서 2-local Pauli 기댓값 추정
        estimated_moments = self.estimate_pauli_expectations_from_shadows(all_shadow_data_list, n_qubits)
        haar_moments = self.get_haar_pauli_expectations(n_qubits)
        
        # 거리 계산 - 여러 가지 메트릭 사용
        distance_metrics = {}
        
        # MSE (Mean Squared Error)
        distance_metrics['mse'] = self.calculate_shadow_distance(estimated_moments, haar_moments, 'mse')
        
        # KL (Kullback-Leibler Divergence)
        distance_metrics['kl'] = self.calculate_shadow_distance(estimated_moments, haar_moments, 'kl')
        
        # JS (Jensen-Shannon Divergence)
        distance_metrics['js'] = self.calculate_shadow_distance(estimated_moments, haar_moments, 'js')
        
        # MMD (Maximum Mean Discrepancy)
        distance_metrics['mmd'] = self.calculate_shadow_distance(estimated_moments, haar_moments, 'mmd')
        
        # 설정에서 기본 거리 메트릭 가져오기, 없으면 'mse' 사용
        # 대소문자 구분을 위해 설정값을 소문자로 변환
        default_metric = config.expressibility.distance_metric.lower() if config.expressibility.distance_metric else 'kl'
        distance = distance_metrics.get(default_metric, distance_metrics['kl']) # 설정된 메트릭이 존재하지 않으면 'mse' 사용
        
        # 신뢰구간 계산
        conf_interval = self.calculate_shadow_confidence_interval(
            estimated_moments, S, M, n_qubits
        )
        
        # 실행 시간
        execution_time = time.time() - start_time
        
        # 결과 구성
        result = {
            "expressibility_value": distance,
            "expressibility_metrics": distance_metrics,  # 다양한 거리 메트릭 결과
            "n_qubits": n_qubits,
            "method": "classical_shadow",
            "S": S,  # 파라미터 샘플 수
            "M": M,  # Shadow 크기
            "conf_interval": list(conf_interval) if conf_interval else None,
            "execution_time": execution_time
        }
        
        print(f"✅ IBM 백엔드 표현력 측정 완료: {distance:.6f}")
        
        return result
