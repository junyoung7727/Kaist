#!/usr/bin/env python3
"""
IBM 백엔드 관리 모듈 (새 IBM Quantum Platform 대응)

이 모듈은 새로운 IBM Quantum Platform의 백엔드 연결, 관리 및 실행을 위한 클래스를 제공합니다.
"""

import os
import sys
import time
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from datetime import datetime

from src.config import config


class IBMBackendManager:
    """
    IBM 양자 백엔드 관리 및 실행 클래스 (새 플랫폼 대응)
    
    이 클래스는 새로운 IBM Quantum Platform에 연결하고, 백엔드를 선택하며,
    회로 실행 및 결과 처리를 관리합니다.
    """
    
    def __init__(self, ibm_token=None, instance=None, use_simulator=False):
        """
        IBM 백엔드 관리자 초기화
        
        Args:
            ibm_token (str, optional): IBM 양자 계정 토큰
            instance (str, optional): IBM Cloud 인스턴스 CRN 또는 이름
            use_simulator (bool, optional): 시뮬레이터를 사용할지 여부
        """
        self.ibm_token = ibm_token or os.environ.get('IBM_TOKEN', '')
        self.instance = instance or os.environ.get('IBM_INSTANCE', '')
        self.service = None
        self.backend = None
        self._backend_name = ""
        self.simulator = AerSimulator()
        self.optimization_level = config.get("ibm_backend", {}).get("optimization_level", 1)
        self.shots = config.get("ibm_backend", {}).get("default_shots", 1024)
        self.use_simulator = use_simulator
        
        # IBM 백엔드 연결 시도
        if self.use_simulator:
            self.backend = self.simulator
            self._backend_name = "aer_simulator"
        elif self.ibm_token:
            connected = self.connect_to_ibm()
            if not connected:
                raise RuntimeError("IBM Quantum Platform 연결 및 초기화에 실패했습니다.")
    
    @property
    def name(self):
        """백엔드 이름을 반환합니다."""
        return self._backend_name or "not_connected"
    
    @property
    def backend_name(self):
        """백엔드 이름을 반환합니다 (legacy 호환성)."""
        return self.name
    
    def connect_to_ibm(self):
        """
        새로운 IBM Quantum Platform에 연결합니다.
        
        Returns:
            bool: 연결 성공 여부
        """
        try:
            if not self.ibm_token:
                print("⚠️ IBM Quantum 토큰이 필요합니다.")
                return False
            
            print("\n새로운 IBM Quantum Platform에 연결 중...")
            
            # 새로운 플랫폼으로 서비스 초기화
            if self.instance:
                print(f"지정된 인스턴스 사용: {self.instance}")
                self.service = QiskitRuntimeService(
                    channel="ibm_quantum",
                    token=self.ibm_token,
                    instance=self.instance
                )
            else:
                print("모든 인스턴스에 접근 (인스턴스 미지정)")
                self.service = QiskitRuntimeService(
                    channel="ibm_quantum",
                    token=self.ibm_token
                )
            
            # 계정 정보 확인
            account = self.service.active_account()
            if account:
                print(f"✅ IBM Quantum Platform 연결 성공!")
                print(f"계정 정보: {account}")
            
            # 자동 백엔드 선택
            self._backend_name = ""
            return self.select_backend()
        
        except Exception as e:
            print(f"❌ IBM Quantum Platform 연결 실패: {str(e)}")
            print("💡 확인사항:")
            print("   - API 토큰이 올바른지 확인")
            print("   - 인스턴스 CRN이나 이름이 올바른지 확인")
            print("   - 새로운 IBM Quantum Platform 계정인지 확인")
            return False
    
    def select_backend(self, backend_name=None):
        """
        IBM 백엔드를 선택합니다.
        
        Args:
            backend_name (str, optional): 사용할 백엔드 이름
            
        Returns:
            bool: 백엔드 선택 성공 여부
        """
        if not self.service:
            print("⚠️ IBM Quantum 서비스가 초기화되지 않았습니다.")
            return False
        
        try:
            backend_name = backend_name or os.environ.get('IBM_BACKEND_NAME', '').strip()
            
            print("\n사용 가능한 IBM 양자 컴퓨터:")
            real_backends = []
            backend_info = {}
            
            # 백엔드 목록 가져오기
            backends = self.service.backends()
            
            for backend in backends:
                try:
                    # 시뮬레이터 제외
                    is_simulator = False
                    if hasattr(backend, 'simulator'):
                        is_simulator = backend.simulator
                    elif hasattr(backend.configuration(), 'simulator'):
                        is_simulator = backend.configuration().simulator
                    else:
                        is_simulator = 'simulator' in backend.name.lower() or 'qasm' in backend.name.lower()
                    
                    if is_simulator:
                        continue
                    
                    # 백엔드 정보 가져오기
                    config = backend.configuration()
                    status = backend.status()
                    
                    qubit_count = config.n_qubits
                    operational = status.operational
                    pending_jobs = status.pending_jobs if operational else float('inf')
                    
                    print(f"- {backend.name}: {qubit_count} qubits")
                    print(f"  Status: {'🟢 Available' if operational else '🔴 Offline'}")
                    if operational:
                        print(f"  Queue length: {pending_jobs}")
                        real_backends.append(backend)
                        backend_info[backend.name] = (backend, qubit_count, pending_jobs)
                
                except Exception as e:
                    print(f"  ⚠️ Error getting backend info: {str(e)}")
            
            if not real_backends:
                print("\n⚠️ 운영 중인 양자 컴퓨터를 찾을 수 없습니다.")
                return False
            
            # 백엔드 선택 로직
            selected_backend = None
            
            if backend_name and backend_name in backend_info:
                selected_backend = backend_info[backend_name][0]
                print(f"\n✅ 선택된 백엔드: {backend_name}")
            else:
                # 자동 선택: 대기열이 가장 짧은 백엔드 선택
                min_queue = float('inf')
                for name, (backend_obj, _, queue_len) in backend_info.items():
                    if queue_len < min_queue:
                        min_queue = queue_len
                        selected_backend = backend_obj
                        backend_name = name
                
                print(f"\n✅ 자동 선택된 백엔드: {backend_name}")
                print(f"  대기열 길이: {min_queue}")
            
            self.backend = selected_backend
            self._backend_name = backend_name
            
            return True
        
        except Exception as e:
            print(f"❌ 백엔드 선택 실패: {str(e)}")
            return False
    
    def run_circuits(self, circuits, shots=None, skip_transpile=False):
        """
        회로를 실행하고 결과를 반환합니다. (새 플랫폼 방식)
        
        Args:
            circuits (List[QuantumCircuit]): 실행할 양자 회로 리스트
            shots (int, optional): 실행 샷 수
            skip_transpile (bool, optional): 회로 번역(transpile) 단계를 건너뛰기 여부. 회로가 이미 transpile 된 경우 True로 설정.
            
        Returns:
            List[Dict]: 각 회로별 실행 결과 딕셔너리 리스트
        """
        if not self.backend:
            print("⚠️ 백엔드가 선택되지 않았습니다.")
            return None
        
        try:
            shots = shots or self.shots
            
            print(f"\n🚀 {len(circuits)}개 회로 실행 중 (shots={shots})...")
            start_time = time.time()
            
            # 2024년 3월 4일 이후 IBM 정책에 따라 회로를 실행하기 전에 transpile 필요
            # 모든 회로는 대상 하드웨어의 네이티브 게이트셋으로 변환해야 함
            if skip_transpile:
                print(f"  회로가 이미 transpile 되어 있으므로 단계 건너뛐")
                circuits_to_run = circuits
            else:
                print(f"  회로를 {self.backend.name} 백엔드용으로 transpile 중...")
                circuits_to_run = transpile(
                    circuits,
                    backend=self.backend,
                    optimization_level=self.optimization_level
                )
                print(f"  Transpile 완료.")
            
            # 새로운 플랫폼에서는 SamplerV2 사용
            sampler = Sampler(mode=self.backend)
            job = sampler.run(circuits_to_run, shots=shots)
            result = job.result()
            
            elapsed = time.time() - start_time
            print(f"✅ 회로 실행 완료: {elapsed:.2f}초 소요")
            
            # 각 회로의 결과를 딕셔너리 형태로 변환
            print(f"  🔄 결과 파싱 중...")
            circuit_results = []
            
            for i in range(len(circuits)):
                try:
                    # SamplerV2 결과에서 각 회로의 카운트 추출
                    pub_result = result[i]
                    
                    # 카운트 추출 - DataBin 오류 해결
                    counts_dict = {}
                    
                    if hasattr(pub_result, 'data') and hasattr(pub_result.data, 'meas'):
                        # 새로운 SamplerV2 API
                        meas_data = pub_result.data.meas
                        
                        # DataBin 객체 처리
                        if hasattr(meas_data, 'get_counts'):
                            try:
                                counts_dict = meas_data.get_counts()
                            except TypeError as e:
                                # DataBin 객체가 callable이 아닌 경우
                                if hasattr(meas_data, 'array'):
                                    # 배열 형태의 데이터를 카운트로 변환
                                    measurements = meas_data.array
                                    n_qubits = circuits[i].num_qubits
                                    
                                    # 측정 결과를 비트 문자열로 변환하고 카운트
                                    from collections import Counter
                                    bit_strings = []
                                    for shot in measurements:
                                        bit_string = ''.join(str(int(bit)) for bit in shot)
                                        bit_strings.append(bit_string)
                                    counts_dict = dict(Counter(bit_strings))
                                else:
                                    print(f"  ⚠️ 회로 {i+1}: DataBin 처리 실패, 빈 결과 사용")
                                    counts_dict = {}
                        elif hasattr(meas_data, 'array'):
                            # 직접 배열 접근
                            measurements = meas_data.array
                            n_qubits = circuits[i].num_qubits
                            
                            from collections import Counter
                            bit_strings = []
                            for shot in measurements:
                                bit_string = ''.join(str(int(bit)) for bit in shot)
                                bit_strings.append(bit_string)
                            counts_dict = dict(Counter(bit_strings))
                        else:
                            counts_dict = {}
                            
                    elif hasattr(result, 'quasi_dists') and i < len(result.quasi_dists):
                        # 이전 버전 호환성
                        quasi_dist = result.quasi_dists[i]
                        # 큐빗 수 추정 (회로에서 추출)
                        n_qubits = circuits[i].num_qubits
                        counts_dict = {bin(k)[2:].zfill(n_qubits): v for k, v in quasi_dist.items()}
                    else:
                        # 대안 방법들 시도
                        try:
                            # 다른 가능한 접근 방법들
                            if hasattr(pub_result, 'data') and 'meas' in pub_result.data:
                                meas_data = pub_result.data['meas']
                                if hasattr(meas_data, 'get_counts'):
                                    counts_dict = meas_data.get_counts()
                                elif hasattr(meas_data, 'counts'):
                                    counts_dict = meas_data.counts
                                elif hasattr(meas_data, 'array'):
                                    # 배열 형태 처리
                                    measurements = meas_data.array
                                    n_qubits = circuits[i].num_qubits
                                    
                                    from collections import Counter
                                    bit_strings = []
                                    for shot in measurements:
                                        bit_string = ''.join(str(int(bit)) for bit in shot)
                                        bit_strings.append(bit_string)
                                    counts_dict = dict(Counter(bit_strings))
                                else:
                                    counts_dict = {}
                            else:
                                counts_dict = {}
                        except Exception as parsing_error:
                            print(f"  ⚠️ 회로 {i+1} 결과 파싱 오류: {parsing_error}")
                            counts_dict = {}
                    
                    # 결과 딕셔너리 생성
                    circuit_result = {
                        'counts': counts_dict,
                        'circuit_index': i
                    }
                    circuit_results.append(circuit_result)
                    
                except Exception as e:
                    print(f"  ⚠️ 회로 {i+1} 결과 처리 오류: {str(e)}")
                    # 빈 결과라도 추가하여 인덱스 맞춤
                    circuit_results.append({
                        'counts': {},
                        'circuit_index': i,
                        'error': str(e)
                    })
            
            print(f"  ✅ {len(circuit_results)}개 회로 결과 파싱 완료")
            return circuit_results
            
        except Exception as e:
            print(f"❌ 회로 실행 실패: {str(e)}")
            return None
    
    def run_circuit_shadow(self, base_circuit, circuit_info, n_qubits, shadow_size):
        """
        Classical Shadow 방법에 필요한 회로 실행 (새 플랫폼 대응)
        
        Args:
            base_circuit: 기본 회로 객체
            circuit_info: 회로 정보
            n_qubits: 큐빗 수
            shadow_size: Shadow 크기
            
        Returns:
            dict: counts와 bases_used 정보가 포함된 사전
        """
        try:
            # Shadow 회로 준비
            shadow_circuit = base_circuit.copy()
            shadow_circuit.measure_all()
            
            # 2024년 3월 4일 이후 IBM 정책에 따라 transpile 필요
            print(f"  회로를 {self.backend.name} 백엔드용으로 transpile 중...")
            transpiled_circuit = transpile(
                shadow_circuit,
                backend=self.backend,
                optimization_level=self.optimization_level
            )
            print(f"  Transpile 완료.")
            
            # 새로운 플랫폼 방식으로 실행 (SamplerV2 사용)
            sampler = Sampler(mode=self.backend)
            job = sampler.run([transpiled_circuit], shots=shadow_size)
            result = job.result()
            
            # SamplerV2 결과 객체에서 카운트 추출
            # PrimitiveResult 구조: result[0].data.meas 를 통해 카운트 접근
            if hasattr(result, 'quasi_dists'):
                # 이전 버전 API 호환성
                quasi_dist = result.quasi_dists[0]
                counts_dict = {bin(k)[2:].zfill(n_qubits): v for k, v in quasi_dist.items()}
            else:
                # 새 SamplerV2 API
                try:
                    # 고수준 사전형 결과에서 카운트 추출
                    memory_data = result.data()[0]["meas"]
                    if hasattr(memory_data, 'get_counts'):
                        counts_dict = memory_data.get_counts()
                    elif hasattr(memory_data, 'counts'):
                        counts_dict = memory_data.counts
                    else:
                        # 월경적인 포맷으로 덧분에 이 부분을 체크
                        print(f"  결과 구조: {dir(result)}")
                        print(f"  데이터 구조: {dir(result.data()[0])}")
                        counts_dict = {}
                except Exception as parsing_error:
                    print(f"  결과 구조 분석 오류: {parsing_error}")
                    print(f"  결과 유형: {type(result)}")
                    print(f"  결과 구조: {dir(result)}")
                    counts_dict = {}
            
            # 기저 정보 추출 (이 예제에서는 Z 기저 고정)
            bases_used = ['Z'] * n_qubits
            
            # Classical Shadow 형식에 맞는 결과 반환
            return {
                'counts': counts_dict,
                'bases_used': bases_used
            }
            
        except Exception as e:
            print(f"❌ Shadow 회로 실행 실패: {str(e)}")
            return None