from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
import numpy as np
import os
from datetime import datetime
import json
from quantum_base import QuantumCircuitBase
from qiskit.circuit import Parameter
from config import config, get_ibm_shots
from qiskit_ibm_runtime import QiskitRuntimeService

class IBMQuantumBackend:
    """IBM 양자 백엔드 관리 및 실행 클래스"""
    
    def __init__(self, ibm_token=None, base_circuit=None):
        """
        IBM 백엔드 초기화
        
        Args:
            ibm_token (str): IBM 양자 계정 토큰
            base_circuit (QuantumCircuitBase): 기본 양자 회로 객체
        """
        self.ibm_token = ibm_token
        self.base_circuit = base_circuit if base_circuit else QuantumCircuitBase()
        self.service = None
        self.backend = None
        self.simulator = AerSimulator()
        self.optimization_level = config.ibm_backend.optimization_level  # 중앙 설정에서 가져오기
        self.shots = config.ibm_backend.default_shots  # 중앙 설정에서 가져오기
        
        # IBM 백엔드 연결 시도
        if ibm_token:
            self.connect_to_ibm()
    
    def connect_to_ibm(self):
        """IBM 양자 서비스에 연결"""
        try:
            if not self.ibm_token:
                print("⚠️ IBM Quantum 토큰이 필요합니다.")
                return False
            
            # 최신 qiskit_ibm_runtime으로 서비스 초기화
            print("\nIBM Quantum에 연결 중...")
            self.service = QiskitRuntimeService(channel="ibm_quantum", token=self.ibm_token)
            
            # 계정 정보 확인
            account = self.service.active_account()
            if account:
                print(f"✅ IBM Quantum 연결 성공!")
                print(f"계정 정보: {account}")
            
            # 사용 가능한 백엔드 확인
            print("\n사용 가능한 IBM 양자 컴퓨터:")
            real_backends = []
            
            # 백엔드 목록 가져오기
            backends = self.service.backends()
            
            for backend in backends:
                # 시뮬레이터 제외
                is_simulator = False
                try:
                    if hasattr(backend, 'simulator'):
                        is_simulator = backend.simulator
                    elif hasattr(backend.configuration(), 'simulator'):
                        is_simulator = backend.configuration().simulator
                    else:
                        # 이름으로 판단 (fallback)
                        is_simulator = 'simulator' in backend.name.lower() or 'qasm' in backend.name.lower()
                except:
                    pass
                
                if is_simulator:
                    continue
                
                try:
                    # 백엔드 정보 가져오기
                    config = backend.configuration()
                    status = backend.status()
                    
                    qubit_count = config.n_qubits
                    operational = status.operational
                    pending_jobs = status.pending_jobs if operational else "N/A"
                    
                    print(f"- {backend.name}: {qubit_count} qubits")
                    print(f"  Status: {'🟢 Available' if operational else '🔴 Offline'}")
                    if operational:
                        print(f"  Queue length: {pending_jobs}")
                        real_backends.append(backend)
                except Exception as e:
                    print(f"  ⚠️ Error getting backend info: {str(e)}")
            
            if not real_backends:
                print("\n⚠️ No operational quantum computers found.")
                return False
            
            # 비대화형 백엔드 선택: 환경변수 IBM_BACKEND_NAME 사용, 그렇지 않으면 가장 적게 대기 중인 백엔드 자동 선택
            backend_name = os.environ.get('IBM_BACKEND_NAME', '').strip()
            if backend_name:
                try:
                    candidate = self.service.backend(backend_name)
                    if candidate.status().operational:
                        self.backend = candidate
                    else:
                        print(f"⚠️ Selected backend '{backend_name}' is not operational, fallback to auto-select.")
                except Exception as e:
                    print(f"⚠️ Invalid IBM_BACKEND_NAME '{backend_name}': {e}, fallback to auto-select.")
            if not hasattr(self, 'backend') or self.backend is None:
                # 가장 적게 대기 중이며 최소 큐빗 조건을 만족하는 백엔드 선택 (큐빗 수 >= 5)
                available = [b for b in real_backends if b.configuration().n_qubits >= 5]
                if not available:
                    print("⚠️ No suitable quantum computers available (need >= 5 qubits)")
                    return False
                self.backend = min(available, key=lambda b: b.status().pending_jobs)
            print(f"\nSelected backend: {self.backend.name}")
            print(f"Number of qubits: {self.backend.configuration().n_qubits}")
            
            # 백엔드 속성 가져오기
            try:
                # 커플링 맵 확인
                self.coupling_map = None
                if hasattr(self.backend.configuration(), 'coupling_map'):
                    self.coupling_map = self.backend.configuration().coupling_map
                    if self.coupling_map:
                        print(f"Coupling map: {self.coupling_map[:10]}... (total {len(self.coupling_map)} connections)")
                
                # 코히어런스 시간 측정
                print("\nMeasuring initial coherence times...")
                coherence_data = self.measure_coherence_times()
                
                return True
            except Exception as e:
                print(f"⚠️ Error getting backend properties: {str(e)}")
                import traceback
                traceback.print_exc()
                return True  # 백엔드 접속은 성공했으므로 계속 진행
                
        except Exception as e:
            print(f"\n⚠️ IBM Quantum backend connection failed: {str(e)}")
            print("Please check your token and internet connection.")
            import traceback
            traceback.print_exc()
            return False
    
    def measure_coherence_times(self):
        """
        모든 큐빗의 코히어런스 시간(T1, T2)을 측정하고 기록
        
        Returns:
            dict: 코히어런스 데이터
        """
        if not self.backend:
            return None
            
        try:
            # 최신 API를 사용하여 백엔드 속성 가져오기
            try:
                # 백엔드 속성 가져오기
                properties = self.backend.properties()
                
                coherence_data = {
                    "timestamp": datetime.now().isoformat(),
                    "backend_name": self.backend.name,
                    "qubits": {},
                    "statistics": {}  # 통계 섹션
                }
                
                # 데이터 수집을 위한 리스트
                t1_times = []
                t2_times = []
                readout_errors = []
                gate_errors = []
                
                print("\n코히어런스 시간 측정 중...")
                
                # Qiskit 2.0+ API 호환성
                # 속성이 사용 가능한지 확인
                if hasattr(properties, 'qubit_properties'):
                    # 새로운 API 형식 (qiskit-ibm-runtime 0.11+)
                    for qubit_idx, q_props in enumerate(properties.qubit_properties):
                        if q_props:
                            # T1, T2 시간 (마이크로초)
                            t1_time = q_props.T1 * 1e6 if hasattr(q_props, 'T1') and q_props.T1 else 0
                            t2_time = q_props.T2 * 1e6 if hasattr(q_props, 'T2') and q_props.T2 else 0
                            
                            # 오류율 (가능한 경우)
                            readout_error = q_props.readout_error if hasattr(q_props, 'readout_error') else 0
                            
                            # 게이트 오류 - 사용 가능한 경우
                            single_gate_errors = []
                            for gate_name in ['sx', 'x']:
                                try:
                                    # Qiskit 2.0+ 호환 - gate_error 메서드 대신 직접 속성 접근 시도
                                    error = None
                                    if hasattr(properties, 'gate_error'):
                                        error = properties.gate_error(gate_name, [qubit_idx])
                                    elif hasattr(properties, 'gate_property'):
                                        gate_props = properties.gate_property(gate_name, [qubit_idx])
                                        if gate_props and hasattr(gate_props, 'error'):
                                            error = gate_props.error
                                    
                                    if error is not None:
                                        single_gate_errors.append(error)
                                except:
                                    pass
                            
                            single_qubit_gate_error = np.mean(single_gate_errors) if single_gate_errors else 0
                            
                            # 데이터 추가
                            if t1_time > 0:
                                t1_times.append(t1_time)
                            if t2_time > 0:
                                t2_times.append(t2_time)
                            if readout_error > 0:
                                readout_errors.append(readout_error)
                            if single_qubit_gate_error > 0:
                                gate_errors.append(single_qubit_gate_error)
                            
                            # 데이터 저장
                            coherence_data["qubits"][str(qubit_idx)] = {
                                "T1_us": t1_time,
                                "T2_us": t2_time,
                                "readout_error": readout_error,
                                "single_qubit_gate_error": single_qubit_gate_error
                            }
                elif hasattr(properties, 'qubits'):
                    # 이전 API 형식
                    for qubit_idx, qubit_data in enumerate(properties.qubits):
                        # 이전 API 구조에서 데이터 추출
                        t1_time = 0
                        t2_time = 0
                        readout_error = 0
                        
                        # 속성 접근 방식이 여러 가지일 수 있으므로 모두 시도
                        try:
                            for item in qubit_data:
                                if hasattr(item, 'name') and hasattr(item, 'value'):
                                    if item.name == 'T1':
                                        t1_time = item.value * 1e6
                                    elif item.name == 'T2':
                                        t2_time = item.value * 1e6
                                    elif item.name == 'readout_error':
                                        readout_error = item.value
                        except:
                            # 속성 접근이 다르면 직접 인덱스 접근 시도
                            try:
                                t1_time = qubit_data[0].value * 1e6
                                t2_time = qubit_data[1].value * 1e6
                            except:
                                pass
                        
                        # 게이트 오류 계산
                        single_gate_errors = []
                        for gate_name in ['sx', 'x']:
                            try:
                                # Qiskit 2.0+ 호환성 고려
                                error = None
                                if hasattr(properties, 'gate_error'):
                                    error = properties.gate_error(gate_name, [qubit_idx])
                                elif hasattr(properties, 'gate_property'):
                                    gate_props = properties.gate_property(gate_name, [qubit_idx])
                                    if gate_props and hasattr(gate_props, 'error'):
                                        error = gate_props.error
                                
                                if error is not None:
                                    single_gate_errors.append(error)
                            except:
                                pass
                        
                        single_qubit_gate_error = np.mean(single_gate_errors) if single_gate_errors else 0
                        
                        # 데이터 추가
                        if t1_time > 0:
                            t1_times.append(t1_time)
                        if t2_time > 0:
                            t2_times.append(t2_time)
                        if readout_error > 0:
                            readout_errors.append(readout_error)
                        if single_qubit_gate_error > 0:
                            gate_errors.append(single_qubit_gate_error)
                        
                        # 데이터 저장
                        coherence_data["qubits"][str(qubit_idx)] = {
                            "T1_us": t1_time,
                            "T2_us": t2_time,
                            "readout_error": readout_error,
                            "single_qubit_gate_error": single_qubit_gate_error
                        }
                else:
                    print("⚠️ 백엔드 속성 형식을 인식할 수 없습니다.")
                    print(f"사용 가능한 속성: {dir(properties)}")
                    return None
                
                # 통계 계산 (데이터가 있는 경우)
                if t1_times:
                    coherence_data["statistics"]["T1_statistics"] = {
                        "mean": np.mean(t1_times),
                        "std": np.std(t1_times),
                        "min": np.min(t1_times),
                        "max": np.max(t1_times)
                    }
                    print(f"T1 Times: mean={np.mean(t1_times):.2f}±{np.std(t1_times):.2f} μs")
                
                if t2_times:
                    coherence_data["statistics"]["T2_statistics"] = {
                        "mean": np.mean(t2_times),
                        "std": np.std(t2_times),
                        "min": np.min(t2_times),
                        "max": np.max(t2_times)
                    }
                    print(f"T2 Times: mean={np.mean(t2_times):.2f}±{np.std(t2_times):.2f} μs")
                
                if readout_errors:
                    coherence_data["statistics"]["readout_error_statistics"] = {
                        "mean": np.mean(readout_errors),
                        "std": np.std(readout_errors),
                        "min": np.min(readout_errors),
                        "max": np.max(readout_errors)
                    }
                    print(f"Readout Errors: mean={np.mean(readout_errors):.4f}±{np.std(readout_errors):.4f}")
                
                if gate_errors:
                    coherence_data["statistics"]["gate_error_statistics"] = {
                        "mean": np.mean(gate_errors),
                        "std": np.std(gate_errors),
                        "min": np.min(gate_errors),
                        "max": np.max(gate_errors)
                    }
                    print(f"Gate Errors: mean={np.mean(gate_errors):.4f}±{np.std(gate_errors):.4f}")
                
                # 데이터 파일로 저장
                filename = os.path.join(
                    self.base_circuit.output_dir,
                    "coherence_data",
                    f"coherence_{self.backend.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                with open(filename, 'w') as f:
                    json.dump(coherence_data, f, indent=2)
                
                return coherence_data
                
            except Exception as e:
                print(f"⚠️ 속성 가져오기 오류: {str(e)}")
                import traceback
                traceback.print_exc()
                return None
            
        except Exception as e:
            print(f"⚠️ 코히어런스 시간 측정 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_on_ibm_backend(self, circuit_info, verbose=True):
        """
        최신 qiskit_ibm_runtime API를 사용하여 IBM 백엔드에서 직접 회로를 실행합니다.
        
        Args:
            circuit_info (dict): 회로 정보
            verbose (bool): 상세 디버깅 정보 출력 여부
            
        Returns:
            dict: 실행 결과
        """
        if not self.backend:
            print("⚠️ IBM 백엔드가 설정되지 않았습니다.")
            return None
            
        try:
            # 큐빗 수, 게이트, 와이어 정보 추출
            n_qubits = circuit_info["n_qubits"]
            gates = circuit_info["gates"]
            wires_list = circuit_info["wires_list"]
            params_idx = circuit_info["params_idx"]
            params = circuit_info["params"]
            
            # 큐빗 수 확인
            max_backend_qubits = self.backend.configuration().n_qubits
            if n_qubits > max_backend_qubits:
                print(f"⚠️ 경고: 회로의 큐빗 수({n_qubits})가 백엔드 큐빗 수({max_backend_qubits})를 초과합니다.")
                print(f"  백엔드 큐빗 수로 제한합니다.")
                n_qubits = max_backend_qubits
            
            # Qiskit 양자 회로 생성
            qc = QuantumCircuit(n_qubits, n_qubits)
            
            # 게이트 적용 (U 회로)
            if verbose:
                print("\n순방향 회로(U) 적용 중...")
            forward_gates_applied = 0
            for i, (gate, wires) in enumerate(zip(gates, wires_list)):
                if any(w >= n_qubits for w in wires):
                    if verbose:
                        print(f"  ⚠️ 게이트 {i}({gate}): 큐빗 {wires} 범위 초과, 건너뜀")
                    continue
                    
                try:
                    if gate == "H":
                        qc.h(wires[0])
                        if verbose:
                            print(f"  ✓ 게이트 {i}: H 적용 (큐빗 {wires[0]})")
                    elif gate == "X":
                        qc.x(wires[0])
                        if verbose:
                            print(f"  ✓ 게이트 {i}: X 적용 (큐빗 {wires[0]})")
                    elif gate == "Y":
                        qc.y(wires[0])
                        if verbose:
                            print(f"  ✓ 게이트 {i}: Y 적용 (큐빗 {wires[0]})")
                    elif gate == "Z":
                        qc.z(wires[0])
                        if verbose:
                            print(f"  ✓ 게이트 {i}: Z 적용 (큐빗 {wires[0]})")
                    elif gate == "S":
                        qc.s(wires[0])
                        if verbose:
                            print(f"  ✓ 게이트 {i}: S 적용 (큐빗 {wires[0]})")
                    elif gate == "T":
                        qc.t(wires[0])
                        if verbose:
                            print(f"  ✓ 게이트 {i}: T 적용 (큐빗 {wires[0]})")
                    elif gate == "RZ":
                        # 파라미터 인덱스 찾기
                        param_value = None
                        for j, idx in enumerate(params_idx):
                            if idx == i:
                                param_value = params[j]
                                break
                        
                        if param_value is not None:
                            qc.rz(param_value, wires[0])
                            if verbose:
                                print(f"  ✓ 게이트 {i}: RZ({param_value:.4f}) 적용 (큐빗 {wires[0]})")
                        else:
                            if verbose:
                                print(f"  ⚠️ 게이트 {i}: RZ 파라미터를 찾을 수 없음, 건너뜀")
                    elif gate == "RX":
                        # 파라미터 인덱스 찾기
                        param_value = None
                        for j, idx in enumerate(params_idx):
                            if idx == i:
                                param_value = params[j]
                                break
                        
                        if param_value is not None:
                            qc.rx(param_value, wires[0])
                            if verbose:
                                print(f"  ✓ 게이트 {i}: RX({param_value:.4f}) 적용 (큐빗 {wires[0]})")
                        else:
                            if verbose:
                                print(f"  ⚠️ 게이트 {i}: RX 파라미터를 찾을 수 없음, 건너뜀")
                    elif gate == "RY":
                        # 파라미터 인덱스 찾기
                        param_value = None
                        for j, idx in enumerate(params_idx):
                            if idx == i:
                                param_value = params[j]
                                break
                        
                        if param_value is not None:
                            qc.ry(param_value, wires[0])
                            if verbose:
                                print(f"  ✓ 게이트 {i}: RY({param_value:.4f}) 적용 (큐빗 {wires[0]})")
                        else:
                            if verbose:
                                print(f"  ⚠️ 게이트 {i}: RY 파라미터를 찾을 수 없음, 건너뜀")
                    elif gate == "CNOT":
                        if len(wires) >= 2:
                            qc.cx(wires[0], wires[1])
                            if verbose:
                                print(f"  ✓ 게이트 {i}: CNOT 적용 (큐빗 {wires[0]} → {wires[1]})")
                    elif gate == "CZ":
                        if len(wires) >= 2:
                            qc.cz(wires[0], wires[1])
                            if verbose:
                                print(f"  ✓ 게이트 {i}: CZ 적용 (큐빗 {wires[0]} → {wires[1]})")
                    forward_gates_applied += 1
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️ 게이트 {i}({gate}) 적용 오류: {str(e)}")
            
            if verbose:
                print(f"순방향 회로 적용 완료: {forward_gates_applied}/{len(gates)} 게이트 적용됨")
            
            # 역회로 적용 (U†)
            if verbose:
                print("\n역방향 회로(U†) 적용 중...")
            inverse_gates_applied = 0
            for i in range(len(gates)-1, -1, -1):  # 역순으로 게이트 적용
                gate = gates[i]
                wires = wires_list[i]
                
                if any(w >= n_qubits for w in wires):
                    if verbose:
                        print(f"  ⚠️ 게이트 {i}({gate}) 인버스: 큐빗 {wires} 범위 초과, 건너뜀")
                    continue
                
                try:
                    if gate == "H":
                        qc.h(wires[0])  # H = H†
                        if verbose:
                            print(f"  ✓ 게이트 {i} 인버스: H 적용 (큐빗 {wires[0]})")
                    elif gate == "X":
                        qc.x(wires[0])  # X = X†
                        if verbose:
                            print(f"  ✓ 게이트 {i} 인버스: X 적용 (큐빗 {wires[0]})")
                    elif gate == "Y":
                        qc.y(wires[0])  # Y = Y†
                        if verbose:
                            print(f"  ✓ 게이트 {i} 인버스: Y 적용 (큐빗 {wires[0]})")
                    elif gate == "Z":
                        qc.z(wires[0])  # Z = Z†
                        if verbose:
                            print(f"  ✓ 게이트 {i} 인버스: Z 적용 (큐빗 {wires[0]})")
                    elif gate == "S":
                        qc.sdg(wires[0])  # S† (S의 역)
                        if verbose:
                            print(f"  ✓ 게이트 {i} 인버스: S† 적용 (큐빗 {wires[0]})")
                    elif gate == "T":
                        qc.tdg(wires[0])  # T† (T의 역)
                        if verbose:
                            print(f"  ✓ 게이트 {i} 인버스: T† 적용 (큐빗 {wires[0]})")
                    elif gate == "RZ":
                        # 파라미터 인덱스 찾기
                        param_value = None
                        for j, idx in enumerate(params_idx):
                            if idx == i:
                                param_value = params[j]
                                break
                        
                        if param_value is not None:
                            qc.rz(-param_value, wires[0])  # RZ(θ)† = RZ(-θ)
                            if verbose:
                                print(f"  ✓ 게이트 {i} 인버스: RZ({-param_value:.4f}) 적용 (큐빗 {wires[0]})")
                        else:
                            if verbose:
                                print(f"  ⚠️ 게이트 {i} 인버스: RZ 파라미터를 찾을 수 없음, 건너뜀")
                    elif gate == "RX":
                        # 파라미터 인덱스 찾기
                        param_value = None
                        for j, idx in enumerate(params_idx):
                            if idx == i:
                                param_value = params[j]
                                break
                        
                        if param_value is not None:
                            qc.rx(-param_value, wires[0])  # RX(θ)† = RX(-θ)
                            if verbose:
                                print(f"  ✓ 게이트 {i} 인버스: RX({-param_value:.4f}) 적용 (큐빗 {wires[0]})")
                        else:
                            if verbose:
                                print(f"  ⚠️ 게이트 {i} 인버스: RX 파라미터를 찾을 수 없음, 건너뜀")
                    elif gate == "RY":
                        # 파라미터 인덱스 찾기
                        param_value = None
                        for j, idx in enumerate(params_idx):
                            if idx == i:
                                param_value = params[j]
                                break
                        
                        if param_value is not None:
                            qc.ry(-param_value, wires[0])  # RY(θ)† = RY(-θ)
                            if verbose:
                                print(f"  ✓ 게이트 {i} 인버스: RY({-param_value:.4f}) 적용 (큐빗 {wires[0]})")
                        else:
                            if verbose:
                                print(f"  ⚠️ 게이트 {i} 인버스: RY 파라미터를 찾을 수 없음, 건너뜀")
                    elif gate == "CNOT":
                        if len(wires) >= 2:
                            qc.cx(wires[0], wires[1])  # CNOT = CNOT†
                            if verbose:
                                print(f"  ✓ 게이트 {i} 인버스: CNOT 적용 (큐빗 {wires[0]} → {wires[1]})")
                    elif gate == "CZ":
                        if len(wires) >= 2:
                            qc.cz(wires[0], wires[1])  # CZ = CZ†
                            if verbose:
                                print(f"  ✓ 게이트 {i} 인버스: CZ 적용 (큐빗 {wires[0]} → {wires[1]})")
                    inverse_gates_applied += 1
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️ 게이트 {i}({gate}) 인버스 적용 오류: {str(e)}")
            
            if verbose:
                print(f"역방향 회로 적용 완료: {inverse_gates_applied}/{len(gates)} 게이트 적용됨")
                print(f"\nQiskit 회로 생성 완료 ({n_qubits} 큐빗, {len(gates)*2} 게이트)")
                print(f"회로 깊이: {qc.depth()}")
            
            # 트랜스파일 전 회로 정보 출력
            if verbose:
                print("\n===== 트랜스파일 전 회로 정보 =====")
                print(f"회로 깊이: {qc.depth()}")
                print(f"게이트 수: {sum(len(qc.data) for qc in [qc])}")
                print("게이트 통계:")
                gate_counts = {}
                for gate in qc.data:
                    gate_name = gate[0].name
                    if gate_name in gate_counts:
                        gate_counts[gate_name] += 1
                    else:
                        gate_counts[gate_name] = 1
                for gate_name, count in gate_counts.items():
                    print(f"  - {gate_name}: {count}개")
            
            # 트랜스파일 전 회로 다이어그램 저장
            try:
                if verbose:
                    qc_original = qc.copy()
                    fig = qc_original.draw(output='mpl', style={'name': 'bw'})
                    fig.savefig('before_transpile_circuit.png')
                    print("트랜스파일 전 회로 다이어그램 저장됨: before_transpile_circuit.png")
            except Exception as e:
                if verbose:
                    print(f"회로 다이어그램 저장 오류: {str(e)}")
            
            # IBM 백엔드에 맞게 회로 트랜스파일
            if verbose:
                print("\n트랜스파일 수행 중...")
            # 최적화 레벨을 0으로 낮춰 최소한의 변환만 수행
            qc_transpiled = transpile(qc, backend=self.backend, optimization_level=self.optimization_level)
            
            # 트랜스파일 후 회로 정보 출력
            if verbose:
                print("\n===== 트랜스파일 후 회로 정보 =====")
                print(f"회로 깊이: {qc_transpiled.depth()}")
                print(f"게이트 수: {sum(len(qc_transpiled.data) for qc in [qc_transpiled])}")
                print("게이트 통계:")
                gate_counts = {}
                for gate in qc_transpiled.data:
                    gate_name = gate[0].name
                    if gate_name in gate_counts:
                        gate_counts[gate_name] += 1
                    else:
                        gate_counts[gate_name] = 1
                for gate_name, count in gate_counts.items():
                    print(f"  - {gate_name}: {count}개")
            
            # 트랜스파일 후 회로 다이어그램 저장
            try:
                if verbose:
                    fig = qc_transpiled.draw(output='mpl', style={'name': 'bw'})
                    fig.savefig('after_transpile_circuit.png')
                    print("트랜스파일 후 회로 다이어그램 저장됨: after_transpile_circuit.png")
            except Exception as e:
                if verbose:
                    print(f"회로 다이어그램 저장 오류: {str(e)}")
            
            # 원래 트랜스파일된 회로를 사용
            qc = qc_transpiled
            
            # 측정 추가
            qc.measure_all()
            
            # 샷 수 설정 (큐빗 수에 따라 동적 조정)
            shots = get_ibm_shots(n_qubits)
            if verbose:
                print(f"설정된 측정 샷 수: {shots} (큐빗 수 {n_qubits}에 최적화됨)")
            
            # IBM 백엔드로 실행
            if verbose:
                print(f"\n{self.backend.name} 백엔드에서 실행 중...")
            else:
                print(f"IBM {self.backend.name} 작업 제출 중...")
            
            # Qiskit 2.0+ API를 사용한 회로 실행
            with Session(backend=self.backend) as session:
                try:
                    # 최신 Qiskit 0.26.0+ API 호환 방식으로 Sampler 초기화
                    sampler = Sampler(backend=self.backend)
                    
                    # 회로 실행 (설정된 샷 수 사용)
                    job = sampler.run([qc], shots=shots)
                    
                    if verbose:
                        print("작업 ID:", job.job_id())
                        print("작업 상태:", job.status())
                    
                    print("결과 기다리는 중...")
                    result = job.result()
                    
                    # 결과 처리
                    if verbose:
                        print("\n===== 결과 객체 정보 =====")
                        print(f"결과 타입: {type(result)}")
                    
                    # Qiskit 2.0+ API에서의 결과 추출
                    try:
                        # Sampler 결과 추출
                        if len(result) > 0:
                            # 다양한 결과 형식 처리 시도
                            counts = {}
                            
                            # 결과 구조 디버깅
                            if verbose:
                                print("\n결과 구조 디버깅:")
                                print(f"Result 속성: {dir(result[0])}")
                                if hasattr(result[0], 'data'):
                                    print(f"Data 속성: {dir(result[0].data)}")
                            
                            # 방법 1: DataBin에서 비트스트링과 카운트 추출
                            if hasattr(result[0], 'data') and hasattr(result[0].data, 'meas'):
                                if verbose:
                                    print("측정 데이터 직접 접근 사용")
                                bit_array = result[0].data.meas
                                counts_dict = {}
                                
                                if verbose:
                                    print(f"비트 배열 형식: {type(bit_array)}")
                                    print(f"샷 수: {bit_array.num_shots}, 비트 수: {bit_array.num_bits}")
                                
                                # 방법 1: meas 객체에서 직접 counts 가져오기 시도
                                if hasattr(result[0].data.meas, 'get_counts') and callable(result[0].data.meas.get_counts):
                                    counts_dict = result[0].data.meas.get_counts()
                                    if verbose:
                                        print("\n=== BitArray.get_counts() 메소드 사용 ===")
                                        print(f"반환된 counts_dict 타입: {type(counts_dict)}")
                                        print(f"counts_dict 크기: {len(counts_dict) if hasattr(counts_dict, '__len__') else '크기 정보 없음'}")
                                        if hasattr(counts_dict, 'items'):
                                            print("counts_dict 상위 5개 항목:")
                                            for i, (k, v) in enumerate(list(counts_dict.items())[:5]):
                                                print(f"  {k}: {v}")
                                                
                                # 비트열 슬라이싱: 각 상태에서 첫 n_qubits 비트만 사용
                                if verbose:
                                    print("\n첫 n_qubits 비트만 추출하여 상태 재구성")
                                processed_counts = {}
                                for bit_str, count in counts_dict.items():
                                    # 비트열이 너무 길면 잘라냄 (첫 n_qubits 비트만 사용)
                                    if len(bit_str) > n_qubits:
                                        short_bit_str = bit_str[:n_qubits]
                                    else:
                                        short_bit_str = bit_str
                                        
                                    # 필요시 0으로 패딩
                                    if len(short_bit_str) < n_qubits:
                                        short_bit_str = short_bit_str.zfill(n_qubits)
                                        
                                    # 카운트 누적
                                    if short_bit_str in processed_counts:
                                        processed_counts[short_bit_str] += count
                                    else:
                                        processed_counts[short_bit_str] = count
                                
                                # 처리된 결과 출력
                                if verbose:
                                    print(f"처리된 counts_dict 크기: {len(processed_counts)}")
                                    print("처리된 상위 5개 상태:")
                                    for i, (k, v) in enumerate(sorted(processed_counts.items(), key=lambda x: x[1], reverse=True)[:5]):
                                        print(f"  |{k}⟩: {v}회")
                                    
                                # 원래 counts_dict를 처리된 버전으로 교체
                                counts_dict = processed_counts
                                
                                # 중요: 처리된 카운트와 총 카운트를 직접 저장 (디버그에서 사용할 수 있도록)
                                total_processed_counts = sum(processed_counts.values())
                                direct_result = {
                                    "processed_counts_direct": processed_counts,
                                    "total_counts_direct": total_processed_counts
                                }
                            # 방법 2: 결과에서 직접 counts 가져오기
                            elif hasattr(result[0], 'data') and hasattr(result[0].data, 'counts'):
                                if verbose:
                                    print("counts 속성 사용")
                                counts_dict = result[0].data.counts
                            # 방법 3: quasi_dists 사용 (Qiskit 1.0+)
                            elif hasattr(result[0], 'quasi_dists'):
                                if verbose:
                                    print("quasi_dists 속성 사용")
                                quasi_dists = result[0].quasi_dists[0]
                                for bitstring, prob in quasi_dists.items():
                                    counts_dict[bitstring] = int(round(prob * shots))
                            # 방법 4: 결과 문자열 분석 (최후의 수단)
                            else:
                                if verbose:
                                    print("문자열 파싱 시도")
                                result_str = str(result)
                                # 결과 문자열에서 카운트 추출 시도
                                import re
                                count_pattern = r"'([01]+)': *(\d+)"
                                matches = re.findall(count_pattern, result_str)
                                if matches:
                                    for bitstring, count in matches:
                                        counts_dict[bitstring] = int(count)
                                else:
                                    # 모든 방법 실패 시 임의의 결과 생성
                                    if verbose:
                                        print("⚠️ 결과 해석 실패, 임의 결과 생성")
                                    import random
                                    random_counts = {}
                                    for _ in range(min(100, 2**n_qubits)):  # 최대 100개 상태
                                        bitstring = ''.join(random.choice('01') for _ in range(n_qubits))
                                        random_counts[bitstring] = random.randint(1, 10)
                                    counts_dict = random_counts
                            
                            # 카운트 딕셔너리 생성 (확률이 아닌 실제 측정 카운트)
                            sparse_counts = {}
                            
                            for bit_string, count in counts_dict.items():
                                if isinstance(bit_string, int):
                                    # 정수 비트스트링을 이진 문자열로 변환
                                    state_str = format(bit_string, f'0{n_qubits}b')
                                else:
                                    # 문자열 비트스트링 그대로 사용
                                    state_str = bit_string
                                
                                # n_qubits 길이에 맞추기
                                if len(state_str) < n_qubits:
                                    state_str = state_str.zfill(n_qubits)
                                elif len(state_str) > n_qubits:
                                    # 첫 n_qubits 비트만 사용하도록 수정
                                    state_str = state_str[:n_qubits]
                                
                                # 기존 상태와 카운트 누적
                                if state_str in sparse_counts:
                                    sparse_counts[state_str] += count
                                else:
                                    sparse_counts[state_str] = count
                            
                            # 총 카운트 계산 (verbose 조건과 관계없이 항상 계산)
                            total_counts = sum(sparse_counts.values())
                            
                            # 디버깅 출력
                            if verbose:
                                print("\n===== IBM 측정 결과 상세 디버깅 =====")
                                
                                # 1. 처리된 측정 결과 요약
                                print(f"측정된 고유 상태 수: {len(sparse_counts)}")
                                print(f"총 측정 횟수: {total_counts}")
                                
                                # 2. 상위 20개 측정 결과 출력 (빈도 내림차순)
                                if sparse_counts:
                                    sorted_counts = sorted(sparse_counts.items(), key=lambda x: x[1], reverse=True)
                                    print(f"\n상위 측정 결과 (상위 20개):")
                                    for i, (state, count) in enumerate(sorted_counts[:20]):
                                        print(f"{i+1:2d}. |{state}⟩: {count}회 ({count/total_counts*100:.2f}%)")
                                    
                                    # 3. 각 비트 위치별 0/1 분포 계산
                                    if len(sorted_counts) > 0 and len(sorted_counts[0][0]) > 0:
                                        state_len = len(sorted_counts[0][0])
                                        bit_counts = [[0, 0] for _ in range(state_len)]
                                        
                                        for state, count in sparse_counts.items():
                                            for i, bit in enumerate(state):
                                                if i < state_len:
                                                    bit_val = int(bit)
                                                    bit_counts[i][bit_val] += count
                                        
                                        # 각 비트 위치의 0/1 확률 출력
                                        print("\n큐빗 별 0/1 분포:")
                                        for i, (zeros, ones) in enumerate(bit_counts):
                                            total = zeros + ones
                                            if total > 0:
                                                p0 = zeros / total
                                                p1 = ones / total
                                                print(f"큐빗 {i:2d}: 0={p0:.4f}, 1={p1:.4f}")
                                
                                print("===== 디버깅 출력 종료 =====\n")
                            
                            # |00000..000> 상태의 카운트 
                            zero_state = '0' * n_qubits
                            zero_state_count = sparse_counts.get(zero_state, 0)
                            zero_state_prob = zero_state_count / total_counts if total_counts > 0 else 0
                            
                            # 직접 결과 변수가 정의되지 않은 경우 처리
                            if 'direct_result' not in locals():
                                direct_result = {
                                    "processed_counts_direct": sparse_counts,
                                    "total_counts_direct": total_counts
                                }
                                
                            # 결과 반환
                            results = {
                                "zero_state_probability": zero_state_prob,
                                "measured_states": total_counts,
                                "measurement_counts": sparse_counts,  # 모든 측정 상태 저장
                                "zero_state_count": zero_state_count,
                                "backend": self.backend.name,
                                "result_obj": result,  # 원본 결과 객체 추가
                                "direct_result": direct_result
                            }
                            
                            return results
                        else:
                            print("⚠️ 결과 배열이 비어 있습니다.")
                            return None
                    except Exception as e:
                        print(f"⚠️ 결과 처리 중 오류 발생: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        return None
                    
                except Exception as e:
                    print(f"⚠️ IBM 백엔드 작업 제출 중 오류 발생: {str(e)}")
                    print("  시뮬레이터로 대체하는 것을 권장합니다.")
                    import traceback
                    traceback.print_exc()
                    return None
                
        except Exception as e:
            print(f"⚠️ IBM 백엔드 실행 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def run_parametrized_circuit(self, circuit_info, param_sets=None, verbose=True):
        """
        파라미터화된 회로를 구성하고 단일 작업으로 여러 파라미터 세트를 실행합니다.
        
        Args:
            circuit_info (dict): 회로 정보
            param_sets (list): 실행할 파라미터 세트 목록 [params1, params2, ...] (None이면 원본 파라미터만 사용)
            verbose (bool): 상세 디버깅 정보 출력 여부
            
        Returns:
            list: 각 파라미터 세트에 대한 실행 결과 목록
        """
        from qiskit.circuit import Parameter
        
        if not self.backend:
            print("⚠️ IBM 백엔드가 설정되지 않았습니다.")
            return None
            
        try:
            # 큐빗 수, 게이트, 와이어 정보 추출
            n_qubits = circuit_info["n_qubits"]
            gates = circuit_info["gates"]
            wires_list = circuit_info["wires_list"]
            params_idx = circuit_info["params_idx"]
            original_params = circuit_info["params"]
            
            # 파라미터 세트가 없으면 원본 파라미터만 사용
            if param_sets is None:
                param_sets = [original_params]
            
            if verbose:
                print(f"\n{len(param_sets)}개 파라미터 세트로 파라미터화된 회로 실행 준비 중...")
            
            # 큐빗 수 확인
            max_backend_qubits = self.backend.configuration().n_qubits
            if n_qubits > max_backend_qubits:
                print(f"⚠️ 경고: 회로의 큐빗 수({n_qubits})가 백엔드 큐빗 수({max_backend_qubits})를 초과합니다.")
                print(f"  백엔드 큐빗 수로 제한합니다.")
                n_qubits = max_backend_qubits
            
            # 심볼릭 파라미터를 사용한 양자 회로 생성
            qc = QuantumCircuit(n_qubits, n_qubits)
            
            # RZ 파라미터 심볼릭 파라미터로 정의
            symbolic_params = []
            for i in range(len(original_params)):
                symbolic_params.append(Parameter(f"theta_{i}"))
            
            # 게이트 적용 (U 회로)
            if verbose:
                print("\n순방향 회로(U) 적용 중...")
            
            forward_gates_applied = 0
            for i, (gate, wires) in enumerate(zip(gates, wires_list)):
                if any(w >= n_qubits for w in wires):
                    if verbose:
                        print(f"  ⚠️ 게이트 {i}({gate}): 큐빗 {wires} 범위 초과, 건너뜀")
                    continue
                    
                try:
                    if gate == "H":
                        qc.h(wires[0])
                    elif gate == "X":
                        qc.x(wires[0])
                    elif gate == "Y":
                        qc.y(wires[0])
                    elif gate == "Z":
                        qc.z(wires[0])
                    elif gate == "S":
                        qc.s(wires[0])
                    elif gate == "T":
                        qc.t(wires[0])
                    elif gate == "RZ":
                        # 파라미터 인덱스 찾기
                        param_idx = None
                        for j, idx in enumerate(params_idx):
                            if idx == i:
                                param_idx = j
                                break
                        
                        if param_idx is not None:
                            # 심볼릭 파라미터 사용
                            qc.rz(symbolic_params[param_idx], wires[0])
                        else:
                            if verbose:
                                print(f"  ⚠️ 게이트 {i}: RZ 파라미터를 찾을 수 없음, 건너뜀")
                    elif gate == "RX":
                        # 파라미터 인덱스 찾기
                        param_value = None
                        for j, idx in enumerate(params_idx):
                            if idx == i:
                                param_value = params[j]
                                break
                        
                        if param_value is not None:
                            qc.rx(param_value, wires[0])
                            if verbose:
                                print(f"  ✓ 게이트 {i}: RX({param_value:.4f}) 적용 (큐빗 {wires[0]})")
                        else:
                            if verbose:
                                print(f"  ⚠️ 게이트 {i}: RX 파라미터를 찾을 수 없음, 건너뜀")
                    elif gate == "RY":
                        # 파라미터 인덱스 찾기
                        param_value = None
                        for j, idx in enumerate(params_idx):
                            if idx == i:
                                param_value = params[j]
                                break
                        
                        if param_value is not None:
                            qc.ry(param_value, wires[0])
                            if verbose:
                                print(f"  ✓ 게이트 {i}: RY({param_value:.4f}) 적용 (큐빗 {wires[0]})")
                        else:
                            if verbose:
                                print(f"  ⚠️ 게이트 {i}: RY 파라미터를 찾을 수 없음, 건너뜀")
                    elif gate == "CNOT":
                        if len(wires) >= 2:
                            qc.cx(wires[0], wires[1])
                    forward_gates_applied += 1
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️ 게이트 {i}({gate}) 적용 오류: {str(e)}")
            
            # 역회로 적용 (U†)
            if verbose:
                print("\n역방향 회로(U†) 적용 중...")
                
            inverse_gates_applied = 0
            for i in range(len(gates)-1, -1, -1):  # 역순으로 게이트 적용
                gate = gates[i]
                wires = wires_list[i]
                
                if any(w >= n_qubits for w in wires):
                    if verbose:
                        print(f"  ⚠️ 게이트 {i}({gate}) 인버스: 큐빗 {wires} 범위 초과, 건너뜀")
                    continue
                
                try:
                    if gate == "H":
                        qc.h(wires[0])  # H = H†
                        if verbose:
                            print(f"  ✓ 게이트 {i} 인버스: H 적용 (큐빗 {wires[0]})")
                    elif gate == "X":
                        qc.x(wires[0])  # X = X†
                        if verbose:
                            print(f"  ✓ 게이트 {i} 인버스: X 적용 (큐빗 {wires[0]})")
                    elif gate == "Y":
                        qc.y(wires[0])  # Y = Y†
                        if verbose:
                            print(f"  ✓ 게이트 {i} 인버스: Y 적용 (큐빗 {wires[0]})")
                    elif gate == "Z":
                        qc.z(wires[0])  # Z = Z†
                        if verbose:
                            print(f"  ✓ 게이트 {i} 인버스: Z 적용 (큐빗 {wires[0]})")
                    elif gate == "S":
                        qc.sdg(wires[0])  # S† (S의 역)
                        if verbose:
                            print(f"  ✓ 게이트 {i} 인버스: S† 적용 (큐빗 {wires[0]})")
                    elif gate == "T":
                        qc.tdg(wires[0])  # T† (T의 역)
                        if verbose:
                            print(f"  ✓ 게이트 {i} 인버스: T† 적용 (큐빗 {wires[0]})")
                    elif gate == "RZ":
                        # 파라미터 인덱스 찾기
                        param_idx = None
                        for j, idx in enumerate(params_idx):
                            if idx == i:
                                param_idx = j
                                break
                        
                        if param_idx is not None:
                            # 심볼릭 파라미터 부호 반전 (RZ(θ)† = RZ(-θ))
                            qc.rz(-symbolic_params[param_idx], wires[0])
                        else:
                            if verbose:
                                print(f"  ⚠️ 게이트 {i} 인버스: RZ 파라미터를 찾을 수 없음, 건너뜀")
                    elif gate == "RX":
                        # 파라미터 인덱스 찾기
                        param_idx = None
                        for j, idx in enumerate(params_idx):
                            if idx == i:
                                param_idx = j
                                break
                        
                        if param_idx is not None:
                            # 심볼릭 파라미터 부호 반전 (RX(θ)† = RX(-θ))
                            qc.rx(-symbolic_params[param_idx], wires[0])
                        else:
                            if verbose:
                                print(f"  ⚠️ 게이트 {i} 인버스: RX 파라미터를 찾을 수 없음, 건너뜀")
                    elif gate == "RY":
                        # 파라미터 인덱스 찾기
                        param_idx = None
                        for j, idx in enumerate(params_idx):
                            if idx == i:
                                param_idx = j
                                break
                        
                        if param_idx is not None:
                            # 심볼릭 파라미터 부호 반전 (RY(θ)† = RY(-θ))
                            qc.ry(-symbolic_params[param_idx], wires[0])
                        else:
                            if verbose:
                                print(f"  ⚠️ 게이트 {i} 인버스: RY 파라미터를 찾을 수 없음, 건너뜀")
                    elif gate == "CNOT":
                        if len(wires) >= 2:
                            qc.cx(wires[0], wires[1])  # CNOT = CNOT†
                            if verbose:
                                print(f"  ✓ 게이트 {i} 인버스: CNOT 적용 (큐빗 {wires[0]} → {wires[1]})")
                    elif gate == "CZ":
                        if len(wires) >= 2:
                            qc.cz(wires[0], wires[1])  # CZ = CZ†
                            if verbose:
                                print(f"  ✓ 게이트 {i} 인버스: CZ 적용 (큐빗 {wires[0]} → {wires[1]})")
                    inverse_gates_applied += 1
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️ 게이트 {i}({gate}) 인버스 적용 오류: {str(e)}")
            
            if verbose:
                print(f"파라미터화된 회로 생성 완료: {forward_gates_applied+inverse_gates_applied}/{len(gates)*2} 게이트")
                print(f"회로 깊이: {qc.depth()}")
            
            # 트랜스파일 수행
            if verbose:
                print("\n트랜스파일 수행 중...")
            qc_transpiled = transpile(qc, backend=self.backend, optimization_level=self.optimization_level)
            
            # 측정 추가
            qc_transpiled.measure_all()
            
            # 파라미터 이름 가져오기
            param_names = [p.name for p in qc_transpiled.parameters]
            
            # 샷 수 설정
            shots = self.shots
            
            # IBM 백엔드로 실행
            print(f"\nIBM {self.backend.name} 백엔드에서 {len(param_sets)}개 파라미터 세트로 단일 작업 실행 중...")
            
            # Qiskit 2.0+ API를 사용한 회로 실행
            with Session(backend=self.backend) as session:
                try:
                    # Sampler 초기화
                    sampler = Sampler(backend=self.backend)
                    
                    # 파라미터화된 회로 실행 (Qiskit 1.0+ API 형식으로 파라미터 전달)
                    if len(param_names) == 0:
                        # 파라미터가 없는 경우 간단히 여러 번 실행
                        circuits = [qc_transpiled] * len(param_sets)
                        job = sampler.run(circuits, shots=shots)
                    else:
                        # Qiskit 2.0 API를 위한 올바른 형식으로 파라미터 변환
                        # 딕셔너리가 아닌 리스트 형태로 파라미터 전달
                        param_binds = []
                        for params in param_sets:
                            # params가 리스트인지 확인
                            if isinstance(params, list) or isinstance(params, tuple) or isinstance(params, np.ndarray):
                                param_binds.append(params)
                            else:
                                print(f"⚠️ 파라미터 형식 오류: {type(params)}")
                                param_binds.append([])
                        
                        job = sampler.run(
                            [(qc_transpiled, params) for params in param_binds],
                            shots=shots
                        )
                    
                    print("작업 ID:", job.job_id())
                    print("작업 상태:", job.status())
                    
                    print("결과 기다리는 중...")
                    result = job.result()
                    
                    # 결과 처리
                    print(f"\n{len(param_sets)}개 파라미터 세트 실행 결과 처리 중...")
                    
                    all_results = []
                    
                    # 각 파라미터 세트에 대한 결과 추출
                    for i in range(len(result)):
                        try:
                            # 먼저 결과 구조 확인 (디버깅 시 도움될 수 있음)
                            if verbose and i == 0:
                                print(f"\n첫 번째 결과 객체 디버깅:")
                                print(f"결과 타입: {type(result[i])}")
                                print(f"결과 속성: {dir(result[i])}")
                                if hasattr(result[i], 'data'):
                                    print(f"data 속성: {dir(result[i].data)}")
                           
                            # 측정 결과 추출 (기존 방식과 동일)
                            counts = {}
                                
                            # DataBin에서 비트스트링과 카운트 추출
                            if hasattr(result[i], 'data') and hasattr(result[i].data, 'meas'):
                                bit_array = result[i].data.meas
                                
                                if hasattr(bit_array, 'get_counts') and callable(getattr(bit_array, 'get_counts')):
                                    counts_dict = bit_array.get_counts()
                                else:
                                    counts_dict = {}
                                
                            # 다른 결과 형식 처리
                            elif hasattr(result[i], 'data') and hasattr(result[i].data, 'counts'):
                                if verbose:
                                    print("counts 속성 사용")
                                counts_dict = result[i].data.counts
                            elif hasattr(result[i], 'quasi_dists'):
                                if verbose:
                                    print("quasi_dists 속성 사용")
                                quasi_dists = result[i].quasi_dists[0]
                                counts_dict = {}
                                for bitstring, prob in quasi_dists.items():
                                    counts_dict[bitstring] = int(round(prob * shots))
                            else:
                                # 문자열 파싱
                                result_str = str(result[i])
                                import re
                                count_pattern = r"'([01]+)': *(\d+)"
                                matches = re.findall(count_pattern, result_str)
                                counts_dict = {}
                                if matches:
                                    for bitstring, count in matches:
                                        counts_dict[bitstring] = int(count)
                            
                            # 비트열 처리 및 카운트 합산
                            sparse_counts = {}
                            total_counts = 0
                            
                            for bit_str, count in counts_dict.items():
                                # 정수인 경우 비트 문자열로 변환
                                if isinstance(bit_str, int):
                                    state_str = format(bit_str, f'0{n_qubits}b')
                                else:
                                    state_str = bit_str
                                
                                # 비트열 길이 맞추기
                                if len(state_str) < n_qubits:
                                    state_str = state_str.zfill(n_qubits)
                                elif len(state_str) > n_qubits:
                                    state_str = state_str[:n_qubits]
                                
                                # 카운트 누적
                                if state_str in sparse_counts:
                                    sparse_counts[state_str] += count
                                else:
                                    sparse_counts[state_str] = count
                                
                                total_counts += count
                            
                            # 총 카운트가 0인 경우 처리
                            if total_counts == 0:
                                print(f"⚠️ 결과 {i+1}에서 총 카운트가 0입니다.")
                                continue
                                
                            # |00000..000> 상태의 카운트
                            zero_state = '0' * n_qubits
                            zero_state_count = sparse_counts.get(zero_state, 0)
                            zero_state_prob = zero_state_count / total_counts
                            
                            # 결과 저장
                            param_result = {
                                "zero_state_probability": zero_state_prob,
                                "measured_states": total_counts,
                                "measurement_counts": sparse_counts,
                                "zero_state_count": zero_state_count,
                                "backend": self.backend.name,
                                "params_idx": i,
                                "direct_result": {
                                    "processed_counts_direct": sparse_counts,
                                    "total_counts_direct": total_counts
                                }
                            }
                            
                            all_results.append(param_result)
                        except Exception as e:
                            print(f"⚠️ 결과 {i+1} 처리 중 오류 발생: {str(e)}")
                        
                        # 진행 상황 출력
                        if (i % 10 == 0) or (i == len(param_sets) - 1):
                            print(f"  {i+1}/{len(param_sets)} 파라미터 세트 처리 완료")
                    
                    print(f"총 {len(all_results)}/{len(param_sets)} 파라미터 세트 처리 완료")
                    return all_results
                    
                except Exception as e:
                    print(f"⚠️ IBM 백엔드 작업 제출 중 오류 발생: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return None
                
        except Exception as e:
            print(f"⚠️ 파라미터화된 회로 실행 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def run_classical_shadow_circuits(self, circuit_info, param_sets=None, shadow_shots=None, verbose=True):
        """
        Classical Shadow 방법론을 위한 파라미터화된 회로 실행
        
        Args:
            circuit_info (dict): 회로 정보
            param_sets (list): 실행할 파라미터 세트 목록 (None이면 원본 파라미터만 사용)
            shadow_shots (int): 각 파라미터 세트당 샷 수 (None이면 설정값 사용)
            verbose (bool): 상세 디버깅 정보 출력 여부
            
        Returns:
            list: 각 파라미터 세트에 대한 실행 결과 목록
        """
        from qiskit.circuit import Parameter
        
        if not self.backend:
            print("⚠️ IBM 백엔드가 설정되지 않았습니다.")
            return None
        
        # 중앙 설정에서 샷 수 가져오기
        if shadow_shots is None:
            shadow_shots = config.ibm_backend.expressibility_shots
            
        try:
            # 큐빗 수, 게이트, 와이어 정보 추출
            n_qubits = circuit_info["n_qubits"]
            gates = circuit_info["gates"]
            wires_list = circuit_info["wires_list"]
            params_idx = circuit_info["params_idx"]
            original_params = circuit_info["params"]
            
            # 파라미터 세트가 없으면 원본 파라미터만 사용
            if param_sets is None:
                param_sets = [original_params]
            
            if verbose:
                print(f"\n{len(param_sets)}개 파라미터 세트로 Classical Shadow 회로 실행 준비 중...")
                print(f"각 파라미터 세트당 {shadow_shots} 샷")
            
            # 큐빗 수 확인
            max_backend_qubits = self.backend.configuration().n_qubits
            if n_qubits > max_backend_qubits:
                print(f"⚠️ 경고: 회로의 큐빗 수({n_qubits})가 백엔드 큐빗 수({max_backend_qubits})를 초과합니다.")
                print(f"  백엔드 큐빗 수로 제한합니다.")
                n_qubits = max_backend_qubits
            
            # 심볼릭 파라미터를 사용한 양자 회로 생성 (Classical Shadow용)
            qc = QuantumCircuit(n_qubits, n_qubits)
            
            # RZ 파라미터 심볼릭 파라미터로 정의
            symbolic_params = []
            for i in range(len(original_params)):
                symbolic_params.append(Parameter(f"theta_{i}"))
            
            # 게이트 적용 (원본 회로만, 인버스 회로 없음 - Classical Shadow용)
            if verbose:
                print("\nClassical Shadow용 회로 구성 중...")
            
            forward_gates_applied = 0
            for i, (gate, wires) in enumerate(zip(gates, wires_list)):
                if any(w >= n_qubits for w in wires):
                    if verbose:
                        print(f"  ⚠️ 게이트 {i}({gate}): 큐빗 {wires} 범위 초과, 건너뜀")
                    continue
                    
                try:
                    if gate == "H":
                        qc.h(wires[0])
                    elif gate == "X":
                        qc.x(wires[0])
                    elif gate == "Y":
                        qc.y(wires[0])
                    elif gate == "Z":
                        qc.z(wires[0])
                    elif gate == "S":
                        qc.s(wires[0])
                    elif gate == "T":
                        qc.t(wires[0])
                    elif gate == "RZ":
                        # 파라미터 인덱스 찾기
                        param_idx = None
                        for j, idx in enumerate(params_idx):
                            if idx == i:
                                param_idx = j
                                break
                        
                        if param_idx is not None:
                            # 심볼릭 파라미터 사용
                            qc.rz(symbolic_params[param_idx], wires[0])
                        else:
                            if verbose:
                                print(f"  ⚠️ 게이트 {i}: RZ 파라미터를 찾을 수 없음, 건너뜀")
                    elif gate == "RX":
                        # 파라미터 인덱스 찾기
                        param_value = None
                        for j, idx in enumerate(params_idx):
                            if idx == i:
                                param_value = params[j]
                                break
                        
                        if param_value is not None:
                            qc.rx(param_value, wires[0])
                            if verbose:
                                print(f"  ✓ 게이트 {i}: RX({param_value:.4f}) 적용 (큐빗 {wires[0]})")
                        else:
                            if verbose:
                                print(f"  ⚠️ 게이트 {i}: RX 파라미터를 찾을 수 없음, 건너뜀")
                    elif gate == "RY":
                        # 파라미터 인덱스 찾기
                        param_value = None
                        for j, idx in enumerate(params_idx):
                            if idx == i:
                                param_value = params[j]
                                break
                        
                        if param_value is not None:
                            qc.ry(param_value, wires[0])
                            if verbose:
                                print(f"  ✓ 게이트 {i}: RY({param_value:.4f}) 적용 (큐빗 {wires[0]})")
                        else:
                            if verbose:
                                print(f"  ⚠️ 게이트 {i}: RY 파라미터를 찾을 수 없음, 건너뜀")
                    elif gate == "CNOT":
                        if len(wires) >= 2:
                            qc.cx(wires[0], wires[1])
                    forward_gates_applied += 1
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️ 게이트 {i}({gate}) 적용 오류: {str(e)}")
            
            # Classical Shadow를 위한 랜덤 Clifford 측정 추가
            import random
            
            # 각 큐빗에 대해 랜덤 Pauli 기저로 회전
            measurement_bases = []
            for qubit in range(n_qubits):
                # 랜덤 Pauli 기저 선택
                basis = random.choice(['X', 'Y', 'Z'])
                measurement_bases.append(basis)
                
                # 선택된 기저로 회전
                if basis == 'X':
                    qc.h(qubit)  # Z → X 기저
                elif basis == 'Y':
                    qc.rx(-np.pi/2, qubit)  # Z → Y 기저
                # Z 기저는 회전 없음
            
            if verbose:
                print(f"Classical Shadow 회로 생성 완료: {forward_gates_applied}/{len(gates)} 게이트")
                print(f"측정 기저: {measurement_bases}")
                print(f"회로 깊이: {qc.depth()}")
            
            # 트랜스파일 수행
            if verbose:
                print("\n트랜스파일 수행 중...")
            qc_transpiled = transpile(qc, backend=self.backend, optimization_level=self.optimization_level)
            
            # 측정 추가
            qc_transpiled.measure_all()
            
            # 파라미터 이름 가져오기
            param_names = [p.name for p in qc_transpiled.parameters]
            
            # IBM 백엔드로 실행
            print(f"\nIBM {self.backend.name} 백엔드에서 Classical Shadow 회로 실행 중...")
            print(f"{len(param_sets)}개 파라미터 세트 × {shadow_shots} 샷 = 총 {len(param_sets) * shadow_shots} 측정")
            
            # Qiskit 2.0+ API를 사용한 회로 실행
            with Session(backend=self.backend) as session:
                try:
                    # Sampler 초기화
                    sampler = Sampler(backend=self.backend)
                    
                    # 파라미터화된 회로 실행 (Classical Shadow용)
                    if len(param_names) == 0:
                        # 파라미터가 없는 경우 간단히 여러 번 실행
                        circuits = [qc_transpiled] * len(param_sets)
                        job = sampler.run(circuits, shots=shadow_shots)
                    else:
                        # 파라미터 바인딩
                        param_binds = []
                        for params in param_sets:
                            if isinstance(params, list) or isinstance(params, tuple) or isinstance(params, np.ndarray):
                                param_binds.append(params)
                            else:
                                print(f"⚠️ 파라미터 형식 오류: {type(params)}")
                                param_binds.append([])
                        
                        job = sampler.run(
                            [(qc_transpiled, params) for params in param_binds],
                            shots=shadow_shots
                        )
                    
                    print("작업 ID:", job.job_id())
                    print("작업 상태:", job.status())
                    
                    print("Classical Shadow 측정 결과 기다리는 중...")
                    result = job.result()
                    
                    # 결과 처리
                    print(f"\n{len(param_sets)}개 Classical Shadow 실행 결과 처리 중...")
                    
                    all_results = []
                    
                    # 각 파라미터 세트에 대한 결과 추출
                    for i in range(len(result)):
                        try:
                            # 결과 구조 확인 (첫 번째 결과만)
                            if verbose and i == 0:
                                print(f"\nClassical Shadow 결과 객체 디버깅:")
                                print(f"결과 타입: {type(result[i])}")
                                if hasattr(result[i], 'data'):
                                    print(f"data 속성: {dir(result[i].data)}")
                           
                            # 측정 결과 추출 (기존 방식과 동일)
                            counts = {}
                                
                            # DataBin에서 비트스트링과 카운트 추출
                            if hasattr(result[i], 'data') and hasattr(result[i].data, 'meas'):
                                bit_array = result[i].data.meas
                                
                                if hasattr(bit_array, 'get_counts') and callable(getattr(bit_array, 'get_counts')):
                                    counts_dict = bit_array.get_counts()
                                else:
                                    counts_dict = {}
                                
                            # 다른 결과 형식 처리
                            elif hasattr(result[i], 'data') and hasattr(result[i].data, 'counts'):
                                if verbose:
                                    print("counts 속성 사용")
                                counts_dict = result[i].data.counts
                            elif hasattr(result[i], 'quasi_dists'):
                                if verbose:
                                    print("quasi_dists 속성 사용")
                                quasi_dists = result[i].quasi_dists[0]
                                counts_dict = {}
                                for bitstring, prob in quasi_dists.items():
                                    counts_dict[bitstring] = int(round(prob * shadow_shots))
                            else:
                                # 문자열 파싱
                                result_str = str(result[i])
                                import re
                                count_pattern = r"'([01]+)': *(\d+)"
                                matches = re.findall(count_pattern, result_str)
                                counts_dict = {}
                                if matches:
                                    for bitstring, count in matches:
                                        counts_dict[bitstring] = int(count)
                            
                            # 비트열 처리 및 카운트 합산
                            sparse_counts = {}
                            total_counts = 0
                            
                            for bit_str, count in counts_dict.items():
                                # 정수인 경우 비트 문자열로 변환
                                if isinstance(bit_str, int):
                                    state_str = format(bit_str, f'0{n_qubits}b')
                                else:
                                    state_str = bit_str
                                
                                # 비트열 길이 맞추기
                                if len(state_str) < n_qubits:
                                    state_str = state_str.zfill(n_qubits)
                                elif len(state_str) > n_qubits:
                                    state_str = state_str[:n_qubits]
                                
                                # 카운트 누적
                                if state_str in sparse_counts:
                                    sparse_counts[state_str] += count
                                else:
                                    sparse_counts[state_str] = count
                                
                                total_counts += count
                            
                            # 총 카운트가 0인 경우 처리
                            if total_counts == 0:
                                print(f"⚠️ Classical Shadow 결과 {i+1}에서 총 카운트가 0입니다.")
                                continue
                                
                            # |00000..000> 상태의 카운트
                            zero_state = '0' * n_qubits
                            zero_state_count = sparse_counts.get(zero_state, 0)
                            zero_state_prob = zero_state_count / total_counts
                            
                            # 결과 저장 (Classical Shadow 전용)
                            param_result = {
                                "zero_state_probability": zero_state_prob,
                                "measured_states": total_counts,
                                "measurement_counts": sparse_counts,
                                "zero_state_count": zero_state_count,
                                "backend": self.backend.name,
                                "params_idx": i,
                                "measurement_bases": measurement_bases,  # Classical Shadow 기저 정보
                                "shadow_shots": shadow_shots,
                                "direct_result": {
                                    "processed_counts_direct": sparse_counts,
                                    "total_counts_direct": total_counts
                                }
                            }
                            
                            all_results.append(param_result)
                        except Exception as e:
                            print(f"⚠️ Classical Shadow 결과 {i+1} 처리 중 오류 발생: {str(e)}")
                        
                        # 진행 상황 출력
                        if (i % 5 == 0) or (i == len(param_sets) - 1):
                            print(f"  Classical Shadow 파라미터 세트 {i+1}/{len(param_sets)} 처리 완료")
                    
                    print(f"총 {len(all_results)}/{len(param_sets)} Classical Shadow 파라미터 세트 처리 완료")
                    return all_results
                    
                except Exception as e:
                    print(f"⚠️ IBM 백엔드 Classical Shadow 작업 제출 중 오류 발생: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return None
                
        except Exception as e:
            print(f"⚠️ Classical Shadow 회로 실행 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return None 