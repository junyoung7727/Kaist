#!/usr/bin/env python3
"""
양자 회로 데이터 생성 중앙 설정 파일
모든 실험 파라미터를 한 곳에서 관리
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ClassicalShadowConfig:
    """Classical Shadow 방법론 설정"""
    # 기본 파라미터
    default_samples: int = 50          # 기본 파라미터 샘플 수
    default_shadow_size: int = 256     # 기본 Shadow 크기 (각 파라미터당)
    
    # 큐빗 수별 최적화된 설정
    small_system_samples: int = 50    # ≤10 큐빗: 더 많은 샘플
    small_system_shadow_size: int = 256
    
    medium_system_samples: int = 50    # 11-20 큐빗: 중간 샘플
    medium_system_shadow_size: int = 256
    
    large_system_samples: int = 50     # ≥21 큐빗: 적은 샘플 (메모리 제한)
    large_system_shadow_size: int = 256
    
    # 메모리 제한
    max_shadow_qubits: int = 25        # Shadow 수집 시 최대 큐빗 수
    max_2local_qubits: int = 15        # 2-local 연산자 계산 시 최대 큐빗 수

@dataclass
class IBMBackendConfig:
    """IBM 백엔드 실행 설정"""
    # 기본 샷 수
    default_shots: int = 128           # 기본 측정 샷 수
    
    # 큐빗 수별 샷 수 최적화 (피델리티 측정용)
    small_circuit_shots: int = 256    # ≤10 큐빗: 충분한 샷 (피델리티 신뢰성)
    medium_circuit_shots: int = 256  # 11-20 큐빗: 중간 샷
    large_circuit_shots: int = 256     # ≥21 큐빗: 적은 샷
    
    # 표현력 측정용 파라미터화된 실행 (총 샷 수 조절)
    expressibility_samples: int = 15   # 표현력 측정용 파라미터 세트 수 (32→10)
    expressibility_shots: int = 64    # 표현력 측정용 샷 수 (128→64)
    
    # 최적화 설정
    optimization_level: int = 0        # 트랜스파일 최적화 레벨
    max_backend_qubits: int = 127      # 백엔드 최대 큐빗 수

@dataclass
class SimulatorConfig:
    """시뮬레이터 설정"""
    # 메모리 제한
    max_simulation_qubits: int = 20    # 시뮬레이터 최대 큐빗 수
    max_fidelity_qubits: int = 30      # 피델리티 계산 최대 큐빗 수
    
    # 샘플링 설정
    default_shots: int = 256          # 기본 샘플링 샷 수
    fidelity_shots: int = 256         # 피델리티 측정 샷 수

@dataclass
class DataGenerationConfig:
    """대량 데이터 생성 설정"""
    # 메가잡 설정
    batch_size: int = 100              # 배치당 회로 수
    max_batches: int = 10              # 최대 배치 수
    
    # 회로 파라미터 범위
    min_qubits: int = 1               # 최소 큐빗 수
    max_qubits: int = 127              # 최대 큐빗 수
    qubit_step: int = 10               # 큐빗 수 증가 단위
    
    min_depth: int = 1                 # 최소 회로 깊이
    max_depth: int = 10                 # 최대 회로 깊이
    
    # 병렬 처리
    max_workers: int = 4               # 최대 워커 수
    timeout_seconds: int = 300         # 작업 타임아웃 (초)
    
    # 저장 설정
    compress_data: bool = True         # 데이터 압축 여부
    save_intermediate: bool = True     # 중간 결과 저장 여부

@dataclass
class ExperimentConfig:
    """실험별 특화 설정"""
    # 표현력 연구용
    expressibility_study: Dict[str, Any] = None
    
    # 스케일링 연구용  
    scaling_study: Dict[str, Any] = None
    
    # 노이즈 연구용
    noise_study: Dict[str, Any] = None

class ConfigManager:
    """설정 관리자 클래스"""
    
    def __init__(self):
        self.classical_shadow = ClassicalShadowConfig()
        self.ibm_backend = IBMBackendConfig()
        self.simulator = SimulatorConfig()
        self.data_generation = DataGenerationConfig()
        self.experiment = ExperimentConfig()
        
        # 환경 변수에서 설정 오버라이드
        self._load_from_environment()
    
    def _load_from_environment(self):
        """환경 변수에서 설정 로드"""
        # Classical Shadow 설정
        if "SHADOW_SAMPLES" in os.environ:
            self.classical_shadow.default_samples = int(os.environ["SHADOW_SAMPLES"])
        if "SHADOW_SIZE" in os.environ:
            self.classical_shadow.default_shadow_size = int(os.environ["SHADOW_SIZE"])
        
        # IBM 백엔드 설정
        if "IBM_SHOTS" in os.environ:
            self.ibm_backend.default_shots = int(os.environ["IBM_SHOTS"])
        if "IBM_EXPRESSIBILITY_SAMPLES" in os.environ:
            self.ibm_backend.expressibility_samples = int(os.environ["IBM_EXPRESSIBILITY_SAMPLES"])
        
        # 시뮬레이터 설정
        if "MAX_SIMULATION_QUBITS" in os.environ:
            self.simulator.max_simulation_qubits = int(os.environ["MAX_SIMULATION_QUBITS"])
        
        # 데이터 생성 설정
        if "BATCH_SIZE" in os.environ:
            self.data_generation.batch_size = int(os.environ["BATCH_SIZE"])
        if "MAX_BATCHES" in os.environ:
            self.data_generation.max_batches = int(os.environ["MAX_BATCHES"])
    
    def get_classical_shadow_params(self, n_qubits: int) -> tuple:
        """큐빗 수에 따른 최적화된 Classical Shadow 파라미터 반환"""
        if n_qubits <= 10:
            return (self.classical_shadow.small_system_samples,
                    self.classical_shadow.small_system_shadow_size)
        elif n_qubits <= 20:
            return (self.classical_shadow.medium_system_samples,
                   self.classical_shadow.medium_system_shadow_size)
        else:
            return (self.classical_shadow.large_system_samples,
                   self.classical_shadow.large_system_shadow_size)
    
    def get_ibm_shots(self, n_qubits: int) -> int:
        """큐빗 수에 따른 최적화된 IBM 샷 수 반환"""
        if n_qubits <= 10:
            return self.ibm_backend.small_circuit_shots
        elif n_qubits <= 20:
            return self.ibm_backend.medium_circuit_shots
        else:
            return self.ibm_backend.large_circuit_shots
    
    def get_simulator_shots(self, n_qubits: int) -> int:
        """큐빗 수에 따른 시뮬레이터 샷 수 반환"""
        return self.simulator.default_shots
    
    def get_batch_config(self) -> Dict[str, Any]:
        """배치 처리 설정 반환"""
        return {
            "batch_size": self.data_generation.batch_size,
            "max_batches": self.data_generation.max_batches,
            "max_workers": self.data_generation.max_workers,
            "timeout_seconds": self.data_generation.timeout_seconds,
            "compress_data": self.data_generation.compress_data,
            "save_intermediate": self.data_generation.save_intermediate
        }
    
    def get_circuit_ranges(self) -> Dict[str, Any]:
        """회로 파라미터 범위 반환"""
        return {
            "qubits": {
                "min": self.data_generation.min_qubits,
                "max": self.data_generation.max_qubits,
                "step": self.data_generation.qubit_step
            },
            "depth": {
                "min": self.data_generation.min_depth,
                "max": self.data_generation.max_depth
            }
        }
    
    def print_current_config(self):
        """현재 설정 출력"""
        print("\n" + "="*60)
        print("🔧 현재 양자 회로 실험 설정")
        print("="*60)
        
        print("\n📊 Classical Shadow 설정:")
        print(f"  기본 샘플 수: {self.classical_shadow.default_samples}")
        print(f"  기본 Shadow 크기: {self.classical_shadow.default_shadow_size}")
        print(f"  최대 Shadow 큐빗: {self.classical_shadow.max_shadow_qubits}")
        
        print("\n🖥️  IBM 백엔드 설정:")
        print(f"  기본 샷 수: {self.ibm_backend.default_shots}")
        print(f"  표현력 측정 샘플: {self.ibm_backend.expressibility_samples}")
        print(f"  표현력 측정 샷: {self.ibm_backend.expressibility_shots}")
        
        print("\n💻 시뮬레이터 설정:")
        print(f"  최대 시뮬레이션 큐빗: {self.simulator.max_simulation_qubits}")
        print(f"  기본 샷 수: {self.simulator.default_shots}")
        
        print("\n📦 데이터 생성 설정:")
        print(f"  배치 크기: {self.data_generation.batch_size}")
        print(f"  최대 배치 수: {self.data_generation.max_batches}")
        print(f"  큐빗 범위: {self.data_generation.min_qubits}-{self.data_generation.max_qubits}")
        print(f"  깊이 범위: {self.data_generation.min_depth}-{self.data_generation.max_depth}")
        print(f"  병렬 워커: {self.data_generation.max_workers}")
        
        print("="*60)
    
    def save_config_to_file(self, filename: str = "experiment_config.json"):
        """설정을 JSON 파일로 저장"""
        import json
        from dataclasses import asdict
        
        config_dict = {
            "classical_shadow": asdict(self.classical_shadow),
            "ibm_backend": asdict(self.ibm_backend),
            "simulator": asdict(self.simulator),
            "data_generation": asdict(self.data_generation),
            "experiment": asdict(self.experiment)
        }
        
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"설정이 {filename}에 저장되었습니다.")
    
    def load_config_from_file(self, filename: str):
        """JSON 파일에서 설정 로드"""
        import json
        
        if not os.path.exists(filename):
            print(f"⚠️ 설정 파일 {filename}을 찾을 수 없습니다.")
            return
        
        with open(filename, 'r') as f:
            config_dict = json.load(f)
        
        # 설정 업데이트
        for key, value in config_dict.get("classical_shadow", {}).items():
            setattr(self.classical_shadow, key, value)
        
        for key, value in config_dict.get("ibm_backend", {}).items():
            setattr(self.ibm_backend, key, value)
        
        for key, value in config_dict.get("simulator", {}).items():
            setattr(self.simulator, key, value)
        
        for key, value in config_dict.get("data_generation", {}).items():
            setattr(self.data_generation, key, value)
        
        print(f"설정이 {filename}에서 로드되었습니다.")

# 전역 설정 인스턴스
config = ConfigManager()

# 편의 함수들
def get_shadow_params(n_qubits: int) -> tuple:
    """큐빗 수에 따른 Classical Shadow 파라미터 반환"""
    return config.get_classical_shadow_params(n_qubits)

def get_ibm_shots(n_qubits: int) -> int:
    """큐빗 수에 따른 IBM 샷 수 반환"""
    return config.get_ibm_shots(n_qubits)

def get_simulator_shots(n_qubits: int) -> int:
    """큐빗 수에 따른 시뮬레이터 샷 수 반환"""
    return config.get_simulator_shots(n_qubits)

def print_config():
    """현재 설정 출력"""
    config.print_current_config()

# 실험별 프리셋 설정
EXPRESSIBILITY_PRESET = {
    "classical_shadow": {
        "default_samples": 100,
        "default_shadow_size": 200
    },
    "ibm_backend": {
        "expressibility_samples": 50,
        "expressibility_shots": 128
    }
}

SCALING_PRESET = {
    "data_generation": {
        "batch_size": 50,
        "max_batches": 20,
        "min_qubits": 30,
        "max_qubits": 127,
        "qubit_step": 5
    }
}

NOISE_STUDY_PRESET = {
    "ibm_backend": {
        "default_shots": 512,
        "expressibility_samples": 20
    }
}

def apply_preset(preset_name: str):
    """프리셋 설정 적용"""
    presets = {
        "expressibility": EXPRESSIBILITY_PRESET,
        "scaling": SCALING_PRESET,
        "noise": NOISE_STUDY_PRESET
    }
    
    if preset_name not in presets:
        print(f"⚠️ 알 수 없는 프리셋: {preset_name}")
        print(f"사용 가능한 프리셋: {list(presets.keys())}")
        return
    
    preset = presets[preset_name]
    
    # 설정 적용
    for section, settings in preset.items():
        section_obj = getattr(config, section)
        for key, value in settings.items():
            setattr(section_obj, key, value)
    
    print(f"✅ '{preset_name}' 프리셋이 적용되었습니다.")
    config.print_current_config()

if __name__ == "__main__":
    # 설정 테스트
    print_config()
    
    # 큐빗 수별 파라미터 테스트
    print("\n🧪 큐빗 수별 최적화된 파라미터:")
    for n_qubits in [5, 7, 10, 15, 20, 30, 50, 80, 127]:
        samples, shadow_size = get_shadow_params(n_qubits)
        ibm_shots = get_ibm_shots(n_qubits)
        print(f"  {n_qubits:3d} 큐빗: Shadow({samples:2d}, {shadow_size:3d}), IBM({ibm_shots:3d} shots)")