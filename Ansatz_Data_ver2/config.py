#!/usr/bin/env python3
"""
통합 설정 관리

모든 설정을 중앙에서 관리하는 통합 설정 시스템입니다.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()



@dataclass
class ExperimentConfig:
    """실험 설정 전용 클래스"""
    num_qubits: Any  # List[int] 또는 int
    depth: int
    shots: int
    num_circuits: int
    two_qubit_ratio: list[float]
    exp_name: Optional[str] = "Default Name"
    optimization_level: int = 1
    fidelity_shots: int = 256
    executor: Optional[Any] = None
    entangle_shots: int = 256
    num_samples: int = 10
    
    def __post_init__(self):
        if isinstance(self.num_qubits, int):
            self.num_qubits = [self.num_qubits]

@dataclass
class Exp_Box:
    # IBM Quantum 1회 제출량 1천만 샷 제한 내 최적화된 실험 설계
    scalability_test = ExperimentConfig(
        num_qubits=[3, 5, 7, 10, 20, 30, 40, 50],  # 12개 큐빗 수준
        depth=[1, 2, 3, 5],  # 2개 깊이 (너무 많으면 샷 수 폭증)
        shots=512,     # 표현력 측정용 (감소)
        num_circuits=1,  # 회로 수 최소화
        optimization_level=3,
        two_qubit_ratio=[0.3, 0.5, 0.8],  # 1개 비율만 (다양성 줄임)
        exp_name="scalability_test",
        fidelity_shots=512,    # 피델리티 측정용
        executor=None,
        entangle_shots=512,    #얽힘도 측정용
        num_samples=7          # 표현력 샘플 수 최소화 (2개면 1개 페어)
    )
    
    # 샷 수 계산 검증용 설정
    exp1 = ExperimentConfig(
        num_qubits=[3],
        depth=[5],
        shots=1024,
        num_circuits=1,
        optimization_level=1,
        two_qubit_ratio=[0.1,0.3],
        exp_name="exp1",
        fidelity_shots=1024,
        executor = None,
        entangle_shots=1024,
        num_samples = 3
    )

    exp2 = ExperimentConfig(
        num_qubits=[10],
        depth=[6],
        shots=2048,
        num_circuits=3,
        optimization_level=1,
        two_qubit_ratio=[0.3],
        exp_name="exp2",
        fidelity_shots=1024,
        executor = None,
        entangle_shots=1024,
        num_samples = 10
    )

    statistical_validation_config = ExperimentConfig(
        num_qubits=[3,5,7,10],#7,10,13,15
        depth=[2,4,6,8],
        shots=2048,
        num_circuits=3,
        optimization_level=2,
        two_qubit_ratio=[0.2,0.5,0.8],
        exp_name="statistical_validation_config",
        fidelity_shots=1024,
        executor = None,
        entangle_shots=1024,
        num_samples = 10
    )

    statistical_validation_config1 = ExperimentConfig(
        num_qubits=[3,5],#7,10,13,15
        depth=[6],
        shots=2048,
        num_circuits=1,
        optimization_level=2,
        two_qubit_ratio=[0.3],
        exp_name="statistical_validation_config123",
        fidelity_shots=1024,
        executor = None,
        entangle_shots=1024,
        num_samples = 10
    )

    def get_setting(self, exp_name='exp1'):
        """실험 설정을 딕셔너리 형태로 반환
        
        Args:
            exp_name: 실험 설정 이름 (기본값: 'exp1')
            
        Returns:
            실험 설정 딕셔너리
        """
        return getattr(self, exp_name)
    

@dataclass
class Config:
    """애플리케이션 설정"""
    
    # 실행 설정
    seed: Optional[int] = None
    
    # 피델리티/표현력 계산 설정
    fidelity_shots: int = 256
    
    # 출력 설정
    output_dir: str = './output'
    save_circuits: bool = True
    save_results: bool = True
    
    # IBM 설정 (IBM 백엔드 사용 시)
    ibm_token: Optional[str] = None
    ibm_backend_name: Optional[str] = None


    def __post_init__(self):
        """설정 후처리"""
        # 환경변수에서 IBM 토큰 로드
        if not self.ibm_token:
            self.ibm_token = os.getenv('IBM_TOKEN')
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """딕셔너리에서 설정 생성"""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            'backend_type': self.backend_type,
            'shots': self.shots,
            'fidelity_shots': self.fidelity_shots,
            'optimization_level': self.optimization_level,
            'seed': self.seed,
            'num_qubits': self.num_qubits,
            'circuit_depth': self.circuit_depth,
            'num_circuits': self.num_circuits,
            'min_fidelity_samples': self.min_fidelity_samples,
            'output_dir': self.output_dir,
            'save_circuits': self.save_circuits,
            'save_results': self.save_results,
            'ibm_token': self.ibm_token,
            'ibm_backend_name': os.getenv('IBM_QUANTUM_TOKEN')
        }


# 기본 설정 인스턴스
default_config = Config()
