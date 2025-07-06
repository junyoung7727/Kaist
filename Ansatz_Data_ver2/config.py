#!/usr/bin/env python3
"""
통합 설정 관리

모든 설정을 중앙에서 관리하는 통합 설정 시스템입니다.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import os


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
    
    def __post_init__(self):
        if isinstance(self.num_qubits, int):
            self.num_qubits = [self.num_qubits]

@dataclass
class Exp_Box:
    exp1 = ExperimentConfig(
        num_qubits=[4,5,6,7,8],
        depth=[1,2,4],
        shots=1024,
        num_circuits=5,
        optimization_level=1,
        two_qubit_ratio=[0.1, 0.5],
        exp_name="exp1"
    )

    exp2 = ExperimentConfig(
        num_qubits=[10],
        depth=[6],
        shots=2048,
        num_circuits=3,
        optimization_level=1,
        two_qubit_ratio=[0.3],
        exp_name="exp2"
    )
    

@dataclass
class Config:
    """애플리케이션 설정"""
    
    # 실행 설정
    backend_type: str = 'simulator' # 'simulator' 또는 'ibm'
    seed: Optional[int] = None
    
    # 피델리티/표현력 계산 설정
    min_fidelity_samples: int = 100
    
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
            self.ibm_token = os.getenv('IBM_QUANTUM_TOKEN')
        
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
            'ibm_backend_name': self.ibm_backend_name
        }


# 기본 설정 인스턴스
default_config = Config()
