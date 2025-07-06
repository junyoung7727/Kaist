#!/usr/bin/env python3
"""
추상 실행자 인터페이스

백엔드별 실행 로직을 완전히 분리하는 추상 인터페이스입니다.
코어 로직은 이 인터페이스만 알면 되고, 구체적인 백엔드는 전혀 몰라도 됩니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from core.circuit_interface import AbstractQuantumCircuit
from config import Config, default_config, ExperimentConfig
from core.qiskit_circuit import QiskitQuantumCircuit
from core.circuit_interface import CircuitSpec


@dataclass
class ExecutionResult:
    """실행 결과를 담는 데이터 클래스"""
    counts: Dict[str, int]          # 측정 결과 카운트
    shots: int                      # 총 샷 수
    execution_time: float           # 실행 시간 (초)
    backend_info: Dict[str, Any]    # 백엔드 정보
    circuit_id: Optional[str] = None  # 회로 식별자 
    success: bool = True            # 실행 성공 여부
    error_message: Optional[str] = None  # 오류 메시지



class AbstractQuantumExecutor(ABC):
    """
    양자 회로 실행을 위한 추상 인터페이스
    
    시뮬레이터든 IBM 하드웨어든 이 인터페이스를 통해 동일하게 실행됩니다.
    """
    
    def __init__(self):
        self._config = default_config
        self._initialized = False
    
    def __enter__(self):
        """컨텍스트 매니저 진입 - 초기화 수행"""
        if not self._initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료 - 자원 정리"""
        # 필요한 자원 정리 로직 추가
        pass
    
    @property
    def config(self) -> Config:
        """실행 설정 반환"""
        return self._config
        
    @abstractmethod
    def run(self, circuits: List[CircuitSpec],experiment_config : ExperimentConfig):
        """간단한 실험 실행 메서드
        
        Args:
            experiment_config: 실험 설정 (Config의 실험 설정)
            
        Returns:
            실행 결과
        """
        pass
            
    @abstractmethod
    async def initialize(self) -> bool:
        """백엔드 초기화"""
        pass
    
    @abstractmethod
    async def execute_circuit(self, circuit: AbstractQuantumCircuit) -> ExecutionResult:
        """단일 회로 실행"""
        pass
    
    @abstractmethod
    async def execute_circuits(self, circuits: List[AbstractQuantumCircuit]) -> List[ExecutionResult]:
        """다중 회로 배치 실행"""
        pass
    
    @abstractmethod
    def get_backend_info(self) -> Dict[str, Any]:
        """백엔드 정보 반환"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """리소스 정리"""
        pass
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self.cleanup()


class QuantumExecutorFactory:
    """
    실행자 팩토리 - 백엔드 선택은 오직 여기서만 발생
    
    이것이 유일하게 백엔드를 구분하는 곳입니다.
    
    사용 예:
        simul = QuantumExecutorFactory("simulator")
        result = await simul.run(config.exp1)
        
        ibm = QuantumExecutorFactory("ibm")
        result = await ibm.run(config.exp2)
    """
    
    _executors = {}
    
    def __init__(self, backend_type: str, config: Config = None):
        """
        실행자 인스턴스 생성
        
        Args:
            backend_type: 'simulator' 또는 'ibm'
            config: 실행 설정 (기본값: default_config)
        """
        if backend_type not in self._executors:
            raise ValueError(f"Unknown backend type: {backend_type}. "
                           f"Available: {list(self._executors.keys())}")
        
        self.backend_type = backend_type
        executor_class = self._executors[backend_type]
        self.executor = executor_class()
        
        if config:
            self.executor._config = config
    
    @classmethod
    def register_executor(cls, name: str, executor_class):
        """실행자 등록"""
        cls._executors[name] = executor_class
    
    @classmethod
    def list_available_backends(cls) -> List[str]:
        """사용 가능한 백엔드 목록 반환"""
        return list(cls._executors.keys())
    
    @classmethod
    def create_executor(cls, backend_type: str):
        if backend_type not in cls._executors:
            raise ValueError(f"Unknown backend type: {backend_type}. "
                           f"Available: {list(cls._executors.keys())}")
        
        executor_class = cls._executors[backend_type]
        executor = executor_class()
        
        return executor



# 데코레이터를 통한 실행자 등록
def register_executor(name: str):
    """실행자 등록 데코레이터"""
    def decorator(cls):
        QuantumExecutorFactory.register_executor(name, cls)
        return cls
    return decorator
