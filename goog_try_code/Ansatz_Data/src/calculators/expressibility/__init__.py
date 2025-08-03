"""
표현력(Expressibility) 계산 모듈 패키지

이 패키지는 양자 회로의 표현력을 계산하기 위한 다양한 방법과 도구를 제공합니다.
기본 계산기 클래스, 시뮬레이터 및 IBM 백엔드용 특화 계산기, 엔트로피 기반 계산 등을 포함합니다.
"""

# 리팩토링된 config 모듈에서 설정 가져오기
from src.config import config
from src.calculators.expressibility.base import ExpressibilityCalculatorBase, calculate_measurement_entropy
from src.calculators.expressibility.entropy import (
    calculate_entropy_expressibility,
    entropy_based_expressibility,
    calculate_angle_entropy,
    calculate_entropy_expressibility_from_ibm_results
)
from src.calculators.expressibility.ibm import (
    IBMExpressibilityCalculator
)

# 환경 설정에 따라 적절한 계산기 선택
# 리팩토링된 config는 디셔너리 형태가 나오므로 그에 맞게 접근
if config.get("experiment_mode", "SIMULATOR").upper() == "SIMULATOR":
    # 시뮬레이터 기반 계산기 사용
    from src.calculators.expressibility.simulator import _calculate_statevector_expressibility
    
    # 시뮬레이터 계산기 클래스 정의
    class ExpressibilityCalculator(ExpressibilityCalculatorBase):
        """
        시뮬레이터 기반 표현력 계산기

        Classical Shadow 또는 상태벡터 기반 표현력 계산을 제공합니다.
        """
        def __init__(self, seed=None):
            """표현력 계산기 초기화 (seed 설정으로 실험 재현성 보장)"""
            super().__init__(seed=seed)
            # 딕셔너리 타입 config 저장
            self.config = config
            
        def calculate_expressibility(self, circuit_info, S=None, M=None, metric='classical_shadow', sigma=1.0):
            """
            회로의 표현력 계산
            
            Args:
                circuit_info (dict): 회로 정보
                S (int): 파라미터 샘플 수 (None이면 기본값 사용)
                M (int): Shadow 크기 (None이면 기본값 사용)
                metric (str): 사용할 메트릭 (classical_shadow, statevector)
                sigma (float): 정규화 계수
                
            Returns:
                dict: 표현력 계산 결과
            """
            if metric == 'statevector':
                # 상태벡터 기반 계산
                return _calculate_statevector_expressibility(circuit_info)
            else:
                # Classical Shadow 기반 계산 (기본)
                from src.calculators.expressibility.simulator import SimulatorExpressibilityCalculator
                sim_calculator = SimulatorExpressibilityCalculator(seed=self.seed)
                return sim_calculator.calculate_expressibility(circuit_info, S=S, M=M, metric=metric, sigma=sigma)
else:
    # IBM 실제 백엔드 계산기 사용
    ExpressibilityCalculator = IBMExpressibilityCalculator
