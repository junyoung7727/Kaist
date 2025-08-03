"""
양자 회로 메트릭 계산 모듈

이 패키지는 양자 회로의 다양한 메트릭을 계산하는 도구들을 제공합니다.
회로 복잡도, 깊이, 게이트 수 등의 기본 메트릭과 함께 더 복잡한 분석도 포함합니다.
"""

# 향후 구현될 기능들을 미리 import 형태로 선언
from src.calculators.metrics.circuit_metrics import (
    calculate_circuit_metrics,
    calculate_gate_counts,
    calculate_circuit_depth,
    calculate_circuit_width,
    calculate_two_qubit_gate_ratio
)
