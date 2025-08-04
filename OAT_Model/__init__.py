# aa/__init__.py
import sys
import os
from pathlib import Path

# Ansatz_Data_ver2 경로 추가
def _setup_ansatz_path():
    current_dir = Path(__file__).parent
    ansatz_path = current_dir.parent / "Ansatz_Data_ver2"
    ansatz_str = str(ansatz_path)
    if ansatz_str not in sys.path:
        sys.path.insert(0, ansatz_str)

_setup_ansatz_path()

# 필요한 모듈들을 aa 네임스페이스로 가져오기
try:
    from core.gates import QuantumGateRegistry
    from core.circuit_interface import CircuitInterface  # 예시
    from core.entanglement_mea import EntanglementMeasurement  # 예시
    
    # aa 패키지에서 직접 사용할 수 있도록 export
    __all__ = [
        'QuantumGateRegistry',
        'CircuitInterface', 
        'EntanglementMeasurement'
    ]
    
except ImportError as e:
    print(f"Warning: Could not import from Ansatz_Data_ver2: {e}")