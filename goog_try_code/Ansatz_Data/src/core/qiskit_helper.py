#!/usr/bin/env python3
"""
Qiskit 회로 도우미 모듈 - Qiskit 회로 변환 및 트랜스파일 로직을 처리합니다.
"""

import sys
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from tqdm import tqdm

# 현재 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qiskit import QuantumCircuit, transpile
from src.config import config

def convert_to_qiskit_circuits(all_circuits: List[Dict[str, Any]], ibm_backend):
    """
    모든 회로를 Qiskit 회로로 변환
    
    Args:
        all_circuits (List[Dict[str, Any]]): 변환할 회로 목록
        ibm_backend: IBM 백엔드 객체
        
    Returns:
        Tuple[List[QuantumCircuit], List[Dict[str, Any]]]: 변환된 Qiskit 회로 목록과 메타데이터 목록
    """
    # 내부에 이미 정의된 build_qiskit_circuit_from_data 함수를 사용
    print("🔄 Qiskit 회로로 변환 및 트랜스파일 중...")
    
    qiskit_circuits = []
    circuit_metadata = []
    
    # 모듈 내부에서 필요한 함수 임포트 (순환 참조 방지)
    from src.core.quantum_properties import calculate_quantum_properties
    
    for i, circuit_data in enumerate(tqdm(all_circuits, desc="회로 변환")):
        try:
            circuit_info = circuit_data
            # 백엔드 큐빗 제약 반영
            max_q = ibm_backend.backend.configuration().n_qubits
            # 빌드 및 트랜스파일
            qc = build_qiskit_circuit_from_data(circuit_info, ibm_backend.backend)
            optimization_level = config.get('transpilation_options', {}).get('optimization_level', 1)
            qc_transpiled = transpile(qc, backend=ibm_backend.backend, optimization_level=optimization_level)
            # 특성 계산
            circuit_properties = calculate_quantum_properties(circuit_info, qc_transpiled)
            # 메타데이터 기록
            enhanced_metadata = circuit_data.copy()
            enhanced_metadata['circuit_properties'] = circuit_properties
            enhanced_metadata['qiskit_circuit'] = qc_transpiled  # Qiskit 회로 객체 추가
            qiskit_circuits.append(qc_transpiled)
            circuit_metadata.append(enhanced_metadata)
        except Exception as e:
            print(f"⚠️ 회로 {i} 변환 실패: {str(e)}")
    
    print(f"✅ {len(qiskit_circuits)}개 회로 변환 완료")
    return qiskit_circuits, circuit_metadata


def build_qiskit_circuit_from_data(circuit_info: Dict[str, Any], backend=None) -> QuantumCircuit:
    """
    회로 데이터에서 Qiskit 양자회로 생성
    
    Args:
        circuit_info (Dict[str, Any]): 회로 정보
        backend: 백엔드 객체 (선택사항)
        
    Returns:
        QuantumCircuit: 생성된 Qiskit 회로
    """
    n_qubits = circuit_info.get("n_qubits", 0)
    gates = circuit_info.get("gates", [])
    wires_list = circuit_info.get("wires_list", [])
    params_idx = circuit_info.get("params_idx", [])
    params = circuit_info.get("params", [])
    
    # 백엔드 큐빗 제한 적용
    if backend:
        max_backend_qubits = backend.configuration().n_qubits
        if n_qubits > max_backend_qubits:
            n_qubits = max_backend_qubits
    
    # Qiskit 양자 회로 생성 (U + U†)
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # 순방향 회로 (U) 적용
    for j, (gate, wires) in enumerate(zip(gates, wires_list)):
        if any(w >= n_qubits for w in wires):
            continue
            
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
            # 파라미터 찾기
            param_value = None
            for k, idx in enumerate(params_idx):
                if idx == j:
                    param_value = params[k]
                    break
            if param_value is not None:
                qc.rz(param_value, wires[0])
        elif gate == "RX":
            param_value = None
            for k, idx in enumerate(params_idx):
                if idx == j:
                    param_value = params[k]
                    break
            if param_value is not None:
                qc.rx(param_value, wires[0])
        elif gate == "RY":
            param_value = None
            for k, idx in enumerate(params_idx):
                if idx == j:
                    param_value = params[k]
                    break
            if param_value is not None:
                qc.ry(param_value, wires[0])
        elif gate == "CZ":
            if len(wires) >= 2:
                qc.cz(wires[0], wires[1])
        elif gate == "CNOT":
            if len(wires) >= 2:
                qc.cx(wires[0], wires[1])
    
    # 역방향 회로 (U†) 적용
    for j in range(len(gates)-1, -1, -1):
        gate = gates[j]
        wires = wires_list[j]
        
        if any(w >= n_qubits for w in wires):
            continue
        
        if gate == "H":
            qc.h(wires[0])
        elif gate == "X":
            qc.x(wires[0])
        elif gate == "Y":
            qc.y(wires[0])
        elif gate == "Z":
            qc.z(wires[0])
        elif gate == "S":
            qc.sdg(wires[0])
        elif gate == "T":
            qc.tdg(wires[0])
        elif gate == "RZ":
            # 파라미터 찾기
            param_value = None
            for k, idx in enumerate(params_idx):
                if idx == j:
                    param_value = params[k]
                    break
            if param_value is not None:
                qc.rz(-param_value, wires[0])
        elif gate == "RX":
            param_value = None
            for k, idx in enumerate(params_idx):
                if idx == j:
                    param_value = params[k]
                    break
            if param_value is not None:
                qc.rx(-param_value, wires[0])
        elif gate == "RY":
            param_value = None
            for k, idx in enumerate(params_idx):
                if idx == j:
                    param_value = params[k]
                    break
            if param_value is not None:
                qc.ry(-param_value, wires[0])
        elif gate == "CZ":
            if len(wires) >= 2:
                qc.cz(wires[0], wires[1])
        elif gate == "CNOT":
            if len(wires) >= 2:
                qc.cx(wires[0], wires[1])
    
    # 측정 추가
    qc.measure_all()
    
    return qc
