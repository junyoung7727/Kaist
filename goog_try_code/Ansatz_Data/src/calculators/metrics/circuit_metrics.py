#!/usr/bin/env python3
"""
양자 회로 메트릭 계산 모듈

이 모듈은 양자 회로의 다양한 메트릭을 계산하는 함수들을 제공합니다.
게이트 수, 회로 깊이, 회로 너비, 2큐빗 게이트 비율 등을 계산할 수 있습니다.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import qiskit


def calculate_circuit_metrics(circuit, include_barriers=False, transpiled=False) -> Dict[str, Any]:
    """
    양자 회로의 다양한 메트릭 계산
    
    Args:
        circuit: 분석할 양자 회로 (Qiskit 회로 객체)
        include_barriers (bool): 장벽(barrier)을 포함할지 여부
        transpiled (bool): 이미 트랜스파일된 회로인지 여부
        
    Returns:
        Dict[str, Any]: 회로 메트릭 정보
    """
    result = {}
    
    # 게이트 수 계산
    gate_counts = calculate_gate_counts(circuit, include_barriers)
    result["gate_counts"] = gate_counts
    result["total_gates"] = sum(gate_counts.values())
    
    # 단일 및 다중 큐빗 게이트 수 계산
    single_qubit_gates = 0
    two_qubit_gates = 0
    multi_qubit_gates = 0  # 3개 이상 큐빗에 작용하는 게이트
    
    for gate, count in gate_counts.items():
        if gate in ["barrier", "measure"] and not include_barriers:
            continue
            
        # 큐빗 수에 따른 구분
        if gate in ["x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "p", "u1", "u2", "u3"]:
            single_qubit_gates += count
        elif gate in ["cx", "cz", "swap", "rxx", "ryy", "rzz", "cp", "crx", "cry", "crz"]:
            two_qubit_gates += count
        else:
            multi_qubit_gates += count
    
    result["single_qubit_gates"] = single_qubit_gates
    result["two_qubit_gates"] = two_qubit_gates
    result["multi_qubit_gates"] = multi_qubit_gates
    
    # 2큐빗 게이트 비율 계산
    result["two_qubit_gate_ratio"] = calculate_two_qubit_gate_ratio(circuit, include_barriers)
    
    # 회로 깊이 및 너비 계산
    result["depth"] = calculate_circuit_depth(circuit, include_barriers)
    result["width"] = calculate_circuit_width(circuit)
    
    # 파라미터 수 계산
    result["parameter_count"] = len(circuit.parameters) if hasattr(circuit, "parameters") else 0
    
    # 트랜스파일 정보 추가
    result["transpiled"] = transpiled
    
    return result


def calculate_gate_counts(circuit, include_barriers=False) -> Dict[str, int]:
    """
    회로의 게이트 종류별 카운트 계산
    
    Args:
        circuit: 분석할 양자 회로
        include_barriers (bool): 장벽(barrier)을 포함할지 여부
        
    Returns:
        Dict[str, int]: 게이트 종류별 카운트 딕셔너리
    """
    # 회로에서 게이트 카운트 추출
    try:
        counts = circuit.count_ops()
        counts = {k.lower(): v for k, v in counts.items()}  # 소문자로 통일
    except:
        # QOp 형태로 바로 변환이 안 되는 경우 수동으로 계산
        counts = {}
        try:
            for instr, qargs, _ in circuit.data:
                gate_name = instr.name.lower()
                counts[gate_name] = counts.get(gate_name, 0) + 1
        except:
            # 위 방법도 안 되면 빈 딕셔너리 반환
            pass
    
    # barrier 제외 옵션
    if not include_barriers and "barrier" in counts:
        del counts["barrier"]
        
    return counts


def calculate_circuit_depth(circuit, include_barriers=False) -> int:
    """
    회로의 깊이 계산
    
    Args:
        circuit: 분석할 양자 회로
        include_barriers (bool): 장벽(barrier)을 포함할지 여부
        
    Returns:
        int: 회로 깊이
    """
    try:
        # 장벽을 포함하지 않는 경우
        if not include_barriers:
            # 장벽 없는 복사본 생성
            from qiskit import QuantumCircuit
            temp_circuit = QuantumCircuit(circuit.num_qubits)
            for instr, qargs, cargs in circuit.data:
                if instr.name.lower() != "barrier":
                    temp_circuit.append(instr, qargs, cargs)
            return temp_circuit.depth()
        
        # 장벽 포함
        return circuit.depth()
    except:
        # 깊이 계산에 실패한 경우
        return -1


def calculate_circuit_width(circuit) -> int:
    """
    회로의 너비(사용된 큐빗 수) 계산
    
    Args:
        circuit: 분석할 양자 회로
        
    Returns:
        int: 회로 너비 (큐빗 수)
    """
    try:
        return circuit.num_qubits
    except:
        # 너비 계산에 실패한 경우
        return -1


def calculate_two_qubit_gate_ratio(circuit, include_barriers=False) -> float:
    """
    2큐빗 게이트 비율 계산
    
    Args:
        circuit: 분석할 양자 회로
        include_barriers (bool): 장벽(barrier)을 포함할지 여부
        
    Returns:
        float: 2큐빗 게이트 비율 (0 ~ 1 사이의 값)
    """
    gate_counts = calculate_gate_counts(circuit, include_barriers)
    
    # 2큐빗 게이트 목록 (일반적인 Qiskit 게이트 기준)
    two_qubit_gates = ["cx", "cz", "swap", "rxx", "ryy", "rzz", "cp", "crx", "cry", "crz", "ch"]
    
    # 2큐빗 게이트 수 계산
    two_qubit_count = sum(gate_counts.get(gate, 0) for gate in two_qubit_gates)
    
    # 전체 게이트 수 (barrier와 measure는 제외)
    total_count = sum(gate_counts.values())
    
    # 비율 계산 (게이트가 없는 경우 0 반환)
    if total_count == 0:
        return 0.0
        
    return two_qubit_count / total_count


def calculate_circuit_expressibility(circuit, n_qubits, method="entropy") -> Dict[str, Any]:
    """
    회로의 표현력(expressibility) 계산 래퍼 함수
    
    Args:
        circuit: 분석할 양자 회로
        n_qubits: 큐빗 수
        method (str): 계산 방법 ("entropy", "fidelity", "shadow")
        
    Returns:
        Dict[str, Any]: 표현력 계산 결과
    
    Notes:
        샷 수는 config.simulator.entropy_shots에서 가져옵니다.
    """
    from src.calculators.expressibility import ExpressibilityCalculator
    
    from src.config import config
    
    # 회로 정보 구성
    circuit_info = {
        "circuit": circuit,
        "n_qubits": n_qubits
    }
    
    # 표현력 계산기 인스턴스 생성
    calculator = ExpressibilityCalculator(seed=42)
    
    # 선택된 방법에 따라 표현력 계산
    if method == "fidelity":
        result = calculator.calculate_expressibility(circuit_info, metric="statevector")
    elif method == "shadow":
        result = calculator.calculate_expressibility(circuit_info, metric="classical_shadow")
    else:  # 기본값: entropy
        # 엔트로피 기반 계산 (측정 결과 분포의 엔트로피)
        from src.calculators.expressibility.entropy import calculate_entropy_expressibility_from_ibm_results
        
        # 회로 실행 및 측정 결과 얻기
        try:
            from qiskit import Aer, execute
            simulator = Aer.get_backend('qasm_simulator')
            # 하드코딩된 샷 수 대신 config에서 가져오기
            entropy_shots = getattr(config.simulator, "entropy_shots", 1024)  # 기본값: 1024
            job = execute(circuit, simulator, shots=entropy_shots)
            result = job.result()
            counts = result.get_counts(circuit)
            
            # 엔트로피 기반 표현력 계산
            return calculate_entropy_expressibility_from_ibm_results(counts, n_qubits)
        except Exception as e:
            return {
                "expressibility_value": float('nan'),
                "error": str(e),
                "method": "entropy"
            }
    
    return result


def calculate_circuit_complexity(circuit) -> Dict[str, Any]:
    """
    회로의 복잡도 점수 계산
    
    Args:
        circuit: 분석할 양자 회로
        
    Returns:
        Dict[str, Any]: 복잡도 점수 및 관련 메트릭
    """
    # 기본 메트릭 계산
    metrics = calculate_circuit_metrics(circuit)
    
    # 복잡도 점수 계산 요소
    depth_factor = metrics["depth"] / max(1, metrics["width"])
    two_qubit_factor = metrics["two_qubit_gates"] * 2
    multi_qubit_factor = metrics["multi_qubit_gates"] * 3
    param_factor = metrics["parameter_count"] * 0.5
    
    # 복잡도 점수 (가중 합산)
    complexity_score = (
        depth_factor + 
        two_qubit_factor + 
        multi_qubit_factor + 
        param_factor
    ) * (1 + metrics["two_qubit_gate_ratio"])
    
    # 결과 구성
    result = {
        "complexity_score": float(complexity_score),
        "normalized_score": float(min(1.0, complexity_score / 100)),  # 정규화된 점수 (0~1)
        "metrics": metrics
    }
    
    return result
