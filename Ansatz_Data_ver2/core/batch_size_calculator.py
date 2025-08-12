#!/usr/bin/env python3
"""
동적 배치 크기 계산기

시스템 메모리와 양자 회로 복잡도를 기반으로 최적의 배치 크기를 계산합니다.
"""

import psutil
import math
import gc
import tracemalloc
from typing import List, Dict, Any, Optional
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


class BatchSizeCalculator:
    """
    동적 배치 크기 계산기
    
    시스템 메모리, 회로 복잡도, 백엔드 제약사항을 종합적으로 고려하여
    최적의 배치 크기를 계산합니다.
    """
    
    def __init__(self, backend_type: str = "ibm"):
        """
        Args:
            backend_type: 백엔드 타입 ("ibm", "aer", "local" 등)
        """
        self.backend_type = backend_type.lower()
        
        # 백엔드별 제약사항 설정
        self._set_backend_constraints()
    
    def _set_backend_constraints(self):
        """백엔드별 제약사항 설정"""
        if self.backend_type == "ibm":
            self.max_shots_per_batch = 10_000_000  # IBM 샷 수 제한
            self.max_payload_bytes = 80_000_000    # IBM 페이로드 제한 (80MB, 안전 마진)
            self.memory_safety_factor = 0.2        # 메모리 안전 마진 (60%)
        elif self.backend_type == "aer":
            self.max_shots_per_batch = 100_000_000  # Aer는 더 관대
            self.max_payload_bytes = 500_000_000    # 500MB
            self.memory_safety_factor = 0.9        # 80%
        else:  # local or other
            self.max_shots_per_batch = float('inf')
            self.max_payload_bytes = 1_000_000_000  # 1GB
            self.memory_safety_factor = 0.2         # 40% (매우 보수적)
    
    def calculate_dynamic_batch_sizes(
        self, 
        circuits: List[QuantumCircuit], 
        shots_per_circuit: int,
        verbose: bool = True
    ) -> List[int]:
        """
        회로별 메모리 사용량을 기반으로 동적 배치 크기 리스트 계산
        
        Args:
            circuits: 실행할 양자 회로 리스트
            shots_per_circuit: 회로당 샷 수
            verbose: 상세 출력 여부
            
        Returns:
            각 배치의 크기를 담은 리스트 [batch1_size, batch2_size, ...]
        """
        if not circuits:
            return [1000]
        
        # 1. 시스템 메모리 정보 수집
        memory_info = self._get_memory_info()
        if verbose:
            self._print_memory_info(memory_info)
        
        # 2. 각 회로별 정확한 메모리 사용량 계산
        circuit_memory_usage = self._calculate_per_circuit_memory_usage(circuits)
        
        # 3. 사용 가능한 메모리 계산
        available_memory_mb = memory_info['available_gb'] * 1024 * self.memory_safety_factor
        
        # 4. 동적 배치 분할
        batch_sizes = self._create_dynamic_batches(
            circuits, circuit_memory_usage, available_memory_mb, shots_per_circuit, verbose
        )
        
        if verbose:
            self._print_dynamic_batch_results(batch_sizes, circuit_memory_usage)
        
        return batch_sizes
    
    def calculate_max_batch_size(
        self, 
        circuits: List[QuantumCircuit], 
        shots_per_circuit: int,
        verbose: bool = True
    ) -> int:
        """
        동적 배치 크기 계산 (기존 호환성 유지)
        
        Args:
            circuits: 실행할 양자 회로 리스트
            shots_per_circuit: 회로당 샷 수
            verbose: 상세 출력 여부
            
        Returns:
            최적화된 배치 크기
        """
        batch_sizes = self.calculate_dynamic_batch_sizes(circuits, shots_per_circuit, verbose)
        return batch_sizes[0] if batch_sizes else 1000
    
    def _get_memory_info(self) -> Dict[str, float]:
        """시스템 메모리 정보 수집"""
        memory_info = psutil.virtual_memory()
        return {
            'total_gb': memory_info.total / (1024**3),
            'available_gb': memory_info.available / (1024**3),
            'usage_percent': memory_info.percent,
            'free_percent': 100 - memory_info.percent
        }
    
    def _analyze_circuit_complexities(self, circuits: List[QuantumCircuit]) -> Dict[str, Any]:
        """
        회로들의 복잡도를 분석하여 메모리 사용량 예측에 활용
        
        Returns:
            회로 복잡도 통계 정보
        """
        if not circuits:
            return {}
        
        # 샘플링으로 성능 최적화 (최대 100개 회로 분석)
        sample_size = min(100, len(circuits))
        sample_circuits = circuits[:sample_size]
        
        qubit_counts = [c.num_qubits for c in sample_circuits]
        gate_counts = [len(c.data) for c in sample_circuits]
        depth_counts = [c.depth() for c in sample_circuits]
        
        # 2큐빗 게이트 비율 계산
        two_qubit_ratios = []
        for circuit in sample_circuits:
            total_gates = len(circuit.data)
            if total_gates == 0:
                two_qubit_ratios.append(0)
            else:
                two_qubit_gates = sum(1 for gate, qubits, _ in circuit.data if len(qubits) == 2)
                two_qubit_ratios.append(two_qubit_gates / total_gates)
        
        # 파라미터화된 게이트 비율 계산
        param_gate_ratios = []
        for circuit in sample_circuits:
            total_gates = len(circuit.data)
            if total_gates == 0:
                param_gate_ratios.append(0)
            else:
                param_gates = sum(1 for gate, qubits, params in circuit.data if params)
                param_gate_ratios.append(param_gates / total_gates)
        
        complexities = {
            'avg_qubits': sum(qubit_counts) / len(qubit_counts),
            'max_qubits': max(qubit_counts),
            'min_qubits': min(qubit_counts),
            'avg_gates': sum(gate_counts) / len(gate_counts),
            'max_gates': max(gate_counts),
            'min_gates': min(gate_counts),
            'avg_depth': sum(depth_counts) / len(depth_counts),
            'max_depth': max(depth_counts),
            'min_depth': min(depth_counts),
            'avg_two_qubit_ratio': sum(two_qubit_ratios) / len(two_qubit_ratios) if two_qubit_ratios else 0,
            'avg_param_ratio': sum(param_gate_ratios) / len(param_gate_ratios) if param_gate_ratios else 0,
            'sample_size': sample_size,
            'total_circuits': len(circuits)
        }
        
        return complexities
    
    def _calculate_shots_based_limit(self, shots_per_circuit: int) -> int:
        """샷 수 제한 기반 배치 크기 계산"""
        if self.max_shots_per_batch == float('inf'):
            return 50000  # 실용적인 상한선
        
        return max(1, self.max_shots_per_batch // shots_per_circuit)
    
    def _calculate_memory_based_limit(self, complexities: Dict[str, Any], memory_info: Dict[str, float]) -> int:
        """
        시스템 메모리를 기반으로 한 배치 크기 계산
        
        메모리 사용량은 큐빗 수에 지수적으로 증가하므로 이를 고려한 계산
        """
        if not complexities:
            return 1000
        
        avg_qubits = complexities['avg_qubits']
        max_qubits = complexities['max_qubits']
        available_gb = memory_info['available_gb']
        
        # 큐빗 수별 메모리 사용량 추정 (경험적 공식)
        memory_per_circuit_mb = self._estimate_memory_per_circuit(complexities)
        
        # 사용 가능한 메모리 계산 (안전 마진 적용)
        usable_memory_mb = available_gb * 1024 * self.memory_safety_factor
        
        # 배치 크기 계산
        max_circuits = int(usable_memory_mb / memory_per_circuit_mb)
        
        # 큐빗 수별 추가 제한 (하드웨어 특성 고려)
        if max_qubits > 30:
            max_circuits = min(max_circuits, 20)   # 매우 큰 회로
        elif max_qubits > 25:
            max_circuits = min(max_circuits, 50)   # 큰 회로
        elif max_qubits > 20:
            max_circuits = min(max_circuits, 200)  # 중간-큰 회로
        elif max_qubits > 15:
            max_circuits = min(max_circuits, 500)  # 중간 회로
        elif max_qubits > 10:
            max_circuits = min(max_circuits, 1000) # 작은-중간 회로
        
        return max(10, max_circuits)
    
    def _estimate_memory_per_circuit(self, complexities: Dict[str, Any]) -> float:
        """회로당 메모리 사용량 추정 (MB 단위)"""
        avg_qubits = complexities['avg_qubits']
        avg_gates = complexities['avg_gates']
        avg_depth = complexities['avg_depth']
        two_qubit_ratio = complexities['avg_two_qubit_ratio']
        param_ratio = complexities['avg_param_ratio']
        
        # 기본 메모리 사용량 (큐빗 수 기반)
        if avg_qubits <= 10:
            # 작은 회로: 선형 증가
            base_memory = 2 + avg_qubits * 1.5
        elif avg_qubits <= 20:
            # 중간 회로: 지수적 증가 시작
            base_memory = 5 * (1.8 ** (avg_qubits - 10))
        elif avg_qubits <= 30:
            # 큰 회로: 급격한 지수적 증가
            base_memory = 100 * (2.5 ** (avg_qubits - 20))
        else:
            # 매우 큰 회로: 극도로 제한적
            base_memory = 10000 * (3 ** (avg_qubits - 30))
        
        # 게이트 복잡도 보정
        gate_factor = 1 + (avg_gates / 1000) * 0.3
        depth_factor = 1 + (avg_depth / 100) * 0.2
        two_qubit_factor = 1 + two_qubit_ratio * 0.5
        param_factor = 1 + param_ratio * 0.3
        
        total_memory = base_memory * gate_factor * depth_factor * two_qubit_factor * param_factor
        
        return max(1.0, total_memory)  # 최소 1MB
    
    def _calculate_payload_based_limit(self, complexities: Dict[str, Any]) -> int:
        """페이로드 제한을 기반으로 한 배치 크기 계산"""
        if not complexities:
            return 1000
        
        # 회로당 페이로드 크기 추정 (바이트)
        payload_per_circuit = self._estimate_payload_per_circuit(complexities)
        
        # 최대 배치 크기 계산
        max_circuits = int(self.max_payload_bytes / payload_per_circuit)
        
        return max(50, min(10000, max_circuits))
    
    def _estimate_payload_per_circuit(self, complexities: Dict[str, Any]) -> float:
        """회로당 페이로드 크기 추정 (바이트 단위)"""
        base_payload = 500  # 기본 메타데이터
        
        # 큐빗당 페이로드
        qubit_payload = complexities['avg_qubits'] * 50
        
        # 게이트당 페이로드 (게이트 타입, 파라미터, 큐빗 정보)
        gate_payload = complexities['avg_gates'] * 100
        
        # 2큐빗 게이트 추가 복잡도
        two_qubit_payload = (complexities['avg_gates'] * 
                           complexities['avg_two_qubit_ratio'] * 50)
        
        # 파라미터화된 게이트 추가 복잡도
        param_payload = (complexities['avg_gates'] * 
                        complexities['avg_param_ratio'] * 30)
        
        # 회로 깊이에 따른 추가 복잡도
        depth_payload = complexities['avg_depth'] * 20
        
        total_payload = (base_payload + qubit_payload + gate_payload + 
                        two_qubit_payload + param_payload + depth_payload)
        
        return max(500, total_payload)  # 최소 500바이트
    
    def _calculate_per_circuit_memory_usage(self, circuits: List[QuantumCircuit]) -> List[float]:
        """
        각 회로별 실제 메모리 사용량 측정 기반 계산 (MB 단위)
        
        Args:
            circuits: 양자 회로 리스트
            
        Returns:
            각 회로의 메모리 사용량 리스트 (MB)
        """
        memory_usage = []
        
        # 샘플 회로들로 실제 메모리 사용량 측정
        sample_size = min(5, len(circuits))  # 최대 5개 샘플로 측정
        sample_circuits = circuits[:sample_size]
        
        measured_memory = {}  # 큐빗 수별 실제 메모리 사용량 캐시
        
        print(f"🔬 실제 메모리 사용량 측정 중... ({sample_size}개 샘플)")
        
        for i, circuit in enumerate(sample_circuits):
            n_qubits = circuit.num_qubits
            
            # 이미 측정한 큐빗 수는 건너뛰기
            if n_qubits in measured_memory:
                continue
                
            actual_memory = self._measure_actual_memory_usage(circuit)
            measured_memory[n_qubits] = actual_memory
            
            print(f"  - {n_qubits}큐빗 회로: {actual_memory:.1f} MB")
        
        # 모든 회로에 대해 메모리 사용량 계산
        for circuit in circuits:
            n_qubits = circuit.num_qubits
            
            if n_qubits in measured_memory:
                # 실제 측정된 값 사용
                base_memory = measured_memory[n_qubits]
            else:
                # 가장 가까운 측정값으로 추정
                base_memory = self._estimate_from_measured_data(n_qubits, measured_memory)
            
            # 게이트 복잡도에 따른 조정
            n_gates = len(circuit.data)
            depth = circuit.depth()
            
            complexity_factor = 1.0
            if n_gates > 1000:
                complexity_factor *= 1.3
            elif n_gates > 500:
                complexity_factor *= 1.2
            elif n_gates > 100:
                complexity_factor *= 1.1
            
            if depth > 100:
                complexity_factor *= 1.2
            elif depth > 50:
                complexity_factor *= 1.1
            
            final_memory = base_memory * complexity_factor
            memory_usage.append(max(10.0, final_memory))  # 최소 10MB
        
        return memory_usage
    
    def _measure_actual_memory_usage(self, circuit: QuantumCircuit) -> float:
        """
        단일 회로의 실제 메모리 사용량 측정
        
        Args:
            circuit: 측정할 양자 회로
            
        Returns:
            실제 메모리 사용량 (MB)
        """
        try:
            # 메모리 측정 시작
            gc.collect()  # 가비지 컬렉션
            tracemalloc.start()
            
            # 시뮬레이터 생성 및 실행
            simulator = AerSimulator(method='statevector')
            
            # 측정을 위한 클래식 비트 추가 (필요한 경우)
            test_circuit = circuit.copy()
            if test_circuit.num_clbits == 0:
                test_circuit.add_bits([0] * test_circuit.num_qubits)
                test_circuit.measure_all()
            
            # 실제 실행 (적은 샷 수로)
            job = simulator.run(test_circuit, shots=1)
            result = job.result()
            
            # 메모리 사용량 측정
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # MB 단위로 변환
            peak_memory_mb = peak / (1024 ** 2)
            
            # 정리
            del simulator, job, result, test_circuit
            gc.collect()
            
            return peak_memory_mb
            
        except Exception as e:
            # 측정 실패 시 이론적 값으로 대체
            print(f"    ⚠️  메모리 측정 실패 ({circuit.num_qubits}큐빗): {e}")
            n_qubits = circuit.num_qubits
            theoretical_memory = (2 ** n_qubits * 16) / (1024 ** 2) * 0.3  # 4배 안전 마진
            return max(10.0, theoretical_memory)
    
    def _estimate_from_measured_data(self, target_qubits: int, measured_data: Dict[int, float]) -> float:
        """
        측정된 데이터를 바탕으로 다른 큐빗 수의 메모리 사용량 추정
        
        Args:
            target_qubits: 추정하려는 큐빗 수
            measured_data: 측정된 큐빗별 메모리 데이터
            
        Returns:
            추정된 메모리 사용량 (MB)
        """
        if not measured_data:
            # 측정 데이터가 없으면 이론적 값 사용
            return (2 ** target_qubits * 16) / (1024 ** 2) * 4
        
        # 가장 가까운 측정값 찾기
        closest_qubits = min(measured_data.keys(), key=lambda x: abs(x - target_qubits))
        closest_memory = measured_data[closest_qubits]
        
        # 큐빗 차이에 따른 스케일링
        qubit_diff = target_qubits - closest_qubits
        
        if qubit_diff == 0:
            return closest_memory
        elif qubit_diff > 0:
            # 더 큰 회로: 지수적 증가
            scaling_factor = 2 ** qubit_diff
            return closest_memory * scaling_factor
        else:
            # 더 작은 회로: 지수적 감소
            scaling_factor = 2 ** abs(qubit_diff)
            return max(10.0, closest_memory / scaling_factor)
    
    def _create_dynamic_batches(
        self, 
        circuits: List[QuantumCircuit], 
        memory_usage: List[float], 
        available_memory_mb: float,
        shots_per_circuit: int,
        verbose: bool
    ) -> List[int]:
        """
        메모리 사용량을 기반으로 동적 배치 생성
        
        Args:
            circuits: 양자 회로 리스트
            memory_usage: 각 회로의 메모리 사용량 (MB)
            available_memory_mb: 사용 가능한 메모리 (MB)
            shots_per_circuit: 회로당 샷 수
            verbose: 상세 출력 여부
            
        Returns:
            배치 크기 리스트
        """
        batch_sizes = []
        current_batch_size = 0
        current_batch_memory = 0.0
        
        # 샷 수 제한 계산
        max_circuits_by_shots = self.max_shots_per_batch // shots_per_circuit if self.max_shots_per_batch != float('inf') else float('inf')
        
        i = 0
        while i < len(circuits):
            circuit_memory = memory_usage[i]
            
            # 현재 회로를 배치에 추가할 수 있는지 확인
            can_add_to_batch = (
                current_batch_memory + circuit_memory <= available_memory_mb and
                current_batch_size + 1 <= max_circuits_by_shots and
                current_batch_size < 50000  # 최대 배치 크기 제한
            )
            
            if can_add_to_batch and current_batch_size > 0:
                # 현재 배치에 추가
                current_batch_size += 1
                current_batch_memory += circuit_memory
                i += 1
            else:
                # 새로운 배치 시작
                if current_batch_size > 0:
                    batch_sizes.append(current_batch_size)
                
                # 단일 회로가 메모리 한계를 초과하는 경우
                if circuit_memory > available_memory_mb:
                    if verbose:
                        print(f"⚠️  회로 {i+1}번이 메모리 한계를 초과합니다 ({circuit_memory:.1f}MB > {available_memory_mb:.1f}MB)")
                        print(f"    큐빗 수: {circuits[i].num_qubits}, 게이트 수: {len(circuits[i].data)}")
                    # 강제로 단일 회로 배치 생성
                    batch_sizes.append(1)
                    i += 1
                else:
                    # 새 배치 시작
                    current_batch_size = 1
                    current_batch_memory = circuit_memory
                    i += 1
        
        # 마지막 배치 추가
        if current_batch_size > 0:
            batch_sizes.append(current_batch_size)
        
        # 빈 배치가 없도록 보장
        if not batch_sizes:
            batch_sizes = [min(10, len(circuits))]
        
        return batch_sizes
    
    def _print_dynamic_batch_results(self, batch_sizes: List[int], memory_usage: List[float]):
        """동적 배치 결과 출력"""
        total_circuits = sum(batch_sizes)
        total_memory = sum(memory_usage)
        avg_memory = total_memory / len(memory_usage) if memory_usage else 0
        
        print(f"🎯 동적 배치 분할 결과:")
        print(f"  - 총 회로 수: {total_circuits:,}개")
        print(f"  - 배치 수: {len(batch_sizes)}개")
        print(f"  - 배치 크기: {batch_sizes}")
        print(f"  - 총 예상 메모리: {total_memory:.1f} MB")
        print(f"  - 회로당 평균 메모리: {avg_memory:.1f} MB")
        
        # 메모리 사용량 통계
        if memory_usage:
            min_memory = min(memory_usage)
            max_memory = max(memory_usage)
            print(f"  - 메모리 범위: {min_memory:.1f} - {max_memory:.1f} MB")
    
    def _print_memory_info(self, memory_info: Dict[str, float]):
        """메모리 정보 출력"""
        print(f"💾 시스템 메모리 상태:")
        print(f"  - 총 메모리: {memory_info['total_gb']:.1f} GB")
        print(f"  - 사용 가능: {memory_info['available_gb']:.1f} GB ({memory_info['free_percent']:.1f}%)")
        print(f"  - 안전 마진: {self.memory_safety_factor*100:.0f}% → 사용 가능: {memory_info['available_gb']*self.memory_safety_factor:.1f} GB")
    
    def _print_complexity_info(self, complexities: Dict[str, Any]):
        """회로 복잡도 정보 출력"""
        print(f"🔍 회로 복잡도 분석 ({complexities['sample_size']}/{complexities['total_circuits']}개 샘플):")
        print(f"  - 큐빗 수: 평균 {complexities['avg_qubits']:.1f}, 최대 {complexities['max_qubits']}")
        print(f"  - 게이트 수: 평균 {complexities['avg_gates']:.1f}, 최대 {complexities['max_gates']}")
        print(f"  - 회로 깊이: 평균 {complexities['avg_depth']:.1f}, 최대 {complexities['max_depth']}")
        print(f"  - 2큐빗 게이트 비율: {complexities['avg_two_qubit_ratio']*100:.1f}%")
        print(f"  - 파라미터 게이트 비율: {complexities['avg_param_ratio']*100:.1f}%")
    
    def _print_final_results(self, max_by_shots: int, max_by_memory: int, 
                           max_by_payload: int, final_size: int):
        """최종 결과 출력"""
        print(f"📊 동적 배치 크기 계산 결과 ({self.backend_type.upper()}):")
        print(f"  - 샷 수 제한: {max_by_shots:,}개")
        print(f"  - 메모리 제한: {max_by_memory:,}개")
        print(f"  - 페이로드 제한: {max_by_payload:,}개")
        print(f"  - 최종 배치 크기: {final_size:,}개/배치")
        
        # 제한 요인 분석
        limiting_factor = min(max_by_shots, max_by_memory, max_by_payload)
        if limiting_factor == max_by_shots:
            print(f"  ⚠️  주요 제한 요인: 샷 수 제한")
        elif limiting_factor == max_by_memory:
            print(f"  ⚠️  주요 제한 요인: 메모리 제한")
        else:
            print(f"  ⚠️  주요 제한 요인: 페이로드 제한")


def calculate_dynamic_batch_sizes(
    circuits: List[QuantumCircuit],
    shots_per_circuit: int,
    backend_type: str = "ibm",
    verbose: bool = True
) -> List[int]:
    """
    동적 배치 크기 리스트 계산 (편의 함수)
    
    Args:
        circuits: 실행할 양자 회로 리스트
        shots_per_circuit: 회로당 샷 수
        backend_type: 백엔드 타입 ("ibm", "aer", "local")
        verbose: 상세 출력 여부
        
    Returns:
        각 배치의 크기를 담은 리스트
    """
    calculator = BatchSizeCalculator(backend_type)
    return calculator.calculate_dynamic_batch_sizes(circuits, shots_per_circuit, verbose)

def calculate_optimal_batch_size(
    circuits: List[QuantumCircuit],
    shots_per_circuit: int,
    backend_type: str = "ibm",
    verbose: bool = True
) -> int:
    """
    최적 배치 크기 계산 (편의 함수, 기존 호환성 유지)
    
    Args:
        circuits: 실행할 양자 회로 리스트
        shots_per_circuit: 회로당 샷 수
        backend_type: 백엔드 타입 ("ibm", "aer", "local")
        verbose: 상세 출력 여부
        
    Returns:
        최적화된 배치 크기
    """
    batch_sizes = calculate_dynamic_batch_sizes(circuits, shots_per_circuit, backend_type, verbose)
    return batch_sizes[0] if batch_sizes else 1000


def get_memory_usage_estimate(circuits: List[QuantumCircuit]) -> Dict[str, Any]:
    """
    회로들의 예상 메모리 사용량 분석
    
    Args:
        circuits: 분석할 양자 회로 리스트
        
    Returns:
        메모리 사용량 분석 결과
    """
    calculator = BatchSizeCalculator()
    complexities = calculator._analyze_circuit_complexities(circuits)
    
    if not complexities:
        return {}
    
    memory_per_circuit = calculator._estimate_memory_per_circuit(complexities)
    payload_per_circuit = calculator._estimate_payload_per_circuit(complexities)
    
    return {
        'circuit_complexities': complexities,
        'memory_per_circuit_mb': memory_per_circuit,
        'payload_per_circuit_bytes': payload_per_circuit,
        'total_memory_estimate_mb': memory_per_circuit * len(circuits),
        'total_payload_estimate_mb': (payload_per_circuit * len(circuits)) / (1024**2)
    }


if __name__ == "__main__":
    # 테스트 코드
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RealAmplitudes
    
    print("🧪 배치 크기 계산기 테스트")
    print("=" * 50)
    
    # 테스트 회로 생성
    test_circuits = []
    
    # 다양한 크기의 회로 생성
    for n_qubits in [4, 8, 12, 16]:
        for _ in range(10):
            circuit = RealAmplitudes(n_qubits, reps=3)
            test_circuits.append(circuit)
    
    print(f"테스트 회로 개수: {len(test_circuits)}개")
    print(f"큐빗 범위: 4-16 qubits")
    
    # 배치 크기 계산 테스트
    batch_size = calculate_optimal_batch_size(
        circuits=test_circuits,
        shots_per_circuit=1024,
        backend_type="ibm",
        verbose=True
    )
    
    print(f"\n✅ 권장 배치 크기: {batch_size}개")
    
    # 메모리 사용량 분석
    print("\n" + "=" * 50)
    print("📊 메모리 사용량 분석")
    memory_analysis = get_memory_usage_estimate(test_circuits)
    
    if memory_analysis:
        print(f"회로당 예상 메모리: {memory_analysis['memory_per_circuit_mb']:.1f} MB")
        print(f"회로당 예상 페이로드: {memory_analysis['payload_per_circuit_bytes']:.0f} bytes")
        print(f"전체 예상 메모리: {memory_analysis['total_memory_estimate_mb']:.1f} MB")
        print(f"전체 예상 페이로드: {memory_analysis['total_payload_estimate_mb']:.1f} MB")
