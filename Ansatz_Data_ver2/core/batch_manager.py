#!/usr/bin/env python3
"""
IBM Quantum 통합 배치 처리 시스템

모든 양자 회로 측정을 단일 배치 실행으로 최적화하여 대기시간을 대폭 단축합니다.
기존 3번의 개별 백엔드 연결을 1번의 통합 실행으로 최적화합니다.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from qiskit import QuantumCircuit
from core.circuit_interface import CircuitSpec
from execution.executor import ExecutionResult, QuantumExecutorFactory
from config import ExperimentConfig


@dataclass
class BatchCircuitInfo:
    """배치 실행을 위한 회로 정보"""
    task_type: str           # "fidelity", "expressibility", "entanglement"
    subtask_id: str         # "swap_test_qubit_0", "pairwise_fidelity_pair_5", etc.
    circuit: QuantumCircuit # 측정할 실제 회로
    circuit_spec: CircuitSpec # 원본 회로 스펙
    metadata: Dict[str, Any] # 결과 처리용 메타데이터
    batch_index: int        # 배치 내 순서 (결과 매핑용)


class QuantumCircuitBatchManager:
    """모든 양자 회로 측정을 통합 관리하는 배치 시스템"""
    
    def __init__(self, exp_config: ExperimentConfig):
        """
        배치 관리자 초기화
        
        Args:
            exp_config: 실험 설정
        """
        self.exp_config = exp_config
        self.batch_circuits: List[BatchCircuitInfo] = []
        self.result_mapping: Dict[str, Any] = {}
        self.task_indices: Dict[str, List[int]] = {
            "fidelity": [],
            "expressibility": [],
            "entanglement": []
        }
        self.batch_results: Optional[List[ExecutionResult]] = None
        
    def collect_task_circuits(self, task_type: str, circuits: List[QuantumCircuit], 
                            circuit_specs: List[CircuitSpec], metadata: Dict[str, Any]) -> List[int]:
        """
        태스크별 회로 수집 (배치 인덱스 반환)
        
        Args:
            task_type: 태스크 유형 ("fidelity", "expressibility", "entanglement")
            circuits: 실행할 양자 회로 리스트
            circuit_specs: 원본 회로 스펙 리스트
            metadata: 결과 처리용 메타데이터
            
        Returns:
            배치 내 인덱스 리스트
        """
        indices = []
        
        for i, (circuit, circuit_spec) in enumerate(zip(circuits, circuit_specs)):
            batch_index = len(self.batch_circuits)
            
            # 서브태스크 ID 생성
            if task_type == "fidelity":
                subtask_id = f"fidelity_circuit_{circuit_spec.circuit_id}"
            elif task_type == "expressibility":
                subtask_id = f"expr_pair_{metadata.get('pair_id', i)}"
            elif task_type == "entanglement":
                subtask_id = f"entangle_qubit_{metadata.get('target_qubit', i)}_circuit_{circuit_spec.circuit_id}"
            else:
                subtask_id = f"{task_type}_{i}"
            
            # 배치 회로 정보 생성
            batch_info = BatchCircuitInfo(
                task_type=task_type,
                subtask_id=subtask_id,
                circuit=circuit,
                circuit_spec=circuit_spec,
                metadata={**metadata, "original_index": i},
                batch_index=batch_index
            )
            
            self.batch_circuits.append(batch_info)
            self.task_indices[task_type].append(batch_index)
            indices.append(batch_index)


            # if task_type == "fidelity":
            #     print(batch_info.circuit)
            #     result = self.exp_config.executor.execute_circuits([batch_info.circuit], self.exp_config)
            #     print(result[0].counts)
            #     exit()
                
            
        print(f"📦 {task_type} 태스크: {len(circuits)}개 회로 수집 완료 (배치 인덱스: {indices[0]}-{indices[-1]})")
        return indices
    
    def execute_unified_batch(self) -> Dict[str, List[ExecutionResult]]:
        """
        통합 배치 실행 (1회 연결로 모든 회로 실행)
        
        Returns:
            태스크별 실행 결과 딕셔너리
        """
        if not self.batch_circuits:
            print("⚠️ 배치에 실행할 회로가 없습니다.")
            return {}
        
        print(f"\n🚀 통합 배치 실행 시작: {len(self.batch_circuits)}개 회로")
        print(f"   - 피델리티: {len(self.task_indices['fidelity'])}개")
        print(f"   - 표현력: {len(self.task_indices['expressibility'])}개")
        print(f"   - 얽힘도: {len(self.task_indices['entanglement'])}개")
        
        # 모든 회로를 한 번에 실행
        all_circuits = [info.circuit for info in self.batch_circuits]

        # 실행자를 통해 배치 실행
        executor = self.exp_config.executor
        self.batch_results = executor.execute_circuits(all_circuits, self.exp_config)

        if self.batch_results:
            print(f"✅ 배치 실행 완료: {len(self.batch_results)}개 결과")
        
        # 태스크별로 결과 분배
        task_results = self.distribute_results()
        return task_results
            
    def distribute_results(self) -> Dict[str, List[ExecutionResult]]:
        """
        배치 실행 결과를 태스크별로 분배
        
        Returns:
            태스크별 결과 딕셔너리
        """
        if not self.batch_results:
            return {"fidelity": [], "expressibility": [], "entanglement": []}
        
        task_results = {
            "fidelity": [],
            "expressibility": [],
            "entanglement": []
        }
        
        # 배치 인덱스 순서대로 결과 분배
        for batch_info, result in zip(self.batch_circuits, self.batch_results):
            task_type = batch_info.task_type
            task_results[task_type].append(result)
        
        print(f"📊 결과 분배 완료:")
        for task_type, results in task_results.items():
            print(f"   - {task_type}: {len(results)}개 결과")
        
        return task_results
    
    def get_task_results(self, task_type: str, indices: List[int]) -> List[ExecutionResult]:
        """
        특정 태스크의 결과를 인덱스 순서대로 반환
        
        Args:
            task_type: 태스크 유형
            indices: 요청할 배치 인덱스 리스트
            
        Returns:
            해당 인덱스의 실행 결과 리스트
        """
        if not self.batch_results:
            return []
        
        results = []
        for idx in indices:
            if 0 <= idx < len(self.batch_results):
                results.append(self.batch_results[idx])
            else:
                print(f"⚠️ 잘못된 배치 인덱스: {idx}")
                # 에러 내성: 빈 결과 추가
                results.append(ExecutionResult(counts={}, metadata={}))
        
        return results
    
    def get_circuit_info_by_index(self, batch_index: int) -> Optional[BatchCircuitInfo]:
        """
        배치 인덱스로 회로 정보 조회
        
        Args:
            batch_index: 배치 인덱스
            
        Returns:
            회로 정보 또는 None
        """
        if 0 <= batch_index < len(self.batch_circuits):
            return self.batch_circuits[batch_index]
        return None
    
    def get_batch_summary(self) -> Dict[str, Any]:
        """
        배치 처리 요약 정보 반환
        
        Returns:
            요약 정보 딕셔너리
        """
        return {
            "total_circuits": len(self.batch_circuits),
            "task_counts": {
                task_type: len(indices) 
                for task_type, indices in self.task_indices.items()
            },
            "execution_completed": self.batch_results is not None,
            "results_count": len(self.batch_results) if self.batch_results else 0
        }


class ResultDistributor:
    """배치 실행 결과를 원래 태스크로 정확히 분배"""
    
    @staticmethod
    def distribute_fidelity_results(batch_results: List[ExecutionResult], 
                                  circuit_specs: List[CircuitSpec],
                                  exp_config: ExperimentConfig) -> List[Dict[str, float]]:
        """
        피델리티 결과 분배 및 계산 (robust fidelity 포함)
        
        Args:
            batch_results: 배치 실행 결과
            circuit_specs: 원본 회로 스펙 리스트
            exp_config: 실험 설정
            
        Returns:
            피델리티 값 리스트 (standard, robust 포함)
        """
        from core.error_fidelity import ErrorFidelityCalculator
        
        fidelities = []
        for result, circuit_spec in zip(batch_results, circuit_specs):
            try:
                # 기본 피델리티 계산
                standard_fidelity = ErrorFidelityCalculator.calculate_from_execution_result(
                    result, circuit_spec.num_qubits, exp_config.shots
                )
                
                # Robust fidelity 계산 (10% 비트 플립 허용)
                robust_fidelity = ResultDistributor._calculate_robust_fidelity(
                    result, circuit_spec.num_qubits
                )
                
                fidelities.append({
                    'standard': standard_fidelity,
                    'robust': robust_fidelity
                })
            except Exception as e:
                print(f"⚠️ 피델리티 계산 실패 (회로 {circuit_spec.circuit_id}): {e}")
                fidelities.append({
                    'standard': 0.0,
                    'robust': 0.0
                })  # 에러 내성
        
        return fidelities
    
    @staticmethod
    def distribute_expressibility_results(batch_results: List[ExecutionResult], 
                                        metadata: Dict[str, Any]) -> float:
        """
        표현력 결과 분배 및 계산
        
        Args:
            batch_results: 배치 실행 결과 (45개 SWAP test 결과)
            metadata: 메타데이터 (회로 스펙 정보 등)
            
        Returns:
            표현력 값 리스트 (각 회로별 1개씩)
        """
        from expressibility.swap_test_fidelity import SwapTestFidelityEstimator
        from expressibility.fidelity_divergence import Divergence_Expressibility
        
        circuit_spec = metadata.get("circuit_spec")
        
        if not circuit_spec:
            print("⚠️ circuit_spec가 메타데이터에 없음")
            return [0.0]
    
        print(f"📊 표현력 결과 분배 시작:")
        print(f"  - 총 배치 결과: {len(batch_results)}개")
            
        # 해당 회로의 SWAP test 결과로부터 페어와이즈 피델리티 계산
        circuit_fidelities = []
        for i, result in enumerate(batch_results):
            try:
                fidelity = SwapTestFidelityEstimator._calculate_fidelity_from_swap_result(result)
                circuit_fidelities.append(fidelity)
                print(f"    ✅ SWAP test {i+1}: fidelity = {fidelity:.4f}")
            except Exception as e:
                print(f"    ⚠️ SWAP test {i+1} 계산 실패: {e}")
                circuit_fidelities.append(0.0)
            
        # KL divergence 계산
        if len(circuit_fidelities) > 0:
            print(f"    📈 {len(circuit_fidelities)}개 피델리티로 KL divergence 계산...")
            divergence_result = Divergence_Expressibility._cal_fidelity_divergence(
                circuit_fidelities, circuit_spec.num_qubits
            )
            kl_div = divergence_result.get("kl_divergence")

        print(f"📊 표현력 결과 분배 완료: {kl_div}")
        return kl_div
    
    @staticmethod
    def distribute_entanglement_results(batch_results: List[ExecutionResult], 
                                      circuit_qubit_mapping: List[Tuple[int, int, int]]) -> List[float]:
        """
        얽힘도 결과 분배 및 계산 (SWAP test 기반)
        
        Args:
            batch_results: 배치 실행 결과 (SWAP test 회로들)
            circuit_qubit_mapping: (circuit_idx, target_qubit, n_qubits) 매핑
            
        Returns:
            Meyer-Wallach entropy 값 리스트
        """
        print(f"🔍 얽힘도 결과 분배: {len(batch_results)}개 결과, {len(circuit_qubit_mapping)}개 매핑")
        
        # 결과와 매핑 수 검증
        if len(batch_results) != len(circuit_qubit_mapping):
            print(f"⚠️ 결과 수와 매핑 수 불일치: {len(batch_results)} vs {len(circuit_qubit_mapping)}")
            # 짧은 쪽에 맞춰서 처리
            min_len = min(len(batch_results), len(circuit_qubit_mapping))
            batch_results = batch_results[:min_len]
            circuit_qubit_mapping = circuit_qubit_mapping[:min_len]
            print(f"  조정된 길이: {min_len}")
        
        # 회로별로 결과 그룹화
        circuit_purities = {}
        successful_calculations = 0
        
        for i, (result, (circuit_idx, target_qubit, n_qubits)) in enumerate(zip(batch_results, circuit_qubit_mapping)):
            if circuit_idx not in circuit_purities:
                circuit_purities[circuit_idx] = {}
            
            try:
                # SWAP test 결과로부터 purity 계산
                purity = ResultDistributor._calculate_purity_from_swap_result(result)
                circuit_purities[circuit_idx][target_qubit] = purity
                successful_calculations += 1
                
                # 첫 5개 결과만 디버깅 출력
                if i < 5:
                    print(f"  [{i}] 회로 {circuit_idx}, 큐빗 {target_qubit}: purity = {purity:.4f}")
                    
            except Exception as e:
                print(f"⚠️ 얽힘도 계산 실패 (회로 {circuit_idx}, 큐빗 {target_qubit}): {e}")
                circuit_purities[circuit_idx][target_qubit] = 1.0  # 에러 내성
        
        print(f"  성공적 purity 계산: {successful_calculations}/{len(batch_results)}")
        
        # Meyer-Wallach entropy 계산
        entanglement_values = []
        for circuit_idx in sorted(circuit_purities.keys()):
            purities = circuit_purities[circuit_idx]
            if purities:
                # MW = 2 * (1 - average_purity)
                average_purity = sum(purities.values()) / len(purities)
                mw_entropy = 2 * (1 - average_purity)
                entanglement_values.append(mw_entropy)
                
                # 첫 3개 회로만 디버깅 출력
                if circuit_idx < 3:
                    print(f"  회로 {circuit_idx}: {len(purities)}개 큐빗, 평균 purity = {average_purity:.4f}, MW = {mw_entropy:.4f}")
            else:
                entanglement_values.append(0.0)  # 에러 내성
                print(f"  회로 {circuit_idx}: purity 데이터 없음")
        
        print(f"✅ 얽힘도 계산 완료: {len(entanglement_values)}개 회로")
        return entanglement_values
    
    @staticmethod
    def _calculate_purity_from_swap_result(result: ExecutionResult) -> float:
        """
        SWAP test 결과로부터 purity 계산
        
        Args:
            result: SWAP test 실행 결과
            
        Returns:
            purity 값
        """
        counts = result.counts
        total_shots = sum(counts.values())
        
        if total_shots == 0:
            return 999  # 에러 내성
        
        # ancilla 큐빗이 0인 확률 계산
        zero_count = 0
        for bitstring, count in counts.items():
            # ancilla는 나중에 추가된 레지스터로 비트스트링의 첫 번째 자리에 위치
            if len(bitstring) > 0 and bitstring[0] == '0':
                zero_count += count
        
        zero_probability = zero_count / total_shots
        
        # purity = 2 * P(0) - 1
        purity = 2 * zero_probability - 1
        if purity > 1:
            print("purity error")
            exit()
        return purity
    
    @staticmethod
    def _calculate_robust_fidelity(result: ExecutionResult, num_qubits: int) -> float:
        """
        Robust fidelity 계산 - 전체 큐빗의 10%까지 1로 바뀐 경우도 정확한 것으로 계산
        
        Args:
            result: 실행 결과
            num_qubits: 큐빗 수
            
        Returns:
            robust fidelity 값
        """
        counts = result.counts
        total_shots = sum(counts.values())
        
        if total_shots == 0:
            return 0.0
        
        # 전체 큐빗의 10% 계산 (최소 1개)
        max_flips = max(1, int(num_qubits * 0.1))
        
        # 올바른 결과 (모든 큐빗이 0)
        correct_state = '0' * num_qubits
        correct_count = counts.get(correct_state, 0)
        
        # 허용 가능한 상태들 (최대 max_flips개의 1을 가진 상태들)
        robust_count = correct_count
        
        for bitstring, count in counts.items():
            if bitstring != correct_state:
                # 1의 개수 계산
                ones_count = bitstring.count('1')
                if ones_count <= max_flips:
                    robust_count += count
        
        robust_fidelity = robust_count / total_shots
        return min(1.0, robust_fidelity)  # 1.0을 초과하지 않도록
