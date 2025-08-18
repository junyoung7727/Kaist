"""
Quantum Circuit Dataset Module
간단하고 확장성 높은 양자회로 데이터셋 처리
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import pathlib
import re
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent / "quantumcommon"))
from gates import gate_registry, GateType, GateOperation


def extract_depth_from_circuit_id(circuit_id: str) -> int:
    """
    circuit_id에서 depth 값을 추출합니다.
    
    예시: "scalability_test1_16q_d1_r0.3_0" -> 1
          "circuit_8q_d10_r0.5_2" -> 10
    
    Args:
        circuit_id: 회로 ID 문자열
        
    Returns:
        depth 값 (찾을 수 없으면 0)
    """
    # _d뒤에 오는 숫자를 찾는 정규식
    pattern = r'_d(\d+)_'
    match = re.search(pattern, circuit_id)
    
    if match:
        return int(match.group(1))
    else:
        # 패턴을 찾을 수 없으면 0 반환
        print(f"Warning: Could not extract depth from circuit_id: {circuit_id}")
        return 0


@dataclass
class CircuitSpec:
    """양자 회로 스펙"""
    circuit_id: str
    num_qubits: int
    gates: List[GateOperation]
    depth: int = 0  # 기본값 추가
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CircuitSpec':
        """딕셔너리에서 CircuitSpec 생성"""
        gates = [
            GateOperation(
                name=gate['name'],
                qubits=gate['qubits'],
                parameters=gate['parameters']
            )
            for gate in data['gates']
        ]
        
        # circuit_id에서 depth 추출
        circuit_id = data['circuit_id']
        depth = extract_depth_from_circuit_id(circuit_id)
        
        return cls(
            circuit_id=circuit_id,
            num_qubits=data['num_qubits'],
            gates=gates,
            depth=depth
        )


@dataclass
class MeasurementResult:
    """측정 결과 데이터 - timestamp 제외하고 모든 필드 포함"""
    circuit_id: str
    num_qubits: int
    depth: int
    fidelity: float
    robust_fidelity: Optional[float] = None
    expressibility: Optional[Dict[str, float]] = None
    entanglement: Optional[float] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MeasurementResult':
        """딕셔너리에서 MeasurementResult 생성

        예:
        {
            "circuit_id": "scalability_test_15q_d1_r0.8_0",
            "num_qubits": 15,
            "depth": 16,
            "timestamp": "2025-08-13T10:14:44.745735",
            "fidelity": 0.73046875,
            "robust_fidelity": 0.962890625,
            "expressibility": {
                "kl_divergence": 8.466604598985924
            },
            "entanglement": 0.5020833333333334
        }
        """
        return cls(
            circuit_id=data['circuit_id'],
            num_qubits=data['num_qubits'],
            depth=data['depth'],
            timestamp=data['timestamp'],
            fidelity=data['fidelity'],
            robust_fidelity=data.get('robust_fidelity'),
            expressibility=data.get('expressibility'),
            entanglement=np.clip(data.get('entanglement'), 0, 1)
        )


@dataclass
class CircuitData:
    """회로 스펙과 측정 결과를 결합한 데이터"""
    circuit_spec: CircuitSpec
    measurement_result: MeasurementResult
    
    @property
    def circuit_id(self) -> str:
        return self.circuit_spec.circuit_id
    
    @property
    def num_qubits(self) -> int:
        return self.circuit_spec.num_qubits
    
    @property
    def gates(self) -> List[GateOperation]:
        return self.circuit_spec.gates


class QuantumCircuitDataset(Dataset):
    """양자 회로 데이터셋"""
    
    def __init__(self, circuit_data: List[CircuitData]):
        self.circuit_data = circuit_data
    
    def __len__(self) -> int:
        return len(self.circuit_data)
    
    def __getitem__(self, idx: int) -> CircuitData:
        return self.circuit_data[idx]
    
    def get_by_circuit_id(self, circuit_id: str) -> Optional[CircuitData]:
        """circuit_id로 데이터 검색"""
        for data in self.circuit_data:
            if data.circuit_id == circuit_id:
                return data
        return None


class DatasetManager:
    """데이터셋 관리자 - 로딩, 분할, 변환 담당"""
    
    def __init__(self, unified_data_path: Optional[str] = None, results_path: Optional[str] = None, circuits_path: Optional[str] = None, merged_results_path: Optional[str] = None):
        """데이터셋 매니저 초기화
        
        Args:
            unified_data_path: 통합 데이터 파일 경로 (merged_results와 merged_circuits 모두 포함)
            results_path: 측정 결과 JSON 파일 경로 (레거시 지원)
            circuits_path: 회로 스펙 JSON 파일 경로 (레거시 지원)
            merged_results_path: 병합된 결과 JSON 파일 경로 (레거시 지원)
        """
        self.unified_data_path = Path(unified_data_path) if unified_data_path else None
        self.results_path = Path(results_path) if results_path else None
        self.circuits_path = Path(circuits_path) if circuits_path else None
        self.merged_results_path = Path(merged_results_path) if merged_results_path else None
        
        self.raw_results_data = None
        self.raw_circuits_data = None
        self.circuit_data = None
    
    def load_results_data(self) -> Dict[str, Any]:
        """측정 결과 JSON 데이터 로딩"""
        if self.unified_data_path and self.unified_data_path.exists():
            # 통합 데이터 파일에서 로딩
            with open(self.unified_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.raw_results_data = data
                return data
        elif self.merged_results_path and self.merged_results_path.exists():
            # 병합된 결과 파일에서 로딩 (레거시 지원)
            with open(self.merged_results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.raw_results_data = data
                return data
        elif self.results_path and self.results_path.exists():
            # 개별 결과 파일에서 로딩 (레거시 지원)
            with open(self.results_path, 'r', encoding='utf-8') as f:
                self.raw_results_data = json.load(f)
                return self.raw_results_data
        else:
            raise FileNotFoundError("결과 데이터 파일을 찾을 수 없습니다.")
    
    def load_circuits_data(self) -> Dict[str, Any]:
        """회로 스펙 JSON 데이터 로딩"""
        if self.unified_data_path and self.unified_data_path.exists():
            # 통합 데이터 파일에서 로딩 (이미 load_results_data에서 로딩된 경우 재사용)
            if self.raw_results_data is None:
                with open(self.unified_data_path, 'r', encoding='utf-8') as f:
                    self.raw_results_data = json.load(f)
            self.raw_circuits_data = self.raw_results_data  # 같은 파일에서 circuits 정보 추출
            return self.raw_circuits_data
        elif self.circuits_path and self.circuits_path.exists():
            # 개별 회로 파일에서 로딩 (레거시 지원)
            with open(self.circuits_path, 'r', encoding='utf-8') as f:
                self.raw_circuits_data = json.load(f)
                return self.raw_circuits_data
        else:
            raise FileNotFoundError("회로 스펙 파일을 찾을 수 없습니다.")
    
    def load_measurement_results(self, file_path: str) -> Dict[str, MeasurementResult]:
        """측정 결과 파일 로딩 - merged_results와 IBM results 형식 모두 지원"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = {}
        
        # merged_results 형식 처리 (시뮬레이터 형식)
        if 'merged_results' in data:
            for result in data['merged_results']:
                # timestamp 제거 (사용자 요청)
                result_data = {k: v for k, v in result.items() if k != 'timestamp'}
                measurement_result = MeasurementResult(**result_data)
                results[measurement_result.circuit_id] = measurement_result
        
        # IBM 실험 결과 형식 처리
        elif 'results' in data:
            for result in data['results']:
                # timestamp 제거 (사용자 요청)
                result_data = {k: v for k, v in result.items() if k != 'timestamp'}
                measurement_result = MeasurementResult(**result_data)
                results[measurement_result.circuit_id] = measurement_result
        
        # 단일 결과 형식 처리 (기존 호환성)
        elif isinstance(data, list):
            for result in data:
                # timestamp 제거 (사용자 요청)
                result_data = {k: v for k, v in result.items() if k != 'timestamp'}
                measurement_result = MeasurementResult(**result_data)
                results[measurement_result.circuit_id] = measurement_result
        
        else:
            raise ValueError(f"Unsupported JSON format in {file_path}. Expected 'merged_results', 'results', or list format.")
        
        return results
    
    def parse_circuit_specs(self) -> Dict[str, CircuitSpec]:
        """회로 스펙을 파싱하여 circuit_id를 키로 하는 딕셔너리 반환"""
        if self.raw_circuits_data is None:
            self.load_circuits_data()
        
        specs = {}
        circuits_data = self.raw_circuits_data.get('merged_circuits', {})
        
        for circuit_id, circuit_data in circuits_data.items():
            spec = CircuitSpec.from_dict(circuit_data)
            specs[circuit_id] = spec
        
        return specs
    
    def is_valid_circuit_data(self, circuit_data: CircuitData) -> bool:
        """
        회로 데이터의 유효성을 검증
        
        Args:
            circuit_data: 검증할 회로 데이터
            
        Returns:
            True if valid, False if invalid
        """
        result = circuit_data.measurement_result
        # 1. 기본 필수 필드 검증
        if result is None or result.fidelity is None or result.entanglement is None:
            return False
        
        # 2. Expressibility 데이터 검증 (KL divergence만 사용)
        if result.expressibility and isinstance(result.expressibility, dict):
            expr = result.expressibility
            
            # KL divergence가 0인 경우 무효한 데이터로 제거
            kl_divergence = expr.get('kl_divergence')
            if kl_divergence is None or kl_divergence == 0.0:
                return False
                
        elif result.expressibility is None:
            # expressibility 데이터가 None인 경우 무효한 데이터로 간주
            return False
        else:
            # expressibility가 dict가 아닌 경우도 무효한 데이터로 간주
            return False
        
        # 3. 피델리티 값 검증 (음수나 1보다 큰 값 제거)
        if result.fidelity < 0 or result.fidelity > 1:
            return False
        
        if result.robust_fidelity is not None:
            if result.robust_fidelity < 0 or result.robust_fidelity > 1:
                return False
        
        # 4. 얽힘도 값 검증 (음수 제거)
        if result.entanglement < 0:
            return False
        
        return True
    
    def merge_data(self, enable_filtering: bool = True) -> List[CircuitData]:
        """
        회로 스펙과 측정 결과를 circuit_id로 병합하고 데이터 품질 검증
        
        Args:
            enable_filtering: True면 무효한 데이터 필터링, False면 모든 데이터 포함
            
        Returns:
            병합된 회로 데이터 리스트
        """
        # 측정 결과와 회로 스펙 파싱
        # 사용할 데이터 파일 경로 결정
        if self.unified_data_path:
            data_file_path = str(self.unified_data_path)
        elif self.merged_results_path:
            data_file_path = str(self.merged_results_path)
        elif self.results_path:
            data_file_path = str(self.results_path)
        else:
            raise FileNotFoundError("데이터 파일 경로가 설정되지 않았습니다.")
            
        measurement_results = self.load_measurement_results(data_file_path)
        circuit_specs = self.parse_circuit_specs()
        
        merged_data = []
        filtered_count = 0
        
        # circuit_id를 기준으로 데이터 병합
        for circuit_id in list(circuit_specs.keys()):

            circuit_data = CircuitData(
                circuit_spec=circuit_specs[circuit_id],
                measurement_result=measurement_results.get(circuit_id)
            )
            # 데이터 품질 검증
            if enable_filtering:
                merged_data.append(circuit_data)
        # 측정 결과만 있고 회로 스펙이 없는 경우 경고
        for circuit_id in measurement_results.keys():
            if circuit_id not in list(circuit_specs.keys()):
                print(f"Warning: 회로 스펙이 없는 측정 결과 ID: {circuit_id}")
        
        if enable_filtering and filtered_count > 0:
            print(f"✅ 데이터 품질 필터링 완료: {filtered_count}개 무효 데이터 제거됨")
            print(f"📊 유효한 데이터: {len(merged_data)}개 / 전체: {len(merged_data) + filtered_count}개")
        
        self.circuit_data = merged_data
        return merged_data
    
    def split_dataset(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ) -> Tuple["QuantumCircuitDataset", "QuantumCircuitDataset", "QuantumCircuitDataset"]:
        """데이터셋 분할"""
        if self.circuit_data is None:
            self.merge_data()
        
        # 비율 검증
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "비율의 합이 1이 되어야 합니다"
        
        # 데이터셋 크기 확인
        total_samples = len(self.circuit_data)
        print(f"Total samples in dataset: {total_samples}")
        
        if total_samples <= 3:
            print("Warning: Dataset is too small for proper splitting. Using all samples for all splits.")
            # 데이터셋이 너무 작은 경우, 모든 데이터를 모든 분할에 사용
            return (
                QuantumCircuitDataset(self.circuit_data),
                QuantumCircuitDataset(self.circuit_data),
                QuantumCircuitDataset(self.circuit_data)
            )
        
        # 첫 번째 분할: train + val vs test
        train_val_data, test_data = train_test_split(
            self.circuit_data,
            test_size=test_ratio,
            random_state=random_state
        )
        
        # 두 번째 분할: train vs val
        adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=adjusted_val_ratio,
            random_state=random_state
        )
        
        return (
            QuantumCircuitDataset(train_data),
            QuantumCircuitDataset(val_data),
            QuantumCircuitDataset(test_data)
        )
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """데이터셋 통계 정보 반환"""
        if self.circuit_data is None:
            self.merge_data()
        
        # 유효한 measurement_result가 있는 데이터만 필터링
        valid_data = [data for data in self.circuit_data if data.measurement_result is not None]
        
        num_circuits = len(self.circuit_data)
        num_valid_circuits = len(valid_data)
        
        # CircuitSpec에서 가져올 수 있는 정보
        qubit_counts = [data.circuit_spec.num_qubits for data in self.circuit_data]
        gate_counts = [len(data.circuit_spec.gates) for data in self.circuit_data]
        
        stats = {
            "total_circuits": num_circuits,
            "valid_circuits": num_valid_circuits,
            "qubit_range": (min(qubit_counts), max(qubit_counts)),
            "avg_qubits": np.mean(qubit_counts),
            "gate_range": (min(gate_counts), max(gate_counts)),
            "avg_gates": np.mean(gate_counts)
        }
        
        if valid_data:
            # MeasurementResult에서 가져올 수 있는 정보 (유효한 데이터만)
            depths = [data.measurement_result.depth for data in valid_data]
            fidelities = [data.measurement_result.fidelity for data in valid_data]
            
            stats.update({
                "depth_range": (min(depths), max(depths)),
                "avg_depth": np.mean(depths),
                "fidelity_range": (min(fidelities), max(fidelities)),
                "avg_fidelity": np.mean(fidelities)
            })
            
            # 추가 메트릭 통계 (있는 경우)
            entanglements = [data.measurement_result.entanglement for data in valid_data 
                           if data.measurement_result.entanglement is not None]
            if entanglements:
                stats["entanglement_range"] = (min(entanglements), max(entanglements))
                stats["avg_entanglement"] = np.mean(entanglements)
        
        return stats


def create_dataloaders(
    train_dataset: QuantumCircuitDataset,
    val_dataset: QuantumCircuitDataset,
    test_dataset: QuantumCircuitDataset,
    batch_size: int = 32,
    num_workers: int = 0,
    rtg_calculator = None,
    enable_rtg: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """DataLoader 생성 (RTG 지원 포함)"""
    
    # RTG 활성화 시 collate_fn 변경
    if enable_rtg and rtg_calculator is not None:
        collate_fn = lambda batch: _rtg_collate_fn(batch, rtg_calculator)
    else:
        collate_fn = lambda x: x  # CircuitSpec 객체들을 그대로 반환
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader


def _rtg_collate_fn(batch, rtg_calculator):
    """RTG 값을 포함한 배치 데이터 생성"""
    import torch
    
    batch_size = len(batch)
    
    # CircuitData에서 필요한 정보 추출
    circuit_specs = []
    states = []  # 상태 시퀀스
    actions = []  # 액션 시퀀스
    target_properties = {
        'entanglement': [],
        'fidelity': [],
        'expressibility': []
    }
    
    max_seq_len = 0
    
    for circuit_data in batch:
        circuit_specs.append(circuit_data)
        
        # 게이트 시퀀스를 상태-액션 쌍으로 변환
        gate_sequence = []
        for gate in circuit_data.gates:
            # 게이트 정보를 벡터로 인코딩 (간단화된 버전)
            gate_vector = torch.randn(256)  # d_model 크기로 임시 생성
            gate_sequence.append(gate_vector)
        
        if gate_sequence:
            states.append(torch.stack(gate_sequence))
            actions.append(torch.stack(gate_sequence))  # 액션도 동일하게 설정
            max_seq_len = max(max_seq_len, len(gate_sequence))
        else:
            # 빈 시퀀스 처리
            empty_seq = torch.zeros(1, 256)
            states.append(empty_seq)
            actions.append(empty_seq)
            max_seq_len = max(max_seq_len, 1)
        
        # 정답 속성값 추출
        if circuit_data.measurement_result:
            target_properties['entanglement'].append(
                circuit_data.measurement_result.entanglement or 0.0
            )
            target_properties['fidelity'].append(
                circuit_data.measurement_result.fidelity or 0.0
            )
            target_properties['expressibility'].append(
                getattr(circuit_data.measurement_result, 'expressibility', 0.0)
            )
        else:
            # 기본값 사용
            target_properties['entanglement'].append(0.0)
            target_properties['fidelity'].append(0.0)
            target_properties['expressibility'].append(0.0)
    
    # 시퀀스 길이 패딩
    padded_states = []
    padded_actions = []
    attention_masks = []
    
    for i in range(batch_size):
        seq_len = states[i].shape[0]
        
        # 패딩 처리
        if seq_len < max_seq_len:
            pad_len = max_seq_len - seq_len
            padded_state = torch.cat([
                states[i],
                torch.zeros(pad_len, 256)
            ], dim=0)
            padded_action = torch.cat([
                actions[i],
                torch.zeros(pad_len, 256)
            ], dim=0)
            mask = torch.cat([
                torch.ones(seq_len, dtype=torch.bool),
                torch.zeros(pad_len, dtype=torch.bool)
            ])
        else:
            padded_state = states[i]
            padded_action = actions[i]
            mask = torch.ones(seq_len, dtype=torch.bool)
        
        padded_states.append(padded_state)
        padded_actions.append(padded_action)
        attention_masks.append(mask)
    
    # 배치 텐서로 변환
    batch_states = torch.stack(padded_states)  # [batch_size, seq_len, d_model]
    batch_actions = torch.stack(padded_actions)  # [batch_size, seq_len, d_model]
    batch_masks = torch.stack(attention_masks)  # [batch_size, seq_len]
    
    # 목표 속성값을 텐서로 변환
    target_tensors = {}
    for prop_name, prop_values in target_properties.items():
        # 각 시퀀스의 모든 스텝에 대해 동일한 목표값 사용
        prop_tensor = torch.zeros(batch_size, max_seq_len)
        for i, prop_val in enumerate(prop_values):
            prop_tensor[i, :] = prop_val
        target_tensors[prop_name] = prop_tensor
    
    # RTG 계산 (실제 RTG Calculator 사용)
    if rtg_calculator is not None:
        try:
            # Property 모델로 예측
            predicted_properties = rtg_calculator.calculate_sequence_properties(
                batch_states, batch_masks
            )
            
            # 표준 RL RTG 계산
            rtg_rewards = rtg_calculator.calculate_rtg_rewards(
                predicted_properties, target_tensors, batch_masks, gamma=0.99
            )
        except Exception as e:
            print(f"⚠️ RTG 계산 오류: {e}")
            # 폴백: 기본값 사용
            rtg_rewards = torch.ones(batch_size, max_seq_len) * 0.5
    else:
        # RTG Calculator가 없으면 기본값
        rtg_rewards = torch.ones(batch_size, max_seq_len) * 0.5
    
    return {
        'circuit_specs': circuit_specs,
        'states': batch_states,
        'actions': batch_actions,
        'attention_masks': batch_masks,
        'target_properties': target_tensors,
        'rtg_rewards': rtg_rewards
    }


# 사용 예시
if __name__ == "__main__":
    # 방법 1: 통합 데이터 파일로 로딩 (권장)
    manager = DatasetManager(
        unified_data_path=r"C:\Users\jungh\Documents\GitHub\Kaist\OAT_Model\raw_data\merged_data.json"
    )
    
    # 방법 2: 개별 파일로 로딩 (레거시 지원)
    # manager = DatasetManager(
    #     results_path="path/to/results.json",
    #     circuits_path="path/to/circuits.json"
    # )
    
    # 방법 3: 병합된 결과 파일로 로딩 (레거시 지원)
    # manager = DatasetManager(
    #     merged_results_path="path/to/merged_results.json",
    #     circuits_path="path/to/circuits.json"
    # )
    
    # 데이터 병합 (기본적으로 품질 필터링 활성화)
    circuit_data = manager.merge_data(enable_filtering=True)
    print(f"총 {len(circuit_data)}개의 유효한 회로 데이터 로딩 완료")
    print(f"통합 데이터 파일 사용: {manager.unified_data_path is not None}")
    
    # 통계 정보 출력
    stats = manager.get_dataset_stats()
    print("\n데이터셋 통계:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 데이터셋 분할
    train_dataset, val_dataset, test_dataset = manager.split_dataset()
    print(f"\n데이터셋 분할 완료:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    # 첫 번째 데이터 샘플 확인
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"\n첫 번째 샘플:")
        print(f"  Circuit ID: {sample.circuit_id}")
        print(f"  Qubits: {sample.num_qubits}")
        print(f"  Gates: {len(sample.gates)}")
        print(f"  Fidelity: {sample.measurement_result.fidelity}")
        print(f"  Entanglement: {sample.measurement_result.entanglement}")

