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
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent / "quantumcommon"))
from gates import gate_registry, GateType, GateOperation


@dataclass
class CircuitSpec:
    """양자 회로 스펙"""
    circuit_id: str
    num_qubits: int
    gates: List[GateOperation]
    
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
        
        return cls(
            circuit_id=data['circuit_id'],
            num_qubits=data['num_qubits'],
            gates=gates
        )


class QuantumCircuitDataset(Dataset):
    """양자 회로 데이터셋"""
    
    def __init__(self, circuit_specs: List[CircuitSpec]):
        self.circuit_specs = circuit_specs
    
    def __len__(self) -> int:
        return len(self.circuit_specs)
    
    def __getitem__(self, idx: int) -> CircuitSpec:
        return self.circuit_specs[idx]


class DatasetManager:
    """데이터셋 관리자 - 로딩, 분할, 변환 담당"""
    
    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        self.raw_data = None
        self.circuit_specs = None
    
    def load_data(self) -> Dict[str, Any]:
        """JSON 데이터 로딩"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        return self.raw_data
    
    def parse_circuits(self) -> List[CircuitSpec]:
        """JSON에서 CircuitSpec 객체로 변환"""
        if self.raw_data is None:
            self.load_data()
        
        circuit_specs = []
        circuits_data = self.raw_data.get('circuits', {})
        
        for circuit_id, circuit_data in circuits_data.items():
            spec = CircuitSpec.from_dict(circuit_data)
            circuit_specs.append(spec)
        
        self.circuit_specs = circuit_specs
        return circuit_specs
    
    def split_dataset(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ) -> Tuple["QuantumCircuitDataset", "QuantumCircuitDataset", "QuantumCircuitDataset"]:
        """데이터셋 분할"""
        if self.circuit_specs is None:
            self.parse_circuits()
        
        # 비율 검증
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "비율의 합이 1이 되어야 합니다"
        
        # 데이터셋 크기 확인
        total_samples = len(self.circuit_specs)
        print(f"Total samples in dataset: {total_samples}")
        
        if total_samples <= 3:
            print("Warning: Dataset is too small for proper splitting. Using all samples for all splits.")
            # 데이터셋이 너무 작은 경우, 모든 데이터를 모든 분할에 사용
            return (
                QuantumCircuitDataset(self.circuit_specs),
                QuantumCircuitDataset(self.circuit_specs),
                QuantumCircuitDataset(self.circuit_specs)
            )
        
        # 최소 샘플 수 계산
        min_train = max(1, int(total_samples * train_ratio))
        min_val = max(1, int(total_samples * val_ratio))
        min_test = max(1, total_samples - min_train - min_val)
        
        if min_train + min_val + min_test > total_samples:
            print(f"Warning: Adjusting split to ensure at least 1 sample per split. "
                  f"New ratio - Train: {min_train}/{total_samples}, "
                  f"Val: {min_val}/{total_samples}, Test: {min_test}/{total_samples}")
            
            # 첫 번째 분할: train vs (val + test)
            train_specs = self.circuit_specs[:min_train]
            remaining = self.circuit_specs[min_train:]
            
            # 두 번째 분할: val vs test
            if len(remaining) >= min_val + min_test:
                val_specs = remaining[:min_val]
                test_specs = remaining[min_val:min_val+min_test]
            else:
                # 남은 샘플이 부족한 경우
                val_specs = remaining[:min_val] if len(remaining) >= min_val else remaining
                test_specs = val_specs  # 테스트와 검증이 동일하게 설정
        else:
            # 정상적인 분할 진행
            try:
                # 첫 번째 분할: train vs (val + test)
                train_specs, temp_specs = train_test_split(
                    self.circuit_specs,
                    test_size=(val_ratio + test_ratio),
                    random_state=random_state
                )
                
                # 두 번째 분할: val vs test
                val_specs, test_specs = train_test_split(
                    temp_specs,
                    test_size=test_ratio / (val_ratio + test_ratio),
                    random_state=random_state
                )
            except ValueError as e:
                print(f"Error during dataset split: {e}")
                print("Falling back to manual split to ensure at least one sample per split")
                
                # 수동 분할
                train_size = max(1, int(total_samples * 0.7))
                val_size = max(1, int(total_samples * 0.15))
                
                train_specs = self.circuit_specs[:train_size]
                val_specs = self.circuit_specs[train_size:train_size+val_size]
                test_specs = self.circuit_specs[train_size+val_size:]
                
                # 테스트 샘플이 없는 경우 검증 샘플 재사용
                if not test_specs:
                    test_specs = val_specs
        
        print(f"Split sizes - Train: {len(train_specs)}, Val: {len(val_specs)}, Test: {len(test_specs)}")
        
        return (
            QuantumCircuitDataset(train_specs),
            QuantumCircuitDataset(val_specs),
            QuantumCircuitDataset(test_specs)
        )
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """데이터셋 정보 반환"""
        if self.circuit_specs is None:
            self.parse_circuits()
        
        num_qubits_list = [spec.num_qubits for spec in self.circuit_specs]
        gate_counts = [len(spec.gates) for spec in self.circuit_specs]
        
        gate_types = set()
        for spec in self.circuit_specs:
            for gate in spec.gates:
                gate_types.add(gate.name)
        
        return {
            'total_circuits': len(self.circuit_specs),
            'num_qubits_range': (min(num_qubits_list), max(num_qubits_list)),
            'gate_count_range': (min(gate_counts), max(gate_counts)),
            'unique_gate_types': sorted(list(gate_types)),
            'avg_gates_per_circuit': np.mean(gate_counts)
        }


def create_dataloaders(
    train_dataset: QuantumCircuitDataset,
    val_dataset: QuantumCircuitDataset,
    test_dataset: QuantumCircuitDataset,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """DataLoader 생성"""
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: x  # CircuitSpec 객체들을 그대로 반환
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x
    )
    
    return train_loader, val_loader, test_loader


# 사용 예시
if __name__ == "__main__":
    # 데이터셋 매니저 생성
    manager = DatasetManager("../data/unified_batch_experiment_results_with_circuits.json")
    
    # 데이터 로딩 및 파싱
    manager.load_data()
    circuit_specs = manager.parse_circuits()
    
    # 데이터셋 정보 출력
    info = manager.get_dataset_info()
    print("Dataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 데이터셋 분할
    train_ds, val_ds, test_ds = manager.split_dataset()
    
    print(f"\nDataset Split:")
    print(f"  Train: {len(train_ds)} circuits")
    print(f"  Val: {len(val_ds)} circuits")
    print(f"  Test: {len(test_ds)} circuits")
    
    # DataLoader 생성
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, batch_size=4
    )
    
    # 첫 번째 배치 확인
    for batch in train_loader:
        print(f"\nFirst batch: {len(batch)} circuits")
        for i, spec in enumerate(batch):
            print(f"  Circuit {i}: {spec.circuit_id}, {spec.num_qubits} qubits, {len(spec.gates)} gates")
        break
