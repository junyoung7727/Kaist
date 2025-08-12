"""
Embedding Pipeline Module
CircuitSpec -> Grid Encoder -> Decision Transformer Embedding
"""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

# 임포트 경로 문제 해결
try:
    # 절대 경로 시도
    from data.quantum_circuit_dataset import CircuitSpec
    from encoding.grid_graph_encoder import GridGraphEncoder
    from encoding.Decision_Transformer_Embed import QuantumGateSequenceEmbedding
except ImportError:
    # 상대 경로 시도
    try:
        from .quantum_circuit_dataset import CircuitSpec
        from ..encoding.grid_graph_encoder import GridGraphEncoder
        from ..encoding.Decision_Transformer_Embed import QuantumGateSequenceEmbedding
    except ImportError:
        # 로컬 경로 시도
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from data.quantum_circuit_dataset import CircuitSpec
        from encoding.grid_graph_encoder import GridGraphEncoder
        from encoding.Decision_Transformer_Embed import QuantumGateSequenceEmbedding

# 🎆 NEW: 게이트 레지스트리 싱글톤 임포트
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))
from gates import QuantumGateRegistry

@dataclass
class EmbeddingConfig:
    """임베딩 설정"""
    d_model: int = 512
    n_gate_types: int = None  # NEW: gate vocab 싱글톤에서 자동 설정
    n_qubits: int = 10
    max_seq_len: int = 1000
    max_time_steps: int = 50

    def __post_init__(self):
        """초기화 후 gate 수를 싱글톤에서 가져오기"""
        if self.n_gate_types is None:
            self.n_gate_types = QuantumGateRegistry.get_singleton_gate_count()
            print(f" EmbeddingConfig: Using gate vocab singleton, n_gate_types = {self.n_gate_types}")


class EmbeddingPipeline:
    """완전한 임베딩 파이프라인"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        
        # 게이트 레지스트리 싱글톤 초기화
        self.gate_registry = QuantumGateRegistry()
        
        # 게이트 vocab 초기화
        self.gate_vocab = self.gate_registry.get_gate_vocab()
        
        # 게이트 수 확인 및 설정 동기화
        actual_gate_count = len(self.gate_vocab)
        if self.config.n_gate_types != actual_gate_count:
            print(f" Config mismatch: expected {self.config.n_gate_types}, got {actual_gate_count}")
            self.config.n_gate_types = actual_gate_count
        print(f" EmbeddingPipeline initialized with {actual_gate_count} gate types from singleton")
        
        # Grid Encoder 초기화
        self.grid_encoder = GridGraphEncoder()
        
        # Decision Transformer Embedding 초기화
        self.dt_embedding = QuantumGateSequenceEmbedding(
            d_model=config.d_model,
            n_gate_types=config.n_gate_types,
            n_qubits=config.n_qubits,
            max_seq_len=config.max_seq_len
        )
    
    def process_single_circuit(self, circuit_spec: CircuitSpec) -> Dict[str, torch.Tensor]:
        """단일 회로 처리"""
        
        # 1. Grid Encoder로 회로 인코딩
        encoded_data = self.grid_encoder.encode(circuit_spec)
        
        # 2. 인코딩된 데이터를 그리드 매트릭스로 변환
        grid_matrix_data = self.grid_encoder.to_grid_matrix(encoded_data)
        
        # 3. Decision Transformer Embedding 적용
        dt_results = self.dt_embedding.process_grid_matrix_data(grid_matrix_data)
        
        # 4. 메타데이터 추가
        dt_results.update({
            'circuit_id': circuit_spec.circuit_id,
            'num_qubits': circuit_spec.num_qubits,
            'num_gates': len(circuit_spec.gates)
        })
        
        return dt_results
    
    def process_batch(self, circuit_specs: List[CircuitSpec]) -> Dict[str, torch.Tensor]:
        """배치 처리"""
        batch_results = []
        
        for spec in circuit_specs:
            try:
                result = self.process_single_circuit(spec)
                batch_results.append(result)
            except Exception as e:
                print(f"Error processing circuit {spec.circuit_id}: {e}")
                continue
        
        if not batch_results:
            return {}
        
        # 배치 차원으로 합치기
        return self._combine_batch_results(batch_results)
    
    def _combine_batch_results(self, batch_results: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """배치 결과 합치기"""
        if not batch_results:
            return {}
        
        combined = {}
        
        # 텐서 데이터 합치기
        tensor_keys = ['input_sequence', 'attention_mask', 'action_prediction_mask', 
                      'state_embedded', 'action_embedded', 'reward_embedded', 'target_actions']
        
        for key in tensor_keys:
            if key in batch_results[0]:
                # 패딩을 통한 배치 처리 (가변 길이 지원)
                tensors = [result[key] for result in batch_results]
                combined[key] = self._pad_and_stack_tensors(tensors)
        
        # 메타데이터 합치기
        meta_keys = ['circuit_id', 'num_qubits', 'num_gates', 'episode_time_len', 'sar_sequence_len']
        for key in meta_keys:
            if key in batch_results[0]:
                combined[key] = [result[key] for result in batch_results]
        
        return combined
    
    def _pad_and_stack_tensors(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        서로 다른 크기의 텐서들을 패딩하여 배치로 합치기
        
        Args:
            tensors: 서로 다른 크기의 텐서 리스트
        
        Returns:
            패딩된 배치 텐서
        """
        if not tensors:
            return torch.tensor([])
        
        # 최대 크기 계산
        max_dims = []
        for dim in range(len(tensors[0].shape)):
            max_size = max(tensor.shape[dim] for tensor in tensors)
            max_dims.append(max_size)
        
        # 패딩된 텐서 리스트 생성
        padded_tensors = []
        for tensor in tensors:
            # 패딩 크기 계산
            padding = []
            for dim in reversed(range(len(tensor.shape))):
                pad_size = max_dims[dim] - tensor.shape[dim]
                padding.extend([0, pad_size])  # (left, right) padding
            
            # 패딩 적용
            if any(p > 0 for p in padding):
                padded_tensor = torch.nn.functional.pad(tensor, padding, value=0)
            else:
                padded_tensor = tensor
            
            padded_tensors.append(padded_tensor)
        
        # 배치 차원으로 합치기
        return torch.stack(padded_tensors, dim=0)


def create_embedding_pipeline(config: EmbeddingConfig = None) -> EmbeddingPipeline:
    """임베딩 파이프라인 팩토리 함수"""
    if config is None:
        config = EmbeddingConfig()
    
    return EmbeddingPipeline(config)


# 사용 예시
if __name__ == "__main__":
    from quantum_circuit_dataset import DatasetManager
    
    # 데이터셋 로딩
    manager = DatasetManager("OAT_Model/dataset/unified_batch_experiment_results_with_circuits.json")
    circuit_specs = manager.parse_circuits()
    
    # 임베딩 파이프라인 생성
    config = EmbeddingConfig(d_model=256)  # NEW: 싱글톤에서 가져온 gate 수로 임베딩 레이어 초기화
    pipeline = create_embedding_pipeline(config)
    
    # 단일 회로 테스트
    if circuit_specs:
        print("Testing single circuit processing...")
        result = pipeline.process_single_circuit(circuit_specs[0])
        
        print(f"Circuit ID: {result['circuit_id']}")
        print(f"Input sequence shape: {result['input_sequence'].shape}")
        print(f"Attention mask shape: {result['attention_mask'].shape}")
        
        # 배치 테스트
        print("\nTesting batch processing...")
        batch_result = pipeline.process_batch(circuit_specs[:3])
        
        if batch_result:
            print(f"Batch input sequence shape: {batch_result['input_sequence'].shape}")
            print(f"Batch attention mask shape: {batch_result['attention_mask'].shape}")
