"""
Quantum Circuit Generator using Trained Decision Transformer

학습된 Decision Transformer 모델을 사용하여 양자 회로를 생성하는 모듈
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import sys
import json
from dataclasses import dataclass

# 프로젝트 모듈 임포트
sys.path.append(str(Path(__file__).parent.parent))
from models.decision_transformer import DecisionTransformer
from data.embedding_pipeline import EmbeddingPipeline, EmbeddingConfig
from data.quantum_circuit_dataset import CircuitSpec

# quantumcommon 모듈 임포트
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))
from gates import QuantumGateRegistry, GateOperation


@dataclass
class GenerationConfig:
    """회로 생성 설정"""
    max_circuit_length: int = 50
    target_num_qubits: int = 4
    temperature: float = 1.0
    top_k: int = 10
    top_p: float = 0.9
    do_sample: bool = True
    
    # 목표 메트릭 (선택적)
    target_fidelity: Optional[float] = None
    target_entanglement: Optional[float] = None
    target_expressibility: Optional[float] = None


class QuantumCircuitGenerator:
    """학습된 Decision Transformer를 사용한 양자 회로 생성기"""
    
    def __init__(self, 
                 model_path: str,
                 config: GenerationConfig = None):
        self.config = config or GenerationConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 게이트 레지스트리 초기화
        self.gate_registry = QuantumGateRegistry()
        self.gate_vocab = self.gate_registry.get_gate_vocab()
        self.idx_to_gate = {idx: gate for gate, idx in self.gate_vocab.items()}
        
        # 모델 로드
        self.model = self._load_model(model_path)
        
        # 임베딩 파이프라인 초기화
        self.embedding_pipeline = self._create_embedding_pipeline()
        
        print(f"QuantumCircuitGenerator initialized on {self.device}")
        print(f"Gate vocabulary: {len(self.gate_vocab)} gates")
    
    def _load_model(self, model_path: str) -> DecisionTransformer:
        """학습된 모델 로드"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 모델 설정 추출
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
        else:
            # 기본 설정 사용
            model_config = {
                'd_model': 512,
                'n_layers': 6,
                'n_heads': 8,
                'n_gate_types': len(self.gate_vocab),
                'dropout': 0.1
            }
        
        # 모델 생성 및 가중치 로드
        model = DecisionTransformer(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded from {model_path}")
        return model
    
    def _create_embedding_pipeline(self) -> EmbeddingPipeline:
        """임베딩 파이프라인 생성"""
        embedding_config = EmbeddingConfig(
            d_model=512,
            n_gate_types=len(self.gate_vocab),
            n_qubits=self.config.target_num_qubits,
            max_seq_len=self.config.max_circuit_length * 3  # S-A-R 패턴
        )
        return EmbeddingPipeline(embedding_config)
    
    def generate_circuit(self, 
                        initial_state: Optional[Dict] = None,
                        target_metrics: Optional[Dict] = None) -> CircuitSpec:
        """
        양자 회로 생성
        
        Args:
            initial_state: 초기 상태 (선택적)
            target_metrics: 목표 메트릭 (fidelity, entanglement 등)
        
        Returns:
            생성된 CircuitSpec
        """
        print(f"Generating quantum circuit...")
        print(f"Target qubits: {self.config.target_num_qubits}")
        print(f"Max length: {self.config.max_circuit_length}")
        
        # 초기 상태 설정
        if initial_state is None:
            initial_state = self._create_initial_state()
        
        # 생성된 게이트들
        generated_gates = []
        
        # 현재 상태 (State-Action-Reward 시퀀스)
        current_sequence = self._initialize_sequence(initial_state, target_metrics)
        
        # 순차적으로 게이트 생성
        for step in range(self.config.max_circuit_length):
            # 다음 게이트 예측
            next_gate = self._predict_next_gate(current_sequence, step)
            
            if next_gate is None or next_gate.name == '[EOS]':
                print(f"Circuit generation completed at step {step}")
                break
            
            # 게이트 추가
            generated_gates.append(next_gate)
            
            # 시퀀스 업데이트
            current_sequence = self._update_sequence(current_sequence, next_gate, step)
            
            print(f"Step {step}: Generated {next_gate.name} on qubits {next_gate.qubits}")
        
        # CircuitSpec 생성
        circuit_spec = CircuitSpec(
            circuit_id=f"generated_circuit_{len(generated_gates)}_gates",
            num_qubits=self.config.target_num_qubits,
            gates=generated_gates
        )
        
        print(f"Circuit generation completed: {len(generated_gates)} gates")
        return circuit_spec
    
    def _create_initial_state(self) -> Dict:
        """초기 상태 생성"""
        return {
            'num_qubits': self.config.target_num_qubits,
            'circuit_depth': 0,
            'gate_count': 0
        }
    
    def _initialize_sequence(self, initial_state: Dict, target_metrics: Optional[Dict]) -> torch.Tensor:
        """초기 시퀀스 생성 (State-Action-Reward 패턴)"""
        # 빈 회로로 시작
        empty_circuit = CircuitSpec(
            circuit_id="initial",
            num_qubits=self.config.target_num_qubits,
            gates=[]
        )
        
        # 임베딩 파이프라인을 통해 초기 시퀀스 생성
        embedded_data = self.embedding_pipeline.process_single_circuit(empty_circuit)
        
        # 초기 시퀀스 추출
        initial_sequence = embedded_data['input_sequence']  # [1, seq_len, d_model]
        
        return initial_sequence.to(self.device)
    
    def _predict_next_gate(self, current_sequence: torch.Tensor, step: int) -> Optional[GateOperation]:
        """다음 게이트 예측"""
        with torch.no_grad():
            # 어텐션 마스크 생성
            seq_len = current_sequence.shape[1]
            attention_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).to(self.device)
            
            # 액션 예측 마스크 (Action 위치에서만 예측)
            action_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=self.device)
            action_positions = list(range(1, seq_len, 3))  # 1, 4, 7, 10, ... (Action 위치)
            if action_positions:
                action_mask[0, action_positions] = True
            
            # 모델 예측
            outputs = self.model(
                input_sequence=current_sequence,
                attention_mask=attention_mask,
                action_prediction_mask=action_mask
            )
            
            # 마지막 Action 위치의 로짓 추출
            action_logits = outputs['action_logits']  # [1, seq_len, n_gate_types]
            
            if action_positions:
                last_action_pos = action_positions[-1]
                if last_action_pos < seq_len:
                    logits = action_logits[0, last_action_pos, :]  # [n_gate_types]
                else:
                    # 시퀀스 끝에 도달
                    return None
            else:
                return None
            
            # 샘플링을 통한 게이트 선택
            gate_id = self._sample_gate_id(logits)
            
            # 게이트 ID를 GateOperation으로 변환
            return self._gate_id_to_operation(gate_id)
    
    def _sample_gate_id(self, logits: torch.Tensor) -> int:
        """로짓에서 게이트 ID 샘플링"""
        if not self.config.do_sample:
            # 그리디 선택
            return torch.argmax(logits).item()
        
        # 온도 적용
        logits = logits / self.config.temperature
        
        # Top-k 필터링
        if self.config.top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, min(self.config.top_k, logits.size(-1)))
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(-1, top_k_indices, top_k_logits)
        
        # Top-p (nucleus) 필터링
        if self.config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 누적 확률이 top_p를 초과하는 토큰들 제거
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')
        
        # 확률 분포에서 샘플링
        probs = F.softmax(logits, dim=-1)
        gate_id = torch.multinomial(probs, 1).item()
        
        return gate_id
    
    def _gate_id_to_operation(self, gate_id: int) -> Optional[GateOperation]:
        """게이트 ID를 GateOperation으로 변환"""
        if gate_id not in self.idx_to_gate:
            return None
        
        gate_name = self.idx_to_gate[gate_id]
        
        # 특수 토큰 처리
        if gate_name in ['[EOS]', '[PAD]', '[EMPTY]']:
            return GateOperation(name=gate_name, qubits=[], parameters=[])
        
        # 🚨 CRITICAL FIX: gate_registry.get_gate_info() 메서드가 존재하지 않음!
        # 올바른 메서드 사용
        try:
            gate_def = self.gate_registry.get_gate(gate_name)
            if gate_def is None:
                return None
            
            # 큐빗 선택
            required_qubits = gate_def.num_qubits
            qubits = self._select_qubits(required_qubits)
            
            # 파라미터 생성
            required_params = gate_def.num_parameters
            parameters = self._generate_parameters(required_params)
            
        except Exception as e:
            print(f"Warning: Failed to get gate info for {gate_name}: {e}")
            return None
        
        return GateOperation(
            name=gate_name,
            qubits=qubits,
            parameters=parameters
        )
    
    def _select_qubits(self, num_qubits: int) -> List[int]:
        """게이트에 필요한 큐빗 선택"""
        available_qubits = list(range(self.config.target_num_qubits))
        
        if num_qubits == 1:
            # 단일 큐빗 게이트
            return [np.random.choice(available_qubits)]
        elif num_qubits == 2:
            # 2큐빗 게이트
            selected = np.random.choice(available_qubits, size=2, replace=False)
            return selected.tolist()
        else:
            # 다중 큐빗 게이트
            selected = np.random.choice(available_qubits, size=min(num_qubits, len(available_qubits)), replace=False)
            return selected.tolist()
    
    def _generate_parameters(self, num_params: int) -> List[float]:
        """게이트 파라미터 생성"""
        if num_params == 0:
            return []
        
        # 0 ~ 2π 범위의 랜덤 파라미터
        parameters = []
        for _ in range(num_params):
            param = np.random.uniform(0, 2 * np.pi)
            parameters.append(param)
        
        return parameters
    
    def _update_sequence(self, current_sequence: torch.Tensor, new_gate: GateOperation, step: int) -> torch.Tensor:
        """시퀀스에 새 게이트 추가 - CRITICAL FIX"""
        # 🚨 CRITICAL: 이전 구현은 완전히 잘못됨!
        # State-Action-Reward 패턴으로 시퀀스를 실제로 업데이트해야 함
        
        try:
            # 현재 시퀀스에서 회로 상태 추출
            batch_size, seq_len, d_model = current_sequence.shape
            
            # 새 게이트를 포함한 임시 회로 생성
            temp_circuit = CircuitSpec(
                circuit_id=f"temp_step_{step}",
                num_qubits=self.config.target_num_qubits,
                gates=[new_gate]  # 새 게이트만 포함
            )
            
            # 임베딩 파이프라인을 통해 새 시퀀스 생성
            embedded_data = self.embedding_pipeline.process_single_circuit(temp_circuit)
            new_sequence = embedded_data['input_sequence'].to(self.device)
            
            # 기존 시퀀스와 새 시퀀스 연결
            # State-Action-Reward 패턴 유지
            if seq_len + new_sequence.shape[1] <= self.config.max_circuit_length * 3:
                updated_sequence = torch.cat([current_sequence, new_sequence], dim=1)
            else:
                # 최대 길이 초과시 기존 시퀀스 유지
                updated_sequence = current_sequence
            
            return updated_sequence
            
        except Exception as e:
            print(f"Warning: Failed to update sequence: {e}")
            # 실패시 기존 시퀀스 반환
            return current_sequence
    
    def generate_multiple_circuits(self, 
                                 num_circuits: int = 5,
                                 target_metrics: Optional[Dict] = None) -> List[CircuitSpec]:
        """여러 회로 생성"""
        circuits = []
        
        for i in range(num_circuits):
            print(f"\nGenerating circuit {i+1}/{num_circuits}")
            circuit = self.generate_circuit(target_metrics=target_metrics)
            circuits.append(circuit)
        
        return circuits
    
    def save_circuits(self, circuits: List[CircuitSpec], output_path: str):
        """생성된 회로들을 JSON 파일로 저장"""
        circuits_data = []
        
        for circuit in circuits:
            circuit_data = {
                'circuit_id': circuit.circuit_id,
                'num_qubits': circuit.num_qubits,
                'gates': [
                    {
                        'name': gate.name,
                        'qubits': gate.qubits,
                        'parameters': gate.parameters
                    }
                    for gate in circuit.gates
                ]
            }
            circuits_data.append(circuit_data)
        
        with open(output_path, 'w') as f:
            json.dump(circuits_data, f, indent=2)
        
        print(f"Saved {len(circuits)} circuits to {output_path}")


def main():
    """사용 예시"""
    # 설정
    config = GenerationConfig(
        max_circuit_length=20,
        target_num_qubits=4,
        temperature=0.8,
        top_k=10,
        do_sample=True
    )
    
    # 모델 경로 (학습된 체크포인트)
    model_path = "checkpoints/best_model.pt"
    
    # 생성기 초기화
    generator = QuantumCircuitGenerator(model_path, config)
    
    # 목표 메트릭 설정 (선택적)
    target_metrics = {
        'target_fidelity': 0.8,
        'target_entanglement': 0.6
    }
    
    # 회로 생성
    circuits = generator.generate_multiple_circuits(
        num_circuits=5,
        target_metrics=target_metrics
    )
    
    # 결과 저장
    generator.save_circuits(circuits, "generated_circuits.json")
    
    # 생성된 회로 정보 출력
    for i, circuit in enumerate(circuits):
        print(f"\nCircuit {i+1}:")
        print(f"  Qubits: {circuit.num_qubits}")
        print(f"  Gates: {len(circuit.gates)}")
        for j, gate in enumerate(circuit.gates[:5]):  # 처음 5개 게이트만 출력
            print(f"    {j+1}. {gate.name} on qubits {gate.qubits}")
        if len(circuit.gates) > 5:
            print(f"    ... and {len(circuit.gates) - 5} more gates")


if __name__ == "__main__":
    main()
