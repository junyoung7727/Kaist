"""
Quantum Circuit Generator using Trained Decision Transformer

학습된 Decision Transformer 모델을 사용하여 양자 회로를 생성하는 모듈
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import sys
import json
import math
from dataclasses import dataclass
import time

# 프로젝트 모듈 임포트
sys.path.append(str(Path(__file__).parent.parent))
from models.decision_transformer import DecisionTransformer
from data.embedding_pipeline import EmbeddingPipeline, EmbeddingConfig
from data.quantum_circuit_dataset import CircuitSpec
from utils.debug_utils import debug_print

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
    
    # 생성 제어 옵션
    use_target_metrics: bool = True    # 목표 메트릭 사용 여부
    metrics_weight: float = 1.0       # 메트릭 가중치 (0.0 ~ 2.0)
    verbose: bool = True              # 자세한 출력 여부


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
        
        # 특수 토큰 ID 저장
        self.eos_token_id = self.gate_vocab.get('[EOS]', len(self.gate_vocab) - 1)
        self.pad_token_id = self.gate_vocab.get('[PAD]', len(self.gate_vocab) - 2)
        self.empty_token_id = self.gate_vocab.get('[EMPTY]', len(self.gate_vocab) - 3)
        
        # 모델 로드
        self.model = self._load_model(model_path)
        
        # 임베딩 파이프라인 초기화
        self.embedding_pipeline = self._create_embedding_pipeline()
        
        # 상태 추적용 임시 변수
        self._current_circuit = None
        
        if self.config.verbose:
            print(f"QuantumCircuitGenerator initialized on {self.device}")
            print(f"Gate vocabulary: {len(self.gate_vocab)} gates")
            if self.config.use_target_metrics:
                print(f"Target metrics enabled with weight: {self.config.metrics_weight}")
                metrics = []
                if self.config.target_fidelity is not None:
                    metrics.append(f"fidelity={self.config.target_fidelity:.2f}")
                if self.config.target_entanglement is not None:
                    metrics.append(f"entanglement={self.config.target_entanglement:.2f}")
                if self.config.target_expressibility is not None:
                    metrics.append(f"expressibility={self.config.target_expressibility:.2f}")
                if metrics:
                    print(f"Default targets: {', '.join(metrics)}")
    
    def _load_model(self, model_path: str) -> DecisionTransformer:
        """학습된 모델 로드"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 모델 설정 추출 (체크포인트 우선)
        ckpt_cfg = None
        if 'model_config' in checkpoint:
            ckpt_cfg = checkpoint['model_config']
        elif 'config' in checkpoint:
            ckpt_cfg = checkpoint['config']

        if ckpt_cfg is not None:
            model_config = {
                'd_model': ckpt_cfg.get('d_model', 512),
                'n_layers': ckpt_cfg.get('n_layers', 6),
                'n_heads': ckpt_cfg.get('n_heads', 8),
                'n_gate_types': ckpt_cfg.get('n_gate_types', len(self.gate_vocab)),
                'dropout': ckpt_cfg.get('dropout', 0.1),
            }
            print(f"[Model Load] Using checkpoint config: n_gate_types={model_config['n_gate_types']} | registry={len(self.gate_vocab)}")
        else:
            # 기본 설정 (체크포인트에 설정이 없는 경우)
            model_config = {
                'd_model': 512,
                'n_layers': 6,
                'n_heads': 8,
                'n_gate_types': len(self.gate_vocab),
                'dropout': 0.1
            }
            print(f"[Model Load] No config in checkpoint. Falling back to registry gate count={len(self.gate_vocab)}")
        
        # 모델 생성 및 가중치 로드
        model = DecisionTransformer(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded from {model_path}")
        return model
    
    def _create_embedding_pipeline(self) -> EmbeddingPipeline:
        """임베딩 파이프라인 생성"""
        embedding_config = EmbeddingConfig()
        return EmbeddingPipeline(config=embedding_config)
        
    def _encode_target_metrics(self, target_metrics: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """Target metrics (fidelity, entanglement, expressibility)를 모델 입력으로 인코딩"""
        if target_metrics is None:
            target_metrics = {}
        
        # 기본값으로 각 메트릭을 0.5로 설정 (중간값)
        metrics = {
            'fidelity': target_metrics.get('target_fidelity', self.config.target_fidelity or 0.5),
            'entanglement': target_metrics.get('target_entanglement', self.config.target_entanglement or 0.5),
            'expressibility': target_metrics.get('target_expressibility', self.config.target_expressibility or 0.5),
        }
        
        # 모델의 d_model 차원에 맞는 임베딩 생성
        d_model = self.embedding_pipeline.config.d_model
        
        # 각 메트릭을 1차원 벡터로 임베딩 (단순 반복)
        metric_values = torch.tensor([
            metrics['fidelity'], 
            metrics['entanglement'],
            metrics['expressibility']
        ], device=self.device).float()
        
        # 메트릭 임베딩을 더 큰 차원으로 확장
        constraints_emb = torch.zeros((1, 3, d_model), device=self.device)
        
        # 각 메트릭 타입별로 다른 포지션에 값 할당
        for i, val in enumerate(metric_values):
            constraints_emb[0, i, :] = val
        
        if self.config.verbose:
            print(f"Target metrics encoded: fidelity={metrics['fidelity']:.2f}, " 
                  f"entanglement={metrics['entanglement']:.2f}, "
                  f"expressibility={metrics['expressibility']:.2f}")
        
        return constraints_emb
        
    def _initialize_sequence(self, target_metrics: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """초기 시퀀스 생성 (비어있는 회로 + 목표 메트릭 조건부)"""
        # 비어있는 회로로 시작
        empty_circuit = CircuitSpec(
            circuit_id="initialization",
            num_qubits=self.config.target_num_qubits,
            gates=[]
        )
        
        # 임베딩 파이프라인을 통한 초기 회로 임베딩
        embedded_data = self.embedding_pipeline.process_single_circuit(empty_circuit)
        
        # 현재 회로 상태 초기화
        self._current_circuit = empty_circuit
        
        return embedded_data['input_sequence'].to(self.device)
    
    def generate_circuit(self, 
                        initial_state: Optional[Dict] = None,
                        target_metrics: Optional[Dict] = None) -> CircuitSpec:
        """
        양자 회로 생성
        
        Args:
            initial_state: 초기 상태 (선택적)
            target_metrics: 목표 메트릭 (fidelity, entanglement, expressibility)
            
        Returns:
            생성된 CircuitSpec
        """
        start_time = time.time()
        if self.config.verbose:
            print(f"Generating quantum circuit...")
            print(f"Target qubits: {self.config.target_num_qubits}")
            print(f"Max length: {self.config.max_circuit_length}")
            
            if target_metrics and self.config.use_target_metrics:
                print(f"Target metrics:")
                for name, value in target_metrics.items():
                    print(f"  - {name}: {value:.4f}")
        
        # 새 회로 생성을 위한 임시 상태 초기화
        self._current_circuit = None
        
        # 초기 시퀀스 생성
        current_sequence = self._initialize_sequence(target_metrics)
        
        # 생성된 게이트들
        generated_gates = []
        
        # 순차적으로 게이트 생성
        for step in range(self.config.max_circuit_length):
            # 다음 게이트 예측 - 메트릭 조건부 방식
            next_gate = self._predict_next_gate(current_sequence, target_metrics)
            
            if next_gate is None:
                if self.config.verbose:
                    print(f"Failed to predict gate at step {step}")
                break
                
            # EOS 토큰 처리
            if next_gate.name in ['[EOS]', '[PAD]', '[EMPTY]']:
                if self.config.verbose:
                    print(f"EOS token encountered at step {step}")
                break
            
            # 게이트 추가
            generated_gates.append(next_gate)
            
            if self.config.verbose:
                param_str = ""
                if next_gate.parameters:
                    param_str = f" with parameters [{', '.join([f'{p:.4f}' for p in next_gate.parameters])}]"
                print(f"Step {step}: Generated {next_gate.name} on qubits {next_gate.qubits}{param_str}")
            
            # 시퀀스 업데이트 - SAR 패턴 유지
            current_sequence = self._update_sequence(current_sequence, next_gate)
        
        # 최종 회로 생성
        timestamp = int(time.time())
        metrics_suffix = ""
        if target_metrics and self.config.use_target_metrics:
            # 목표 메트릭 정보를 회로 ID에 추가
            metrics = []
            if 'target_fidelity' in target_metrics:
                metrics.append(f"f{target_metrics['target_fidelity']:.2f}")
            if 'target_entanglement' in target_metrics:
                metrics.append(f"e{target_metrics['target_entanglement']:.2f}")
            if 'target_expressibility' in target_metrics:
                metrics.append(f"x{target_metrics['target_expressibility']:.2f}")
            
            if metrics:
                metrics_suffix = "_" + "_".join(metrics)
        
        circuit_id = f"generated_circuit_{len(generated_gates)}g_{self.config.target_num_qubits}q{metrics_suffix}_{timestamp}"
        
        circuit_spec = CircuitSpec(
            circuit_id=circuit_id,
            num_qubits=self.config.target_num_qubits,
            gates=generated_gates
        )
        
        elapsed = time.time() - start_time
        print(f"Circuit generation completed: {len(generated_gates)} gates in {elapsed:.2f} seconds")
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
    
    def _predict_next_gate(self, current_sequence: torch.Tensor, target_metrics: Optional[Dict] = None) -> Optional[GateOperation]:
        """다음 게이트 예측"""
        with torch.no_grad():
            # 어텐션 마스크 생성
            seq_len = current_sequence.shape[1]
            attention_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).unsqueeze(0).to(self.device)
            
            # 타겟 메트릭 인코딩 (제약 조건)
            circuit_constraints = None
            if self.config.use_target_metrics and target_metrics:
                circuit_constraints = self._encode_target_metrics(target_metrics)
            
            # predict_next_action 메서드 사용 (단일 액션 예측에 최적화)
            outputs = self.model.predict_next_action(
                input_sequence=current_sequence,
                attention_mask=attention_mask,
                circuit_constraints=circuit_constraints
            )
            
            # 게이트 타입, 큐빗 위치, 파라미터 예측값 추출
            gate_logits = outputs['gate_logits'][0]  # [vocab_size]
            position_preds = outputs['position_preds'][0]  # [max_qubits_per_gate]
            param_value = outputs['param_value'][0].item()  # 단일 값
            
            if self.config.verbose:
                # 샘플링 전 가장 높은 확률의 게이트 표시
                top_k_values, top_k_indices = torch.topk(F.softmax(gate_logits, dim=-1), k=3)
                top_gates = [(self.idx_to_gate.get(idx.item(), "UNK"), prob.item()) for idx, prob in zip(top_k_indices, top_k_values)]
                print(f"Top predicted gates: {top_gates}")
            
            # 샘플링을 통한 게이트 선택
            gate_id = self._sample_gate_id(gate_logits)
            
            # 게이트 ID를 GateOperation으로 변환 (모델 예측 사용)
            return self._gate_id_to_operation(gate_id, position_preds, param_value)
    
    def _sample_gate_id(self, logits: torch.Tensor) -> int:
        """게이트 샘플링 (temperature, top-k, top-p)"""
        if not self.config.do_sample:  # 그리디 샘플링
            return logits.argmax().item()
            
        # 온도 스케일링
        if self.config.temperature != 1.0:
            logits = logits / self.config.temperature
        
        # Top-K 샘플링
        if self.config.top_k > 0:
            top_k = min(self.config.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
            
        # Top-p (nucleus) 샘플링
        if 0.0 < self.config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 임계값보다 큰 확률 제거
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            # 첫 번째 토큰은 항상 유지
            sorted_indices_to_remove[0] = False
            
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove[sorted_indices[sorted_indices_to_remove]] = True
            logits[indices_to_remove] = float('-inf')
            
        # 소프트맥스 -> 샘플링
        probs = F.softmax(logits, dim=-1)
        gate_id = torch.multinomial(probs, 1).item()
        
        return gate_id
        
    def _select_qubits_from_predictions(self, position_preds: torch.Tensor, required_qubits: int) -> List[int]:
        """모델 예측에서 큐빗 위치 선택"""
        # 예측된 위치 벡터를 정규화된 확률로 변환
        max_qubits = self.config.target_num_qubits
        
        # 위치 예측을 적절한 크기로 자르기
        position_preds = position_preds[:max_qubits]
        
        # 위치 예측을 0-1 범위로 시그모이드 변환
        position_probs = torch.sigmoid(position_preds)
        
        if not self.config.do_sample:
            # 그리디: 가장 높은 확률의 큐빗 선택
            _, top_indices = torch.topk(position_probs, k=required_qubits)
            return sorted([idx.item() for idx in top_indices])
        
        # 샘플링: 확률에 따라 큐빗 선택
        selected_qubits = []
        remaining_positions = list(range(max_qubits))
        remaining_probs = position_probs.clone()
        
        # 필요한 수의 큐빗 샘플링
        for _ in range(required_qubits):
            if not remaining_positions:
                # 남은 위치가 없으면 랜덤 선택 (백업)
                return np.random.choice(max_qubits, size=required_qubits, replace=False).tolist()
            
            # 남은 위치에서 확률에 따라 샘플링
            probs = remaining_probs[remaining_positions]
            probs = F.softmax(probs, dim=-1)
            
            # 샘플링 및 선택된 큐빗 추가
            selected_idx = torch.multinomial(probs, 1).item()
            qubit_idx = remaining_positions[selected_idx]
            selected_qubits.append(qubit_idx)
            
            # 선택된 위치 제거
            remaining_positions.remove(qubit_idx)
        
        return sorted(selected_qubits)
        
    def _generate_parameters_from_predictions(self, param_value: float, required_params: int) -> List[float]:
        """모델 예측에서 게이트 파라미터 생성"""
        if required_params == 0:
            return []
        
        # 첫 번째 파라미터는 모델 예측 그대로 사용
        params = [param_value]
        
        # 추가 파라미터는 첫 파라미터에서 변형하여 생성
        # (현재는 간단히 구현 - 실제로는 다중 파라미터 예측이 필요할 수 있음)
        for i in range(1, required_params):
            # 파라미터 값을 약간 변형 (이 부분은 모델이 다중 파라미터를 예측하도록 개선 필요)
            additional_param = param_value * (1.0 + 0.1 * i)
            params.append(additional_param)
        
        return params
    
    def _gate_id_to_operation(self, gate_id: int, position_preds: Optional[torch.Tensor] = None, param_value: Optional[float] = None) -> Optional[GateOperation]:
        """게이트 ID를 GateOperation으로 변환 - 모델 예측 사용"""
        if gate_id not in self.idx_to_gate:
            return None
        
        gate_name = self.idx_to_gate[gate_id]
        
        # 특수 토큰 처리
        if gate_name in ['[EOS]', '[PAD]', '[EMPTY]']:
            return GateOperation(name=gate_name, qubits=[], parameters=[])
            
        try:
            gate_def = self.gate_registry.get_gate(gate_name)
            if gate_def is None:
                return None
            
            # 필요한 큐빗 수
            required_qubits = gate_def.num_qubits
            
            # 모델 예측에서 큐빗 위치 추출 또는 랜덤 선택
            if position_preds is not None:
                qubits = self._select_qubits_from_predictions(position_preds, required_qubits)
            else:
                # 큐빗 포지션 랜덤 샘플링 (대체 방법)
                qubits = np.random.choice(
                    self.config.target_num_qubits, 
                    size=required_qubits, 
                    replace=False
                ).tolist()
            
            # 파라미터 수에 따라 모델 예측 사용
            required_params = gate_def.num_parameters
            if param_value is not None:
                parameters = self._generate_parameters_from_predictions(param_value, required_params)
            else:
                # 대체 방법: 랜덤 파라미터 생성
                parameters = [np.random.uniform(0, 2 * np.pi) for _ in range(required_params)]
            
        except Exception as e:
            print(f"Warning: Failed to process gate {gate_name}: {e}")
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
            return [int(np.random.choice(available_qubits))]
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
            param = float(np.random.uniform(0, 2 * np.pi))
            parameters.append(param)
        
        return parameters
    
    def _update_sequence(self, current_sequence: torch.Tensor, new_gate: GateOperation) -> torch.Tensor:
        """시퀀스에 새 게이트 추가 - SAR 패턴 유지"""
        try:
            # 현재 회로에 게이트 추가
            if not hasattr(self, '_current_circuit') or self._current_circuit is None:
                self._current_circuit = CircuitSpec(
                    circuit_id="generation_in_progress",
                    num_qubits=self.config.target_num_qubits,
                    gates=[]
                )
            
            # 새 게이트 추가
            self._current_circuit.gates.append(new_gate)
            
            # 전체 회로를 임베딩하여 SAR (State-Action-Reward) 패턴 유지
            embedded_data = self.embedding_pipeline.process_single_circuit(self._current_circuit)
            new_sequence = embedded_data['input_sequence'].to(self.device)
            
            if self.config.verbose and len(self._current_circuit.gates) % 5 == 0:
                print(f"Sequence updated: {len(self._current_circuit.gates)} gates processed")
                # 시퀀스 길이 확인
                print(f"Sequence shape: {new_sequence.shape}")
                
            return new_sequence
                
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
                        'qubits': [int(q) for q in gate.qubits],
                        'parameters': [float(p) for p in gate.parameters]
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
