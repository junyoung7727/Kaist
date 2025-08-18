"""Context-Aware Quantum Circuit Generator (Unified Decision Transformer)

Decision Transformer 기반 통합 생성기:
- Decision Transformer의 generate_autoregressive 메서드 사용
- Property predictor 기반 실시간 보상 계산
- 학습과 동일한 SAR 시퀀스 구조 유지
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from pathlib import Path
import sys

# Add quantumcommon to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))
from circuit_interface import CircuitSpec, GateOperation

# Import models and utilities
from models.decision_transformer import DecisionTransformer
from utils.reward_calculator import RewardCalculator


class ContextAwareGenerator(nn.Module):
    """Decision Transformer 기반 통합 생성기"""
    
    def __init__(
        self,
        decision_transformer: DecisionTransformer,
        reward_calculator: RewardCalculator,
        max_sequence_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50
    ):
        super().__init__()
        
        self.decision_transformer = decision_transformer
        self.reward_calculator = reward_calculator
        
        self.max_sequence_length = max_sequence_length
        self.temperature = temperature
        self.top_k = top_k
    
    def generate_circuit(
        self,
        target_properties: Dict[str, float],
        num_qubits: int = 4,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> CircuitSpec:
        """
        목표 속성을 만족하는 양자 회로 생성
        
        Args:
            target_properties: 목표 속성 {'entanglement': 0.8, 'fidelity': 0.9, 'expressibility': 0.7}
            num_qubits: 큐빗 수
            max_length: 최대 게이트 수 (기본값: self.max_sequence_length)
            temperature: 샘플링 온도 (기본값: self.temperature)
            top_k: Top-k 샘플링 (기본값: self.top_k)
            
        Returns:
            생성된 CircuitSpec
        """
        # 파라미터 설정
        max_length = max_length or self.max_sequence_length
        temperature = temperature or self.temperature
        top_k = top_k or self.top_k
        
        print(f"🚀 Starting circuit generation with Decision Transformer")
        print(f"   Target properties: {target_properties}")
        print(f"   Num qubits: {num_qubits}, Max length: {max_length}")
        
        # Decision Transformer의 autoregressive 생성 사용
        generated_gates = self.decision_transformer.generate_autoregressive(
            prompt_tokens=None,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            reward_calculator=self.reward_calculator,
            target_properties=target_properties,
            num_qubits=num_qubits
        )
        
        # 생성된 게이트들을 CircuitSpec으로 변환
        circuit_spec = self._gates_to_circuit_spec(generated_gates, num_qubits)
        
        print(f"✅ Circuit generation completed: {len(generated_gates)} gates")
        return circuit_spec
    
    def _gates_to_circuit_spec(self, gates: List[Dict], num_qubits: int) -> CircuitSpec:
        """생성된 게이트 리스트를 CircuitSpec으로 변환"""
        gate_operations = []
        
        for gate_info in gates:
            gate_op = GateOperation(
                gate_type=gate_info['gate'],
                qubits=gate_info['qubits'],
                parameters=gate_info.get('params', [])
            )
            gate_operations.append(gate_op)
        
        return CircuitSpec(
            num_qubits=num_qubits,
            gates=gate_operations,
            depth=len(gate_operations)
        )
