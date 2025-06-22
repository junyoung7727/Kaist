import gym
from gym import spaces
import numpy as np
import torch
from qiskit import QuantumCircuit

from .constants import SAMPLING_BATCH_SIZE
from .predictor import CircuitPredictor

class QuantumCircuitEnv(gym.Env):
    """
    양자 회로 최적화를 위한 RL 환경
    - 예측기 모델을 사용한 회로 평가
    - 사용자 요구사항 기반 보상 함수
    """

    def __init__(self, diffusion_model, predictor_model_path=None, n_qubits=5, max_gates=20,
                 user_requirements={"fidelity": 0.9, "expressibility": 0.8, "entanglement": 0.7}):
        super().__init__()

        self.diffusion_model = diffusion_model
        self.n_qubits = n_qubits
        self.max_gates = max_gates
        self.user_requirements = user_requirements

        # CircuitPredictor must be defined in the scope where QuantumCircuitEnv is instantiated.
        self.predictor = CircuitPredictor(predictor_model_path)

        self.action_space = spaces.Box(
            low=np.array([0.1, 0.1, 0.1]),
            high=np.array([2.0, 2.0, 2.0]),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(16,), dtype=np.float32
        )

        self.current_circuit = None
        self.current_metrics = None
        self.step_count = 0
        self.max_steps = 50  # Consider moving to constants.py

    def reset(self, seed=None):
        """환경 초기화"""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.current_circuit = self._generate_random_circuit()
        self.current_metrics = self._evaluate_circuit_with_predictor(self.current_circuit)
        self.step_count = 0

        obs = self._get_observation()
        info = {"metrics": self.current_metrics}

        return obs, info

    def step(self, action):
        """액션 실행"""
        self.step_count += 1
        # If action passed as (action_array, ...) tuple or list, extract the array
        if isinstance(action, (tuple, list)):
            action = action[0]
        # Ensure action is a flat numpy array
        action = np.array(action).flatten()
        complexity_factor, entanglement_factor, gate_density = action

        new_circuit = self._generate_circuit_with_params(
            complexity_factor, entanglement_factor, gate_density
        )
        new_metrics = self._evaluate_circuit_with_predictor(new_circuit)
        reward = self._compute_reward(new_metrics)

        if self._is_better_circuit(new_metrics, self.current_metrics):
            self.current_circuit = new_circuit
            self.current_metrics = new_metrics
            reward += 0.1  # Improvement bonus

        terminated = self._check_requirements_met()
        truncated = (self.step_count >= self.max_steps)

        obs = self._get_observation()
        info = {
            "metrics": self.current_metrics,
            "requirements_met": terminated,
            "improvement": self._is_better_circuit(new_metrics, self.current_metrics) # Can be re-eval if current_metrics changed
        }
        return obs, reward, terminated, truncated, info

    def _generate_random_circuit(self):
        """랜덤 초기 회로 생성"""
        circuit = QuantumCircuit(self.n_qubits)
        for _ in range(np.random.randint(5, self.max_gates)):
            gate_type = np.random.choice(['h', 'x', 'rx', 'ry', 'cx'])
            qubit = np.random.randint(self.n_qubits)
            if gate_type == 'h': circuit.h(qubit)
            elif gate_type == 'x': circuit.x(qubit)
            elif gate_type in ['rx', 'ry']:
                angle = np.random.uniform(0, 2 * np.pi)
                if gate_type == 'rx': circuit.rx(angle, qubit)
                else: circuit.ry(angle, qubit)
            elif gate_type == 'cx' and self.n_qubits > 1:
                control, target = np.random.choice(self.n_qubits, 2, replace=False)
                circuit.cx(control, target)
        return circuit

    def _generate_circuit_with_params(self, complexity_factor, entanglement_factor, gate_density):
        """파라미터 기반 회로 생성 (디퓨전 모델 활용)"""
        try:
            seq_len_to_sample = max(1, min(int(self.max_gates * gate_density), self.max_gates))
            circuits = self.diffusion_model.sample_batch(
                self.n_qubits, seq_len_to_sample, batch_size=SAMPLING_BATCH_SIZE
            )

            # Batch feature extraction and vectorized prediction
            features = [self.predictor.circuit_to_features(c) for c in circuits]
            gate_types_batch = torch.cat([f['gate_types'] for f in features], dim=0)
            qubit_batch = torch.cat([f['qubit_features'] for f in features], dim=0)
            params_batch = torch.cat([f['params'] for f in features], dim=0)
            with torch.no_grad():
                outputs = self.predictor.model(
                    gate_types=gate_types_batch,
                    qubit_features=qubit_batch,
                    params=params_batch
                )
            # Build metrics list from batch outputs
            if isinstance(outputs, torch.Tensor):
                preds = outputs.cpu().numpy()
                metrics_list = [
                    {'fidelity': float(p[0]), 'expressibility': float(p[1]), 'entanglement': float(p[2])}
                    for p in preds
                ]
            else:
                # Fallback to per-circuit predictor if output format is unexpected
                metrics_list = [self._evaluate_circuit_with_predictor(c) for c in circuits]
            rewards = [self._compute_reward(m) for m in metrics_list]
            best_idx = int(np.argmax(rewards))
            circuit = circuits[best_idx]
        except Exception as e:
            print(f"⚠️ 디퓨전 모델 생성 실패: {e}")
            circuit = self._generate_parametric_circuit(complexity_factor, entanglement_factor, gate_density)
        return circuit

    def _generate_parametric_circuit(self, complexity_factor, entanglement_factor, gate_density):
        """파라미터 기반 회로 생성 (폴백 방법)"""
        circuit = QuantumCircuit(self.n_qubits)
        num_gates = int(self.max_gates * gate_density)
        for _ in range(num_gates):
            if np.random.random() < entanglement_factor * 0.5 and self.n_qubits > 1:
                control, target = np.random.choice(self.n_qubits, 2, replace=False)
                circuit.cx(control, target)
            else:
                qubit = np.random.randint(self.n_qubits)
                gate_type = np.random.choice(['h', 'x', 'rx', 'ry'])
                if gate_type == 'h': circuit.h(qubit)
                elif gate_type == 'x': circuit.x(qubit)
                elif gate_type in ['rx', 'ry']:
                    angle = np.random.uniform(0, 2 * np.pi) * complexity_factor
                    if gate_type == 'rx': circuit.rx(angle, qubit)
                    else: circuit.ry(angle, qubit)
        return circuit

    def _evaluate_circuit_with_predictor(self, circuit):
        """ 예측기 모델을 사용한 회로 평가"""
        try:
            predictions = self.predictor.predict(circuit)
            cnot_count = sum(1 for instruction in circuit.data if instruction.operation.name.lower() in ['cx', 'cz'])
            max_possible_cnots = self.n_qubits * (self.n_qubits - 1) // 2
            entanglement = min(cnot_count / max(max_possible_cnots, 1), 1.0)
            return {
                'fidelity': predictions['fidelity'],
                'expressibility': predictions['normalized_expressibility'],
                'entanglement': entanglement,
                'expressibility_distance': predictions['expressibility_distance']
            }
        except Exception as e:
            print(f"⚠️ 예측기 평가 실패: {e}")
            return self._fallback_evaluation(circuit)

    def _fallback_evaluation(self, circuit):
        """폴백 평가 방법 (예측기 실패시)"""
        depth = circuit.depth()
        gate_count = len(circuit.data)
        cnot_count = sum(1 for instruction in circuit.data if instruction.operation.name.lower() in ['cx', 'cz'])
        fidelity = max(0.1, 1.0 - depth * 0.05)
        expressibility = min(1.0, gate_count * 0.1)
        entanglement = min(1.0, cnot_count * 0.2)
        return {
            'fidelity': fidelity,
            'expressibility': expressibility,
            'entanglement': entanglement,
            'expressibility_distance': 0.001
        }

    def _compute_reward(self, metrics):
        """보상 함수 (사용자 요구사항 기반)"""
        reward = 0.0
        for metric_name, target_value in self.user_requirements.items():
            current_value = metrics.get(metric_name, 0.0)
            diff = abs(current_value - target_value)
            metric_reward = max(0, 1.0 - diff)
            reward += metric_reward
            if current_value >= target_value: reward += 0.5
        if all(metrics.get(k, 0) >= v * 0.8 for k, v in self.user_requirements.items()): reward += 1.0
        return reward

    def _is_better_circuit(self, new_metrics, old_metrics):
        """새 회로가 더 좋은지 판단"""
        if old_metrics is None: return True
        weights = {'fidelity': 0.4, 'expressibility': 0.4, 'entanglement': 0.2}
        new_score = sum(new_metrics.get(k, 0) * w for k, w in weights.items())
        old_score = sum(old_metrics.get(k, 0) * w for k, w in weights.items())
        return new_score > old_score

    def _get_observation(self):
        """현재 상태 관찰"""
        metrics_obs = [
            self.current_metrics.get('fidelity', 0.0) if self.current_metrics else 0.0,
            self.current_metrics.get('expressibility', 0.0) if self.current_metrics else 0.0,
            self.current_metrics.get('entanglement', 0.0) if self.current_metrics else 0.0
        ]
        requirements_obs = [
            self.user_requirements.get('fidelity', 0.9),
            self.user_requirements.get('expressibility', 0.8),
            self.user_requirements.get('entanglement', 0.7)
        ]
        if self.current_circuit is not None:
            circuit_features = [
                self.current_circuit.num_qubits / 20.0,
                self.current_circuit.depth() / 50.0,
                len(self.current_circuit.data) / 100.0,
                sum(1 for inst in self.current_circuit.data if inst.operation.name.lower() in ['cx', 'cz']) / 20.0,
                self.step_count / self.max_steps,
                0.0, 0.0, 0.0, 0.0, 0.0 # Padding
            ]
        else:
            circuit_features = [0.0] * 10
        obs = np.array(metrics_obs + requirements_obs + circuit_features, dtype=np.float32)
        return obs

    def _check_requirements_met(self):
        """요구사항 만족 여부 확인"""
        if self.current_metrics is None: return False
        return all(
            self.current_metrics.get(k, 0) >= v
            for k, v in self.user_requirements.items()
        )
