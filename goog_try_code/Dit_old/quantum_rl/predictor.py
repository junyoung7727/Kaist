import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from qiskit import QuantumCircuit
import os

# Assuming quantum_rl.models.py contains QuantumRepresentationTransformer
# and quantum_rl.constants.py contains necessary constants.
from quantum_rl.models import QuantumRepresentationTransformer
from quantum_rl.constants import (
    DEVICE, MAX_GATES, MAX_QUBITS, NUM_GATE_TYPES,
    LATENT_DIM, NUM_LAYERS, NUM_HEADS
)

class CircuitPredictor:
    def __init__(self, model_path: str = None):
        self.model = QuantumRepresentationTransformer(
            dim=LATENT_DIM,
            depth=NUM_LAYERS,
            heads=NUM_HEADS,
            num_gate_types=NUM_GATE_TYPES,
            max_qubits=MAX_QUBITS,
            max_gates=MAX_GATES,
            # Add other necessary parameters for QuantumRepresentationTransformer if any
            # e.g., mlp_dim, dim_head, dropout, from constants or defaults
            mlp_dim=LATENT_DIM * 4, # A common setting
            dim_head=LATENT_DIM // NUM_HEADS if NUM_HEADS > 0 else LATENT_DIM, # A common setting
            dropout=0.1 # A common setting
        ).to(DEVICE)

        if model_path and os.path.exists(model_path):
            try:
                # Load state dict non-strictly to allow missing or unexpected keys
                state_dict = torch.load(model_path, map_location=DEVICE)
                self.model.load_state_dict(state_dict, strict=False)
                print(f"CircuitPredictor: Loaded model from {model_path}")
            except Exception as e:
                print(f"CircuitPredictor: Error loading model from {model_path}: {e}. Using fresh model.")
        else:
            if model_path:
                print(f"CircuitPredictor: Model path {model_path} not found. Using fresh model.")
            else:
                print("CircuitPredictor: No model path provided. Using fresh model.")
        self.model.eval()

        self.gate_to_idx = {
            'h': 0, 'x': 1, 'y': 2, 'z': 3, 's': 4, 't': 5,
            'rx': 6, 'ry': 7, 'rz': 8, 'cx': 9, 'cz': 10,
            'swap': 11, 'ccx': 12, 'barrier': 13, 'measure': 14
        }
        # Ensure NUM_GATE_TYPES is consistent with this mapping.
        # Current constants.py has NUM_GATE_TYPES = 15, which matches.

    def circuit_to_features(self, circuit: QuantumCircuit):
        gate_type_indices = []
        qubit_indices_list = []
        parameters_list = []

        for instruction in circuit.data:
            op_name = instruction.operation.name
            # Map Qubit objects to their indices in the circuit
            op_qubits = [circuit.qubits.index(q) for q in instruction.qubits]
            op_params = [float(p) for p in instruction.operation.params]

            if op_name not in self.gate_to_idx:
                # print(f"Warning: Unknown gate '{op_name}' encountered. Skipping.")
                continue # Skip unknown gates

            gate_type_indices.append(self.gate_to_idx[op_name])
            qubit_indices_list.append(op_qubits)
            parameters_list.append(op_params)

            if len(gate_type_indices) >= MAX_GATES:
                break
        
        num_actual_gates = len(gate_type_indices)
        pad_len = MAX_GATES - num_actual_gates

        gate_types_tensor = torch.tensor(gate_type_indices, dtype=torch.long)
        gate_types_tensor = F.pad(gate_types_tensor, (0, pad_len), value=0) # Pad with 0 (h gate / default)

        qubit_features_tensor = torch.zeros((MAX_GATES, MAX_QUBITS), dtype=torch.float32)
        for i in range(num_actual_gates):
            for q_idx in qubit_indices_list[i]:
                if q_idx < MAX_QUBITS:
                    qubit_features_tensor[i, q_idx] = 1.0
        
        max_params_per_gate = 2 # Consistent with some transformer designs
        params_tensor = torch.zeros((MAX_GATES, max_params_per_gate), dtype=torch.float32)
        for i in range(num_actual_gates):
            gate_params = parameters_list[i]
            for j, p_val in enumerate(gate_params):
                if j < max_params_per_gate:
                    params_tensor[i, j] = p_val
        
        features = {
            'gate_types': gate_types_tensor.unsqueeze(0).to(DEVICE),
            'qubit_features': qubit_features_tensor.unsqueeze(0).to(DEVICE),
            'params': params_tensor.unsqueeze(0).to(DEVICE)
        }
        return features

    def predict(self, circuit: QuantumCircuit):
        if not isinstance(circuit, QuantumCircuit):
            raise ValueError("Input must be a Qiskit QuantumCircuit object.")

        features = self.circuit_to_features(circuit)
        
        with torch.no_grad():
            try:
                # This assumes QuantumRepresentationTransformer.forward accepts these named arguments.
                # Adjust if the model's forward signature is different.
                predicted_output = self.model(
                    gate_types=features['gate_types'],
                    qubit_features=features['qubit_features'],
                    params=features['params']
                )
            except TypeError as e:
                # Fallback or re-raise, depending on how robust this needs to be.
                print(f"CircuitPredictor: Error during model prediction (likely mismatched forward signature): {e}")
                # Return dummy/error metrics
                return {"fidelity": 0.0, "expressibility": 0.0, "entanglement": 0.0, "error": str(e)}

        # Process predicted_output based on its format (tensor or dict)
        metrics = {}
        if isinstance(predicted_output, torch.Tensor):
            predicted_values = predicted_output.squeeze(0).cpu().numpy() if predicted_output.shape[0] == 1 else predicted_output.cpu().numpy()
            if len(predicted_values) >= 3:
                metrics = {
                    "fidelity": float(predicted_values[0]),
                    "expressibility": float(predicted_values[1]),
                    "entanglement": float(predicted_values[2])
                }
            elif len(predicted_values) == 1:
                 metrics = { "score": float(predicted_values[0]) }
            else:
                metrics = {"error": "Output tensor shape mismatch from model."}
        elif isinstance(predicted_output, dict):
            metrics = {k: v.item() if isinstance(v, torch.Tensor) else float(v) for k, v in predicted_output.items()}
        else:
            metrics = {"error": f"Unexpected output type from model: {type(predicted_output)}"}
            
        # Ensure compatibility with environment keys
        metrics['normalized_expressibility'] = metrics.get('expressibility', 0.0)
        metrics['expressibility_distance'] = 0.0
        
        return metrics
