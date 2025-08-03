"""
Experiment Results Dataset for Quantum Circuit Analysis
Supports experiment_results.json format with expressibility, two_qubit_ratio, and fidelity targets
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import sys  

sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))

from gates import QuantumGateRegistry

class ExperimentResultsDataset(Dataset):
    """
    Dataset for experiment results with quantum circuit properties
    Supports expressibility, two_qubit_ratio, and fidelity as targets
    """
    
    def __init__(self,
                 data_path: str,
                 circuit_spec_path: str = None,
                 target_properties: List[str] = None,
                 normalize_targets: bool = True,
                 augment_data: bool = False,
                 train_mode: bool = True):
        """
        Initialize experiment results dataset
        
        Args:
            data_path: Path to experiment_results.json file
            target_properties: List of target properties to predict
            normalize_targets: Whether to normalize target values
            augment_data: Whether to apply data augmentation
            train_mode: Whether in training mode (affects augmentation)
        """
        self.data_path = data_path
        self.circuit_spec_path = circuit_spec_path
        self.target_properties = target_properties or [
            'expressibility', 'two_qubit_ratio', 'simulator_error_fidelity'
        ]
        self.normalize_targets = normalize_targets
        self.augment_data = augment_data and train_mode
        self.train_mode = train_mode
        
        # Initialize gate registry
        self.gate_registry = QuantumGateRegistry()
        self.gate_vocab = self.gate_registry.get_gate_vocab()
        self.vocab_size = len(self.gate_vocab)
        
        # Load circuit specifications if provided
        self.circuit_specs = {}
        if self.circuit_spec_path and os.path.exists(self.circuit_spec_path):
            self.circuit_specs = self._load_circuit_specs()
        
        # Load and process data
        self.circuits, self.targets = self._load_experiment_data()
        
        # Compute normalization statistics
        if self.normalize_targets:
            self.target_stats = self._compute_target_stats()
        
        print(f"Loaded {len(self.circuits)} quantum circuit experiments")
        print(f"Target properties: {self.target_properties}")
        print(f"Vocabulary size: {self.vocab_size}")
        if self.circuit_specs:
            print(f"Loaded {len(self.circuit_specs)} circuit specifications")
    
    def _load_circuit_specs(self) -> Dict[str, Dict[str, Any]]:
        """Load circuit specifications from circuit_spec.json"""
        with open(self.circuit_spec_path, 'r') as f:
            spec_data = json.load(f)
        
        circuit_specs = {}
        for circuit in spec_data.get('circuits', []):
            circuit_id = circuit['circuit_id']
            circuit_specs[circuit_id] = circuit
        
        return circuit_specs
    
    def _convert_gates_to_indices(self, gates: List[Dict[str, Any]]) -> List[int]:
        """Convert gate specifications to gate indices"""
        gate_indices = []
        
        # Gate name to index mapping (based on circuit_specs.json)
        gate_name_to_idx = {
            'i': 0, 'id': 0, 'identity': 0,
            'x': 1, 'pauli_x': 1,
            'y': 2, 'pauli_y': 2,
            'z': 3, 'pauli_z': 3,
            'h': 4, 'hadamard': 4,
            's': 5, 'phase': 5,
            't': 6,
            'tdg': 7, 't_dagger': 7,  # T-dagger gate
            'p': 8,  # Phase gate (with parameter)
            'rx': 9, 'rotation_x': 9,
            'ry': 10, 'rotation_y': 10,
            'rz': 11, 'rotation_z': 11,
            'cnot': 12, 'cx': 12,
            'cz': 13,
            'swap': 14,
            'crx': 15,
            'cry': 16,
            'crz': 17
        }
        
        for gate in gates:
            gate_name = gate['name'].lower()
            gate_idx = gate_name_to_idx.get(gate_name, 0)  # Default to identity
            gate_indices.append(gate_idx)
        
        return gate_indices
    
    def _load_experiment_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
        """Load experiment results from JSON file"""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        circuits = []
        targets = []
        
        for result in data['results']:
            # Extract circuit properties
            circuit = {
                'circuit_id': result['circuit_id'],
                'num_qubits': result['num_qubits'],
                'gate_count': result['gate_count'],
                'two_qubit_ratio': result['two_qubit_ratio'],
                'depth': self._estimate_depth_from_id(result['circuit_id']),
                'gates': self._get_gate_sequence(result)
            }
            
            # Extract target properties
            target = {}
            for prop in self.target_properties:
                if prop == 'expressibility':
                    target[prop] = result['expressibility_divergence']['expressibility']
                elif prop == 'two_qubit_ratio':
                    target[prop] = result['two_qubit_ratio']
                elif prop == 'simulator_error_fidelity':
                    target[prop] = result['simulator_error_fidelity']
                elif prop == 'kl_divergence':
                    target[prop] = result['expressibility_divergence']['kl_divergence']
                elif prop == 'js_divergence':
                    target[prop] = result['expressibility_divergence']['js_divergence']
                else:
                    # Try to get from expressibility_divergence
                    target[prop] = result['expressibility_divergence'].get(prop, 0.0)
            
            circuits.append(circuit)
            targets.append(target)
        
        return circuits, targets
    
    def _estimate_depth_from_id(self, circuit_id: str) -> int:
        """Extract depth from circuit ID (e.g., 'exp1_4q_d2_r0.1_4' -> depth=2)"""
        try:
            parts = circuit_id.split('_')
            for part in parts:
                if part.startswith('d') and part[1:].isdigit():
                    return int(part[1:])
            return 1  # Default depth
        except:
            return 1
    
    def _get_gate_sequence(self, result: Dict[str, Any]) -> List[int]:
        """
        Get gate sequence from circuit_spec if available, otherwise generate synthetic sequence
        """
        circuit_id = result['circuit_id']
        
        # Try to get real gate sequence from circuit_spec
        if circuit_id in self.circuit_specs:
            circuit_spec = self.circuit_specs[circuit_id]
            gates = circuit_spec.get('gates', [])
            return self._convert_gates_to_indices(gates)
        
        # Fallback: generate synthetic gate sequence
        return self._generate_synthetic_gate_sequence(result)
    
    def _generate_synthetic_gate_sequence(self, result: Dict[str, Any]) -> List[int]:
        """
        Generate a realistic gate sequence based on circuit properties
        This is a heuristic approach when actual gate sequences aren't available
        """
        num_qubits = result['num_qubits']
        gate_count = result['gate_count']
        two_qubit_ratio = result['two_qubit_ratio']
        
        gates = []
        
        # Calculate number of two-qubit gates
        num_two_qubit = int(gate_count * two_qubit_ratio)
        num_single_qubit = gate_count - num_two_qubit
        
        # Add single-qubit gates (H, X, Y, Z, RX, RY, RZ)
        single_qubit_gates = [0, 1, 2, 3, 4, 7, 8, 9]  # Gate indices
        for _ in range(num_single_qubit):
            gates.append(random.choice(single_qubit_gates))
        
        # Add two-qubit gates (CNOT, CZ)
        two_qubit_gates = [10, 11]  # Gate indices for CNOT, CZ
        for _ in range(num_two_qubit):
            gates.append(random.choice(two_qubit_gates))
        
        # Shuffle to create realistic ordering
        random.shuffle(gates)
        
        return gates
    
    def _compute_target_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute normalization statistics for targets"""
        stats = {}
        
        for prop in self.target_properties:
            values = [target[prop] for target in self.targets]
            stats[prop] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return stats
    
    def _normalize_targets(self, targets: Dict[str, float]) -> Dict[str, float]:
        """Normalize target values"""
        if not self.normalize_targets:
            return targets
        
        normalized = {}
        for prop, value in targets.items():
            stats = self.target_stats[prop]
            if stats['std'] > 1e-8:  # Avoid division by zero
                normalized[prop] = (value - stats['mean']) / stats['std']
            else:
                normalized[prop] = 0.0
        
        return normalized
    
    def _denormalize_targets(self, normalized_targets: Dict[str, float]) -> Dict[str, float]:
        """Denormalize target values"""
        if not self.normalize_targets:
            return normalized_targets
        
        denormalized = {}
        for prop, norm_value in normalized_targets.items():
            stats = self.target_stats[prop]
            denormalized[prop] = norm_value * stats['std'] + stats['mean']
        
        return denormalized
    
    def _augment_circuit(self, circuit: Dict[str, Any]) -> Dict[str, Any]:
        """Apply data augmentation to circuit"""
        if not self.augment_data:
            return circuit
        
        augmented = circuit.copy()
        
        # Gate sequence augmentation
        if random.random() < 0.3:
            gates = augmented['gates'].copy()
            
            # Random gate substitution (10% chance per gate)
            for i in range(len(gates)):
                if random.random() < 0.1:
                    # Substitute with similar gate
                    if gates[i] in [1, 2, 3]:  # Pauli gates
                        gates[i] = random.choice([1, 2, 3])
                    elif gates[i] in [7, 8, 9]:  # Rotation gates
                        gates[i] = random.choice([7, 8, 9])
            
            augmented['gates'] = gates
        
        # Small perturbations to continuous properties
        if random.random() < 0.2:
            # Add small noise to two_qubit_ratio (within reasonable bounds)
            noise = random.gauss(0, 0.01)
            augmented['two_qubit_ratio'] = max(0, min(1, 
                augmented['two_qubit_ratio'] + noise))
        
        return augmented
    
    def __len__(self) -> int:
        """Get dataset size"""
        return len(self.circuits)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single circuit sample with targets"""
        circuit = self.circuits[idx]
        targets = self.targets[idx]
        
        # Apply augmentation
        if self.augment_data:
            circuit = self._augment_circuit(circuit)
        
        # Normalize targets
        normalized_targets = self._normalize_targets(targets)
        
        # Prepare circuit features
        max_gates = 64  # Maximum gate sequence length
        gates = circuit['gates'][:max_gates]  # Truncate if too long
        
        # Pad gate sequence
        if len(gates) < max_gates:
            gates.extend([self.vocab_size] * (max_gates - len(gates)))  # Pad with special token
        
        # Create sample
        sample = {
            # Circuit features
            'gates': torch.tensor(gates, dtype=torch.long),
            'num_qubits': torch.tensor(circuit['num_qubits'], dtype=torch.long),
            'gate_count': torch.tensor(circuit['gate_count'], dtype=torch.long),
            'depth': torch.tensor(circuit['depth'], dtype=torch.long),
            'two_qubit_ratio': torch.tensor(circuit['two_qubit_ratio'], dtype=torch.float32),
            
            # Target properties
            'targets': torch.tensor([normalized_targets[prop] for prop in self.target_properties], 
                                  dtype=torch.float32),
            
            # Metadata
            'circuit_id': circuit['circuit_id'],
            'original_length': len(circuit['gates'])
        }
        
        return sample
    
    def get_target_names(self) -> List[str]:
        """Get list of target property names"""
        return self.target_properties
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size (including padding token)"""
        return self.vocab_size + 1
    
    def get_target_stats(self) -> Dict[str, Dict[str, float]]:
        """Get target normalization statistics"""
        return getattr(self, 'target_stats', {})
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.circuits:
            return {}
        
        # Circuit statistics
        num_qubits = [c['num_qubits'] for c in self.circuits]
        gate_counts = [c['gate_count'] for c in self.circuits]
        depths = [c['depth'] for c in self.circuits]
        two_qubit_ratios = [c['two_qubit_ratio'] for c in self.circuits]
        
        # Target statistics
        target_stats = {}
        for prop in self.target_properties:
            values = [t[prop] for t in self.targets]
            target_stats[prop] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        stats = {
            'num_circuits': len(self.circuits),
            'num_qubits': {
                'mean': np.mean(num_qubits),
                'std': np.std(num_qubits),
                'min': np.min(num_qubits),
                'max': np.max(num_qubits)
            },
            'gate_counts': {
                'mean': np.mean(gate_counts),
                'std': np.std(gate_counts),
                'min': np.min(gate_counts),
                'max': np.max(gate_counts)
            },
            'depths': {
                'mean': np.mean(depths),
                'std': np.std(depths),
                'min': np.min(depths),
                'max': np.max(depths)
            },
            'two_qubit_ratios': {
                'mean': np.mean(two_qubit_ratios),
                'std': np.std(two_qubit_ratios),
                'min': np.min(two_qubit_ratios),
                'max': np.max(two_qubit_ratios)
            },
            'targets': target_stats,
            'vocab_size': self.get_vocab_size()
        }
        
        return stats


class ExperimentResultsCollator:
    """Custom collator for experiment results batches"""
    
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch of experiment results"""
        collated = {}
        
        # Handle tensor fields that can be stacked
        tensor_keys = ['gates', 'num_qubits', 'gate_count', 'depth', 'two_qubit_ratio', 'targets']
        
        for key in tensor_keys:
            if key in batch[0]:
                collated[key] = torch.stack([item[key] for item in batch], dim=0)
        
        # Handle non-tensor fields
        if 'circuit_id' in batch[0]:
            collated['circuit_id'] = [item['circuit_id'] for item in batch]
        
        if 'original_length' in batch[0]:
            collated['original_length'] = [item['original_length'] for item in batch]
        
        return collated


def create_experiment_dataloaders(train_path: str,
                                val_path: str,
                                target_properties: List[str] = None,
                                batch_size: int = 32,
                                num_workers: int = 4,
                                **dataset_kwargs) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create training and validation dataloaders for experiment results
    
    Args:
        train_path: Path to training experiment results
        val_path: Path to validation experiment results
        target_properties: List of target properties to predict
        batch_size: Batch size
        num_workers: Number of worker processes
        **dataset_kwargs: Additional dataset arguments
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Default target properties
    if target_properties is None:
        target_properties = ['expressibility', 'two_qubit_ratio', 'simulator_error_fidelity']
    
    # Create datasets
    train_dataset = ExperimentResultsDataset(
        train_path, 
        target_properties=target_properties,
        train_mode=True, 
        augment_data=True, 
        **dataset_kwargs
    )
    
    val_dataset = ExperimentResultsDataset(
        val_path, 
        target_properties=target_properties,
        train_mode=False, 
        augment_data=False, 
        **dataset_kwargs
    )
    
    # Create collator
    collator = ExperimentResultsCollator(pad_token_id=train_dataset.get_vocab_size() - 1)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    dataset_path = r"Dit_Model_ver2\data\raw\experiment_results.json"
    
    if os.path.exists(dataset_path):
        dataset = ExperimentResultsDataset(
            dataset_path,
            target_properties=['expressibility', 'two_qubit_ratio', 'simulator_error_fidelity']
        )
        
        print("Dataset statistics:")
        stats = dataset.get_statistics()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue}")
            else:
                print(f"  {key}: {value}")
        
        # Test sample
        sample = dataset[0]
        print(f"\nSample:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} - {value}")
            else:
                print(f"  {key}: {value}")
    else:
        print(f"Dataset file not found: {dataset_path}")
        print("Please ensure the experiment_results.json file exists.")
