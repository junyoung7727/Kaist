"""
Quantum Circuit Dataset for Training DiT Models
Supports experiment_results.json format with expressibility, two_qubit_ratio, and fidelity targets
"""

import os
import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
import numpy as np
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))

from gates import QuantumGateRegistry


class QuantumCircuitDataset(Dataset):
    """
    Dataset for quantum circuit data with advanced preprocessing
    """
    
    def __init__(self,
                 data_path: str,
                 max_circuit_length: int = 256,
                 max_qubits: int = 32,
                 augment_data: bool = True,
                 cache_data: bool = True,
                 normalize: bool = True):
        """
        Initialize quantum circuit dataset
        
        Args:
            data_path: Path to dataset directory or file
            max_circuit_length: Maximum circuit length
            max_qubits: Maximum number of qubits
            augment_data: Whether to apply data augmentation
            cache_data: Whether to cache processed data
            normalize: Whether to normalize circuit representations
        """
        self.data_path = data_path
        self.max_circuit_length = max_circuit_length
        self.max_qubits = max_qubits
        self.augment_data = augment_data
        self.cache_data = cache_data
        self.normalize = normalize
        
        # Initialize gate registry
        self.gate_registry = QuantumGateRegistry()
        self.gate_vocab = self.gate_registry.get_gate_vocab()
        self.vocab_size = len(self.gate_vocab)
        
        # Load and process data
        self.circuits = self._load_data()
        
        print(f"Loaded {len(self.circuits)} quantum circuits")
        print(f"Vocabulary size: {self.vocab_size}")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load circuit data from files"""
        circuits = []
        
        if os.path.isfile(self.data_path):
            # Single file
            circuits = self._load_single_file(self.data_path)
        elif os.path.isdir(self.data_path):
            # Directory with multiple files
            for file_path in Path(self.data_path).glob("*.json"):
                circuits.extend(self._load_single_file(str(file_path)))
            
            # Also check for pickle files
            for file_path in Path(self.data_path).glob("*.pkl"):
                circuits.extend(self._load_pickle_file(str(file_path)))
        else:
            # Generate synthetic data if path doesn't exist
            print(f"Data path {self.data_path} not found, generating synthetic data...")
            circuits = self._generate_synthetic_data(1000)
        
        # Process and filter circuits
        processed_circuits = []
        for circuit in circuits:
            processed = self._process_circuit(circuit)
            if processed is not None:
                processed_circuits.append(processed)
        
        return processed_circuits
    
    def _load_single_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load circuits from a single JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            else:
                return []
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []
    
    def _load_pickle_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load circuits from a pickle file"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            else:
                return []
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []
    
    def _generate_synthetic_data(self, num_circuits: int) -> List[Dict[str, Any]]:
        """Generate synthetic quantum circuits for testing"""
        circuits = []
        
        for _ in range(num_circuits):
            # Random circuit parameters
            num_qubits = random.randint(2, min(8, self.max_qubits))
            circuit_length = random.randint(5, min(50, self.max_circuit_length))
            
            # Generate random gate sequence
            gates = []
            for _ in range(circuit_length):
                gate_idx = random.randint(0, self.vocab_size - 1)
                gates.append(gate_idx)
            
            # Create circuit representation
            circuit = {
                'gates': gates,
                'qubits': num_qubits,
                'length': circuit_length,
                'parameters': [random.random() for _ in range(circuit_length)],
                'metadata': {
                    'synthetic': True,
                    'depth': random.randint(3, circuit_length // num_qubits + 1)
                }
            }
            
            circuits.append(circuit)
        
        return circuits
    
    def _process_circuit(self, circuit: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process and validate a single circuit"""
        try:
            # Extract basic information
            gates = circuit.get('gates', [])
            qubits = circuit.get('qubits', 2)
            
            # Validate circuit
            if not gates or len(gates) == 0:
                return None
            
            if len(gates) > self.max_circuit_length:
                gates = gates[:self.max_circuit_length]
            
            if qubits > self.max_qubits:
                qubits = self.max_qubits
            
            # Ensure all gate indices are valid
            valid_gates = []
            for gate in gates:
                if isinstance(gate, int) and 0 <= gate < self.vocab_size:
                    valid_gates.append(gate)
                elif isinstance(gate, str):
                    # Convert gate name to index
                    gate_idx = self.gate_vocab.get(gate, 0)
                    valid_gates.append(gate_idx)
                else:
                    valid_gates.append(0)  # Default to identity
            
            # Pad or truncate to fixed length
            if len(valid_gates) < self.max_circuit_length:
                # Pad with special padding token (vocab_size)
                padding_length = self.max_circuit_length - len(valid_gates)
                valid_gates.extend([self.vocab_size] * padding_length)
            
            # Process parameters
            parameters = circuit.get('parameters', [])
            if len(parameters) < len(gates):
                # Pad with zeros
                parameters.extend([0.0] * (len(gates) - len(parameters)))
            elif len(parameters) > len(gates):
                parameters = parameters[:len(gates)]
            
            # Pad parameters to match circuit length
            if len(parameters) < self.max_circuit_length:
                parameters.extend([0.0] * (self.max_circuit_length - len(parameters)))
            
            # Create processed circuit
            processed_circuit = {
                'gates': valid_gates,
                'parameters': parameters,
                'qubits': qubits,
                'original_length': len(gates),
                'metadata': circuit.get('metadata', {})
            }
            
            return processed_circuit
            
        except Exception as e:
            print(f"Error processing circuit: {e}")
            return None
    
    def _augment_circuit(self, circuit: Dict[str, Any]) -> Dict[str, Any]:
        """Apply data augmentation to a circuit"""
        if not self.augment_data:
            return circuit
        
        augmented = circuit.copy()
        
        # Random transformations
        if random.random() < 0.3:
            # Gate substitution (replace some gates with equivalent ones)
            gates = augmented['gates'].copy()
            for i in range(len(gates)):
                if gates[i] < self.vocab_size and random.random() < 0.1:
                    # Simple substitution rules (this would be more sophisticated in practice)
                    if gates[i] == 1:  # X gate
                        gates[i] = random.choice([1, 2])  # X or Y
                    elif gates[i] == 4:  # H gate
                        if random.random() < 0.5:
                            gates[i] = 4  # Keep H
            augmented['gates'] = gates
        
        if random.random() < 0.2:
            # Parameter noise
            parameters = augmented['parameters'].copy()
            for i in range(len(parameters)):
                if parameters[i] != 0.0:
                    noise = random.gauss(0, 0.1)
                    parameters[i] = max(-np.pi, min(np.pi, parameters[i] + noise))
            augmented['parameters'] = parameters
        
        if random.random() < 0.1:
            # Circuit truncation (simulate shorter circuits)
            original_length = augmented['original_length']
            if original_length > 5:
                new_length = random.randint(3, original_length - 1)
                gates = augmented['gates'].copy()
                parameters = augmented['parameters'].copy()
                
                # Truncate and re-pad
                gates = gates[:new_length] + [self.vocab_size] * (self.max_circuit_length - new_length)
                parameters = parameters[:new_length] + [0.0] * (self.max_circuit_length - new_length)
                
                augmented['gates'] = gates
                augmented['parameters'] = parameters
                augmented['original_length'] = new_length
        
        return augmented
    
    def _normalize_circuit(self, circuit: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize circuit representation"""
        if not self.normalize:
            return circuit
        
        normalized = circuit.copy()
        
        # Normalize parameters to [-1, 1]
        parameters = np.array(normalized['parameters'])
        if np.any(parameters != 0):
            # Normalize non-zero parameters
            non_zero_mask = parameters != 0
            parameters[non_zero_mask] = parameters[non_zero_mask] / np.pi
        
        normalized['parameters'] = parameters.tolist()
        
        return normalized
    
    def __len__(self) -> int:
        """Get dataset size"""
        return len(self.circuits)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single circuit sample"""
        circuit = self.circuits[idx]
        
        # Apply augmentation
        if self.augment_data and random.random() < 0.5:
            circuit = self._augment_circuit(circuit)
        
        # Apply normalization
        circuit = self._normalize_circuit(circuit)
        
        # Convert to tensors
        sample = {
            'circuit': torch.tensor(circuit['gates'], dtype=torch.long),
            'parameters': torch.tensor(circuit['parameters'], dtype=torch.float32),
            'qubits': torch.tensor(circuit['qubits'], dtype=torch.long),
            'length': torch.tensor(circuit['original_length'], dtype=torch.long),
        }
        
        return sample
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size (including padding token)"""
        return self.vocab_size + 1  # +1 for padding token
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.circuits:
            return {}
        
        lengths = [c['original_length'] for c in self.circuits]
        qubits = [c['qubits'] for c in self.circuits]
        
        # Gate distribution
        all_gates = []
        for circuit in self.circuits:
            all_gates.extend([g for g in circuit['gates'] if g < self.vocab_size])
        
        gate_counts = {}
        for gate in all_gates:
            gate_counts[gate] = gate_counts.get(gate, 0) + 1
        
        stats = {
            'num_circuits': len(self.circuits),
            'avg_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'avg_qubits': np.mean(qubits),
            'std_qubits': np.std(qubits),
            'min_qubits': np.min(qubits),
            'max_qubits': np.max(qubits),
            'gate_distribution': gate_counts,
            'vocab_size': self.get_vocab_size()
        }
        
        return stats


class QuantumCircuitCollator:
    """Custom collator for quantum circuit batches"""
    
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch of circuits"""
        # Stack all tensors
        collated = {}
        
        for key in batch[0].keys():
            if key == 'circuit':
                # Handle variable length circuits
                circuits = [item[key] for item in batch]
                collated[key] = torch.stack(circuits, dim=0)
            else:
                # Stack other tensors normally
                collated[key] = torch.stack([item[key] for item in batch], dim=0)
        
        return collated


def create_quantum_dataloaders(train_path: str,
                             val_path: str,
                             batch_size: int = 32,
                             num_workers: int = 4,
                             **dataset_kwargs) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create training and validation dataloaders
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        batch_size: Batch size
        num_workers: Number of worker processes
        **dataset_kwargs: Additional dataset arguments
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = QuantumCircuitDataset(train_path, augment_data=True, **dataset_kwargs)
    val_dataset = QuantumCircuitDataset(val_path, augment_data=False, **dataset_kwargs)
    
    # Create collator
    collator = QuantumCircuitCollator(pad_token_id=train_dataset.get_vocab_size() - 1)
    
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
    dataset = QuantumCircuitDataset("data/raw/experiment_results.json", max_circuit_length=64)
    
    print("Dataset statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test sample
    sample = dataset[0]
    print(f"\nSample circuit:")
    for key, value in sample.items():
        print(f"  {key}: {value.shape} - {value}")
