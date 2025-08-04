"""
Gate Prediction Dataset for Quantum Transformer
Dataset loader for next-gate prediction using dummy_experiment_results.json format
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import sys  

sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))

from gates import QuantumGateRegistry


class GatePredictionDataset(Dataset):
    """
    Dataset for next-gate prediction using quantum circuit sequences
    Supports dummy_experiment_results.json format
    """
    
    def __init__(self,
                 data_path: str,
                 max_circuit_length: int = 256,
                 max_qubits: int = 32,
                 max_parameters: int = 8,
                 augment_data: bool = False,
                 train_mode: bool = True,
                 min_sequence_length: int = 5):
        """
        Initialize gate prediction dataset
        
        Args:
            data_path: Path to dummy_experiment_results.json file
            max_circuit_length: Maximum circuit length to consider
            max_qubits: Maximum number of qubits
            max_parameters: Maximum number of parameters per gate
            augment_data: Whether to apply data augmentation
            train_mode: Whether in training mode
            min_sequence_length: Minimum sequence length for training
        """
        self.data_path = data_path
        self.max_circuit_length = max_circuit_length
        self.max_qubits = max_qubits
        self.max_parameters = max_parameters
        self.augment_data = augment_data
        self.train_mode = train_mode
        self.min_sequence_length = min_sequence_length
        
        # Initialize gate registry
        self.gate_registry = QuantumGateRegistry()
        self.gate_vocab = self.gate_registry.get_gate_vocab()
        self.gate_to_idx = {gate: idx for idx, gate in enumerate(self.gate_vocab)}
        self.idx_to_gate = {idx: gate for gate, idx in self.gate_to_idx.items()}
        
        # Add special tokens
        self.pad_token_idx = len(self.gate_vocab)
        self.unk_token_idx = len(self.gate_vocab) + 1
        
        # Load and process data
        self.data = self._load_data()
        self.sequences = self._create_sequences()
        
        print(f"Loaded {len(self.sequences)} training sequences from {len(self.data)} circuits")
        print(f"Gate vocabulary size: {len(self.gate_vocab)}")
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from unified JSON file format"""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        processed_data = []
        
        # Check if data has the unified format with separate 'circuits' and 'results'
        if 'circuits' in data and 'results' in data:
            # New unified format
            circuits = data['circuits']
            results = data['results']
            
            # Process each circuit
            for circuit_id, circuit_data in circuits.items():
                # Find corresponding result data
                if circuit_id not in results:
                    continue
                
                result_data = results[circuit_id]
                
                # Extract circuit properties
                circuit_info = {
                    'circuit_id': circuit_id,
                    'num_qubits': circuit_data['num_qubits'],
                    'gates': circuit_data['gates'],
                    'fidelity': result_data.get('fidelity', 0.0),
                    'robust_fidelity': result_data.get('robust_fidelity', 0.0),
                    'expressibility': self._extract_expressibility(result_data.get('expressibility', {})),
                    'entanglement': result_data.get('entanglement', 0.0),
                    'depth': result_data.get('depth', len(circuit_data['gates'])),
                    'two_qubit_ratio': self._calculate_two_qubit_ratio(circuit_data['gates']),
                    'timestamp': result_data.get('timestamp', '')
                }
                
                processed_data.append(circuit_info)
        
        else:
            # Legacy format (original dummy_experiment_results.json)
            # Process each circuit from the old format
            for circuit_id, circuit_data in data.get('circuits', {}).items():
                # Find corresponding result data
                result_data = None
                for result in data.get('results', []):
                    if result['circuit_id'] == circuit_id:
                        result_data = result
                        break
                
                if result_data is None:
                    continue
                    
                # Extract circuit properties
                circuit_info = {
                    'circuit_id': circuit_id,
                    'num_qubits': circuit_data['num_qubits'],
                    'gates': circuit_data['gates'],
                    'fidelity': result_data.get('fidelity', 0.0),
                    'expressibility': result_data.get('expressibility', {}).get('expressibility', 0.0),
                    'entanglement': result_data.get('entanglement', 0.0),
                    'depth': result_data.get('depth', len(circuit_data['gates'])),
                    'two_qubit_ratio': self._calculate_two_qubit_ratio(circuit_data['gates'])
                }
                
                processed_data.append(circuit_info)
        
        return processed_data
    
    def _extract_expressibility(self, expressibility_data: Dict) -> float:
        """Extract expressibility value from different formats"""
        if isinstance(expressibility_data, dict):
            # Try different possible keys
            if 'expressibility' in expressibility_data:
                return expressibility_data['expressibility']
            elif 'kl_divergence' in expressibility_data:
                # Convert KL divergence to expressibility (simplified)
                kl_div = expressibility_data['kl_divergence']
                return max(0.0, 1.0 - kl_div / 2.0)  # Simple conversion
            elif 'js_divergence' in expressibility_data:
                return expressibility_data['js_divergence']
            else:
                return 0.0
        elif isinstance(expressibility_data, (int, float)):
            return float(expressibility_data)
        else:
            return 0.0
    
    def _calculate_two_qubit_ratio(self, gates: List[Dict]) -> float:
        """Calculate two-qubit gate ratio"""
        total_gates = len(gates)
        if total_gates == 0:
            return 0.0
            
        two_qubit_gates = sum(1 for gate in gates if len(gate['qubits']) >= 2)
        return two_qubit_gates / total_gates
    
    def _create_sequences(self) -> List[Dict[str, Any]]:
        """Create training sequences from circuits"""
        sequences = []
        
        for circuit in self.data:
            gates = circuit['gates']
            
            # Skip circuits that are too short
            if len(gates) < self.min_sequence_length:
                continue
                
            # Create multiple sequences from each circuit
            for start_idx in range(len(gates) - self.min_sequence_length + 1):
                for end_idx in range(start_idx + self.min_sequence_length, 
                                   min(start_idx + self.max_circuit_length + 1, len(gates) + 1)):
                    
                    sequence_gates = gates[start_idx:end_idx]
                    
                    sequence_info = {
                        'circuit_id': circuit['circuit_id'],
                        'gates': sequence_gates,
                        'num_qubits': circuit['num_qubits'],
                        'target_fidelity': circuit['fidelity'],
                        'target_expressibility': circuit['expressibility'],
                        'target_entanglement': circuit['entanglement'],
                        'target_depth': circuit['depth'],
                        'two_qubit_ratio': circuit['two_qubit_ratio']
                    }
                    
                    sequences.append(sequence_info)
        
        return sequences
    
    def _encode_gate_sequence(self, gates: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode gate sequence into tensors
        
        Args:
            gates: List of gate dictionaries
            
        Returns:
            Tuple of (gate_indices, qubit_indices, parameters)
        """
        seq_len = len(gates)
        
        # Initialize tensors
        gate_indices = torch.full((seq_len,), self.pad_token_idx, dtype=torch.long)
        qubit_indices = torch.zeros((seq_len, self.max_qubits), dtype=torch.long)
        parameters = torch.zeros((seq_len, self.max_parameters), dtype=torch.float)
        
        for i, gate in enumerate(gates):
            # Gate type
            gate_name = gate['name']
            if gate_name in self.gate_to_idx:
                gate_indices[i] = self.gate_to_idx[gate_name]
            else:
                gate_indices[i] = self.unk_token_idx
            
            # Qubits
            gate_qubits = gate['qubits']
            for j, qubit in enumerate(gate_qubits):
                if j < self.max_qubits:
                    qubit_indices[i, j] = qubit
            
            # Parameters
            gate_params = gate.get('parameters', [])
            for j, param in enumerate(gate_params):
                if j < self.max_parameters:
                    parameters[i, j] = param
        
        return gate_indices, qubit_indices, parameters
    
    def _create_requirements_dict(self, sequence_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Create requirements dictionary for the sequence"""
        requirements = {
            'target_fidelity': torch.tensor([sequence_info['target_fidelity']], dtype=torch.float),
            'target_expressibility': torch.tensor([sequence_info['target_expressibility']], dtype=torch.float),
            'target_entanglement': torch.tensor([sequence_info['target_entanglement']], dtype=torch.float),
            'target_depth': torch.tensor([sequence_info['target_depth']], dtype=torch.float),
            'num_qubits': torch.tensor(sequence_info['num_qubits'], dtype=torch.long),
            'max_depth': torch.tensor([sequence_info['target_depth'] * 1.5], dtype=torch.float),  # Allow some flexibility
            'two_qubit_ratio': torch.tensor([sequence_info['two_qubit_ratio']], dtype=torch.float)
        }
        
        return requirements
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample
        
        Returns:
            Dictionary containing:
                - input_gates: Gate sequence for input [seq_len-1]
                - input_qubits: Qubit sequence for input [seq_len-1, max_qubits]
                - input_parameters: Parameter sequence for input [seq_len-1, max_parameters]
                - target_gate: Target gate index [1]
                - target_qubits: Target qubit indices [max_qubits]
                - target_parameters: Target parameters [max_parameters]
                - requirements: Requirements dictionary
                - seq_len: Sequence length
        """
        sequence_info = self.sequences[idx]
        gates = sequence_info['gates']
        
        # Apply data augmentation if enabled
        if self.augment_data and self.train_mode:
            gates = self._augment_sequence(gates)
        
        # Encode the full sequence
        gate_indices, qubit_indices, parameters = self._encode_gate_sequence(gates)
        
        # Split into input and target
        seq_len = len(gates)
        
        # Input: all gates except the last one
        input_gates = gate_indices[:-1]
        input_qubits = qubit_indices[:-1]
        input_parameters = parameters[:-1]
        
        # Target: the last gate
        target_gate = gate_indices[-1]
        target_qubits = qubit_indices[-1]
        target_parameters = parameters[-1]
        
        # Create requirements
        requirements = self._create_requirements_dict(sequence_info)
        
        return {
            'input_gates': input_gates,
            'input_qubits': input_qubits,
            'input_parameters': input_parameters,
            'target_gate': target_gate,
            'target_qubits': target_qubits,
            'target_parameters': target_parameters,
            'requirements': requirements,
            'seq_len': torch.tensor(seq_len - 1, dtype=torch.long),
            'circuit_id': sequence_info['circuit_id']
        }
    
    def _augment_sequence(self, gates: List[Dict]) -> List[Dict]:
        """Apply data augmentation to gate sequence"""
        if not self.train_mode:
            return gates
        
        augmented_gates = gates.copy()
        
        # Random gate parameter perturbation
        if random.random() < 0.3:
            for gate in augmented_gates:
                if gate.get('parameters'):
                    for i in range(len(gate['parameters'])):
                        noise = random.gauss(0, 0.1)
                        gate['parameters'][i] += noise
        
        # Random qubit permutation (with small probability)
        if random.random() < 0.1:
            # Create a random permutation for qubits
            max_qubit = max(max(gate['qubits']) for gate in augmented_gates if gate['qubits'])
            perm = list(range(max_qubit + 1))
            random.shuffle(perm)
            
            for gate in augmented_gates:
                gate['qubits'] = [perm[q] for q in gate['qubits']]
        
        return augmented_gates


class GatePredictionCollator:
    """Custom collator for gate prediction batches"""
    
    def __init__(self, pad_token_id: int, max_seq_len: int = 256):
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch of gate prediction samples"""
        batch_size = len(batch)
        
        # Find maximum sequence length in batch
        max_len = min(max(item['seq_len'].item() for item in batch), self.max_seq_len)
        
        # Initialize batch tensors
        input_gates = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        input_qubits = torch.zeros((batch_size, max_len, batch[0]['input_qubits'].size(-1)), dtype=torch.long)
        input_parameters = torch.zeros((batch_size, max_len, batch[0]['input_parameters'].size(-1)), dtype=torch.float)
        
        target_gates = torch.zeros(batch_size, dtype=torch.long)
        target_qubits = torch.zeros((batch_size, batch[0]['target_qubits'].size(-1)), dtype=torch.long)
        target_parameters = torch.zeros((batch_size, batch[0]['target_parameters'].size(-1)), dtype=torch.float)
        
        seq_lens = torch.zeros(batch_size, dtype=torch.long)
        
        # Batch requirements
        requirements = {}
        for key in batch[0]['requirements'].keys():
            if batch[0]['requirements'][key].dim() == 0:
                requirements[key] = torch.stack([item['requirements'][key] for item in batch])
            else:
                requirements[key] = torch.cat([item['requirements'][key].unsqueeze(0) for item in batch], dim=0)
        
        # Fill batch tensors
        for i, item in enumerate(batch):
            seq_len = min(item['seq_len'].item(), max_len)
            seq_lens[i] = seq_len
            
            input_gates[i, :seq_len] = item['input_gates'][:seq_len]
            input_qubits[i, :seq_len] = item['input_qubits'][:seq_len]
            input_parameters[i, :seq_len] = item['input_parameters'][:seq_len]
            
            target_gates[i] = item['target_gate']
            target_qubits[i] = item['target_qubits']
            target_parameters[i] = item['target_parameters']
        
        # Create attention mask
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        for i, seq_len in enumerate(seq_lens):
            attention_mask[i, :seq_len] = True
        
        return {
            'input_gates': input_gates,
            'input_qubits': input_qubits,
            'input_parameters': input_parameters,
            'target_gates': target_gates,
            'target_qubits': target_qubits,
            'target_parameters': target_parameters,
            'requirements': requirements,
            'attention_mask': attention_mask,
            'seq_lens': seq_lens
        }


def create_gate_prediction_dataloaders(
    train_path: str,
    val_path: str = None,
    batch_size: int = 32,
    num_workers: int = 4,
    max_circuit_length: int = 256,
    max_qubits: int = 32,
    max_parameters: int = 8,
    **dataset_kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation dataloaders for gate prediction
    
    Args:
        train_path: Path to training data JSON file
        val_path: Path to validation data JSON file (optional)
        batch_size: Batch size
        num_workers: Number of worker processes
        max_circuit_length: Maximum circuit length
        max_qubits: Maximum number of qubits
        max_parameters: Maximum parameters per gate
        **dataset_kwargs: Additional dataset arguments
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = GatePredictionDataset(
        train_path,
        max_circuit_length=max_circuit_length,
        max_qubits=max_qubits,
        max_parameters=max_parameters,
        train_mode=True,
        **dataset_kwargs
    )
    
    val_dataset = None
    if val_path and os.path.exists(val_path):
        val_dataset = GatePredictionDataset(
            val_path,
            max_circuit_length=max_circuit_length,
            max_qubits=max_qubits,
            max_parameters=max_parameters,
            train_mode=False,
            **dataset_kwargs
        )
    
    # Create collator
    collator = GatePredictionCollator(
        pad_token_id=train_dataset.pad_token_idx,
        max_seq_len=max_circuit_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True
        )
    
    print(f"Created dataloaders:")
    print(f"  Training: {len(train_dataset)} samples, {len(train_loader)} batches")
    if val_loader:
        print(f"  Validation: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    dataset_path = r"c:\Users\jungh\Documents\GitHub\Kaist\dummy_experiment_results.json"
    
    if os.path.exists(dataset_path):
        # Create dataset
        dataset = GatePredictionDataset(
            dataset_path,
            max_circuit_length=64,
            max_qubits=8,
            augment_data=True
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test sample
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Input gates shape: {sample['input_gates'].shape}")
        print(f"Input qubits shape: {sample['input_qubits'].shape}")
        print(f"Input parameters shape: {sample['input_parameters'].shape}")
        print(f"Target gate: {sample['target_gate']}")
        print(f"Requirements keys: {sample['requirements'].keys()}")
        
        # Test dataloader
        train_loader, _ = create_gate_prediction_dataloaders(
            dataset_path,
            batch_size=4,
            num_workers=0
        )
        
        batch = next(iter(train_loader))
        print(f"\nBatch keys: {batch.keys()}")
        print(f"Batch input gates shape: {batch['input_gates'].shape}")
        print(f"Batch target gates shape: {batch['target_gates'].shape}")
        print(f"Batch requirements keys: {batch['requirements'].keys()}")
        
    else:
        print(f"Dataset file not found: {dataset_path}")
