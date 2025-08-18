#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import copy
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple
from .quantum_circuit_dataset import CircuitData, QuantumCircuitDataset
import random

class AugmentedQuantumDataset(Dataset):
    """In-memory data augmentation for quantum circuits"""
    
    def __init__(self, base_dataset: QuantumCircuitDataset, 
                 mixup_samples: int = 500,
                 noise_samples: int = 500, 
                 param_random_samples: int = 1000):
        self.base_dataset = base_dataset
        self.mixup_samples = mixup_samples
        self.noise_samples = noise_samples
        self.param_random_samples = param_random_samples
        
        # Generate augmented data in memory
        self.augmented_data = self._generate_augmented_data()
        
    def _generate_augmented_data(self) -> List[CircuitData]:
        """Generate all augmented data in memory"""
        augmented = []
        base_data = list(self.base_dataset.circuit_data)
        
        # 1. Original data
        augmented.extend(base_data)
        print(f"📊 Original data: {len(base_data)} samples")
        
        # 2. Mixup augmentation (500 samples)
        mixup_data = self._generate_mixup_data(base_data, self.mixup_samples)
        augmented.extend(mixup_data)
        print(f"🔀 Mixup augmentation: {len(mixup_data)} samples")
        
        # 3. Noise augmentation (500 samples)
        noise_data = self._generate_noise_data(base_data, self.noise_samples)
        augmented.extend(noise_data)
        print(f"🔊 Noise augmentation: {len(noise_data)} samples")
        
        # 4. Parameter randomization (1000 samples)
        param_data = self._generate_param_random_data(base_data, self.param_random_samples)
        augmented.extend(param_data)
        print(f"🎲 Parameter randomization: {len(param_data)} samples")
        
        print(f"✅ Total augmented dataset: {len(augmented)} samples")
        return augmented
    
    def _generate_mixup_data(self, base_data: List[CircuitData], num_samples: int) -> List[CircuitData]:
        """Generate mixup augmented data"""
        mixup_data = []
        
        for i in range(num_samples):
            # Randomly select two samples
            idx1, idx2 = random.sample(range(len(base_data)), 2)
            sample1, sample2 = base_data[idx1], base_data[idx2]
            
            # Skip if either sample has no measurement result
            if not sample1.measurement_result or not sample2.measurement_result:
                continue
                
            # Mixup lambda
            lam = np.random.beta(1.0, 1.0)  # Beta distribution for mixup
            
            # Create mixed circuit data
            mixed_data = self._mixup_circuit_data(sample1, sample2, lam, i)
            if mixed_data:
                mixup_data.append(mixed_data)
        
        return mixup_data
    
    def _mixup_circuit_data(self, sample1: CircuitData, sample2: CircuitData, 
                           lam: float, aug_id: int) -> CircuitData:
        """Mix two circuit data samples"""
        # Create new circuit spec (use sample1 as base)
        mixed_spec = copy.deepcopy(sample1.circuit_spec)
        mixed_spec.circuit_id = f"mixup_{aug_id}_{sample1.circuit_id}_{sample2.circuit_id}"
        
        # Mix measurement results
        result1 = sample1.measurement_result
        result2 = sample2.measurement_result
        
        mixed_result = copy.deepcopy(result1)
        mixed_result.circuit_id = mixed_spec.circuit_id
        
        # Mix properties with lambda
        mixed_result.fidelity = lam * result1.fidelity + (1 - lam) * result2.fidelity
        
        if result1.entanglement is not None and result2.entanglement is not None:
            mixed_result.entanglement = lam * result1.entanglement + (1 - lam) * result2.entanglement
        
        if result1.robust_fidelity is not None and result2.robust_fidelity is not None:
            mixed_result.robust_fidelity = lam * result1.robust_fidelity + (1 - lam) * result2.robust_fidelity
        
        # Mix expressibility (KL divergence)
        if (result1.expressibility and result2.expressibility and 
            'kl_divergence' in result1.expressibility and 'kl_divergence' in result2.expressibility):
            mixed_expr = copy.deepcopy(result1.expressibility)
            mixed_expr['kl_divergence'] = (lam * result1.expressibility['kl_divergence'] + 
                                         (1 - lam) * result2.expressibility['kl_divergence'])
            mixed_result.expressibility = mixed_expr
        
        return CircuitData(circuit_spec=mixed_spec, measurement_result=mixed_result)
    
    def _generate_noise_data(self, base_data: List[CircuitData], num_samples: int) -> List[CircuitData]:
        """Generate noise-augmented data"""
        noise_data = []
        
        for i in range(num_samples):
            # Randomly select a sample
            base_sample = random.choice(base_data)
            
            if not base_sample.measurement_result:
                continue
            
            # Add noise to properties
            noisy_data = self._add_noise_to_sample(base_sample, i)
            if noisy_data:
                noise_data.append(noisy_data)
        
        return noise_data
    
    def _add_noise_to_sample(self, sample: CircuitData, aug_id: int) -> CircuitData:
        """Add noise to a single sample"""
        # Create noisy circuit spec
        noisy_spec = copy.deepcopy(sample.circuit_spec)
        noisy_spec.circuit_id = f"noise_{aug_id}_{sample.circuit_id}"
        
        # Create noisy measurement result
        noisy_result = copy.deepcopy(sample.measurement_result)
        noisy_result.circuit_id = noisy_spec.circuit_id
        
        # Add Gaussian noise to properties
        noise_std = 0.01  # 1% noise
        
        # Fidelity noise (clamp to [0, 1])
        fidelity_noise = np.random.normal(0, noise_std)
        noisy_result.fidelity = np.clip(sample.measurement_result.fidelity + fidelity_noise, 0.0, 1.0)
        
        # Entanglement noise (clamp to [0, inf))
        if sample.measurement_result.entanglement is not None:
            entanglement_noise = np.random.normal(0, noise_std)
            noisy_result.entanglement = max(0.0, sample.measurement_result.entanglement + entanglement_noise)
        
        # Robust fidelity noise
        if sample.measurement_result.robust_fidelity is not None:
            robust_fidelity_noise = np.random.normal(0, noise_std)
            noisy_result.robust_fidelity = np.clip(
                sample.measurement_result.robust_fidelity + robust_fidelity_noise, 0.0, 1.0
            )
        
        # Expressibility noise
        if (sample.measurement_result.expressibility and 
            'kl_divergence' in sample.measurement_result.expressibility):
            noisy_expr = copy.deepcopy(sample.measurement_result.expressibility)
            kl_noise = np.random.normal(0, noise_std * 10)  # Larger noise for KL divergence
            original_kl = sample.measurement_result.expressibility['kl_divergence']
            noisy_expr['kl_divergence'] = max(0.0, original_kl + kl_noise)
            noisy_result.expressibility = noisy_expr
        
        return CircuitData(circuit_spec=noisy_spec, measurement_result=noisy_result)
    
    def _generate_param_random_data(self, base_data: List[CircuitData], num_samples: int) -> List[CircuitData]:
        """Generate parameter randomization augmented data"""
        param_data = []
        
        for i in range(num_samples):
            # Randomly select a sample
            base_sample = random.choice(base_data)
            
            # Randomize gate parameters
            param_random_data = self._randomize_gate_parameters(base_sample, i)
            if param_random_data:
                param_data.append(param_random_data)
        
        return param_data
    
    def _randomize_gate_parameters(self, sample: CircuitData, aug_id: int) -> CircuitData:
        """Randomize gate parameters while keeping circuit structure"""
        # Create new circuit spec with randomized parameters
        random_spec = copy.deepcopy(sample.circuit_spec)
        random_spec.circuit_id = f"param_rand_{aug_id}_{sample.circuit_id}"
        
        # Randomize gate parameters
        for gate in random_spec.gates:
            if gate.parameters:  # Only if gate has parameters
                # Randomize each parameter
                new_params = []
                for param in gate.parameters:
                    if isinstance(param, (int, float)):
                        # For rotation gates, randomize angle in [0, 2π]
                        if gate.name.lower() in ['rx', 'ry', 'rz', 'u1', 'u2', 'u3']:
                            new_param = np.random.uniform(0, 2 * np.pi)
                        else:
                            # For other parameterized gates, add small noise
                            noise = np.random.normal(0, 0.1)
                            new_param = param + noise
                        new_params.append(new_param)
                    else:
                        new_params.append(param)  # Keep non-numeric parameters
                gate.parameters = new_params
        
        # Keep the same measurement result (since structure is same, properties should be similar)
        # But add small noise to account for parameter changes
        random_result = copy.deepcopy(sample.measurement_result)
        random_result.circuit_id = random_spec.circuit_id
        
        # Add small noise to properties due to parameter changes
        param_noise_std = 0.005  # Smaller noise for parameter changes
        
        if random_result.fidelity is not None:
            fidelity_noise = np.random.normal(0, param_noise_std)
            random_result.fidelity = np.clip(random_result.fidelity + fidelity_noise, 0.0, 1.0)
        
        if random_result.entanglement is not None:
            entanglement_noise = np.random.normal(0, param_noise_std)
            random_result.entanglement = max(0.0, random_result.entanglement + entanglement_noise)
        
        if (random_result.expressibility and 'kl_divergence' in random_result.expressibility):
            expr_noise = np.random.normal(0, param_noise_std * 5)
            original_kl = random_result.expressibility['kl_divergence']
            random_result.expressibility['kl_divergence'] = max(0.0, original_kl + expr_noise)
        
        return CircuitData(circuit_spec=random_spec, measurement_result=random_result)
    
    def __len__(self) -> int:
        return len(self.augmented_data)
    
    def __getitem__(self, idx: int) -> CircuitData:
        return self.augmented_data[idx]

def create_augmented_datasets(base_train_dataset: QuantumCircuitDataset,
                            base_val_dataset: QuantumCircuitDataset,
                            base_test_dataset: QuantumCircuitDataset,
                            mixup_samples: int = 500,
                            noise_samples: int = 500,
                            param_random_samples: int = 1000) -> Tuple[AugmentedQuantumDataset, QuantumCircuitDataset, QuantumCircuitDataset]:
    """Create augmented datasets (only augment training set)"""
    
    print("🚀 Creating augmented datasets...")
    
    # Only augment training dataset
    augmented_train = AugmentedQuantumDataset(
        base_train_dataset,
        mixup_samples=mixup_samples,
        noise_samples=noise_samples,
        param_random_samples=param_random_samples
    )
    
    # Keep validation and test sets unchanged
    print(f"📊 Dataset sizes:")
    print(f"  Train (augmented): {len(augmented_train)}")
    print(f"  Validation: {len(base_val_dataset)}")
    print(f"  Test: {len(base_test_dataset)}")
    
    return augmented_train, base_val_dataset, base_test_dataset

if __name__ == "__main__":
    # Test augmentation
    from quantum_circuit_dataset import DatasetManager
    
    # Load base dataset
    manager = DatasetManager(unified_data_path="dummy_experiment_results.json")
    train_dataset, val_dataset, test_dataset = manager.split_dataset()
    
    # Create augmented dataset
    aug_train, aug_val, aug_test = create_augmented_datasets(
        train_dataset, val_dataset, test_dataset,
        mixup_samples=100,  # Smaller for testing
        noise_samples=100,
        param_random_samples=200
    )
    
    print(f"✅ Augmentation test completed!")
    print(f"Original train size: {len(train_dataset)}")
    print(f"Augmented train size: {len(aug_train)}")
    
    # Test a sample
    if len(aug_train) > 0:
        sample = aug_train[0]
        print(f"Sample circuit ID: {sample.circuit_id}")
        print(f"Sample fidelity: {sample.measurement_result.fidelity}")
