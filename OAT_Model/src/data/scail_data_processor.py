#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import copy
from typing import Dict, List, Any
from pathlib import Path

class SCAILDataProcessor:
    """Process SCAIL data for enhanced fidelity prediction"""
    
    def __init__(self):
        self.scail_prefix = "scail"
    
    def is_scail_data(self, circuit_id: str) -> bool:
        """Check if circuit_id belongs to SCAIL dataset"""
        return circuit_id.startswith(self.scail_prefix)
    
    def filter_scail_for_fidelity(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter data so SCAIL circuits only predict fidelity and robust_fidelity"""
        filtered_data = []
        
        for item in data:
            circuit_id = item.get('circuit_id', '')
            
            if self.is_scail_data(circuit_id):
                # SCAIL data: only keep fidelity-related properties
                filtered_item = copy.deepcopy(item)
                properties = filtered_item.get('properties', {})
                
                # Keep only fidelity and robust_fidelity
                fidelity_properties = {}
                if 'fidelity' in properties:
                    fidelity_properties['fidelity'] = properties['fidelity']
                if 'robust_fidelity' in properties:
                    fidelity_properties['robust_fidelity'] = properties['robust_fidelity']
                
                # Remove other properties for SCAIL data
                filtered_item['properties'] = fidelity_properties
                filtered_data.append(filtered_item)
            else:
                # Non-SCAIL data: keep all properties but set fidelity to 1.0
                filtered_item = copy.deepcopy(item)
                properties = filtered_item.get('properties', {})
                
                # Set fidelity to 1.0 for non-SCAIL data (ideal quantum circuits)
                if 'fidelity' in properties:
                    properties['fidelity'] = 1.0
                if 'robust_fidelity' in properties:
                    properties['robust_fidelity'] = 1.0
                
                filtered_item['properties'] = properties
                filtered_data.append(filtered_item)
        
        return filtered_data
    
    def augment_scail_data(self, data: List[Dict[str, Any]], multiplier: int = 10) -> List[Dict[str, Any]]:
        """Augment SCAIL data by duplicating with slight variations"""
        augmented_data = []
        
        for item in data:
            circuit_id = item.get('circuit_id', '')
            
            if self.is_scail_data(circuit_id):
                # Add original
                augmented_data.append(item)
                
                # Add duplicates with slight noise
                for i in range(multiplier - 1):
                    augmented_item = copy.deepcopy(item)
                    
                    # Add slight noise to fidelity values (±0.001)
                    properties = augmented_item.get('properties', {})
                    
                    if 'fidelity' in properties:
                        noise = (np.random.random() - 0.5) * 0.002  # ±0.001
                        properties['fidelity'] = max(0.0, min(1.0, properties['fidelity'] + noise))
                    
                    if 'robust_fidelity' in properties:
                        noise = (np.random.random() - 0.5) * 0.002  # ±0.001
                        properties['robust_fidelity'] = max(0.0, min(1.0, properties['robust_fidelity'] + noise))
                    
                    # Update circuit_id to indicate augmentation
                    augmented_item['circuit_id'] = f"{circuit_id}_aug_{i+1}"
                    augmented_data.append(augmented_item)
            else:
                # Non-SCAIL data: keep as is
                augmented_data.append(item)
        
        return augmented_data
    
    def get_scail_loss_weights(self, circuit_ids: List[str], base_weights: Dict[str, float]) -> Dict[str, List[float]]:
        """Get per-sample loss weights with higher weights for SCAIL data"""
        weights = {prop: [] for prop in base_weights.keys()}
        
        for circuit_id in circuit_ids:
            if self.is_scail_data(circuit_id):
                # Higher weights for SCAIL data
                for prop in weights.keys():
                    if prop in ['fidelity', 'robust_fidelity']:
                        weights[prop].append(base_weights[prop] * 5.0)  # 5x weight for SCAIL fidelity
                    else:
                        weights[prop].append(0.0)  # No weight for other properties in SCAIL
            else:
                # Normal weights for non-SCAIL data
                for prop in weights.keys():
                    if prop in ['fidelity', 'robust_fidelity']:
                        weights[prop].append(base_weights[prop] * 0.1)  # Lower weight for non-SCAIL fidelity
                    else:
                        weights[prop].append(base_weights[prop])  # Normal weight for other properties
        
        return weights
    
    def process_dataset(self, input_file: str, output_file: str, augment_multiplier: int = 10):
        """Complete processing pipeline for SCAIL data"""
        print(f"🔄 Processing SCAIL data from {input_file}")
        
        # Load data
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Count original SCAIL samples
        scail_count = sum(1 for item in data if self.is_scail_data(item.get('circuit_id', '')))
        print(f"📊 Original SCAIL samples: {scail_count}")
        
        # Filter for fidelity-only prediction on SCAIL
        filtered_data = self.filter_scail_for_fidelity(data)
        print(f"✅ Filtered data for SCAIL fidelity prediction")
        
        # Augment SCAIL data
        augmented_data = self.augment_scail_data(filtered_data, augment_multiplier)
        
        # Count final SCAIL samples
        final_scail_count = sum(1 for item in augmented_data if self.is_scail_data(item.get('circuit_id', '')))
        print(f"📈 Final SCAIL samples: {final_scail_count} ({final_scail_count/scail_count:.1f}x increase)")
        
        # Save processed data
        with open(output_file, 'w') as f:
            json.dump(augmented_data, f, indent=2)
        
        print(f"💾 Processed data saved to {output_file}")
        return augmented_data

# Import numpy for noise generation
import numpy as np

if __name__ == "__main__":
    processor = SCAILDataProcessor()
    
    # Process the main dataset
    input_file = "dummy_experiment_results.json"
    output_file = "processed_experiment_results.json"
    
    if Path(input_file).exists():
        processor.process_dataset(input_file, output_file, augment_multiplier=10)
    else:
        print(f"❌ Input file not found: {input_file}")
