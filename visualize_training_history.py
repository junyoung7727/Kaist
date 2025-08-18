#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def load_training_history(filepath):
    """Load training history from JSON file"""
    with open(filepath, 'r') as f:
        history = json.load(f)
    return history

def extract_metrics(history):
    """Extract relevant metrics from history data"""
    epochs = [entry.get('epoch', i) for i, entry in enumerate(history)]
    
    # Training and validation loss
    train_loss = [entry.get('train_loss', None) for entry in history]
    val_loss = [entry.get('val_loss', None) for entry in history]
    
    # Property metrics
    metrics = {
        'entanglement': {
            'train': [entry.get('train_entanglement', None) for entry in history],
            'val': [entry.get('val_entanglement', None) for entry in history],
            'mae': [entry.get('val_entanglement_mae', None) for entry in history],
            'rmse': [entry.get('val_entanglement_rmse', None) for entry in history],
            'r2': [entry.get('val_entanglement_r2', None) for entry in history],
            'corr': [entry.get('val_entanglement_corr', None) for entry in history],
        },
        'fidelity': {
            'train': [entry.get('train_fidelity', None) for entry in history],
            'val': [entry.get('val_fidelity', None) for entry in history],
            'mae': [entry.get('val_fidelity_mae', None) for entry in history],
            'rmse': [entry.get('val_fidelity_rmse', None) for entry in history],
            'r2': [entry.get('val_fidelity_r2', None) for entry in history],
            'corr': [entry.get('val_fidelity_corr', None) for entry in history],
        },
        'expressibility': {
            'train': [entry.get('train_expressibility', None) for entry in history],
            'val': [entry.get('val_expressibility', None) for entry in history],
            'mae': [entry.get('val_expressibility_mae', None) for entry in history],
            'rmse': [entry.get('val_expressibility_rmse', None) for entry in history],
            'r2': [entry.get('val_expressibility_r2', None) for entry in history],
            'corr': [entry.get('val_expressibility_corr', None) for entry in history],
        }
    }
    
    # Learning rate and duration
    learning_rate = [entry.get('learning_rate', None) for entry in history]
    duration = [entry.get('duration_sec', None) for entry in history]
    
    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'metrics': metrics,
        'learning_rate': learning_rate,
        'duration': duration
    }

def plot_losses(data, output_dir):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(data['epochs'], data['train_loss'], 'b-', label='Train Loss')
    plt.plot(data['epochs'], data['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'loss_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_property_metrics(data, output_dir):
    """Plot property metrics (entanglement, fidelity, expressibility)"""
    properties = ['entanglement', 'fidelity', 'expressibility']
    
    # Plot each property
    for prop in properties:
        plt.figure(figsize=(10, 6))
        plt.plot(data['epochs'], data['metrics'][prop]['train'], 'b-', label=f'Train {prop.capitalize()}')
        plt.plot(data['epochs'], data['metrics'][prop]['val'], 'r-', label=f'Val {prop.capitalize()}')
        plt.title(f'{prop.capitalize()} During Training')
        plt.xlabel('Epoch')
        plt.ylabel(f'{prop.capitalize()} Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, f'{prop}_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Combined property plot
    plt.figure(figsize=(12, 8))
    
    for i, prop in enumerate(properties):
        plt.subplot(3, 1, i+1)
        plt.plot(data['epochs'], data['metrics'][prop]['train'], 'b-', label=f'Train {prop.capitalize()}')
        plt.plot(data['epochs'], data['metrics'][prop]['val'], 'r-', label=f'Val {prop.capitalize()}')
        plt.title(f'{prop.capitalize()}')
        plt.xlabel('Epoch' if i == 2 else '')
        plt.ylabel(f'{prop.capitalize()}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_property_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_evaluation_metrics(data, output_dir):
    """Plot evaluation metrics (MAE, RMSE, R2, Correlation)"""
    properties = ['entanglement', 'fidelity', 'expressibility']
    metric_types = ['mae', 'rmse', 'r2', 'corr']
    metric_labels = ['Mean Absolute Error', 'Root Mean Square Error', 'R² Score', 'Correlation']
    
    # Plot each metric type
    for mt, label in zip(metric_types, metric_labels):
        plt.figure(figsize=(10, 6))
        
        for prop in properties:
            plt.plot(data['epochs'], data['metrics'][prop][mt], label=f'{prop.capitalize()}')
        
        plt.title(f'{label} During Training')
        plt.xlabel('Epoch')
        plt.ylabel(label)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, f'{mt}_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Per property evaluation metrics
    for prop in properties:
        plt.figure(figsize=(12, 10))
        
        for i, (mt, label) in enumerate(zip(metric_types, metric_labels)):
            plt.subplot(2, 2, i+1)
            plt.plot(data['epochs'], data['metrics'][prop][mt], 'g-')
            plt.title(f'{prop.capitalize()} - {label}')
            plt.xlabel('Epoch')
            plt.ylabel(label)
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prop}_evaluation_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_learning_rate(data, output_dir):
    """Plot learning rate over epochs"""
    plt.figure(figsize=(10, 4))
    plt.plot(data['epochs'], data['learning_rate'], 'g-')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_epoch_duration(data, output_dir):
    """Plot epoch duration over training"""
    plt.figure(figsize=(10, 4))
    plt.bar(data['epochs'], data['duration'], color='teal', alpha=0.7)
    plt.title('Epoch Duration')
    plt.xlabel('Epoch')
    plt.ylabel('Duration (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'epoch_duration.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # File path
    history_file = 'training_history.json'
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'training_visualization_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process data
    history = load_training_history(history_file)
    data = extract_metrics(history)
    
    # Generate plots
    plot_losses(data, output_dir)
    plot_property_metrics(data, output_dir)
    plot_evaluation_metrics(data, output_dir)
    plot_learning_rate(data, output_dir)
    plot_epoch_duration(data, output_dir)
    
    print(f"Visualizations saved to {output_dir}/")

if __name__ == "__main__":
    main()
