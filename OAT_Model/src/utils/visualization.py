"""
Visualization utilities for DiT training and evaluation
Advanced plotting and analysis tools
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import json
from pathlib import Path


# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_training_curves(loss_history_path: str, 
                        save_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (15, 8)) -> None:
    """
    Plot comprehensive training curves
    
    Args:
        loss_history_path: Path to loss history JSON
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Load loss history
    with open(loss_history_path, 'r') as f:
        history = json.load(f)
    
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Training Loss by Epoch
    train_epochs = [x['epoch'] for x in train_losses]
    train_loss_values = [x['loss'] for x in train_losses]
    
    ax1.plot(train_epochs, train_loss_values, 'b-', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Over Epochs')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(train_epochs, train_loss_values, 1)
    p = np.poly1d(z)
    ax1.plot(train_epochs, p(train_epochs), "r--", alpha=0.8, linewidth=1)
    
    # 2. Validation Loss by Step
    val_steps = [x['step'] for x in val_losses]
    val_loss_values = [x['loss'] for x in val_losses]
    
    ax2.plot(val_steps, val_loss_values, 'r-', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss Over Training Steps')
    ax2.grid(True, alpha=0.3)
    
    # 3. Combined Loss Comparison
    # Interpolate validation loss to match training epochs
    if val_steps and train_epochs:
        val_interp = np.interp(train_epochs, 
                              [x['epoch'] for x in val_losses], 
                              val_loss_values)
        
        ax3.plot(train_epochs, train_loss_values, 'b-', label='Training', linewidth=2)
        ax3.plot(train_epochs, val_interp, 'r-', label='Validation', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training vs Validation Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Loss Distribution
    ax4.hist(train_loss_values, bins=30, alpha=0.7, label='Training', density=True)
    if val_loss_values:
        ax4.hist(val_loss_values, bins=30, alpha=0.7, label='Validation', density=True)
    ax4.set_xlabel('Loss Value')
    ax4.set_ylabel('Density')
    ax4.set_title('Loss Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()


def plot_quantum_metrics(metrics: Dict[str, Any], 
                        save_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (16, 10)) -> None:
    """
    Plot quantum circuit evaluation metrics
    
    Args:
        metrics: Dictionary of evaluation metrics
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # Extract metrics for plotting
    quantum_props = metrics.get('quantum_properties', {})
    
    # 1. Circuit Depth Distribution
    if 'avg_depth_mean' in quantum_props:
        depth_mean = quantum_props['avg_depth_mean']
        depth_std = quantum_props['avg_depth_std']
        
        x = np.linspace(max(0, depth_mean - 3*depth_std), 
                       depth_mean + 3*depth_std, 100)
        y = np.exp(-0.5 * ((x - depth_mean) / depth_std)**2) / (depth_std * np.sqrt(2*np.pi))
        
        axes[0].plot(x, y, 'b-', linewidth=2)
        axes[0].axvline(depth_mean, color='r', linestyle='--', alpha=0.7)
        axes[0].set_xlabel('Circuit Depth')
        axes[0].set_ylabel('Density')
        axes[0].set_title(f'Depth Distribution (μ={depth_mean:.2f})')
        axes[0].grid(True, alpha=0.3)
    
    # 2. Gate Diversity
    if 'gate_diversity_mean' in quantum_props:
        diversity_mean = quantum_props['gate_diversity_mean']
        diversity_std = quantum_props['gate_diversity_std']
        
        axes[1].bar(['Generated'], [diversity_mean], yerr=[diversity_std], 
                   capsize=5, alpha=0.7, color='green')
        axes[1].set_ylabel('Gate Diversity')
        axes[1].set_title('Gate Diversity Score')
        axes[1].grid(True, alpha=0.3)
    
    # 3. Entanglement Capability
    if 'entanglement_capability_mean' in quantum_props:
        ent_mean = quantum_props['entanglement_capability_mean']
        ent_std = quantum_props['entanglement_capability_std']
        
        axes[2].bar(['Generated'], [ent_mean], yerr=[ent_std], 
                   capsize=5, alpha=0.7, color='purple')
        axes[2].set_ylabel('Entanglement Capability')
        axes[2].set_title('Entanglement Generation')
        axes[2].grid(True, alpha=0.3)
    
    # 4. Circuit Validity
    if 'circuit_validity_mean' in quantum_props:
        validity_mean = quantum_props['circuit_validity_mean']
        validity_std = quantum_props['circuit_validity_std']
        
        axes[3].bar(['Generated'], [validity_mean], yerr=[validity_std], 
                   capsize=5, alpha=0.7, color='orange')
        axes[3].set_ylabel('Validity Score')
        axes[3].set_title('Circuit Validity')
        axes[3].set_ylim(0, 1.1)
        axes[3].grid(True, alpha=0.3)
    
    # 5. Quality Metrics Radar Chart
    if all(key in quantum_props for key in ['avg_depth_mean', 'gate_diversity_mean', 
                                           'entanglement_capability_mean', 'circuit_validity_mean']):
        
        categories = ['Depth', 'Diversity', 'Entanglement', 'Validity', 'Efficiency']
        values = [
            min(1.0, quantum_props['avg_depth_mean'] / 10),
            quantum_props['gate_diversity_mean'] / 5,  # Normalize
            quantum_props['entanglement_capability_mean'],
            quantum_props['circuit_validity_mean'],
            quantum_props.get('avg_efficiency_mean', 0.5)
        ]
        
        # Close the radar chart
        values += values[:1]
        categories += categories[:1]
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=True)
        
        axes[4].plot(angles, values, 'o-', linewidth=2, alpha=0.7)
        axes[4].fill(angles, values, alpha=0.25)
        axes[4].set_xticks(angles[:-1])
        axes[4].set_xticklabels(categories[:-1])
        axes[4].set_ylim(0, 1)
        axes[4].set_title('Quality Metrics Radar')
        axes[4].grid(True, alpha=0.3)
    
    # 6. Comparison with Baseline (if available)
    comparison = metrics.get('comparison', {})
    if comparison:
        metric_names = []
        gen_values = []
        ref_values = []
        
        for key, value in comparison.items():
            if key.endswith('_generated'):
                base_key = key.replace('_generated', '')
                ref_key = base_key + '_reference'
                
                if ref_key in comparison:
                    metric_names.append(base_key.replace('_mean', ''))
                    gen_values.append(value)
                    ref_values.append(comparison[ref_key])
        
        if metric_names:
            x = np.arange(len(metric_names))
            width = 0.35
            
            axes[5].bar(x - width/2, gen_values, width, label='Generated', alpha=0.7)
            axes[5].bar(x + width/2, ref_values, width, label='Reference', alpha=0.7)
            
            axes[5].set_xlabel('Metrics')
            axes[5].set_ylabel('Values')
            axes[5].set_title('Generated vs Reference')
            axes[5].set_xticks(x)
            axes[5].set_xticklabels(metric_names, rotation=45)
            axes[5].legend()
            axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Quantum metrics plot saved to: {save_path}")
    
    plt.show()


def plot_circuit_analysis(circuits: List[Dict[str, Any]], 
                         title: str = "Circuit Analysis",
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (14, 8)) -> None:
    """
    Analyze and plot circuit properties
    
    Args:
        circuits: List of circuit dictionaries
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    """
    if not circuits:
        print("No circuits to analyze")
        return
    
    # Extract properties
    lengths = [len(c.get('gates', [])) for c in circuits]
    qubits = [c.get('qubits', 0) for c in circuits]
    
    # Gate analysis
    all_gates = []
    for circuit in circuits:
        all_gates.extend(circuit.get('gates', []))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Circuit Length Distribution
    ax1.hist(lengths, bins=30, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Circuit Length')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Circuit Length Distribution')
    ax1.axvline(np.mean(lengths), color='r', linestyle='--', 
                label=f'Mean: {np.mean(lengths):.1f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Qubit Count Distribution
    ax2.hist(qubits, bins=range(min(qubits), max(qubits)+2), 
             alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Qubits')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Qubit Count Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 3. Gate Type Distribution
    gate_counts = {}
    for gate in all_gates:
        gate_counts[gate] = gate_counts.get(gate, 0) + 1
    
    if gate_counts:
        gates_sorted = sorted(gate_counts.items(), key=lambda x: x[1], reverse=True)
        top_gates = gates_sorted[:15]  # Top 15 gates
        
        gate_names = [f'Gate_{g[0]}' for g in top_gates]
        gate_freqs = [g[1] for g in top_gates]
        
        ax3.bar(range(len(gate_names)), gate_freqs, alpha=0.7)
        ax3.set_xlabel('Gate Types')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Gate Type Distribution (Top 15)')
        ax3.set_xticks(range(len(gate_names)))
        ax3.set_xticklabels(gate_names, rotation=45)
        ax3.grid(True, alpha=0.3)
    
    # 4. Length vs Qubits Scatter
    ax4.scatter(qubits, lengths, alpha=0.6)
    ax4.set_xlabel('Number of Qubits')
    ax4.set_ylabel('Circuit Length')
    ax4.set_title('Circuit Length vs Qubits')
    
    # Add trend line
    if len(qubits) > 1:
        z = np.polyfit(qubits, lengths, 1)
        p = np.poly1d(z)
        ax4.plot(sorted(qubits), p(sorted(qubits)), "r--", alpha=0.8)
    
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Circuit analysis plot saved to: {save_path}")
    
    plt.show()


def create_training_dashboard(log_dir: str, 
                            save_path: Optional[str] = None) -> None:
    """
    Create comprehensive training dashboard
    
    Args:
        log_dir: Directory containing training logs
        save_path: Path to save the dashboard
    """
    log_path = Path(log_dir)
    
    # Load loss history
    loss_file = log_path / "loss_history.json"
    if not loss_file.exists():
        print(f"Loss history file not found: {loss_file}")
        return
    
    with open(loss_file, 'r') as f:
        history = json.load(f)
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Training curves
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:])
    ax3 = fig.add_subplot(gs[1, :2])
    ax4 = fig.add_subplot(gs[1, 2:])
    ax5 = fig.add_subplot(gs[2, :2])
    ax6 = fig.add_subplot(gs[2, 2:])
    
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    
    # Plot training loss
    train_epochs = [x['epoch'] for x in train_losses]
    train_values = [x['loss'] for x in train_losses]
    
    ax1.plot(train_epochs, train_values, 'b-', linewidth=2)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # Plot validation loss
    val_steps = [x['step'] for x in val_losses]
    val_values = [x['loss'] for x in val_losses]
    
    ax2.plot(val_steps, val_values, 'r-', linewidth=2)
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    
    # Combined view
    ax3.plot(train_epochs, train_values, 'b-', label='Training', linewidth=2)
    if val_values:
        val_epochs = [x['epoch'] for x in val_losses]
        ax3.plot(val_epochs, val_values, 'r-', label='Validation', linewidth=2)
    ax3.set_title('Training vs Validation')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Loss smoothing
    if len(train_values) > 10:
        window = min(10, len(train_values) // 5)
        smoothed = pd.Series(train_values).rolling(window=window).mean()
        ax4.plot(train_epochs, train_values, 'b-', alpha=0.3, label='Raw')
        ax4.plot(train_epochs, smoothed, 'b-', linewidth=2, label='Smoothed')
        ax4.set_title('Smoothed Training Loss')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Loss statistics
    stats_text = f"""
    Training Statistics:
    • Final Loss: {train_values[-1]:.4f}
    • Best Loss: {min(train_values):.4f}
    • Improvement: {((train_values[0] - train_values[-1]) / train_values[0] * 100):.1f}%
    
    Validation Statistics:
    • Final Val Loss: {val_values[-1]:.4f if val_values else 'N/A'}
    • Best Val Loss: {min(val_values):.4f if val_values else 'N/A'}
    """
    
    ax5.text(0.1, 0.5, stats_text, transform=ax5.transAxes, fontsize=12,
             verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax5.set_title('Training Statistics')
    ax5.axis('off')
    
    # Configuration info
    config = history.get('config', {})
    config_text = f"""
    Model Configuration:
    • Model: {config.get('model_config', {}).get('d_model', 'N/A')} dimensions
    • Layers: {config.get('model_config', {}).get('n_layers', 'N/A')}
    • Heads: {config.get('model_config', {}).get('n_heads', 'N/A')}
    • Batch Size: {config.get('batch_size', 'N/A')}
    • Learning Rate: {config.get('learning_rate', 'N/A')}
    • Optimizer: {config.get('optimizer', 'N/A')}
    """
    
    ax6.text(0.1, 0.5, config_text, transform=ax6.transAxes, fontsize=12,
             verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax6.set_title('Model Configuration')
    ax6.axis('off')
    
    plt.suptitle('DiT Training Dashboard', fontsize=20, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training dashboard saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Visualization utilities loaded successfully!")
    
    # Test with dummy data
    dummy_history = {
        'train_losses': [{'epoch': i, 'loss': 2.0 * np.exp(-i/10) + 0.1} for i in range(50)],
        'val_losses': [{'step': i*100, 'epoch': i, 'loss': 2.2 * np.exp(-i/10) + 0.15} for i in range(0, 50, 5)]
    }
    
    # Save dummy history
    with open('dummy_history.json', 'w') as f:
        json.dump(dummy_history, f)
    
    # Test plotting
    plot_training_curves('dummy_history.json')
