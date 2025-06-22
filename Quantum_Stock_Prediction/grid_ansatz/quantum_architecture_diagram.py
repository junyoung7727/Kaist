#!/usr/bin/env python3
"""
Quantum AI System Architecture Visualization
Complete Quantum AI System Architecture Diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle
import numpy as np

def create_quantum_architecture_diagram():
    """Create complete quantum AI system architecture diagram"""
    
    # Create large canvas
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Color palette
    colors = {
        'input': '#E3F2FD',      # Light blue
        'embedding': '#BBDEFB',   # Blue
        'transformer': '#90CAF9', # Dark blue
        'diffusion': '#FFE0B2',   # Light orange
        'output': '#C8E6C9',      # Light green
        'analysis': '#F8BBD9',    # Light pink
        'connection': '#757575'   # Gray
    }
    
    # Title
    ax.text(10, 15.5, 'Quantum AI System Architecture', 
            fontsize=24, fontweight='bold', ha='center')
    ax.text(10, 15, 'Integrated AI System for Quantum Circuit Generation and Analysis', 
            fontsize=16, ha='center', style='italic')
    
    # ==================== Input Layer ====================
    input_y = 13.5
    
    # Quantum circuit input
    circuit_box = FancyBboxPatch((0.5, input_y-0.3), 3, 0.6, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['input'], 
                                edgecolor='black', linewidth=2)
    ax.add_patch(circuit_box)
    ax.text(2, input_y, 'Quantum Circuit Input\nGate|Qubit|Parameter', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Measurement data input
    measurement_box = FancyBboxPatch((4.5, input_y-0.3), 3, 0.6, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=colors['input'], 
                                   edgecolor='black', linewidth=2)
    ax.add_patch(measurement_box)
    ax.text(6, input_y, 'Measurement Data\nIBM Quantum Results', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Circuit features input
    features_box = FancyBboxPatch((8.5, input_y-0.3), 3, 0.6, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['input'], 
                                edgecolor='black', linewidth=2)
    ax.add_patch(features_box)
    ax.text(10, input_y, 'Circuit Features\n33-Dimensional Features', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # ==================== Embedding Layer ====================
    embedding_y = 12
    
    # Gate embedding
    gate_emb = FancyBboxPatch((0.5, embedding_y-0.25), 2.5, 0.5, 
                             boxstyle="round,pad=0.05", 
                             facecolor=colors['embedding'], 
                             edgecolor='black')
    ax.add_patch(gate_emb)
    ax.text(1.75, embedding_y, 'Gate Embedding\nGate Type Encoding', 
            ha='center', va='center', fontsize=9)
    
    # Qubit embedding
    qubit_emb = FancyBboxPatch((3.5, embedding_y-0.25), 2.5, 0.5, 
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['embedding'], 
                              edgecolor='black')
    ax.add_patch(qubit_emb)
    ax.text(4.75, embedding_y, 'Qubit Embedding\nQubit Position Encoding', 
            ha='center', va='center', fontsize=9)
    
    # Parameter embedding
    param_emb = FancyBboxPatch((6.5, embedding_y-0.25), 2.5, 0.5, 
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['embedding'], 
                              edgecolor='black')
    ax.add_patch(param_emb)
    ax.text(7.75, embedding_y, 'Param Embedding\nParameter Projection', 
            ha='center', va='center', fontsize=9)
    
    # Positional encoding
    pos_enc = FancyBboxPatch((9.5, embedding_y-0.25), 3, 0.5, 
                            boxstyle="round,pad=0.05", 
                            facecolor=colors['embedding'], 
                            edgecolor='black')
    ax.add_patch(pos_enc)
    ax.text(11, embedding_y, 'Quantum-Aware\nPositional Encoding', 
            ha='center', va='center', fontsize=9)
    
    # ==================== Core Transformer System ====================
    transformer_y = 10
    
    # Representation learning transformer (left)
    repr_transformer = FancyBboxPatch((0.5, transformer_y-1), 5.5, 2, 
                                     boxstyle="round,pad=0.1", 
                                     facecolor=colors['transformer'], 
                                     edgecolor='black', linewidth=2)
    ax.add_patch(repr_transformer)
    ax.text(3.25, transformer_y+0.5, 'Quantum Representation Transformer', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(3.25, transformer_y, '• Multi-Scale Attention\n• Graph Neural Networks\n• Physics Constraints\n• Self-Supervised Learning', 
            ha='center', va='center', fontsize=9)
    
    # Diffusion transformer (right)
    diffusion_transformer = FancyBboxPatch((7, transformer_y-1), 5.5, 2, 
                                          boxstyle="round,pad=0.1", 
                                          facecolor=colors['diffusion'], 
                                          edgecolor='black', linewidth=2)
    ax.add_patch(diffusion_transformer)
    ax.text(9.75, transformer_y+0.5, 'Quantum Circuit DiT', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(9.75, transformer_y, '• Adaptive Layer Norm (adaLN)\n• Timestep Embedding\n• Classifier-Free Guidance\n• Cosine Beta Schedule', 
            ha='center', va='center', fontsize=9)
    
    # ==================== Middle Processing Layer ====================
    middle_y = 7.5
    
    # Graph topology encoder
    graph_encoder = FancyBboxPatch((0.5, middle_y-0.3), 3, 0.6, 
                                  boxstyle="round,pad=0.05", 
                                  facecolor='#E1F5FE', 
                                  edgecolor='black')
    ax.add_patch(graph_encoder)
    ax.text(2, middle_y, 'Graph Topology\nEncoder (GNN)', 
            ha='center', va='center', fontsize=9)
    
    # Cross attention
    cross_attention = FancyBboxPatch((4, middle_y-0.3), 3, 0.6, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor='#F3E5F5', 
                                   edgecolor='black')
    ax.add_patch(cross_attention)
    ax.text(5.5, middle_y, 'Cross Attention\nSequence ↔ Features', 
            ha='center', va='center', fontsize=9)
    
    # Diffusion process
    diffusion_process = FancyBboxPatch((7.5, middle_y-0.3), 3, 0.6, 
                                     boxstyle="round,pad=0.05", 
                                     facecolor='#FFF3E0', 
                                     edgecolor='black')
    ax.add_patch(diffusion_process)
    ax.text(9, middle_y, 'Diffusion Process\nNoise → Circuit', 
            ha='center', va='center', fontsize=9)
    
    # Bayesian uncertainty
    uncertainty = FancyBboxPatch((11, middle_y-0.3), 3, 0.6, 
                               boxstyle="round,pad=0.05", 
                               facecolor='#EFEBE9', 
                               edgecolor='black')
    ax.add_patch(uncertainty)
    ax.text(12.5, middle_y, 'Bayesian Uncertainty\nMonte Carlo Dropout', 
            ha='center', va='center', fontsize=9)
    
    # ==================== Output and Analysis Layer ====================
    output_y = 5.5
    
    # Circuit representation vector
    repr_output = FancyBboxPatch((0.5, output_y-0.3), 3, 0.6, 
                               boxstyle="round,pad=0.05", 
                               facecolor=colors['output'], 
                               edgecolor='black')
    ax.add_patch(repr_output)
    ax.text(2, output_y, 'Circuit Representation\nHigh-Dim Vector', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Generated circuit
    generated_circuit = FancyBboxPatch((4.5, output_y-0.3), 3, 0.6, 
                                     boxstyle="round,pad=0.05", 
                                     facecolor=colors['output'], 
                                     edgecolor='black')
    ax.add_patch(generated_circuit)
    ax.text(6, output_y, 'Generated Circuit\nNew Quantum Circuit', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Physics score
    physics_score = FancyBboxPatch((8.5, output_y-0.3), 3, 0.6, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor=colors['output'], 
                                 edgecolor='black')
    ax.add_patch(physics_score)
    ax.text(10, output_y, 'Physics Score\nPhysics Consistency', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # ==================== Expressibility Analysis System ====================
    analysis_y = 3.5
    
    # Integrated expressibility system
    expressibility_system = FancyBboxPatch((2, analysis_y-0.8), 8, 1.6, 
                                         boxstyle="round,pad=0.1", 
                                         facecolor=colors['analysis'], 
                                         edgecolor='black', linewidth=2)
    ax.add_patch(expressibility_system)
    ax.text(6, analysis_y+0.4, 'Quantum Expressibility System', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Classical Shadow
    cs_box = FancyBboxPatch((2.5, analysis_y-0.4), 2, 0.6, 
                           boxstyle="round,pad=0.05", 
                           facecolor='#E8F5E8', 
                           edgecolor='black')
    ax.add_patch(cs_box)
    ax.text(3.5, analysis_y-0.1, 'Classical\nShadow', 
            ha='center', va='center', fontsize=9)
    
    # Entropy method
    entropy_box = FancyBboxPatch((5, analysis_y-0.4), 2, 0.6, 
                               boxstyle="round,pad=0.05", 
                               facecolor='#FFF8E1', 
                               edgecolor='black')
    ax.add_patch(entropy_box)
    ax.text(6, analysis_y-0.1, 'Entropy\nMethod', 
            ha='center', va='center', fontsize=9)
    
    # Transformer based
    transformer_expr = FancyBboxPatch((7.5, analysis_y-0.4), 2, 0.6, 
                                    boxstyle="round,pad=0.05", 
                                    facecolor='#F3E5F5', 
                                    edgecolor='black')
    ax.add_patch(transformer_expr)
    ax.text(8.5, analysis_y-0.1, 'Transformer\nBased', 
            ha='center', va='center', fontsize=9)
    
    # ==================== Final Results ====================
    result_y = 1.5
    
    # Comprehensive analysis results
    final_result = FancyBboxPatch((3, result_y-0.4), 6, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#E8F5E8', 
                                edgecolor='black', linewidth=2)
    ax.add_patch(final_result)
    ax.text(6, result_y, 'Comprehensive Quantum Analysis Results\nComplete Circuit Analysis Output', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # ==================== Draw Connection Lines ====================
    
    # Input → Embedding
    for x in [2, 6, 10]:
        arrow = ConnectionPatch((x, input_y-0.3), (x, embedding_y+0.25), 
                              "data", "data", 
                              arrowstyle="->", shrinkA=5, shrinkB=5, 
                              mutation_scale=20, fc=colors['connection'])
        ax.add_patch(arrow)
    
    # Embedding → Transformer
    for x in [1.75, 4.75, 7.75]:
        if x < 6:  # To representation transformer
            arrow = ConnectionPatch((x, embedding_y-0.25), (3.25, transformer_y+1), 
                                  "data", "data", 
                                  arrowstyle="->", shrinkA=5, shrinkB=5, 
                                  mutation_scale=20, fc=colors['connection'])
        else:  # To diffusion transformer
            arrow = ConnectionPatch((x, embedding_y-0.25), (9.75, transformer_y+1), 
                                  "data", "data", 
                                  arrowstyle="->", shrinkA=5, shrinkB=5, 
                                  mutation_scale=20, fc=colors['connection'])
        ax.add_patch(arrow)
    
    # Transformer → Middle processing
    arrow1 = ConnectionPatch((3.25, transformer_y-1), (2, middle_y+0.3), 
                           "data", "data", 
                           arrowstyle="->", shrinkA=5, shrinkB=5, 
                           mutation_scale=20, fc=colors['connection'])
    ax.add_patch(arrow1)
    
    arrow2 = ConnectionPatch((9.75, transformer_y-1), (9, middle_y+0.3), 
                           "data", "data", 
                           arrowstyle="->", shrinkA=5, shrinkB=5, 
                           mutation_scale=20, fc=colors['connection'])
    ax.add_patch(arrow2)
    
    # Middle processing → Output
    for i, x in enumerate([2, 6, 10]):
        arrow = ConnectionPatch((2+i*3.5, middle_y-0.3), (x, output_y+0.3), 
                              "data", "data", 
                              arrowstyle="->", shrinkA=5, shrinkB=5, 
                              mutation_scale=20, fc=colors['connection'])
        ax.add_patch(arrow)
    
    # Output → Analysis
    for x in [2, 6, 10]:
        arrow = ConnectionPatch((x, output_y-0.3), (6, analysis_y+0.8), 
                              "data", "data", 
                              arrowstyle="->", shrinkA=5, shrinkB=5, 
                              mutation_scale=20, fc=colors['connection'])
        ax.add_patch(arrow)
    
    # Analysis → Final results
    arrow = ConnectionPatch((6, analysis_y-0.8), (6, result_y+0.4), 
                          "data", "data", 
                          arrowstyle="->", shrinkA=5, shrinkB=5, 
                          mutation_scale=20, fc=colors['connection'])
    ax.add_patch(arrow)
    
    # ==================== Side Information ====================
    
    # Left side: Key AI techniques
    ax.text(13.5, 13, 'Key AI Techniques', fontsize=14, fontweight='bold')
    techniques = [
        '• Multi-Head Cross-Attention',
        '• Graph Neural Networks',
        '• Hierarchical Multi-Scale Attention',
        '• Physics-Informed Neural Networks',
        '• Self-Supervised Pre-training',
        '• Bayesian Uncertainty Quantification',
        '• Monte Carlo Dropout',
        '• Diffusion Transformer (DiT)',
        '• Adaptive Layer Normalization',
        '• Classifier-Free Guidance'
    ]
    
    for i, technique in enumerate(techniques):
        ax.text(13.5, 12.5-i*0.3, technique, fontsize=9)
    
    # Right side: Data flow
    ax.text(16.5, 13, 'Data Flow', fontsize=14, fontweight='bold')
    flow_steps = [
        '1. Quantum Circuit Input',
        '2. Multi-Modal Embedding',
        '3. Transformer Processing',
        '4. Graph Topology',
        '5. Physics Constraints',
        '6. Representation Extraction',
        '7. Circuit Generation',
        '8. Expressibility Analysis',
        '9. Comprehensive Results'
    ]
    
    for i, step in enumerate(flow_steps):
        ax.text(16.5, 12.5-i*0.3, step, fontsize=9)
    
    # Legend
    legend_y = 0.5
    ax.text(1, legend_y, 'Legend:', fontsize=12, fontweight='bold')
    
    legend_items = [
        ('Input Layer', colors['input']),
        ('Embedding Layer', colors['embedding']),
        ('Transformer', colors['transformer']),
        ('Diffusion', colors['diffusion']),
        ('Output', colors['output']),
        ('Analysis', colors['analysis'])
    ]
    
    for i, (label, color) in enumerate(legend_items):
        rect = patches.Rectangle((2+i*2.5, legend_y-0.1), 0.3, 0.2, 
                               facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(2.4+i*2.5, legend_y, label, fontsize=9, va='center')
    
    plt.tight_layout()
    plt.savefig('quantum_ai_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_detailed_transformer_diagram():
    """Detailed transformer internal structure diagram"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
    
    # ==================== Representation Transformer (Left) ====================
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.axis('off')
    ax1.set_title('Quantum Representation Transformer\nRepresentation Learning Transformer', 
                  fontsize=16, fontweight='bold', pad=20)
    
    # Input layer
    input_box = FancyBboxPatch((1, 10.5), 8, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#E3F2FD', 
                              edgecolor='black', linewidth=2)
    ax1.add_patch(input_box)
    ax1.text(5, 10.9, 'Input: Gate + Qubit + Param Embeddings', 
             ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Transformer layers
    layer_colors = ['#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5']
    layer_names = [
        'Multi-Scale Attention\n(Gate/Block/Circuit Level)',
        'Feed Forward Network\n(GELU + Dropout)',
        'Physics Constraints\n(Unitary/Hermitian)',
        'Layer Normalization\n+ Residual Connection'
    ]
    
    for i, (color, name) in enumerate(zip(layer_colors, layer_names)):
        y_pos = 9 - i * 1.5
        layer_box = FancyBboxPatch((1, y_pos), 8, 1, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=color, 
                                  edgecolor='black')
        ax1.add_patch(layer_box)
        ax1.text(5, y_pos + 0.5, name, 
                ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Connection lines
        if i < len(layer_colors) - 1:
            arrow = ConnectionPatch((5, y_pos), (5, y_pos - 0.5), 
                                  "data", "data", 
                                  arrowstyle="->", shrinkA=5, shrinkB=5, 
                                  mutation_scale=20, fc='black')
            ax1.add_patch(arrow)
    
    # Output
    output_box = FancyBboxPatch((1, 2), 8, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#C8E6C9', 
                               edgecolor='black', linewidth=2)
    ax1.add_patch(output_box)
    ax1.text(5, 2.4, 'Output: Circuit Representation Vector (512D)', 
             ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Final connection line
    arrow = ConnectionPatch((5, 3.5), (5, 2.8), 
                          "data", "data", 
                          arrowstyle="->", shrinkA=5, shrinkB=5, 
                          mutation_scale=20, fc='black')
    ax1.add_patch(arrow)
    
    # ==================== Diffusion Transformer (Right) ====================
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.axis('off')
    ax2.set_title('Quantum Circuit DiT\nDiffusion Transformer', 
                  fontsize=16, fontweight='bold', pad=20)
    
    # Input layer
    input_box2 = FancyBboxPatch((1, 10.5), 8, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#FFF3E0', 
                               edgecolor='black', linewidth=2)
    ax2.add_patch(input_box2)
    ax2.text(5, 10.9, 'Input: Noisy Circuit + Timestep + Condition', 
             ha='center', va='center', fontsize=12, fontweight='bold')
    
    # DiT layers
    dit_colors = ['#FFE0B2', '#FFCC80', '#FFB74D', '#FFA726']
    dit_names = [
        'Timestep Embedding\n(Sinusoidal + MLP)',
        'DiT Block with adaLN-Zero\n(Adaptive Layer Norm)',
        'Multi-Head Attention\n(Conditional Attention)',
        'Final Layer\n(3 Output Heads)'
    ]
    
    for i, (color, name) in enumerate(zip(dit_colors, dit_names)):
        y_pos = 9 - i * 1.5
        layer_box = FancyBboxPatch((1, y_pos), 8, 1, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=color, 
                                  edgecolor='black')
        ax2.add_patch(layer_box)
        ax2.text(5, y_pos + 0.5, name, 
                ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Connection lines
        if i < len(dit_colors) - 1:
            arrow = ConnectionPatch((5, y_pos), (5, y_pos - 0.5), 
                                  "data", "data", 
                                  arrowstyle="->", shrinkA=5, shrinkB=5, 
                                  mutation_scale=20, fc='black')
            ax2.add_patch(arrow)
    
    # Output
    output_box2 = FancyBboxPatch((1, 2), 8, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#C8E6C9', 
                                edgecolor='black', linewidth=2)
    ax2.add_patch(output_box2)
    ax2.text(5, 2.4, 'Output: Denoised Circuit (Gate|Qubit|Param)', 
             ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Final connection line
    arrow2 = ConnectionPatch((5, 3.5), (5, 2.8), 
                           "data", "data", 
                           arrowstyle="->", shrinkA=5, shrinkB=5, 
                           mutation_scale=20, fc='black')
    ax2.add_patch(arrow2)
    
    plt.tight_layout()
    plt.savefig('transformer_details.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_expressibility_flow_diagram():
    """Expressibility calculation flow diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.set_title('Quantum Expressibility Calculation Flow\nQuantum Expressibility Computation Pipeline', 
                 fontsize=18, fontweight='bold', pad=20)
    
    # Input
    input_circuit = FancyBboxPatch((1, 8.5), 3, 1, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='#E3F2FD', 
                                  edgecolor='black', linewidth=2)
    ax.add_patch(input_circuit)
    ax.text(2.5, 9, 'Quantum Circuit\nInput Circuit', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Three methods
    methods_y = 6.5
    
    # Classical Shadow
    cs_box = FancyBboxPatch((0.5, methods_y), 3, 1.5, 
                           boxstyle="round,pad=0.1", 
                           facecolor='#E8F5E8', 
                           edgecolor='black', linewidth=2)
    ax.add_patch(cs_box)
    ax.text(2, methods_y + 0.75, 'Classical Shadow\nExpressibility', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(2, methods_y + 0.3, '• Shadow Measurement\n• Density Matrix Reconstruction\n• KL Divergence Calculation', 
            ha='center', va='center', fontsize=9)
    
    # Entropy method
    entropy_box = FancyBboxPatch((4.5, methods_y), 3, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#FFF8E1', 
                                edgecolor='black', linewidth=2)
    ax.add_patch(entropy_box)
    ax.text(6, methods_y + 0.75, 'Entropy Method\nGeometric Diversity', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(6, methods_y + 0.3, '• Angle Entropy\n• Distance Entropy\n• Geometric Diversity', 
            ha='center', va='center', fontsize=9)
    
    # Transformer method
    transformer_box = FancyBboxPatch((8.5, methods_y), 3, 1.5, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor='#F3E5F5', 
                                    edgecolor='black', linewidth=2)
    ax.add_patch(transformer_box)
    ax.text(10, methods_y + 0.75, 'Transformer Method\nLearned Representation', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(10, methods_y + 0.3, '• Representation Extraction\n• Diversity Measurement\n• Learned Features', 
            ha='center', va='center', fontsize=9)
    
    # Integration system
    integration_box = FancyBboxPatch((4, 4), 4, 1, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor='#F8BBD9', 
                                   edgecolor='black', linewidth=2)
    ax.add_patch(integration_box)
    ax.text(6, 4.5, 'Expressibility Integration System\nUnified Analysis Framework', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Final results
    result_box = FancyBboxPatch((4, 2), 4, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#C8E6C9', 
                               edgecolor='black', linewidth=2)
    ax.add_patch(result_box)
    ax.text(6, 2.5, 'Comprehensive Analysis\nComplete Expressibility Report', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Connection lines
    # Input → Three methods
    for x in [2, 6, 10]:
        arrow = ConnectionPatch((2.5, 8.5), (x, methods_y + 1.5), 
                              "data", "data", 
                              arrowstyle="->", shrinkA=5, shrinkB=5, 
                              mutation_scale=20, fc='black')
        ax.add_patch(arrow)
    
    # Three methods → Integration
    for x in [2, 6, 10]:
        arrow = ConnectionPatch((x, methods_y), (6, 5), 
                              "data", "data", 
                              arrowstyle="->", shrinkA=5, shrinkB=5, 
                              mutation_scale=20, fc='black')
        ax.add_patch(arrow)
    
    # Integration → Results
    arrow = ConnectionPatch((6, 4), (6, 3), 
                          "data", "data", 
                          arrowstyle="->", shrinkA=5, shrinkB=5, 
                          mutation_scale=20, fc='black')
    ax.add_patch(arrow)
    
    # Side information
    ax.text(12.5, 8, 'Output Metrics', fontsize=14, fontweight='bold')
    metrics = [
        '• Classical Shadow Expressibility',
        '• Entropy Expressibility',
        '• Transformer Expressibility',
        '• Expressibility Ratio',
        '• Physics Consistency Score',
        '• Uncertainty Quantification',
        '• Correlation Analysis'
    ]
    
    for i, metric in enumerate(metrics):
        ax.text(12.5, 7.5-i*0.4, metric, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('expressibility_flow.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Generating Quantum AI System Architecture Diagrams...")
    
    # Complete architecture diagram
    print("1. Complete System Architecture...")
    create_quantum_architecture_diagram()
    
    # Detailed transformer diagram
    print("2. Detailed Transformer Structure...")
    create_detailed_transformer_diagram()
    
    # Expressibility calculation flow
    print("3. Expressibility Calculation Flow...")
    create_expressibility_flow_diagram()
    
    print("\nAll diagrams generated successfully!")
    print("Generated files:")
    print("  quantum_ai_architecture.png - Complete System Architecture")
    print("  transformer_details.png - Detailed Transformer Structure")
    print("  expressibility_flow.png - Expressibility Calculation Flow") 