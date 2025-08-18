import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Set up the figure with high DPI for publication quality
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 9
fig, ax = plt.subplots(1, 1, figsize=(20, 14), dpi=300)
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis('off')

# Color scheme (professional blue/gray palette)
colors = {
    'input': '#E8F4FD',
    'embedding': '#B3D9FF', 
    'transformer': '#4A90E2',
    'output': '#2E5C8A',
    'text': '#2C3E50',
    'arrow': '#34495E',
    'accent': '#E74C3C'
}

def draw_rounded_box(ax, x, y, width, height, text, color, text_color='black', fontsize=9, title_fontsize=11):
    """Draw a rounded rectangle with text"""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor='black',
        linewidth=1.5
    )
    ax.add_patch(box)
    
    # Split text into title and content
    lines = text.split('\n')
    title = lines[0]
    content = '\n'.join(lines[1:]) if len(lines) > 1 else ''
    
    # Draw title (emphasized)
    if content:
        ax.text(x + width/2, y + height*0.75, title, 
                ha='center', va='center', fontsize=title_fontsize, 
                color=text_color, weight='bold')
        # Draw content
        ax.text(x + width/2, y + height*0.35, content, 
                ha='center', va='center', fontsize=fontsize, 
                color=text_color, weight='normal')
    else:
        ax.text(x + width/2, y + height/2, title, 
                ha='center', va='center', fontsize=title_fontsize, 
                color=text_color, weight='bold')

def draw_arrow(ax, start_x, start_y, end_x, end_y, color='black', width=2):
    """Draw an arrow between two points"""
    arrow = patches.FancyArrowPatch(
        (start_x, start_y), (end_x, end_y),
        arrowstyle='->', mutation_scale=20,
        color=color, linewidth=width
    )
    ax.add_patch(arrow)

# Title
ax.text(10, 13.5, 'Quantum Circuit Decision Transformer Architecture', 
        ha='center', va='center', fontsize=20, weight='bold', color=colors['text'])

# 1. Input Layer with data examples
draw_rounded_box(ax, 0.5, 11, 3, 2, 'INPUT DATA\n{"gates": [{"gate_name": "H", "qubits": [0]},\n{"gate_name": "CNOT", "qubits": [0,1]}],\n"num_qubits": 2, "depth": 2}', 
                colors['input'], fontsize=8, title_fontsize=12)

# 2. Embedding Pipeline with examples
draw_rounded_box(ax, 4.5, 11, 4, 2, 'SAR EMBEDDING PIPELINE\nState: [0,0,0,...] → [0.1,0.3,-0.2,...]\nAction: H→[1,0,0,0.0] → [0.2,-0.1,0.4,...]\nReward: RTG=0.8 → [0.3,0.1,-0.5,...]', 
                colors['embedding'], fontsize=8, title_fontsize=12)

# 3. Gate Embedding Detail
draw_rounded_box(ax, 0.5, 8.5, 3, 2, 'GATE EMBEDDING\nH Gate: [1,0,0,0.0]\n→ Type: nn.Embed(1)→[256d]\n→ Pos: Linear([0,0])→[128d]\n→ Param: Linear([0.0])→[128d]\n→ Concat: [512d]', 
                colors['embedding'], fontsize=8, title_fontsize=12)

# 4. State Generation Detail
draw_rounded_box(ax, 4.5, 8.5, 4, 2, 'CUMULATIVE STATE GENERATION\nS₀: Empty → [0,0,0,...,0] (512d)\nS₁: [H] → embed([H]) → [0.1,0.3,-0.2,...]\nS₂: [H,CNOT] → embed([H,CNOT]) → [0.2,0.1,0.4,...]', 
                colors['embedding'], fontsize=8, title_fontsize=12)

# 5. Transformer Core
draw_rounded_box(ax, 9.5, 9, 4.5, 3, 'TRANSFORMER CORE\n6 Layers × {\n  MultiHead Attention (8 heads)\n  + Causal Mask\n  + Residual & LayerNorm\n  FeedForward (512→2048→512)\n  + Residual & LayerNorm\n}', 
                colors['transformer'], 'white', fontsize=9, title_fontsize=12)

# 6. Attention Detail
draw_rounded_box(ax, 15, 10, 4, 2, 'MULTI-HEAD ATTENTION\nInput: [batch, seq_len, 512]\nQ,K,V: Linear(512→512)\nAttention: softmax(QK^T/√64)V\n+ Causal Mask for autoregression', 
                colors['transformer'], 'white', fontsize=8, title_fontsize=12)

# 7. Output Heads with examples
draw_rounded_box(ax, 9.5, 5.5, 1.3, 2, 'GATE HEAD\nLogits: [20]\nSoftmax\n→ P(H)=0.7\n→ P(X)=0.2\n→ P(CNOT)=0.1', 
                colors['output'], 'white', fontsize=8, title_fontsize=11)
draw_rounded_box(ax, 11.2, 5.5, 1.3, 2, 'POSITION HEAD\nLogits: [100]\nMask + Softmax\n→ P(q0)=0.6\n→ P(q1)=0.4', 
                colors['output'], 'white', fontsize=8, title_fontsize=11)
draw_rounded_box(ax, 12.8, 5.5, 1.3, 2, 'PARAMETER HEAD\nRegression\nTanh\n→ θ=π/4\n→ φ=0.0', 
                colors['output'], 'white', fontsize=8, title_fontsize=11)

# 8. Loss Functions
draw_rounded_box(ax, 15, 5.5, 4, 2, 'MULTI-TASK LOSS\nGate: CrossEntropy(pred, target)\nPosition: CrossEntropy(pred, target)\nParameter: MSE(pred, target)\nTotal = α×L_gate + β×L_pos + γ×L_param', 
                colors['accent'], 'white', fontsize=8, title_fontsize=12)

# 9. Inference Flow (bottom section)
ax.text(10, 4.5, 'AUTOREGRESSIVE GENERATION WITH RECURSIVE STATE UPDATE', 
        ha='center', va='center', fontsize=14, weight='bold', color=colors['text'])

# Inference steps with recursive arrows
draw_rounded_box(ax, 1, 2.5, 2.5, 1.5, 'STEP 1\nCircuit: []\nState: [0,0,...,0]\n→ Predict: H on q0', 
                colors['input'], fontsize=8, title_fontsize=11)
draw_rounded_box(ax, 4.5, 2.5, 2.5, 1.5, 'STEP 2\nCircuit: [H]\nState: embed([H])\n→ Predict: CNOT', 
                colors['input'], fontsize=8, title_fontsize=11)
draw_rounded_box(ax, 8, 2.5, 2.5, 1.5, 'STEP i\nCircuit: [H,CNOT,...]\nState: embed(all_gates)\n→ Predict: Next Gate', 
                colors['input'], fontsize=8, title_fontsize=11)

# Incremental update
draw_rounded_box(ax, 12, 2.5, 6, 1.5, 'INCREMENTAL STATE UPDATE\nnew_state = embedding.create_incremental_state_embedding(\n  current_circuit_gates, predicted_gate, num_qubits)\n→ Real-time Circuit State for Next Prediction', 
                colors['embedding'], fontsize=8, title_fontsize=11)

# Property Prediction Integration
draw_rounded_box(ax, 1, 0.3, 4, 1.4, 'PROPERTY PREDICTION MODEL\nInput: Circuit State [512d]\nOutput: [fidelity=0.85, expressibility=0.72, entanglement=0.63]\n→ RTG Reward Calculation', 
                colors['accent'], 'white', fontsize=8, title_fontsize=11)

# SAR Pattern Detail
draw_rounded_box(ax, 6, 0.3, 5, 1.4, 'SAR SEQUENCE PATTERN\n[S₀, A₀, R₀, S₁, A₁, R₁, S₂, A₂, R₂, ..., EOS]\nExample: [[0,0,...], [H_emb], [0.8], [H_state], [CNOT_emb], [0.9], ...]\nLength: 3 × num_gates + 1', 
                colors['embedding'], fontsize=8, title_fontsize=11)

# Model Specifications
draw_rounded_box(ax, 12, 0.3, 6, 1.4, 'MODEL SPECIFICATIONS\nd_model=512, n_layers=6, n_heads=8, d_ff=2048\nGate_vocab=20, Max_qubits=50, Dropout=0.1\nBatch_size=32, Learning_rate=1e-4, Optimizer=AdamW', 
                colors['transformer'], 'white', fontsize=8, title_fontsize=11)

# Draw arrows for main flow
draw_arrow(ax, 3.5, 12, 4.5, 12, colors['arrow'], 2)  # Input → Embedding
draw_arrow(ax, 8.5, 12, 9.5, 12, colors['arrow'], 2)  # Embedding → Transformer
draw_arrow(ax, 11.75, 9, 11.75, 7.5, colors['arrow'], 2)  # Transformer → Outputs

# Side arrows
draw_arrow(ax, 3.5, 9.5, 4.5, 9.5, colors['arrow'], 1.5)  # Gate Embedding → State Generation
draw_arrow(ax, 14, 10.5, 15, 10.5, colors['arrow'], 1.5)  # Transformer → Attention Detail

# Recursive inference arrows (showing the loop)
draw_arrow(ax, 3.5, 3.25, 4.5, 3.25, colors['accent'], 2)  # Step 1 → Step 2
draw_arrow(ax, 7, 3.25, 8, 3.25, colors['accent'], 2)  # Step 2 → Step i
draw_arrow(ax, 10.5, 3.25, 12, 3.25, colors['accent'], 2)  # Step i → Update

# Recursive loop arrow (from update back to next step)
arrow_loop = patches.FancyArrowPatch(
    (15, 2.5), (15, 1.8), 
    connectionstyle="arc3,rad=0.3",
    arrowstyle='->', mutation_scale=20,
    color=colors['accent'], linewidth=2
)
ax.add_patch(arrow_loop)

arrow_loop2 = patches.FancyArrowPatch(
    (15, 1.8), (8, 1.8),
    arrowstyle='->', mutation_scale=20,
    color=colors['accent'], linewidth=2
)
ax.add_patch(arrow_loop2)

arrow_loop3 = patches.FancyArrowPatch(
    (8, 1.8), (8, 2.5),
    arrowstyle='->', mutation_scale=20,
    color=colors['accent'], linewidth=2
)
ax.add_patch(arrow_loop3)

# Add recursive loop label
ax.text(11.5, 1.5, 'Recursive Loop', ha='center', va='center', 
        fontsize=10, weight='bold', color=colors['accent'])

# Output flow arrows
draw_arrow(ax, 10.15, 5.5, 10.15, 4.8, colors['arrow'], 1.5)  # Gate Head → Loss
draw_arrow(ax, 11.85, 5.5, 11.85, 4.8, colors['arrow'], 1.5)  # Position Head → Loss
draw_arrow(ax, 13.45, 5.5, 13.45, 4.8, colors['arrow'], 1.5)  # Parameter Head → Loss

# Property prediction arrow
draw_arrow(ax, 5, 1, 6, 1, colors['accent'], 1.5)  # Property → SAR

# Loss arrows
draw_arrow(ax, 12.5, 6.25, 15, 6.25, colors['arrow'], 1.5)  # Outputs → Loss

plt.tight_layout()
plt.savefig('decision_transformer_architecture.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('decision_transformer_architecture.pdf', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

print("✅ Architecture diagram saved as 'decision_transformer_architecture.png' and 'decision_transformer_architecture.pdf'")
