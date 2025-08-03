# 🚀 DiT Quantum Circuit Generation Model

**State-of-the-art Diffusion Transformer for Quantum Circuit Generation**

A cutting-edge implementation of DiT (Diffusion Transformer) specifically designed for quantum circuit generation, featuring the latest advances in transformer architecture and diffusion models.

## 🌟 Key Features

### 🔬 **Advanced Architecture**
- **RoPE (Rotary Positional Embedding)** - Latest positional encoding technique
- **SwiGLU Activation** - Superior to standard FFN activations
- **Flash Attention** - Memory-efficient attention computation
- **Pre-Layer Normalization** - Better training stability
- **GLU Gating** - Advanced information flow control

### ⚡ **Training Optimizations**
- **Automatic Mixed Precision (AMP)** - Faster training with lower memory
- **Torch Compile** - PyTorch 2.0+ optimization
- **Exponential Moving Average (EMA)** - Better model stability
- **Gradient Checkpointing** - Memory-efficient training for large models
- **Advanced Schedulers** - Cosine annealing with warm restarts

### 📊 **Comprehensive Evaluation**
- **Quantum-specific Metrics** - Circuit depth, entanglement capability, gate diversity
- **Training Visualization** - Real-time loss tracking and analysis
- **Circuit Analysis** - Detailed quantum circuit property evaluation
- **Comparison Tools** - Generated vs reference circuit comparison

## 🏗️ Project Structure

```
Dit_Model_ver2/
├── quantum_dit/                    # Main package
│   ├── models/
│   │   └── dit_model.py           # DiT architecture implementation
│   ├── encoding/
│   │   └── Embeding.py            # Quantum circuit embedding
│   ├── data/
│   │   └── quantum_dataset.py     # Dataset and data loading
│   ├── utils/
│   │   ├── diffusion.py           # Diffusion scheduling
│   │   ├── metrics.py             # Quantum circuit metrics
│   │   └── visualization.py       # Plotting and analysis
│   └── __init__.py
├── training/
│   └── train_dit.py               # Advanced training pipeline
├── evaluation/
│   └── evaluate_dit.py            # Comprehensive evaluation
├── run_training.py                # Main execution script
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. **Installation**

```bash
# Clone the repository
git clone <repository-url>
cd Dit_Model_ver2

# Install dependencies
pip install -r requirements.txt

# Install quantum_common package (if not already installed)
cd ../quantum_common
pip install -e .
cd ../Dit_Model_ver2
```

### 2. **Training**

```bash
# Train small model (recommended for testing)
python run_training.py train --model_size small --epochs 50

# Train medium model (balanced performance)
python run_training.py train --model_size medium --epochs 100

# Train large model (maximum performance)
python run_training.py train --model_size large --epochs 200 --batch_size 16
```

### 3. **Evaluation**

```bash
# Evaluate trained model
python run_training.py evaluate \
    --model_path checkpoints/best_model.pt \
    --test_data data/test \
    --num_generated 1000
```

### 4. **Generation**

```bash
# Generate quantum circuits
python run_training.py generate \
    --model_path checkpoints/best_model.pt \
    --num_samples 100 \
    --output_path generated_circuits.json
```

### 5. **Visualization**

```bash
# Create training visualizations
python run_training.py visualize --log_dir logs --dashboard

# Simple training curves
python run_training.py visualize --log_dir logs --output_path training_curves.png
```

## 🔧 Model Configurations

### **Small Model** (Testing & Development)
- **Parameters**: ~25M
- **d_model**: 256
- **Layers**: 6
- **Heads**: 4
- **Max Circuit Length**: 128
- **Memory**: ~4GB GPU

### **Medium Model** (Balanced Performance)
- **Parameters**: ~100M
- **d_model**: 512
- **Layers**: 12
- **Heads**: 8
- **Max Circuit Length**: 256
- **Memory**: ~8GB GPU

### **Large Model** (Maximum Performance)
- **Parameters**: ~300M
- **d_model**: 768
- **Layers**: 18
- **Heads**: 12
- **Max Circuit Length**: 512
- **Memory**: ~16GB GPU

## 📊 Training & Evaluation Metrics

### **Training Metrics**
- **Training Loss** - MSE loss for noise prediction
- **Validation Loss** - Generalization performance
- **Learning Rate** - Adaptive scheduling
- **Gradient Norm** - Training stability

### **Quantum Circuit Metrics**
- **Circuit Depth** - Quantum circuit execution depth
- **Gate Diversity** - Shannon entropy of gate distribution
- **Entanglement Capability** - Two-qubit gate ratio
- **Circuit Validity** - Structural correctness
- **Quantum Volume** - Overall circuit complexity

### **Generation Quality**
- **Fidelity** - Similarity to reference circuits
- **Expressivity** - Diversity across generated set
- **Efficiency** - Gates per qubit ratio
- **Validity Ratio** - Percentage of valid circuits

## 🎯 Advanced Features

### **Diffusion Scheduling**
- **Linear Schedule** - Standard DDPM scheduling
- **Cosine Schedule** - Improved noise scheduling
- **Sigmoid Schedule** - Custom quantum-optimized schedule

### **Sampling Methods**
- **DDPM Sampling** - Standard diffusion sampling
- **DDIM Sampling** - Faster deterministic sampling
- **Guidance** - Conditional generation control

### **Data Augmentation**
- **Gate Substitution** - Equivalent gate replacement
- **Parameter Noise** - Continuous parameter perturbation
- **Circuit Truncation** - Variable length training

## 📈 Performance Optimization

### **Memory Optimization**
```python
# Enable gradient checkpointing for large models
config.gradient_checkpointing = True

# Use automatic mixed precision
config.use_amp = True

# Reduce batch size if OOM
config.batch_size = 16  # or 8 for very large models
```

### **Speed Optimization**
```python
# Enable model compilation (PyTorch 2.0+)
config.compile_model = True

# Use Flash Attention
config.use_flash_attention = True

# Increase number of workers
config.num_workers = 8
```

## 🔬 Research Features

### **Latest Transformer Techniques**
- **RoPE**: Rotary positional embeddings for better length generalization
- **SwiGLU**: Swish-gated linear units for improved FFN performance
- **Pre-LN**: Pre-layer normalization for training stability
- **GLU Gating**: Gated linear units for better information flow

### **Quantum-Specific Innovations**
- **Multi-Modal Embedding**: Separate embeddings for gates, positions, roles, parameters
- **Sinusoidal PE**: Cached positional encodings for efficiency
- **Quantum Metrics**: Domain-specific evaluation metrics
- **Circuit Validation**: Structural correctness checking

## 📝 Usage Examples

### **Custom Training Configuration**
```python
from quantum_dit.models.dit_model import DiTConfig
from training.train_dit import TrainingConfig, AdvancedTrainer

# Create custom model config
model_config = DiTConfig(
    d_model=512,
    n_layers=12,
    n_heads=8,
    use_flash_attention=True,
    use_rotary_pe=True,
    use_swiglu=True
)

# Create training config
training_config = TrainingConfig(
    model_config=model_config,
    batch_size=32,
    learning_rate=1e-4,
    use_amp=True,
    use_ema=True
)

# Train model
trainer = AdvancedTrainer(training_config)
trainer.train()
```

### **Custom Evaluation**
```python
from evaluation.evaluate_dit import QuantumCircuitEvaluator

# Load model and evaluate
evaluator = QuantumCircuitEvaluator("checkpoints/best_model.pt")
results = evaluator.run_comprehensive_evaluation("data/test", num_generated=1000)

# Print results
for category, metrics in results.items():
    print(f"{category}: {metrics}")
```

## 🐛 Troubleshooting

### **Common Issues**

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python run_training.py train --batch_size 8
   
   # Enable gradient checkpointing
   # (automatically enabled for large models)
   ```

2. **Import Errors**
   ```bash
   # Ensure quantum_common is installed
   cd ../quantum_common && pip install -e .
   
   # Check Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

3. **Slow Training**
   ```bash
   # Enable optimizations
   python run_training.py train --use_amp --compile_model
   
   # Use more workers
   python run_training.py train --num_workers 8
   ```

## 📚 References

- **DiT**: "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
- **RoPE**: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
- **SwiGLU**: "GLU Variants Improve Transformer" (Shazeer, 2020)
- **Flash Attention**: "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al., 2022)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎉 Acknowledgments

- **Windsurf AI Team** for the foundational architecture
- **Quantum Computing Community** for domain expertise
- **PyTorch Team** for the excellent framework
- **Research Community** for the latest techniques

---

**Built with ❤️ for the Quantum AI Community**

For questions, issues, or contributions, please open an issue on GitHub or contact the development team.
