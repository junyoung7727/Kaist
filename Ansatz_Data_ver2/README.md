# Ansatz_Data_ver2: Quantum Circuit Analysis & Validation Framework

🚀 **A comprehensive quantum circuit analysis system with unified batch processing, statistical validation, and multi-backend support for quantum expressibility, fidelity, and entanglement measurements.**

## 🎯 **Project Overview**

This project provides a modular, scalable framework for quantum circuit analysis with focus on:
- **Expressibility**: KL-divergence based circuit expressiveness measurement
- **Fidelity**: Error fidelity calculation via inverse circuits
- **Entanglement**: Meyer-Wallach entropy via SWAP test protocols
- **Statistical Validation**: Rigorous validation framework with publication-quality visualizations
- **Batch Processing**: Optimized IBM Quantum execution (3→1 backend connections)

## 🏗️ **Architecture Overview**

```
Ansatz_Data_ver2/
├── main.py                     # 🎯 Main orchestration & unified batch processing
├── config.py                   # ⚙️ Centralized configuration management
├── core/                       # 🧠 Core quantum circuit logic
├── execution/                  # 🚀 Backend execution engines
├── expressibility/             # 📊 Expressibility measurement modules
├── utils/                      # 🛠️ Utility functions
├── validation_results/         # 📈 Statistical validation outputs
└── requirements.txt            # 📦 Dependencies
```

## 🚀 **Quick Start**

### **1. Installation**
```bash
# Clone repository
git clone <repository-url>
cd Ansatz_Data_ver2

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your IBM Quantum token
```

### **2. Basic Usage**
```bash
# Run unified batch experiment
python main.py

# Run statistical validation
python run_statistical_validation.py

# Calculate shot requirements
python shot_calculator.py
```

### **3. Configuration**
Edit `config.py` for experiment parameters:
```python
from config import Exp_Box

# Use pre-configured experiments
config = Exp_Box.scalability_test  # For scalability analysis
config = Exp_Box.statistical_validation_config  # For validation
```

```python
config = Config(
    backend_type='simulator',  # 'simulator' or 'ibm'
    shots=1024,
    num_qubits=4,
    circuit_depth=10,
    num_circuits=100
)
```

## 🧠 Core Concepts

### Abstract Circuit Interface

All circuits implement `AbstractQuantumCircuit`, ensuring backend-agnostic operation:

```python
from core.circuit import CircuitBuilder

# Build a circuit specification (no backend knowledge)
builder = CircuitBuilder()
builder.set_qubits(2)
builder.add_gate('h', 0)
builder.add_gate('cx', [0, 1])
spec = builder.build_spec()
```

### Unified Execution

The executor factory handles backend selection transparently:
```python
from execution.executor import ExecutorFactory, ExecutionConfig

# Create executor (only place backend matters)
config = ExecutionConfig(shots=1024)
executor = ExecutorFactory.create_executor('simulator', config)

# Execute (same interface for all backends)
async with executor:
    result = await executor.execute_circuit(circuit)
```

### 📚 **Module Documentation**

### 🧠 **Core Module** (`core/`)
**Backend-agnostic quantum circuit logic and mathematical implementations**

#### **`circuit_interface.py`** - Abstract Circuit Interface
```python
from core.circuit_interface import CircuitSpec, AbstractQuantumCircuit

# Unified circuit specification
circuit_spec = CircuitSpec(
    num_qubits=5,
    gates=[gate("h", 0), gate("cx", [0, 1])],
    circuit_id="my_circuit"
)
```
**Features:**
- Backend-agnostic circuit representation
- Standardized gate operations
- Circuit builder pattern
- JSON serialization support

#### **`batch_manager.py`** - Unified Batch Processing
```python
from core.batch_manager import QuantumCircuitBatchManager

# Collect circuits from all tasks
batch_manager = QuantumCircuitBatchManager(exp_config)
batch_manager.collect_task_circuits("fidelity", circuits, specs, metadata)
batch_manager.collect_task_circuits("expressibility", circuits, specs, metadata)

# Execute once, distribute results
results = batch_manager.execute_unified_batch(executor)
```
**Features:**
- **Performance**: 3→1 backend connections (67% reduction)
- **Memory Efficient**: Optimized circuit collection
- **Result Distribution**: Automatic result mapping to original tasks
- **Error Tolerance**: Individual circuit failures don't crash batch

#### **`random_circuit_generator.py`** - Circuit Generation
```python
from core.random_circuit_generator import generate_random_circuit

# Generate parameterized random circuits
circuit = generate_random_circuit(
    num_qubits=5,
    depth=10,
    two_qubit_ratio=0.3,
    seed=42
)
```
**Features:**
- Configurable gate ratios
- Reproducible generation (seeded)
- Multiple circuit topologies
- Hardware-compatible gate sets

#### **`error_fidelity.py`** - Fidelity Calculation
```python
from core.error_fidelity import run_error_fidelity

# Calculate fidelity via inverse circuits
fidelity_results = run_error_fidelity(
    circuit_specs, exp_config, batch_manager
)
```
**Features:**
- Standard and robust fidelity metrics
- Inverse circuit generation
- Batch processing support
- Statistical error analysis

#### **`entangle_hardware.py`** - Entanglement Measurement
```python
from core.entangle_hardware import meyer_wallace_entropy_swap_test

# SWAP test based Meyer-Wallach entropy
mw_entropy = meyer_wallace_entropy_swap_test(circuit, exp_config)
```
**Features:**
- Hardware-compatible SWAP test protocol
- Meyer-Wallach entropy calculation
- Batch SWAP test execution
- Purity calculation from measurement results

#### **`statistical_validation_framework.py`** - Validation Framework
```python
from core.statistical_validation_framework import validate_entanglement

# Comprehensive statistical validation
results = validate_entanglement(
    exp_config=config,
    num_repetitions=5,
    save_path='validation_results/entanglement'
)
```
**Features:**
- **Publication Quality**: IEEE/Nature style visualizations
- **Statistical Rigor**: Pearson correlation, R², RMSE, MAE
- **Automated Reporting**: Text file summaries with quality assessment
- **Multi-metric Support**: Fidelity, expressibility, entanglement

### 🚀 **Execution Module** (`execution/`)
**Backend abstraction and execution engines**

#### **`executor.py`** - Abstract Executor Interface
```python
from execution.executor import QuantumExecutorFactory, AbstractQuantumExecutor

# Factory pattern for backend selection
executor = QuantumExecutorFactory.create_executor("ibm")  # or "simulator"
results = executor.run(circuits, exp_config)
```
**Features:**
- **Factory Pattern**: Clean backend selection
- **Context Manager**: Automatic resource management
- **Unified Interface**: Same API for all backends
- **Result Standardization**: Consistent `ExecutionResult` format

#### **`ibm_executor.py`** - IBM Quantum Hardware
```python
@register_executor("ibm")
class IBMExecutor(AbstractQuantumExecutor):
    # Automatic registration with factory
```
**Features:**
- **IBM Runtime Integration**: Latest IBM Quantum services
- **Batch Optimization**: Efficient circuit submission
- **Error Handling**: Robust job management
- **Queue Management**: Automatic backend selection

#### **`simulator_executor.py`** - Local Simulation
```python
@register_executor("simulator")
class SimulatorExecutor(AbstractQuantumExecutor):
    # High-performance local simulation
```
**Features:**
- **Qiskit Aer Integration**: Fast local simulation
- **Memory Optimization**: Efficient large circuit handling
- **Noise Modeling**: Optional noise simulation
- **Parallel Execution**: Multi-circuit batch processing

### 📊 **Expressibility Module** (`expressibility/`)
**Quantum circuit expressibility measurement**

#### **`fidelity_divergence.py`** - KL-Divergence Calculation
```python
from expressibility.fidelity_divergence import Divergence_Expressibility

# Calculate expressibility via KL-divergence
expressibility = Divergence_Expressibility(
    circuit_specs, exp_config, batch_manager
)
```
**Features:**
- **Haar Random Comparison**: Theoretical expressibility baseline
- **KL-Divergence Metrics**: Information-theoretic expressibility
- **Batch Processing**: Efficient multi-circuit analysis
- **Statistical Sampling**: Configurable sample sizes

#### **`swap_test_fidelity.py`** - SWAP Test Implementation
```python
from expressibility.swap_test_fidelity import create_swap_test_circuit

# Hardware-compatible SWAP test
swap_circuit = create_swap_test_circuit(state1, state2)
```
**Features:**
- **Hardware Compatible**: Real quantum device implementation
- **Fidelity Estimation**: Direct state overlap measurement
- **Ancilla Management**: Efficient ancilla qubit usage
- **Measurement Optimization**: Minimal shot requirements

### 🛠️ **Utils Module** (`utils/`)
**Utility functions and result handling**

#### **`result_handler.py`** - Result Processing
```python
from utils.result_handler import ResultHandler

# Comprehensive result management
handler = ResultHandler(output_dir="results")
handler.save_experiment_results(results, "experiment_1")
```
**Features:**
- **JSON Serialization**: Structured result storage
- **Visualization**: Automatic plot generation
- **Data Export**: CSV/Excel export capabilities
- **Metadata Management**: Comprehensive experiment tracking

### ⚙️ **Configuration System** (`config.py`)
**Centralized configuration management**

```python
from config import ExperimentConfig, Exp_Box, default_config

# Pre-configured experiments
config = Exp_Box.scalability_test
config = Exp_Box.statistical_validation_config

# Custom configuration
custom_config = ExperimentConfig(
    num_qubits=[3, 5, 7],
    depth=5,
    shots=1024,
    num_circuits=10
)
```
**Features:**
- **Experiment Templates**: Pre-configured setups
- **Environment Integration**: IBM token management
- **Validation**: Parameter validation and defaults
- **Extensibility**: Easy addition of new parameters
