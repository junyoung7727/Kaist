# Quantum Circuit Backend - Modular Architecture

A beautiful, maintainable, and modular quantum circuit backend system with strict separation between simulator and IBM quantum hardware logic.

## 🎯 Key Features

- **Zero Backend Awareness**: Core logic never knows which backend is being used
- **Unified Circuit Interface**: All circuits use the same abstract interface regardless of backend
- **Dependency Injection**: Backend selection happens only at runtime through factory pattern
- **Mathematical Correctness**: Pure mathematical implementations for gates, inverse circuits, fidelity, and expressibility
- **Clean Architecture**: Beautiful, maintainable code with clear separation of concerns

## 🏗️ Architecture

```
Ansatz_Data_ver2/
├── main_new.py                 # Entry point (backend selection only here)
├── config.py                   # Centralized configuration
├── core/                       # Backend-agnostic pure logic
│   ├── circuit.py              # Abstract circuit interface
│   ├── gates.py                # Gate definitions & registry
│   ├── inverse.py              # Inverse circuit generation
│   ├── fidelity.py             # Fidelity calculation
│   └── expressibility.py       # Expressibility calculation
├── execution/                  # Execution engine (backend abstraction)
│   ├── executor.py             # Abstract executor interface
│   ├── simulator_executor.py   # Simulator implementation
│   └── ibm_executor.py         # IBM hardware implementation
└── requirements.txt
```

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the demo:**
   ```bash
   python main_new.py
   ```

3. **Choose your backend:**
   - `1` for Simulator (no setup required)
   - `2` for IBM Quantum (requires IBM account and token)

## 🔧 Configuration

Edit `config.py` to customize:

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

### Pure Mathematical Functions

Core calculations are backend-independent:

```python
from core.fidelity import calculate_fidelity
from core.expressibility import calculate_expressibility

# Calculate fidelity from measurement counts
fidelity = calculate_fidelity(counts, num_qubits)

# Calculate expressibility from fidelity list
expressibility = calculate_expressibility(fidelities, num_qubits)
```

## 🎨 Design Principles

1. **Single Responsibility**: Each module has one clear purpose
2. **Dependency Inversion**: High-level modules don't depend on low-level details
3. **Interface Segregation**: Clean, minimal interfaces
4. **Open/Closed**: Easy to extend with new backends without modifying existing code

## 🔬 Example: Adding a New Backend

```python
from execution.executor import AbstractQuantumExecutor, register_executor

@register_executor('my_backend')
class MyBackendExecutor(AbstractQuantumExecutor):
    async def execute_circuit(self, circuit):
        # Your implementation here
        pass
```

That's it! No changes needed anywhere else in the codebase.

## 📊 What It Does

1. **Generates** random quantum circuits
2. **Creates** inverse circuits for fidelity measurement
3. **Executes** circuits on chosen backend (simulator or IBM)
4. **Calculates** fidelity for each circuit
5. **Computes** expressibility using Kolmogorov-Smirnov test
6. **Saves** results to JSON files

## 🎯 Benefits

- **Maintainable**: Clear separation makes code easy to understand and modify
- **Testable**: Each component can be tested independently
- **Extensible**: Add new backends, gates, or metrics without breaking existing code
- **Efficient**: No unnecessary backend checks or conditional logic in hot paths
- **Beautiful**: Clean, readable code that follows best practices

## 🔍 No More Backend Checks!

❌ **Old way (bad):**
```python
if backend_type == 'simulator':
    # simulator logic
elif backend_type == 'ibm':
    # IBM logic
```

✅ **New way (beautiful):**
```python
# Backend-agnostic code everywhere
result = await executor.execute_circuit(circuit)
fidelity = calculate_fidelity(result.counts, num_qubits)
```

The backend selection happens exactly once, in the factory. Everything else is pure, beautiful, backend-agnostic code.
