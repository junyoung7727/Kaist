#!/usr/bin/env python3
"""
Core quantum circuit modules

This package contains the core, backend-agnostic quantum circuit functionality:
- Abstract circuit interfaces
- Gate definitions and registry
- Inverse circuit generation
- Fidelity calculation
- Expressibility calculation
"""

from .circuit_interface import (
    AbstractQuantumCircuit,
    CircuitSpec,
    GateOperation,
    CircuitBuilder,
    create_circuit_spec,
    gate
)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "quantumcommon"))

from gates import (
    QuantumGateRegistry,
    GateDefinition,
    GateType,
    gate_registry,
    get_gate_info,
    get_inverse_gate,
    get_inverse_parameters,
    is_parametric,
    validate_gate_operation
)

from .inverse import (
    InverseCircuitGenerator,
    create_inverse_circuit_spec,
    create_fidelity_circuit_spec
)

from .error_fidelity import (
    ErrorFidelityCalculator,
    calculate_error_fidelity_from_result
)

__all__ = [
    # Circuit interfaces
    'AbstractQuantumCircuit',
    'CircuitSpec',
    'GateOperation',
    'CircuitBuilder',
    'create_circuit_spec',
    'gate',
    
    # Gate registry
    'QuantumGateRegistry',
    'GateDefinition',
    'GateType',
    'gate_registry',
    'get_gate_info',
    'get_inverse_gate',
    'get_inverse_parameters',
    'is_parametric',
    'validate_gate_operation',
    
    # Inverse circuits
    'InverseCircuitGenerator',
    'create_inverse_circuit_spec',
    'create_fidelity_circuit_spec',
    
    # Fidelity and Expressibility
    'ErrorFidelityCalculator',
]
