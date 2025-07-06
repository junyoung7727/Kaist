#!/usr/bin/env python3
"""
Execution engine modules

This package contains the execution engine that provides a unified interface
for running quantum circuits on different backends (simulator and IBM hardware).
"""

from .executor import (
    AbstractQuantumExecutor,
    ExecutionResult,
    Config,
    default_config,
    QuantumExecutorFactory,
    register_executor
)

# Import concrete executors to register them
from . import simulator_executor
from . import ibm_executor

__all__ = [
    'AbstractQuantumExecutor',
    'ExecutionResult', 
    'Config',
    'default_config',
    'QuantumExecutorFactory',
    'register_executor'
]
