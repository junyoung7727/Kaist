#!/usr/bin/env python3
"""
Example: reconstruct a Qiskit circuit from saved metadata and display it.
"""
import json
from qiskit import QuantumCircuit


def reconstruct_circuit(circuit_info: dict) -> QuantumCircuit:
    """
    Rebuilds a Qiskit QuantumCircuit from metadata.
    """
    # basic info
    n_qubits = circuit_info.get('n_qubits', 0)
    qc = QuantumCircuit(n_qubits)
    gates = circuit_info.get('gates', [])
    wires_list = circuit_info.get('wires_list', [])
    params = circuit_info.get('params', [])
    params_idx = circuit_info.get('params_idx', [])
    # mappings
    alias_map = {'cnot': 'cx', 'measure': 'measure_all'}
    param_gates = {'rx', 'ry', 'rz'}
    param_dict = {i: params[j] for j, i in enumerate(params_idx)}
    # apply gates
    for i, gate in enumerate(gates):
        gl = gate.lower()
        ws = wires_list[i] if i < len(wires_list) else []
        pv = param_dict.get(i)
        method_name = alias_map.get(gl, gl)
        method = getattr(qc, method_name, None)
        if not method:
            continue
        try:
            if method_name in param_gates:
                if pv is None or not ws:
                    continue
                method(pv, ws[0])
            else:
                if ws:
                    method(*ws)
                else:
                    method()
        except Exception as e:
            print(f"Warning: could not apply {gate} at index {i}: {e}")
    return qc


if __name__ == '__main__':
    # example metadata
    sample = {
        'n_qubits': 3,
        'gates': ['H', 'CNOT', 'RZ', 'X', 'MEASURE'],
        'wires_list': [[0], [0, 1], [1], [2], []],
        'params': [3.1415],
        'params_idx': [2],
    }
    qc = reconstruct_circuit(sample)
    print(qc)
    # draw circuit
    try:
        print(qc.draw(output='text'))
    except Exception:
        pass
