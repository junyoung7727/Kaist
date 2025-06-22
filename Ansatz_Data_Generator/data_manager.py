import pandas as pd
import json


def save_experiment_hdf5(circuits, sim_results=None, hw_results=None, file_path='experiments.h5'):
    """
    Save experiment data to HDF5 in table format for columnar evolution.

    circuits: list of dict, circuit metadata
    sim_results: list of dict, simulator expressibility results
    hw_results: list of dict, hardware execution results
    file_path: str, path to output HDF5 file
    """
    # Prepare circuits DataFrame
    circuit_records = []
    for c in circuits:
        record = {
            'circuit_id': c.get('circuit_id'),
            'config_group': c.get('config_group'),
            'n_qubits': c.get('n_qubits'),
            'depth': c.get('depth'),
            'two_qubit_ratio_target': c.get('two_qubit_ratio_target'),
            'gates': json.dumps(c.get('gates', [])),
            'wires_list': json.dumps(c.get('wires_list', [])),
            'params': json.dumps(c.get('params', [])),
            'params_idx': json.dumps(c.get('params_idx', []))
        }
        circuit_records.append(record)
    circuits_df = pd.DataFrame(circuit_records)

    # Save to HDF5 using pandas HDFStore
    with pd.HDFStore(file_path, mode='w') as store:
        store.put('circuits', circuits_df, format='table', data_columns=True)
        # Simulator results
        if sim_results is not None:
            sim_df = pd.DataFrame(sim_results)
            store.put('simulator_results', sim_df, format='table', data_columns=True)
        # Hardware results
        if hw_results is not None:
            hw_df = pd.DataFrame(hw_results)
            store.put('hardware_results', hw_df, format='table', data_columns=True)

    print(f"🗄️ Saved experiment HDF5: {file_path}")
