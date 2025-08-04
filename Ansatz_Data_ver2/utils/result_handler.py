#!/usr/bin/env python3
"""
Result Handler Module

This module provides utilities for handling experiment results:
- Saving results to files
- Formatting results for display
- Computing summary statistics
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union
from core.circuit_interface import CircuitSpec

# Custom JSON encoder to handle non-serializable objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                                np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, complex):
                return str(obj)
            # Check for CircuitSpec object from circuit_interface module
            elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'CircuitSpec':
                return {
                    "circuit_id": obj.circuit_id,
                    "num_qubits": obj.num_qubits,
                    "gates": [{
                        "name": gate.name,
                        "qubits": gate.qubits,
                        "parameters": gate.parameters
                    } for gate in obj.gates]
                }
            # Check for GateOperation object from gates module
            elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'GateOperation':
                return {
                    "name": obj.name,
                    "qubits": obj.qubits,
                    "parameters": obj.parameters
                }
            return str(obj)  # Safely convert any other type to string
        except:
            return "<unserializable object>"

class ResultHandler:
    """
    A class for handling experiment results.
    """

    @staticmethod
    def safe_get(obj: Any, key: str, default_value: Any = "N/A") -> Any:
        """
        Safely get a value from a dictionary or an object.
        
        Args:
            obj: The dictionary or object to get the value from
            key: The key to get the value for
            default_value: The default value to return if the key is not found
            
        Returns:
            The value for the key or the default value
        """
        if obj is None:
            return default_value
            
        # Try dictionary access first
        if isinstance(obj, dict):
            return obj.get(key, default_value)
        
        # Try object attribute access
        try:
            return getattr(obj, key, default_value)
        except:
            # If all fails, return default
            return default_value

    @staticmethod
    def format_circuit_info(circuit_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format circuit information for saving or display.
        
        Args:
            circuit_info: Raw circuit information
            
        Returns:
            Formatted circuit information
        """
        formatted_info = {}
        
        # Handle fidelity
        if "simulator_error_fidelity" in circuit_info:
            formatted_info["simulator_error_fidelity"] = circuit_info["simulator_error_fidelity"]

        # Handle expressibility divergence
        if "expressibility_divergence" in circuit_info:
            div = circuit_info["expressibility_divergence"]
            if isinstance(div, dict):
                formatted_info["expressibility_divergence"] = div
            else:
                formatted_info["expressibility_divergence"] = {
                    "expressibility": ResultHandler.safe_get(div, "expressibility"),
                    "kl_div_circuit_haar": ResultHandler.safe_get(div, "kl_div_circuit_haar"),
                    "kl_div_haar_circuit": ResultHandler.safe_get(div, "kl_div_haar_circuit")
                }
        
        # Handle expressibility shadow
        if "expressibility_shadow" in circuit_info:
            formatted_info["expressibility_shadow"] = circuit_info["expressibility_shadow"]
        
        # Copy any other fields
        for key, value in circuit_info.items():
            if key not in formatted_info and key not in ["circuit_specs", "simulator_error_fidelity", "expressibility_divergence", "expressibility_shadow"]:
                formatted_info[key] = value
                
        return formatted_info

    @staticmethod
    def save_experiment_results(experiment_results: List[Dict[str, Any]], 
                               exp_config: Any, 
                               output_dir: str = "output",
                               filename: str = "experiment_results.json") -> str:
        """
        Save experiment results to a JSON file.
        
        Args:
            experiment_results: List of experiment results
            exp_config: Experiment configuration
            output_dir: Output directory
            filename: Output filename
            
        Returns:
            Path to the saved file
        """
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Format all results with additional JSON serialization safety
        formatted_results = []
        for result in experiment_results:
            try:
                formatted_result = ResultHandler.format_circuit_info(result)
                # Ensure all values are JSON serializable
                formatted_results.append(formatted_result)
            except Exception as e:
                print(f"Warning: Failed to format circuit result: {e}")
                # Add error information instead of skipping entirely
                formatted_results.append({
                    "circuit_id": ResultHandler.safe_get(result, "circuit_id", "unknown"),
                    "error": f"Failed to format: {str(e)}"
                })
        
        # Calculate summary statistics
        summary = {}
            
        # Calculate average expressibility divergence if available
        try:
            div_values = [
                float(r["expressibility_divergence"]["expressibility"]) for r in formatted_results
                if "expressibility_divergence" in r 
                and "expressibility" in r["expressibility_divergence"]
                and r["expressibility_divergence"]["expressibility"] not in ["N/A", None]
                and isinstance(r["expressibility_divergence"]["expressibility"], (int, float, str))
                and str(r["expressibility_divergence"]["expressibility"]).replace(".", "").isdigit()
            ]
            if div_values:
                summary["average_expressibility_div"] = float(np.mean(div_values))
        except Exception as e:
            print(f"Warning: Failed to calculate average expressibility: {e}")
            summary["average_expressibility_div_error"] = str(e)
            
        # Calculate average simulator_error_fidelity if available
        try:
            fidelity_values = [
                float(r["simulator_error_fidelity"]) for r in formatted_results
                if "simulator_error_fidelity" in r 
                and r["simulator_error_fidelity"] not in ["N/A", None]
                and isinstance(r["simulator_error_fidelity"], (int, float, str)) 
                and str(r["simulator_error_fidelity"]).replace(".", "").isdigit()
            ]
            if fidelity_values:
                summary["average_simulator_error_fidelity"] = float(np.mean(fidelity_values))
        except Exception as e:
            print(f"Warning: Failed to calculate average simulator_error_fidelity: {e}")
            summary["average_simulator_error_fidelity_error"] = str(e)
            
        # Calculate average shadow expressibility if available
        try:
            shadow_values = [
                float(r["expressibility_shadow"]["summary"]["local2_expressibility"]) 
                for r in formatted_results
                if "expressibility_shadow" in r 
                and "summary" in r["expressibility_shadow"] 
                and "local2_expressibility" in r["expressibility_shadow"]["summary"]
                and r["expressibility_shadow"]["summary"]["local2_expressibility"] not in ["N/A", None]
                and isinstance(r["expressibility_shadow"]["summary"]["local2_expressibility"], (int, float, str))
                and str(r["expressibility_shadow"]["summary"]["local2_expressibility"]).replace(".", "").isdigit()
            ]
            if shadow_values:
                summary["average_expressibility_shadow"] = float(np.mean(shadow_values))
        except Exception as e:
            print(f"Warning: Failed to calculate average shadow expressibility: {e}")
            summary["average_expressibility_shadow_error"] = str(e)
        
        # Build final data structure
        output_data = {
            "experiment_name": ResultHandler.safe_get(exp_config, "exp_name"),
            "experiment_config": {
                "num_qubits": ResultHandler.safe_get(exp_config, "num_qubits"),
                "depth": ResultHandler.safe_get(exp_config, "depth"),
                "shots": ResultHandler.safe_get(exp_config, "shots"),
                "num_circuits": ResultHandler.safe_get(exp_config, "num_circuits"),
                "optimization_level": ResultHandler.safe_get(exp_config, "optimization_level"),
                "two_qubit_ratio": ResultHandler.safe_get(exp_config, "two_qubit_ratio")
            },
            "results": formatted_results,
            "summary": summary
        }
        
        # Save to file with improved error handling
        output_path = os.path.join(output_dir, filename)

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, cls=CustomJSONEncoder)
        print(f"Successfully saved experiment results to {output_path}")

        return output_path

    @staticmethod
    def save_experiment_results_with_circuits(experiment_results: List[Dict[str, Any]], 
                                            circuit_specs: List[CircuitSpec],
                                            exp_config: Any, 
                                            output_dir: str = "output",
                                            filename: str = "experiment_results_with_circuits.json") -> str:
        """
        Save experiment results with matching circuit information to a JSON file.
        
        Args:
            experiment_results: List of experiment results
            circuit_specs: List of CircuitSpec objects
            exp_config: Experiment configuration
            output_dir: Output directory
            filename: Output filename
            
        Returns:
            Path to the saved file
        """
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a mapping from circuit_id to circuit_spec
        circuit_map = {spec.circuit_id: spec for spec in circuit_specs}
        
        # Separate circuits and results dictionaries
        circuits_dict = {}
        results_dict = {}
        
        for result in experiment_results:
            try:
                formatted_result = ResultHandler.format_circuit_info(result)
                circuit_id = formatted_result.get('circuit_id')
                
                if not circuit_id:
                    print(f"Warning: Result without circuit_id found, skipping")
                    continue
                
                # Store result data (without circuit info)
                results_dict[circuit_id] = formatted_result
                
                # Store circuit information separately if found
                if circuit_id in circuit_map:
                    circuit_spec = circuit_map[circuit_id]
                    circuits_dict[circuit_id] = {
                        'circuit_id': circuit_spec.circuit_id,
                        'num_qubits': circuit_spec.num_qubits,
                        'gates': [{
                            'name': gate.name,
                            'qubits': gate.qubits,
                            'parameters': gate.parameters
                        } for gate in circuit_spec.gates],
                        'qasm': circuit_spec.to_qasm() if hasattr(circuit_spec, 'to_qasm') else None
                    }
                
            except Exception as e:
                print(f"Warning: Failed to format circuit result: {e}")
                # Still try to store with error info
                circuit_id = ResultHandler.safe_get(result, "circuit_id", f"error_{len(results_dict)}")
                results_dict[circuit_id] = {
                    "circuit_id": circuit_id,
                    "error": f"Failed to format: {str(e)}"
                }
        
        # Calculate summary statistics
        summary = {
            'total_results': len(results_dict),
            'total_circuits': len(circuits_dict),
            'matched_pairs': len(set(results_dict.keys()) & set(circuits_dict.keys())),
            'results_without_circuits': len(set(results_dict.keys()) - set(circuits_dict.keys())),
            'circuits_without_results': len(set(circuits_dict.keys()) - set(results_dict.keys()))
        }
        
        # Create the final data structure with separated top-level objects
        data = {
            "experiment_name": getattr(exp_config, 'exp_name', 'unknown'),
            "circuits": circuits_dict,
            "results": results_dict,
            "summary": summary
        }
        
        # Save to file
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
        
        print(f"💾 Results with circuits saved to: {filepath}")
        print(f"   - Total results: {summary['total_results']}")
        print(f"   - Total circuits: {summary['total_circuits']}")
        print(f"   - Matched pairs: {summary['matched_pairs']}")
        print(f"   - Results without circuits: {summary['results_without_circuits']}")
        print(f"   - Circuits without results: {summary['circuits_without_results']}")
        
        return filepath

    @staticmethod
    def save_circuit_specs(circuit_specs: List[CircuitSpec], exp_config: Any, output_dir: str = "output", 
                          filename: str = "circuit_specs.json") -> str:
        """
        Save a list of CircuitSpec objects to a JSON file.
        
        Args:
            circuit_specs: List of CircuitSpec objects
            exp_config: Experiment configuration
            output_dir: Output directory
            filename: Output filename
            
        Returns:
            Path to the saved file
        """
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Build output data structure

        circuit_specs_dict = []
        for circuit_spec in circuit_specs:
            circuit_specs_dict.append(circuit_spec.to_dict())

        output_data = {
            "experiment_name": ResultHandler.safe_get(exp_config, "exp_name", "circuit_specs"),
            "experiment_config": {
                "num_qubits": ResultHandler.safe_get(exp_config, "num_qubits"),
                "depth": ResultHandler.safe_get(exp_config, "depth"),
                "num_circuits": len(circuit_specs),
                "exp_name": ResultHandler.safe_get(exp_config, "exp_name"),
                "two_qubit_ratio": ResultHandler.safe_get(exp_config, "two_qubit_ratio")
            },
            "circuits": circuit_specs_dict
        }
        
        # Save to file with error handling
        output_path = os.path.join(output_dir, filename)
        try:
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2, cls=CustomJSONEncoder)
            print(f"Successfully saved circuit specs to {output_path}")
            
            # Validate the output file
            with open(output_path, 'r') as f:
                data = json.load(f)
                print(f"  - Saved {len(data.get('circuits', []))} circuits")
                
        except Exception as e:
            print(f"Error saving circuit specs: {e}")
        return output_path
        
    @staticmethod
    def print_result_summary(experiment_results: List[Dict[str, Any]]) -> None:
        """
        Print a summary of experiment results.
        
        Args:
            experiment_results: List of experiment results
        """
        # Format all results
        formatted_results = [
            ResultHandler.format_circuit_info(result) 
            for result in experiment_results
        ]
        
        print("=== 실험 요약 ===")
        print(f"총 회로 수: {len(formatted_results)}")
        print(f"성공한 회로 수: {len([r for r in formatted_results if 'simulator_error_fidelity' in r and r['simulator_error_fidelity'] not in [None, 'N/A', 'error']])}")
    
            
        # Calculate and print average expressibility divergence if available
        div_values = [
            r["expressibility_divergence"]["expressibility"] for r in formatted_results
            if "expressibility_divergence" in r 
            and "expressibility" in r["expressibility_divergence"]
            and r["expressibility_divergence"]["expressibility"] not in ["N/A", None]
        ]
        if div_values:
            print(f"평균 표현력(다이버전스): {float(np.mean(div_values)):.6f}")
            
        # Calculate and print average shadow expressibility if available
        shadow_values = [
            r["expressibility_shadow"]["summary"]["local2_expressibility"] 
            for r in formatted_results
            if "expressibility_shadow" in r 
            and "summary" in r["expressibility_shadow"] 
            and "local2_expressibility" in r["expressibility_shadow"]["summary"]
            and r["expressibility_shadow"]["summary"]["local2_expressibility"] not in ["N/A", None]
        ]
        if shadow_values:
            print(f"평균 표현력(쉐도우): {float(np.mean(shadow_values)):.6f}")
