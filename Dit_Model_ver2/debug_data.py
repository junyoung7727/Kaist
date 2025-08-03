#!/usr/bin/env python3
"""
Debug script to analyze data and find NaN issues
"""

import json
import numpy as np
from pathlib import Path


def analyze_raw_data():
    """Analyze raw JSON data for NaN and problematic values"""
    print("🔍 Analyzing Raw JSON Data for Issues...")
    
    # Load experiment results
    with open(r"C:\Users\jungh\Documents\GitHub\Kaist\Dit_Model_ver2\data\raw\experiment_results.json", 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    print(f"Found {len(results)} experiment results")
    
    # Analyze target properties
    target_properties = ['expressibility', 'two_qubit_ratio', 'simulator_error_fidelity']
    
    print("\n📊 Raw Data Analysis:")
    
    for prop in target_properties:
        print(f"\n  {prop}:")
        values = []
        nan_count = 0
        inf_count = 0
        missing_count = 0
        
        for i, result in enumerate(results):
            if prop == 'expressibility':
                # Get from expressibility_divergence
                expr_div = result.get('expressibility_divergence', {})
                if 'kl_divergence' in expr_div:
                    val = expr_div['kl_divergence']
                elif 'js_divergence' in expr_div:
                    val = expr_div['js_divergence']
                else:
                    val = None
                    missing_count += 1
            elif prop == 'two_qubit_ratio':
                val = result.get('two_qubit_ratio')
            elif prop == 'simulator_error_fidelity':
                val = result.get('simulator_error_fidelity')
            else:
                val = None
                missing_count += 1
            
            if val is not None:
                if np.isnan(val):
                    nan_count += 1
                    print(f"    ❌ Sample {i} ({result.get('circuit_id', 'unknown')}) has NaN {prop}: {val}")
                elif np.isinf(val):
                    inf_count += 1
                    print(f"    ❌ Sample {i} ({result.get('circuit_id', 'unknown')}) has Inf {prop}: {val}")
                else:
                    values.append(val)
        
        if values:
            values = np.array(values)
            print(f"    Valid values: {len(values)}")
            print(f"    Mean: {np.mean(values):.6f}")
            print(f"    Std: {np.std(values):.6f}")
            print(f"    Min: {np.min(values):.6f}")
            print(f"    Max: {np.max(values):.6f}")
        
        print(f"    NaN count: {nan_count}")
        print(f"    Inf count: {inf_count}")
        print(f"    Missing count: {missing_count}")
    
    return results


def suggest_fixes(results):
    """Suggest fixes based on data analysis"""
    print("\n🔧 Suggested Fixes:")
    
    # Check if we have any problematic values
    has_nan = False
    has_inf = False
    has_extreme_values = False
    
    target_properties = ['expressibility', 'two_qubit_ratio', 'simulator_error_fidelity']
    
    for prop in target_properties:
        values = []
        for result in results:
            if prop == 'expressibility':
                expr_div = result.get('expressibility_divergence', {})
                val = expr_div.get('kl_divergence') or expr_div.get('js_divergence')
            elif prop == 'two_qubit_ratio':
                val = result.get('two_qubit_ratio')
            elif prop == 'simulator_error_fidelity':
                val = result.get('simulator_error_fidelity')
            else:
                val = None
            
            if val is not None:
                if np.isnan(val):
                    has_nan = True
                elif np.isinf(val):
                    has_inf = True
                elif abs(val) > 1000:
                    has_extreme_values = True
                else:
                    values.append(val)
    
    if has_nan:
        print("  1. ❌ NaN values detected - Replace NaN with 0 or mean value")
        print("     Add: targets = torch.nan_to_num(targets, nan=0.0)")
    
    if has_inf:
        print("  2. ❌ Infinite values detected - Clamp extreme values")
        print("     Add: targets = torch.clamp(targets, min=-10, max=10)")
    
    if has_extreme_values:
        print("  3. ⚠️  Extreme values detected - Consider normalization")
        print("     Add target normalization: (x - mean) / std")
    
    print("  4. 🎯 Reduce learning rate to prevent gradient explosion")
    print("     Try: learning_rate = 1e-5 or 1e-6")
    
    print("  5. 🛡️  Add gradient clipping")
    print("     Add: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)")
    
    print("  6. 🔧 Use smaller model for debugging")
    print("     Try: d_model=64, n_layers=2, n_heads=2")


def main():
    """Main debug function"""
    print("🐛 Debugging NaN Loss Issue")
    print("=" * 50)
    
    try:
        # Analyze raw data
        results = analyze_raw_data()
        
        # Suggest fixes
        suggest_fixes(results)
        
        print("\n" + "=" * 50)
        print("🎯 Debug Complete!")
        print("Apply the suggested fixes to resolve NaN loss issues.")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
