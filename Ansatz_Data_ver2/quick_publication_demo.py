#!/usr/bin/env python3
"""
Quick Publication Visualization Demo

Simple example showing publication-quality plots without running full validation.
Perfect for testing the visualization system.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add visualization module
project_root = Path(__file__).parent
sys.path.append(str(project_root / "visualization"))

from visualization.publication_visualizer import PublicationVisualizer
from core.statistical_validation_framework import ValidationResult


def create_sample_data():
    """Create realistic sample data for demonstration."""
    np.random.seed(42)
    
    # Simulate quantum expressibility measurements
    n_circuits = 8
    n_measurements_per_circuit = 5
    
    results = []
    
    for i in range(n_circuits):
        # Theoretical expressibility (KL divergence)
        theoretical_value = np.random.uniform(0.5, 2.5)
        
        # Simulated measurements with realistic noise
        noise_level = 0.15
        measured_values = []
        
        for j in range(n_measurements_per_circuit):
            # Shot noise + systematic errors
            shot_noise = np.random.normal(0, noise_level * theoretical_value)
            systematic_bias = 0.05 * theoretical_value  # Small systematic error
            measured = theoretical_value + shot_noise + systematic_bias
            measured_values.append(max(0.1, measured))  # Ensure positive
        
        # Create validation result
        result = ValidationResult(
            circuit_info={
                'ansatz': 'hardware_efficient',
                'num_qubits': 4,
                'num_layers': 2,
                'circuit_id': i
            },
            exact_values=[theoretical_value],
            measured_values=measured_values,
            statistics={
                'mean_measured': np.mean(measured_values),
                'std_measured': np.std(measured_values),
                'theoretical': theoretical_value
            },
            metadata={'simulation': True}
        )
        results.append(result)
    
    return results


def demo_nature_style():
    """Create Nature journal style visualization."""
    print("🔬 Creating Nature-style publication figure...")
    
    # Create sample data
    results = create_sample_data()
    
    # Initialize visualizer
    visualizer = PublicationVisualizer(style="nature")
    
    # Create figure
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    
    fig = visualizer.create_statistical_validation_figure(
        results=results,
        metric_name="Expressibility (KL Divergence)",
        save_path=str(output_dir / "nature_style_demo")
    )
    
    return fig


def demo_ieee_style():
    """Create IEEE journal style visualization."""
    print("🔬 Creating IEEE-style publication figure...")
    
    # Create sample data
    results = create_sample_data()
    
    # Initialize visualizer
    visualizer = PublicationVisualizer(style="ieee")
    
    # Create figure
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    
    fig = visualizer.create_statistical_validation_figure(
        results=results,
        metric_name="Quantum Circuit Expressibility",
        save_path=str(output_dir / "ieee_style_demo")
    )
    
    return fig


def demo_comparison_plot():
    """Create ansatz comparison visualization."""
    print("🔬 Creating ansatz comparison figure...")
    
    # Create data for different ansatz types
    ansatz_results = {}
    
    # Hardware Efficient (2 layers)
    np.random.seed(42)
    ansatz_results["HE (2 layers)"] = create_sample_data()
    
    # Hardware Efficient (3 layers) - slightly different performance
    np.random.seed(43)
    results_3layer = []
    for i in range(6):
        theoretical = np.random.uniform(0.8, 3.0)  # Higher expressibility
        measured = [theoretical + np.random.normal(0, 0.12 * theoretical) for _ in range(4)]
        result = ValidationResult(
            circuit_info={'ansatz': 'hardware_efficient_3layer', 'circuit_id': i},
            exact_values=[theoretical],
            measured_values=measured,
            statistics={},
            metadata={}
        )
        results_3layer.append(result)
    ansatz_results["HE (3 layers)"] = results_3layer
    
    # Real Amplitudes - different characteristics
    np.random.seed(44)
    results_ra = []
    for i in range(5):
        theoretical = np.random.uniform(0.3, 2.0)  # Lower expressibility
        measured = [theoretical + np.random.normal(0, 0.18 * theoretical) for _ in range(4)]
        result = ValidationResult(
            circuit_info={'ansatz': 'real_amplitudes', 'circuit_id': i},
            exact_values=[theoretical],
            measured_values=measured,
            statistics={},
            metadata={}
        )
        results_ra.append(result)
    ansatz_results["Real Amplitudes"] = results_ra
    
    # Create comparison visualization
    visualizer = PublicationVisualizer(style="nature")
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    
    fig = visualizer.create_expressibility_comparison_figure(
        results_dict=ansatz_results,
        save_path=str(output_dir / "ansatz_comparison_demo")
    )
    
    return fig


def main():
    """Run quick demonstration."""
    print("🎯 Quick Publication Visualization Demo")
    print("=" * 45)
    
    try:
        # Create results directory
        results_dir = project_root / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Demo 1: Nature style
        nature_fig = demo_nature_style()
        
        # Demo 2: IEEE style
        ieee_fig = demo_ieee_style()
        
        # Demo 3: Comparison plot
        comparison_fig = demo_comparison_plot()
        
        print("\n" + "=" * 45)
        print("✅ Publication figures created successfully!")
        print(f"📁 Saved in: {results_dir}")
        
        print("\n📄 Generated files:")
        for file_path in results_dir.glob("*_demo.*"):
            print(f"   • {file_path.name}")
        
        print("\n🎨 Features demonstrated:")
        print("   • Nature journal style (serif fonts, classic layout)")
        print("   • IEEE style (sans-serif, technical appearance)")
        print("   • Statistical correlation analysis")
        print("   • Error distribution visualization")
        print("   • Professional statistical summary tables")
        print("   • Multi-format export (PDF/PNG/SVG)")
        print("   • Ansatz comparison analysis")
        
        print("\n📊 Statistical metrics included:")
        print("   • Pearson correlation coefficient")
        print("   • 95% confidence intervals")
        print("   • RMSE and MAE")
        print("   • Bias and variance analysis")
        print("   • Normality tests")
        
        # Show plots
        plt.show()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
