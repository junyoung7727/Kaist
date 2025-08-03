#!/usr/bin/env python3
"""
Publication-Quality Scientific Visualization Module

Professional visualization tools for quantum circuit analysis following
IEEE and Nature publication standards.

Author: Quantum Research Team
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns
from scipy import stats
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import from core module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.statistical_validation_framework import ValidationResult


class PublicationVisualizer:
    """
    Publication-quality scientific visualization for quantum circuit analysis.
    
    Follows IEEE and Nature journal standards:
    - High-resolution vector graphics
    - Professional typography
    - Consistent color schemes
    - Statistical rigor
    - Clear data presentation
    """
    
    def __init__(self, style: str = "nature"):
        """
        Initialize publication visualizer.
        
        Args:
            style: Publication style ('nature', 'ieee', 'prl', 'custom')
        """
        self.style = style
        self._setup_publication_style()
        
    def _setup_publication_style(self):
        """Setup publication-quality matplotlib parameters."""
        
        # Base configuration for all styles
        base_config = {
            'font.size': 10,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Computer Modern Roman'],
            'mathtext.fontset': 'cm',
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.edgecolor': 'black',
            'axes.labelsize': 10,
            'axes.titlesize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'legend.frameon': True,
            'legend.fancybox': False,
            'legend.shadow': False,
            'legend.edgecolor': 'black',
            'legend.facecolor': 'white',
            'legend.framealpha': 0.9,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'grid.color': 'gray',
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.format': 'pdf',
            'savefig.bbox': 'tight',
            'text.usetex': False,  # Set to True if LaTeX is available
        }
        
        # Style-specific configurations
        if self.style == "nature":
            style_config = {
                'figure.facecolor': 'white',
                'axes.facecolor': 'white',
                'text.color': 'black',
                'axes.labelcolor': 'black',
                'xtick.color': 'black',
                'ytick.color': 'black',
                'axes.prop_cycle': plt.cycler('color', [
                    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
                ])
            }
        elif self.style == "ieee":
            style_config = {
                'figure.facecolor': 'white',
                'axes.facecolor': 'white',
                'text.color': 'black',
                'axes.labelcolor': 'black',
                'xtick.color': 'black',
                'ytick.color': 'black',
                'font.family': 'sans-serif',
                'font.sans-serif': ['Arial', 'Helvetica'],
                'axes.prop_cycle': plt.cycler('color', [
                    '#0072BD', '#D95319', '#EDB120', '#7E2F8E',
                    '#77AC30', '#4DBEEE', '#A2142F'
                ])
            }
        elif self.style == "prl":
            style_config = {
                'figure.facecolor': 'white',
                'axes.facecolor': 'white',
                'text.color': 'black',
                'axes.labelcolor': 'black',
                'xtick.color': 'black',
                'ytick.color': 'black',
                'axes.prop_cycle': plt.cycler('color', [
                    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'
                ])
            }
        
        # Merge configurations
        final_config = {**base_config, **style_config}
        plt.rcParams.update(final_config)
        
        # Set color palette
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'info': '#17a2b8',
            'warning': '#ffc107',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
    def create_statistical_validation_figure(self, 
                                           results: List[ValidationResult], 
                                           metric_name: str = "Quantum Metric",
                                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create publication-quality statistical validation figure.
        
        Args:
            results: List of validation results
            metric_name: Name of the quantum metric
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        if not results:
            raise ValueError("No validation results provided")
            
        # Extract and prepare data
        exact_values, measured_values = self._prepare_data(results)
        
        if len(exact_values) == 0:
            raise ValueError("No valid data points found")
            
        # Calculate statistics
        stats_dict = self._calculate_statistics(exact_values, measured_values)
        
        # Create figure with publication layout
        fig = plt.figure(figsize=(12, 8))
        
        # Create grid layout for subplots
        gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], width_ratios=[2, 2, 1],
                             hspace=0.3, wspace=0.3)
        
        # Main correlation plot
        ax_main = fig.add_subplot(gs[0, :2])
        self._plot_correlation_analysis(ax_main, exact_values, measured_values, stats_dict)
        
        # Error distribution
        ax_error = fig.add_subplot(gs[0, 2])
        self._plot_error_distribution(ax_error, exact_values, measured_values)
        
        # Statistical summary table
        ax_stats = fig.add_subplot(gs[1, :])
        self._plot_statistics_table(ax_stats, stats_dict, metric_name)
        
        # Add figure title
        fig.suptitle(f'Statistical Validation: {metric_name}', 
                    fontsize=14, fontweight='bold', y=0.95)
        
        # Save figure
        if save_path:
            self._save_figure(fig, save_path)
            
        return fig
        
    def _prepare_data(self, results: List[ValidationResult]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and validate data from results."""
        exact_values = []
        measured_values = []
        
        for result in results:
            exact_vals = result.exact_values
            measured_vals = result.measured_values
            
            # Handle single exact value case
            if len(exact_vals) == 1:
                exact_vals = exact_vals * len(measured_vals)
                
            exact_values.extend(exact_vals)
            measured_values.extend(measured_vals)
            
        return np.array(exact_values), np.array(measured_values)
        
    def _calculate_statistics(self, exact: np.ndarray, measured: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive statistical metrics."""
        n_samples = len(exact)
        
        # Correlation analysis
        if np.std(exact) == 0 or np.std(measured) == 0:
            r_value, p_value = 0.0, 1.0
            ci_lower, ci_upper = 0.0, 0.0
        else:
            r_value, p_value = stats.pearsonr(exact, measured)
            
            # 95% confidence interval using Fisher's z-transformation
            z = np.arctanh(r_value)
            se = 1 / np.sqrt(n_samples - 3) if n_samples > 3 else 1.0
            z_ci = 1.96 * se
            ci_lower = np.tanh(z - z_ci)
            ci_upper = np.tanh(z + z_ci)
            
        # Error metrics
        errors = measured - exact
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))
        mape = np.mean(np.abs(errors / exact)) * 100 if np.all(exact != 0) else np.inf
        
        # Bias and variance
        bias = np.mean(errors)
        variance = np.var(errors)
        
        # Normality test for errors
        if n_samples >= 8:
            shapiro_stat, shapiro_p = stats.shapiro(errors)
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan
            
        return {
            'n_samples': n_samples,
            'r_value': r_value,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'bias': bias,
            'variance': variance,
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p
        }
        
    def _plot_correlation_analysis(self, ax, exact: np.ndarray, measured: np.ndarray, 
                                 stats_dict: Dict[str, float]):
        """Plot correlation analysis with regression line."""
        
        # Scatter plot
        ax.scatter(exact, measured, c=self.colors['primary'], s=40, alpha=0.7,
                  edgecolors='white', linewidths=0.5, zorder=3)
        
        # Perfect agreement line
        min_val = min(exact.min(), measured.min())
        max_val = max(exact.max(), measured.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               color=self.colors['success'], linewidth=2, alpha=0.8,
               linestyle='--', label='Perfect Agreement', zorder=2)
        
        # Regression line
        if stats_dict['r_value'] != 0:
            slope, intercept, _, _, _ = stats.linregress(exact, measured)
            line_x = np.linspace(min_val, max_val, 100)
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, color=self.colors['danger'], linewidth=2,
                   alpha=0.8, label=f'Linear Fit (r={stats_dict["r_value"]:.3f})', zorder=2)
        
        # Formatting
        ax.set_xlabel('Theoretical Value', fontweight='bold')
        ax.set_ylabel('Measured Value', fontweight='bold')
        ax.set_title('Correlation Analysis', fontweight='bold', pad=20)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3, zorder=1)
        
        # Add correlation info as text box
        textstr = f'r = {stats_dict["r_value"]:.4f}\np = {stats_dict["p_value"]:.2e}\nn = {stats_dict["n_samples"]}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=props)
        
    def _plot_error_distribution(self, ax, exact: np.ndarray, measured: np.ndarray):
        """Plot error distribution histogram."""
        errors = measured - exact
        
        # Histogram
        n_bins = min(15, max(5, len(errors) // 5))
        counts, bins, patches = ax.hist(errors, bins=n_bins, 
                                       color=self.colors['secondary'], alpha=0.7,
                                       edgecolor='black', linewidth=0.5, density=True)
        
        # Normal distribution overlay
        mu, sigma = np.mean(errors), np.std(errors)
        x = np.linspace(errors.min(), errors.max(), 100)
        normal_curve = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, normal_curve, color=self.colors['danger'], linewidth=2,
               label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
        
        # Mean line
        ax.axvline(mu, color=self.colors['dark'], linewidth=2, 
                  linestyle=':', alpha=0.8, label=f'Mean: {mu:.4f}')
        
        # Zero line
        ax.axvline(0, color=self.colors['success'], linewidth=2,
                  linestyle='-', alpha=0.8, label='Zero Error')
        
        # Formatting
        ax.set_xlabel('Error (Measured - Theoretical)', fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.set_title('Error Distribution', fontweight='bold', pad=20)
        ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
        ax.grid(True, alpha=0.3)
        
    def _plot_statistics_table(self, ax, stats_dict: Dict[str, float], metric_name: str):
        """Create a professional statistics summary table."""
        ax.axis('off')
        
        # Prepare table data
        table_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['Sample Size (n)', f"{stats_dict['n_samples']}", 'Number of data points'],
            ['Pearson r', f"{stats_dict['r_value']:.4f}", 'Linear correlation strength'],
            ['p-value', f"{stats_dict['p_value']:.2e}", 'Statistical significance'],
            ['95% CI', f"[{stats_dict['ci_lower']:.3f}, {stats_dict['ci_upper']:.3f}]", 'Confidence interval for r'],
            ['RMSE', f"{stats_dict['rmse']:.4f}", 'Root mean square error'],
            ['MAE', f"{stats_dict['mae']:.4f}", 'Mean absolute error'],
            ['Bias', f"{stats_dict['bias']:.4f}", 'Systematic error'],
            ['Variance', f"{stats_dict['variance']:.4f}", 'Error variance']
        ]
        
        # Create table
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Header styling
        for i in range(3):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
            
        # Alternate row colors
        for i in range(1, len(table_data)):
            color = '#F2F2F2' if i % 2 == 0 else 'white'
            for j in range(3):
                table[(i, j)].set_facecolor(color)
                
        ax.set_title(f'Statistical Summary: {metric_name}', 
                    fontweight='bold', pad=20, fontsize=12)
        
    def _save_figure(self, fig: plt.Figure, save_path: str):
        """Save figure in multiple formats for publication."""
        base_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
        
        # Save as PDF (vector format for publications)
        pdf_path = f"{base_path}.pdf"
        fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        
        # Save as PNG (for presentations/web)
        png_path = f"{base_path}.png"
        fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        
        # Save as SVG (scalable vector graphics)
        svg_path = f"{base_path}.svg"
        fig.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        
        print(f"\n📄 Publication-quality figures saved:")
        print(f"   PDF: {pdf_path}")
        print(f"   PNG: {png_path}")
        print(f"   SVG: {svg_path}")
        
    def create_expressibility_comparison_figure(self, 
                                              results_dict: Dict[str, List[ValidationResult]],
                                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comparative analysis figure for multiple ansatz types.
        
        Args:
            results_dict: Dictionary mapping ansatz names to validation results
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Ansatz Expressibility Comparison', fontsize=16, fontweight='bold')
        
        ansatz_names = list(results_dict.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(ansatz_names)))
        
        # Plot 1: Expressibility values comparison
        ax1 = axes[0, 0]
        for i, (name, results) in enumerate(results_dict.items()):
            exact_vals, measured_vals = self._prepare_data(results)
            ax1.scatter(exact_vals, measured_vals, c=[colors[i]], 
                       label=name, s=50, alpha=0.7)
        
        # Perfect agreement line
        all_exact = np.concatenate([self._prepare_data(results)[0] for results in results_dict.values()])
        all_measured = np.concatenate([self._prepare_data(results)[1] for results in results_dict.values()])
        min_val, max_val = min(all_exact.min(), all_measured.min()), max(all_exact.max(), all_measured.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Agreement')
        
        ax1.set_xlabel('Theoretical Expressibility')
        ax1.set_ylabel('Measured Expressibility')
        ax1.set_title('Correlation Analysis')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Error distributions
        ax2 = axes[0, 1]
        for i, (name, results) in enumerate(results_dict.items()):
            exact_vals, measured_vals = self._prepare_data(results)
            errors = measured_vals - exact_vals
            ax2.hist(errors, bins=15, alpha=0.6, label=name, color=colors[i], density=True)
        
        ax2.set_xlabel('Error (Measured - Theoretical)')
        ax2.set_ylabel('Density')
        ax2.set_title('Error Distribution Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: RMSE comparison
        ax3 = axes[1, 0]
        rmse_values = []
        for name, results in results_dict.items():
            exact_vals, measured_vals = self._prepare_data(results)
            rmse = np.sqrt(np.mean((measured_vals - exact_vals)**2))
            rmse_values.append(rmse)
        
        bars = ax3.bar(ansatz_names, rmse_values, color=colors, alpha=0.7)
        ax3.set_ylabel('RMSE')
        ax3.set_title('Root Mean Square Error')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, rmse in zip(bars, rmse_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{rmse:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Correlation coefficients
        ax4 = axes[1, 1]
        r_values = []
        for name, results in results_dict.items():
            exact_vals, measured_vals = self._prepare_data(results)
            if np.std(exact_vals) > 0 and np.std(measured_vals) > 0:
                r_value, _ = stats.pearsonr(exact_vals, measured_vals)
            else:
                r_value = 0.0
            r_values.append(r_value)
        
        bars = ax4.bar(ansatz_names, r_values, color=colors, alpha=0.7)
        ax4.set_ylabel('Pearson Correlation (r)')
        ax4.set_title('Correlation Strength')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, r_val in zip(bars, r_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{r_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
            
        return fig


# Convenience functions for easy use
def create_publication_validation_plot(results: List[ValidationResult], 
                                     metric_name: str = "Quantum Metric",
                                     style: str = "nature",
                                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a publication-quality validation plot.
    
    Args:
        results: Validation results
        metric_name: Name of the quantum metric
        style: Publication style ('nature', 'ieee', 'prl')
        save_path: Path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    visualizer = PublicationVisualizer(style=style)
    return visualizer.create_statistical_validation_figure(results, metric_name, save_path)


def create_ansatz_comparison_plot(results_dict: Dict[str, List[ValidationResult]],
                                style: str = "nature", 
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a comparative analysis plot for multiple ansatz types.
    
    Args:
        results_dict: Dictionary mapping ansatz names to validation results
        style: Publication style ('nature', 'ieee', 'prl')
        save_path: Path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    visualizer = PublicationVisualizer(style=style)
    return visualizer.create_expressibility_comparison_figure(results_dict, save_path)


if __name__ == "__main__":
    # Example usage
    print("Publication Visualizer Module")
    print("Available styles: nature, ieee, prl")
    print("Use create_publication_validation_plot() for single metric validation")
    print("Use create_ansatz_comparison_plot() for comparative analysis")
