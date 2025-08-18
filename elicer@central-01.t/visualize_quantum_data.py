"""
Comprehensive Quantum Circuit Data Visualization
Academic-quality plots for merged quantum circuit execution data analysis
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set academic style
plt.style.use('seaborn-v0_8-whitegrid')
# Custom color palette: blue and red
custom_colors = ['#1f77b4', '#d62728']  # Blue and red
sns.set_palette(custom_colors)

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'text.usetex': False,  # Set to True if LaTeX is available
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

class QuantumDataVisualizer:
    def __init__(self, data_path, output_dir="quantum_analysis_plots"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load and process data
        self.load_data()
        self.process_data()
        
    def load_data(self):
        """Load merged quantum circuit data"""
        with open(self.data_path, 'r') as f:
            self.raw_data = json.load(f)
        
        # Extract results list - handle both 'results' and 'merged_results' keys
        self.results = self.raw_data.get('merged_results', self.raw_data.get('results', []))
        print(f"Loaded {len(self.results)} quantum circuit results")
        
    def process_data(self):
        """Process raw data into structured format"""
        processed_data = []
        
        for result in self.results:
            # Determine data source type
            circuit_id = result.get('circuit_id', '')
            if circuit_id.startswith('simulator_data_set'):
                source_type = 'Simulator'
            elif circuit_id.startswith('scalability_test'):
                source_type = 'Hardware'
            else:
                source_type = 'Unknown'
            
            # Get KL divergence from nested expressibility object
            expressibility_data = result.get('expressibility', {})
            kl_div = expressibility_data.get('kl_divergence', 0)
            kl_div_reverse = expressibility_data.get('kl_divergence_reverse', 0)
            js_div = expressibility_data.get('js_divergence', 0)
            l2_norm = expressibility_data.get('l2_norm', 0)
            valid_samples = expressibility_data.get('valid_samples', 0)
            
            processed_data.append({
                'circuit_id': circuit_id,
                'source_type': source_type,
                'num_qubits': result.get('num_qubits', 0),
                'depth': result.get('depth', 0),
                'fidelity': result.get('fidelity', 0),
                'robust_fidelity': result.get('robust_fidelity', 0),
                'expressibility': kl_div,  # Use KL divergence as expresss
                'kl_divergence': kl_div,
                'kl_divergence_reverse': kl_div_reverse,
                'js_divergence': js_div,
                'l2_norm': l2_norm,
                'valid_samples': valid_samples,
                'entanglement': result.get('entanglement', 0),
                'timestamp': result.get('timestamp', '')
            })
        
        self.df = pd.DataFrame(processed_data)
        
        # Handle filtering with safeguards to prevent all data from being filtered out
        if len(self.df) > 0:
            initial_count = len(self.df)
            
            # Filter out circuits with invalid values
            valid_mask = (
                (self.df['kl_divergence'] != 0) | 
                (self.df['entanglement'] != 0)
            )
            
            # Only apply the filter if it won't remove all data
            filtered_df = self.df[valid_mask]
            
            if len(filtered_df) > 0:
                # We have some data after filtering
                self.df = filtered_df
                filtered_count = initial_count - len(self.df)
                if filtered_count > 0:
                    print(f"Filtered out {filtered_count} invalid circuits where both KL=0 AND entanglement=0")
            else:
                # Filter would remove all data, keep original data
                print("Warning: Filtering criteria would remove all data. Using all circuits without filtering.")
                
            print(f"Valid circuits remaining: {len(self.df)}")
            
            # Additional diagnostics to understand the data
            kl_zero_count = len(self.df[self.df['kl_divergence'] == 0])
            ent_zero_count = len(self.df[self.df['entanglement'] == 0])
            print(f"Circuits with KL=0: {kl_zero_count}")
            print(f"Circuits with entanglement=0: {ent_zero_count}")
        
        print(f"Processed {len(self.df)} valid circuit results")
        print(f"Source distribution: {self.df['source_type'].value_counts().to_dict()}")
        
    def save_plot(self, fig, filename, tight_layout=True):
        """Save plot with consistent formatting"""
        if tight_layout:
            fig.tight_layout()
        filepath = self.output_dir / f"{filename}.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {filepath}")
        
    def plot_qubit_scaling_analysis(self):
        """Analyze metrics vs number of qubits - individual plots"""
        metrics = [
            ('fidelity', 'Fidelity'),
            ('kl_divergence', 'Expressibility'),
            ('entanglement', 'Entanglement'),
            ('robust_fidelity', 'Robust Fidelity')
        ]
        
        for idx, (metric, title) in enumerate(metrics):
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Plot by source type
            for source in self.df['source_type'].unique():
                if source == 'Unknown':
                    continue
                data = self.df[self.df['source_type'] == source]
                if len(data) > 0:
                    ax.scatter(data['num_qubits'], data[metric], 
                             alpha=0.6, label=source, s=50)
                    
                    # Add average line with triangle markers
                    if len(data) > 5:  # Only if enough data points
                        qubit_groups = data.groupby('num_qubits')[metric].mean()
                        ax.plot(qubit_groups.index, qubit_groups.values, 
                               marker='^', linestyle='-', linewidth=2, markersize=8,
                               alpha=0.8, label=f'{source} Average')
            
            ax.set_xlabel('Number of Qubits')
            ax.set_ylabel(title)
            ax.set_title(f'{title} vs Qubit Count')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            self.save_plot(fig, f'01_{idx+1}_{metric}_vs_qubits')
        
    def plot_depth_analysis(self):
        """Analyze circuit depth effects - individual plots"""
        
        # 1. Depth distribution
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for source in self.df['source_type'].unique():
            if source == 'Unknown':
                continue
            data = self.df[self.df['source_type'] == source]
            ax.hist(data['depth'], alpha=0.6, label=source, bins=20)
        ax.set_xlabel('Circuit Depth')
        ax.set_ylabel('Frequency')
        ax.set_title('Circuit Depth Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.save_plot(fig, '02_1_depth_distribution')
        
        # 2. Depth vs Expressibility
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for source in self.df['source_type'].unique():
            if source == 'Unknown':
                continue
            data = self.df[self.df['source_type'] == source]
            ax.scatter(data['depth'], data['expressibility'], 
                       alpha=0.6, label=source, s=30)
            
            # Add average line with triangle markers
            if len(data) > 5:
                depth_groups = data.groupby('depth')['expressibility'].mean()
                ax.plot(depth_groups.index, depth_groups.values, 
                        marker='^', linestyle='-', linewidth=2, markersize=8,
                        alpha=0.8, label=f'{source} Average')
        ax.set_xlabel('Circuit Depth')
        ax.set_ylabel('Expressibility')
        ax.set_title('Expressibility vs Circuit Depth')
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.save_plot(fig, '02_2_expressibility_vs_depth')
        
        # 3. Depth vs Entanglement
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for source in self.df['source_type'].unique():
            if source == 'Unknown':
                continue
            data = self.df[self.df['source_type'] == source]
            ax.scatter(data['depth'], data['entanglement'], 
                       alpha=0.6, label=source, s=30)
            
            # Add average line with triangle markers
            if len(data) > 5:
                depth_groups = data.groupby('depth')['entanglement'].mean()
                ax.plot(depth_groups.index, depth_groups.values, 
                        marker='^', linestyle='-', linewidth=2, markersize=8,
                        alpha=0.8, label=f'{source} Average')
        ax.set_xlabel('Circuit Depth')
        ax.set_ylabel('Entanglement')
        ax.set_title('Entanglement vs Circuit Depth')
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.save_plot(fig, '02_3_entanglement_vs_depth')
        
        # 4. Depth vs Fidelity
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for source in self.df['source_type'].unique():
            if source == 'Unknown':
                continue
            data = self.df[self.df['source_type'] == source]
            ax.scatter(data['depth'], data['fidelity'], 
                       alpha=0.6, label=source, s=30)
            
            # Add average line with triangle markers
            if len(data) > 5:
                depth_groups = data.groupby('depth')['fidelity'].mean()
                ax.plot(depth_groups.index, depth_groups.values, 
                        marker='^', linestyle='-', linewidth=2, markersize=8,
                        alpha=0.8, label=f'{source} Average')
        ax.set_xlabel('Circuit Depth')
        ax.set_ylabel('Fidelity')
        ax.set_title('Fidelity vs Circuit Depth')
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.save_plot(fig, '02_4_fidelity_vs_depth')
        
    def plot_correlation_matrix(self):
        """Plot correlation matrix of quantum metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Quantum Metrics Correlation Analysis', fontsize=16, y=1.02)
        
        # Define the metrics to include in correlation analysis
        # Replace 'expressibility' with 'kl_divergence' which we're using as expressibility
        metric_cols = ['num_qubits', 'depth', 'fidelity', 'robust_fidelity', 
                      'entanglement', 'kl_divergence', 'js_divergence', 'l2_norm']
        
        # Create a copy of dataframe with renamed columns for clearer visualization
        corr_df = self.df.copy()
        
        # Simulator data correlation
        sim_data = corr_df[corr_df['source_type'] == 'Simulator'][metric_cols]
        if len(sim_data) > 0:
            # Rename kl_divergence to Expressibility for visualization
            sim_data = sim_data.rename(columns={'kl_divergence': 'Expressibility'})
            metric_cols_display = [col if col != 'kl_divergence' else 'Expressibility' for col in metric_cols]
            
            # Drop columns with all zeros or NaNs to avoid correlation issues
            sim_data = sim_data.loc[:, (sim_data != 0).any(axis=0)]
            sim_data = sim_data.dropna(axis=1, how='all')
            
            if not sim_data.empty and len(sim_data.columns) > 1:
                corr_sim = sim_data.corr()
                sns.heatmap(corr_sim, annot=True, cmap='RdBu_r', center=0, 
                          ax=axes[0], fmt='.2f', square=True)
                axes[0].set_title('Simulator Data Correlations')
            else:
                axes[0].text(0.5, 0.5, "Insufficient data for correlation analysis", 
                            ha='center', va='center', fontsize=12)
                axes[0].set_xticks([])
                axes[0].set_yticks([])
        else:
            axes[0].text(0.5, 0.5, "No simulator data available", 
                        ha='center', va='center', fontsize=12)
            axes[0].set_xticks([])
            axes[0].set_yticks([])
        
        # Hardware data correlation
        hw_data = corr_df[corr_df['source_type'] == 'Hardware'][metric_cols]
        if len(hw_data) > 0:
            # Rename kl_divergence to Expressibility for visualization
            hw_data = hw_data.rename(columns={'kl_divergence': 'Expressibility'})
            
            # Drop columns with all zeros or NaNs to avoid correlation issues
            hw_data = hw_data.loc[:, (hw_data != 0).any(axis=0)]
            hw_data = hw_data.dropna(axis=1, how='all')
            
            if not hw_data.empty and len(hw_data.columns) > 1:
                corr_hw = hw_data.corr()
                sns.heatmap(corr_hw, annot=True, cmap='RdBu_r', center=0, 
                          ax=axes[1], fmt='.2f', square=True)
                axes[1].set_title('Hardware Data Correlations')
            else:
                axes[1].text(0.5, 0.5, "Insufficient data for correlation analysis", 
                            ha='center', va='center', fontsize=12)
                axes[1].set_xticks([])
                axes[1].set_yticks([])
        else:
            axes[1].text(0.5, 0.5, "No hardware data available", 
                        ha='center', va='center', fontsize=12)
            axes[1].set_xticks([])
            axes[1].set_yticks([])
        
        self.save_plot(fig, '03_correlation_matrix')
        
    def plot_expressibility_analysis(self):
        """Detailed expressibility analysis using KL divergence as expressibility metric"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Expressibility Analysis', fontsize=16, y=0.98)
        
        # Expressibility distribution (using KL divergence)
        ax1 = axes[0, 0]
        for source in self.df['source_type'].unique():
            if source == 'Unknown':
                continue
            data = self.df[self.df['source_type'] == source]
            # Filter out zero values for better visualization
            valid_data = data[data['kl_divergence'] > 0]
            if len(valid_data) > 0:
                ax1.hist(valid_data['kl_divergence'], alpha=0.6, label=source, bins=20)
        ax1.set_xlabel('Expressibility (KL Divergence)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Expressibility Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # JS Divergence vs Expressibility (KL Divergence)
        ax2 = axes[0, 1]
        for source in self.df['source_type'].unique():
            if source == 'Unknown':
                continue
            data = self.df[self.df['source_type'] == source]
            # Filter out cases where both values are zero
            valid_data = data[(data['js_divergence'] > 0) | (data['kl_divergence'] > 0)]
            if len(valid_data) > 0:
                ax2.scatter(valid_data['js_divergence'], valid_data['kl_divergence'], 
                          alpha=0.6, label=source, s=30)
        ax2.set_xlabel('JS Divergence')
        ax2.set_ylabel('Expressibility (KL Divergence)')
        ax2.set_title('JS Divergence vs Expressibility')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Expressibility (KL Divergence) vs Entanglement
        ax3 = axes[1, 0]
        for source in self.df['source_type'].unique():
            if source == 'Unknown':
                continue
            data = self.df[self.df['source_type'] == source]
            # Filter out cases where both values are zero
            valid_data = data[(data['kl_divergence'] > 0) | (data['entanglement'] > 0)]
            if len(valid_data) > 0:
                ax3.scatter(valid_data['kl_divergence'], valid_data['entanglement'], 
                          alpha=0.6, label=source, s=30)
        ax3.set_xlabel('Expressibility (KL Divergence)')
        ax3.set_ylabel('Entanglement')
        ax3.set_title('Expressibility vs Entanglement')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # L2 Norm vs Expressibility (KL Divergence)
        ax4 = axes[1, 1]
        for source in self.df['source_type'].unique():
            if source == 'Unknown':
                continue
            data = self.df[self.df['source_type'] == source]
            # Filter out cases where both values are zero
            valid_data = data[(data['l2_norm'] > 0) | (data['kl_divergence'] > 0)]
            if len(valid_data) > 0:
                ax4.scatter(valid_data['l2_norm'], valid_data['kl_divergence'], 
                          alpha=0.6, label=source, s=30)
        ax4.set_xlabel('L2 Norm')
        ax4.set_ylabel('Expressibility (KL Divergence)')
        ax4.set_title('L2 Norm vs Expressibility')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        self.save_plot(fig, '04_expressibility_analysis')
        
    def plot_fidelity_analysis(self):
        """Fidelity analysis comparing simulator vs hardware"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Fidelity Analysis: Simulator vs Hardware', fontsize=16, y=0.98)
        
        # Fidelity distribution
        ax1 = axes[0, 0]
        for source in self.df['source_type'].unique():
            if source == 'Unknown':
                continue
            data = self.df[self.df['source_type'] == source]
            ax1.hist(data['fidelity'], alpha=0.6, label=source, bins=20)
        ax1.set_xlabel('Fidelity')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Fidelity Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Robust Fidelity vs Fidelity
        ax2 = axes[0, 1]
        for source in self.df['source_type'].unique():
            if source == 'Unknown':
                continue
            data = self.df[self.df['source_type'] == source]
            ax2.scatter(data['fidelity'], data['robust_fidelity'], 
                       alpha=0.6, label=source, s=30)
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect correlation')
        ax2.set_xlabel('Fidelity')
        ax2.set_ylabel('Robust Fidelity')
        ax2.set_title('Robust Fidelity vs Fidelity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Fidelity vs Number of Qubits (with depth as colormap)
        ax3 = axes[1, 0]
        for source in self.df['source_type'].unique():
            if source == 'Unknown':
                continue
            data = self.df[self.df['source_type'] == source]
            if len(data) > 0:
                scatter = ax3.scatter(data['num_qubits'], data['fidelity'], 
                                    c=data['depth'], alpha=0.7, s=50, 
                                    label=source, cmap='viridis')
        ax3.set_xlabel('Number of Qubits')
        ax3.set_ylabel('Fidelity')
        ax3.set_title('Fidelity vs Qubit Count (Depth as Color)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar for depth
        if len(self.df) > 0:
            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('Circuit Depth')
        
        # Fidelity degradation with depth
        ax4 = axes[1, 1]
        # Calculate average fidelity by depth bins
        for source in self.df['source_type'].unique():
            if source == 'Unknown':
                continue
            data = self.df[self.df['source_type'] == source]
            if len(data) > 10:  # Only if enough data points
                depth_bins = pd.cut(data['depth'], bins=10)
                fidelity_mean = data.groupby(depth_bins)['fidelity'].mean()
                depth_centers = [interval.mid for interval in fidelity_mean.index]
                ax4.plot(depth_centers, fidelity_mean.values, 
                        marker='o', label=source, linewidth=2)
        ax4.set_xlabel('Circuit Depth (binned)')
        ax4.set_ylabel('Average Fidelity')
        ax4.set_title('Fidelity Degradation vs Depth')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        self.save_plot(fig, '05_fidelity_analysis')
        
    def plot_entanglement_analysis(self):
        """Entanglement analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Entanglement Analysis', fontsize=16, y=0.98)
        
        # Entanglement distribution
        ax1 = axes[0, 0]
        for source in self.df['source_type'].unique():
            if source == 'Unknown':
                continue
            data = self.df[self.df['source_type'] == source]
            ax1.hist(data['entanglement'], alpha=0.6, label=source, bins=20)
        ax1.set_xlabel('Entanglement')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Entanglement Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Entanglement vs Qubits
        ax2 = axes[0, 1]
        for source in self.df['source_type'].unique():
            if source == 'Unknown':
                continue
            data = self.df[self.df['source_type'] == source]
            ax2.scatter(data['num_qubits'], data['entanglement'], 
                       alpha=0.6, label=source, s=30)
        ax2.set_xlabel('Number of Qubits')
        ax2.set_ylabel('Entanglement')
        ax2.set_title('Entanglement vs Qubit Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Entanglement vs Depth
        ax3 = axes[1, 0]
        for source in self.df['source_type'].unique():
            if source == 'Unknown':
                continue
            data = self.df[self.df['source_type'] == source]
            ax3.scatter(data['depth'], data['entanglement'], 
                       alpha=0.6, label=source, s=30)
        ax3.set_xlabel('Circuit Depth')
        ax3.set_ylabel('Entanglement')
        ax3.set_title('Entanglement vs Circuit Depth')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Entanglement efficiency (entanglement per qubit)
        ax4 = axes[1, 1]
        self.df['entanglement_per_qubit'] = self.df['entanglement'] / self.df['num_qubits']
        for source in self.df['source_type'].unique():
            if source == 'Unknown':
                continue
            data = self.df[self.df['source_type'] == source]
            ax4.scatter(data['num_qubits'], data['entanglement_per_qubit'], 
                       alpha=0.6, label=source, s=30)
        ax4.set_xlabel('Number of Qubits')
        ax4.set_ylabel('Entanglement per Qubit')
        ax4.set_title('Entanglement Efficiency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        self.save_plot(fig, '06_entanglement_analysis')
        
    def plot_hardware_vs_simulator(self):
        """Direct comparison between hardware and simulator results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Hardware vs Simulator Comparison', fontsize=16, y=0.98)
        
        # Box plots for key metrics
        metrics = ['fidelity', 'kl_divergence', 'entanglement', 'robust_fidelity']
        metric_labels = ['Fidelity', 'KL Divergence', 'Entanglement', 'Robust Fidelity']
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx // 2, idx % 2]
            
            # Prepare data for box plot
            plot_data = []
            labels = []
            for source in ['Simulator', 'Hardware']:
                data = self.df[self.df['source_type'] == source][metric]
                if len(data) > 0:
                    plot_data.append(data)
                    labels.append(source)
            
            if len(plot_data) > 0:
                bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
                colors = ['#1f77b4', '#d62728']  # Blue and red
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
            
            ax.set_ylabel(label)
            ax.set_title(f'{label} Comparison')
            ax.grid(True, alpha=0.3)
        
        self.save_plot(fig, '07_hardware_vs_simulator')
        
    def plot_divergence_relationships(self):
        """Analyze relationships between KL, JS, and L2 norm divergences"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Divergence Metrics Relationships', fontsize=16, y=0.98)
        
        # Filter data with valid divergence values - loosen criteria to show more data
        # Just check if any of the values is greater than 0, not all of them
        valid_data = self.df[(self.df['kl_divergence'] > 0) | (self.df['js_divergence'] > 0) | (self.df['l2_norm'] > 0)]
        
        if len(valid_data) > 0:
            # KL vs JS Divergence
            ax1 = axes[0, 0]
            for source in valid_data['source_type'].unique():
                if source == 'Unknown':
                    continue
                data = valid_data[valid_data['source_type'] == source]
                if len(data) > 0:
                    ax1.scatter(data['kl_divergence'], data['js_divergence'], 
                              alpha=0.6, label=source, s=30)
            ax1.set_xlabel('KL Divergence')
            ax1.set_ylabel('JS Divergence')
            ax1.set_title('KL vs JS Divergence')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # KL vs L2 Norm
            ax2 = axes[0, 1]
            for source in valid_data['source_type'].unique():
                if source == 'Unknown':
                    continue
                data = valid_data[valid_data['source_type'] == source]
                if len(data) > 0:
                    ax2.scatter(data['kl_divergence'], data['l2_norm'], 
                              alpha=0.6, label=source, s=30)
            ax2.set_xlabel('KL Divergence')
            ax2.set_ylabel('L2 Norm')
            ax2.set_title('KL Divergence vs L2 Norm')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # JS vs L2 Norm
            ax3 = axes[1, 0]
            for source in valid_data['source_type'].unique():
                if source == 'Unknown':
                    continue
                data = valid_data[valid_data['source_type'] == source]
                if len(data) > 0:
                    ax3.scatter(data['js_divergence'], data['l2_norm'], 
                              alpha=0.6, label=source, s=30)
            ax3.set_xlabel('JS Divergence')
            ax3.set_ylabel('L2 Norm')
            ax3.set_title('JS Divergence vs L2 Norm')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 3D relationship (projected to 2D with color coding)
            ax4 = axes[1, 1]
            for source in valid_data['source_type'].unique():
                if source == 'Unknown':
                    continue
                data = valid_data[valid_data['source_type'] == source]
                if len(data) > 0:
                    scatter = ax4.scatter(data['kl_divergence'], data['js_divergence'], 
                                        c=data['l2_norm'], alpha=0.6, label=source, s=30)
            ax4.set_xlabel('KL Divergence')
            ax4.set_ylabel('JS Divergence')
            ax4.set_title('KL vs JS (L2 Norm as Color)')
            plt.colorbar(scatter, ax=ax4, label='L2 Norm')
            ax4.grid(True, alpha=0.3)
        else:
            # Handle case when there's no valid data
            for ax in axes.flatten():
                ax.text(0.5, 0.5, "No valid divergence data available", 
                        ha='center', va='center', fontsize=12)
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xticks([])
                ax.set_yticks([])
        
        self.save_plot(fig, '08_divergence_relationships')
        
    def plot_normalized_distributions(self):
        """Plot normalized distributions for better comparison between hardware and simulator"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Normalized Metric Distributions', fontsize=16, y=1.02)
        
        metrics = ['kl_divergence', 'entanglement', 'robust_fidelity']
        metric_labels = ['KL Divergence', 'Entanglement', 'Robust Fidelity']
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx]
            
            # Plot normalized histograms
            for source in ['Simulator', 'Hardware']:
                data = self.df[self.df['source_type'] == source][metric]
                if len(data) > 0:
                    ax.hist(data, alpha=0.6, label=source, bins=20, density=True)
            
            ax.set_xlabel(label)
            ax.set_ylabel('Normalized Frequency')
            ax.set_title(f'{label} Distribution (Normalized)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            for source in ['Simulator', 'Hardware']:
                data = self.df[self.df['source_type'] == source][metric]
                if len(data) > 0:
                    mean_val = data.mean()
                    std_val = data.std()
                    ax.axvline(mean_val, linestyle='--', alpha=0.7, 
                              label=f'{source} μ={mean_val:.3f}')
        
        self.save_plot(fig, '09_normalized_distributions')
        
    def plot_summary_statistics(self):
        """Summary statistics and overview"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Dataset Summary Statistics', fontsize=16, y=0.98)
        
        # Data source distribution
        ax1 = axes[0, 0]
        source_counts = self.df['source_type'].value_counts()
        colors = ['#1f77b4', '#d62728', '#2ca02c']  # Blue, red, green
        ax1.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%',
                colors=colors[:len(source_counts)])
        ax1.set_title('Data Source Distribution')
        
        # Qubit distribution
        ax2 = axes[0, 1]
        ax2.hist(self.df['num_qubits'], bins=20, alpha=0.7, color='#1f77b4', edgecolor='black')
        ax2.set_xlabel('Number of Qubits')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Qubit Count Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Metric ranges
        ax3 = axes[1, 0]
        metrics = ['fidelity', 'kl_divergence', 'entanglement']
        metric_labels = ['Fidelity', 'KL Divergence', 'Entanglement']
        means = [self.df[metric].mean() for metric in metrics]
        stds = [self.df[metric].std() for metric in metrics]
        
        x_pos = np.arange(len(metrics))
        ax3.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='#1f77b4')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(metric_labels)
        ax3.set_ylabel('Value')
        ax3.set_title('Mean Metric Values (±1σ)')
        ax3.grid(True, alpha=0.3)
        
        # Valid samples distribution (for expressibility)
        ax4 = axes[1, 1]
        valid_samples = self.df[self.df['valid_samples'] > 0]['valid_samples']
        if len(valid_samples) > 0:
            ax4.hist(valid_samples, bins=20, alpha=0.7, color='#d62728', edgecolor='black')
            ax4.set_xlabel('Valid Samples')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Valid Samples Distribution')
            ax4.grid(True, alpha=0.3)
        
        self.save_plot(fig, '10_summary_statistics')
        
    def plot_average_metrics(self):
        """Separate visualization focusing only on average metric values using grayscale and distinct markers"""
        metrics = ['fidelity', 'kl_divergence', 'entanglement', 'robust_fidelity']
        metric_labels = ['Fidelity', 'Expressibility', 'Entanglement', 'Robust Fidelity']
        
        # Define marker styles and colors for different sources
        markers = {'Simulator': '^', 'Hardware': 'x'}
        colors = {'Simulator': 'black', 'Hardware': 'dimgray'}
        line_styles = {'Simulator': '-', 'Hardware': '--'}
        
        # 1. Average metrics vs number of qubits - combined plots for simulator and hardware
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            plt.figure(figsize=(10, 6))
            
            # Plot each source type
            for source in self.df['source_type'].unique():
                if source == 'Unknown':
                    continue
                
                data = self.df[self.df['source_type'] == source]
                if len(data) > 0:  # Show even with few data points
                    # Group by qubit count and calculate means
                    grouped_data = data.groupby('num_qubits')[metric].mean().reset_index()
                    
                    plt.plot(grouped_data['num_qubits'], grouped_data[metric], 
                            marker=markers.get(source, 'o'), 
                            linestyle=line_styles.get(source, '-'),
                            color=colors.get(source, 'black'),
                            linewidth=2, markersize=10,
                            label=f'{source}')
            
            plt.xlabel('Number of Qubits')
            plt.ylabel(label)
            plt.title(f'Average {label} vs Qubit Count')
            plt.grid(True, alpha=0.3)
            plt.legend()
            self.save_plot(plt.gcf(), f'11_{i+1}_avg_{metric}_vs_qubits')
        
        # 2. Average metrics vs circuit depth - combined plots for simulator and hardware
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            plt.figure(figsize=(10, 6))
            
            # Plot each source type
            for source in self.df['source_type'].unique():
                if source == 'Unknown':
                    continue
                
                data = self.df[self.df['source_type'] == source]
                if len(data) > 0:  # Show even with few data points
                    # Group by depth and calculate means
                    grouped_data = data.groupby('depth')[metric].mean().reset_index()
                    
                    plt.plot(grouped_data['depth'], grouped_data[metric], 
                            marker=markers.get(source, 'o'), 
                            linestyle=line_styles.get(source, '-'),
                            color=colors.get(source, 'black'),
                            linewidth=2, markersize=10,
                            label=f'{source}')
            
            plt.xlabel('Circuit Depth')
            plt.ylabel(label)
            plt.title(f'Average {label} vs Circuit Depth')
            plt.grid(True, alpha=0.3)
            plt.legend()
            self.save_plot(plt.gcf(), f'12_{i+1}_avg_{metric}_vs_depth')
    
    def plot_metrics_by_qubit_with_depth_colormap(self):
        """Create visualizations with qubits on x-axis, metrics on y-axis, and depth as colormap"""
        # Define the metrics we want to visualize
        metrics = ['fidelity', 'kl_divergence', 'entanglement']
        metric_labels = ['Fidelity', 'Expressibility', 'Entanglement']
        
        # Create a separate plot for each metric
        for metric, label in zip(metrics, metric_labels):
            plt.figure(figsize=(12, 8))
            
            # Create separate plots for each source type (Simulator vs Hardware)
            for source in self.df['source_type'].unique():
                if source == 'Unknown':
                    continue
                    
                data = self.df[self.df['source_type'] == source]
                if len(data) > 0:
                    # Create scatter plot with qubits on x-axis, metric on y-axis, and depth as color
                    scatter = plt.scatter(data['num_qubits'], data[metric], 
                                      c=data['depth'], cmap='viridis', 
                                      s=80, alpha=0.7, 
                                      label=source, edgecolors='black')
            
            plt.colorbar(label='Circuit Depth')
            plt.xlabel('Number of Qubits')
            plt.ylabel(label)
            plt.title(f'{label} vs Number of Qubits (Depth as Color)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save the plot
            self.save_plot(plt.gcf(), f'13_{metrics.index(metric)+1}_{metric}_vs_qubits_depth_colormap')
    
    def generate_all_plots(self):
        """Generate all visualization plots"""
        print("Generating comprehensive quantum circuit data visualizations...")
        print(f"Output directory: {self.output_dir.absolute()}")
        
        self.plot_qubit_scaling_analysis()
        self.plot_depth_analysis()
        self.plot_correlation_matrix()
        self.plot_expressibility_analysis()
        self.plot_fidelity_analysis()
        self.plot_entanglement_analysis()
        self.plot_hardware_vs_simulator()
        self.plot_divergence_relationships()
        self.plot_normalized_distributions()
        self.plot_summary_statistics()
        self.plot_average_metrics()
        self.plot_metrics_by_qubit_with_depth_colormap()
        
        # Generate summary report
        self.generate_summary_report()
        
        # Generate comprehensive statistics file
        self.generate_statistics_file()
        
        print(f"\nVisualization complete! Generated {len(list(self.output_dir.glob('*.png')))} plots.")
        
    def generate_summary_report(self):
        """Generate a summary report of the analysis"""
        report = []
        report.append("# Quantum Circuit Data Analysis Summary\n")
        
        # Dataset overview
        report.append("## Dataset Overview")
        report.append(f"- Total circuits analyzed: {len(self.df)}")
        report.append(f"- Data sources: {', '.join(self.df['source_type'].unique())}")
        report.append(f"- Qubit range: {self.df['num_qubits'].min()} - {self.df['num_qubits'].max()}")
        report.append(f"- Depth range: {self.df['depth'].min()} - {self.df['depth'].max()}")
        report.append("")
        
        # Source breakdown
        report.append("## Source Breakdown")
        for source in self.df['source_type'].unique():
            count = len(self.df[self.df['source_type'] == source])
            report.append(f"- {source}: {count} circuits ({count/len(self.df)*100:.1f}%)")
        report.append("")
        
        # Metric statistics
        report.append("## Metric Statistics")
        metrics = ['fidelity', 'expressibility', 'entanglement', 'kl_divergence']
        for metric in metrics:
            mean_val = self.df[metric].mean()
            std_val = self.df[metric].std()
            min_val = self.df[metric].min()
            max_val = self.df[metric].max()
            report.append(f"- {metric.title()}: μ={mean_val:.3f}, σ={std_val:.3f}, range=[{min_val:.3f}, {max_val:.3f}]")
        
        # Save report
        report_path = self.output_dir / "analysis_summary.md"
        with open(self.output_dir / "analysis_summary.md", 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Summary report saved to: {self.output_dir / 'analysis_summary.md'}")
    
    def generate_statistics_file(self):
        """Generate comprehensive statistics summary text file"""
        stats_lines = []
        stats_lines.append("=" * 80)
        stats_lines.append("QUANTUM CIRCUIT DATA ANALYSIS - COMPREHENSIVE STATISTICS")
        stats_lines.append("=" * 80)
        stats_lines.append("")
        
        # Dataset Overview
        stats_lines.append("DATASET OVERVIEW")
        stats_lines.append("-" * 40)
        stats_lines.append(f"Total circuits analyzed: {len(self.df)}")
        
        # Source type breakdown
        source_counts = self.df['source_type'].value_counts()
        for source, count in source_counts.items():
            percentage = (count / len(self.df)) * 100
            stats_lines.append(f"  - {source}: {count} circuits ({percentage:.1f}%)")
        
        # Qubit range
        if len(self.df) > 0:
            min_qubits = self.df['num_qubits'].min()
            max_qubits = self.df['num_qubits'].max()
            stats_lines.append(f"Qubit range: {min_qubits} - {max_qubits}")
            
            # Depth range
            min_depth = self.df['depth'].min()
            max_depth = self.df['depth'].max()
            stats_lines.append(f"Depth range: {min_depth} - {max_depth}")
        
        stats_lines.append("")
        
        # Detailed Statistics by Source Type
        for source in ['Simulator', 'Hardware']:
            source_data = self.df[self.df['source_type'] == source]
            if len(source_data) == 0:
                continue
                
            stats_lines.append(f"{source.upper()} DATA STATISTICS")
            stats_lines.append("-" * 40)
            stats_lines.append(f"Number of circuits: {len(source_data)}")
            
            # Metrics statistics
            metrics = {
                'fidelity': 'Fidelity',
                'robust_fidelity': 'Robust Fidelity', 
                'kl_divergence': 'KL Divergence (Expressibility)',
                'entanglement': 'Entanglement',
                'js_divergence': 'JS Divergence',
                'l2_norm': 'L2 Norm'
            }
            
            for metric, label in metrics.items():
                if metric in source_data.columns:
                    data = source_data[metric]
                    valid_data = data[data != 0]  # Exclude zero values
                    
                    if len(valid_data) > 0:
                        stats_lines.append(f"\n{label}:")
                        stats_lines.append(f"  Mean: {valid_data.mean():.6f}")
                        stats_lines.append(f"  Std:  {valid_data.std():.6f}")
                        stats_lines.append(f"  Min:  {valid_data.min():.6f}")
                        stats_lines.append(f"  Max:  {valid_data.max():.6f}")
                        stats_lines.append(f"  Median: {valid_data.median():.6f}")
                        stats_lines.append(f"  Valid samples: {len(valid_data)}/{len(data)}")
                        
                        # Percentiles
                        q25 = valid_data.quantile(0.25)
                        q75 = valid_data.quantile(0.75)
                        stats_lines.append(f"  Q1 (25%): {q25:.6f}")
                        stats_lines.append(f"  Q3 (75%): {q75:.6f}")
                        stats_lines.append(f"  IQR: {q75 - q25:.6f}")
                    else:
                        stats_lines.append(f"\n{label}: No valid data (all zeros)")
            
            # Circuit complexity statistics
            stats_lines.append(f"\nCircuit Complexity:")
            stats_lines.append(f"  Qubits - Mean: {source_data['num_qubits'].mean():.2f}, Range: {source_data['num_qubits'].min()}-{source_data['num_qubits'].max()}")
            stats_lines.append(f"  Depth  - Mean: {source_data['depth'].mean():.2f}, Range: {source_data['depth'].min()}-{source_data['depth'].max()}")
            
            stats_lines.append("")
        
        # Comparative Statistics
        if len(self.df[self.df['source_type'] == 'Simulator']) > 0 and len(self.df[self.df['source_type'] == 'Hardware']) > 0:
            stats_lines.append("COMPARATIVE ANALYSIS (SIMULATOR vs HARDWARE)")
            stats_lines.append("-" * 50)
            
            for metric, label in metrics.items():
                if metric in self.df.columns:
                    sim_data = self.df[self.df['source_type'] == 'Simulator'][metric]
                    hw_data = self.df[self.df['source_type'] == 'Hardware'][metric]
                    
                    sim_valid = sim_data[sim_data != 0]
                    hw_valid = hw_data[hw_data != 0]
                    
                    if len(sim_valid) > 0 and len(hw_valid) > 0:
                        sim_mean = sim_valid.mean()
                        hw_mean = hw_valid.mean()
                        difference = abs(sim_mean - hw_mean)
                        relative_diff = (difference / max(sim_mean, hw_mean)) * 100 if max(sim_mean, hw_mean) > 0 else 0
                        
                        stats_lines.append(f"\n{label}:")
                        stats_lines.append(f"  Simulator mean: {sim_mean:.6f}")
                        stats_lines.append(f"  Hardware mean:  {hw_mean:.6f}")
                        stats_lines.append(f"  Absolute diff:  {difference:.6f}")
                        stats_lines.append(f"  Relative diff:  {relative_diff:.2f}%")
                        
                        # Statistical significance (simple comparison)
                        if sim_mean > hw_mean:
                            stats_lines.append(f"  Simulator > Hardware by {((sim_mean/hw_mean - 1) * 100):.1f}%")
                        else:
                            stats_lines.append(f"  Hardware > Simulator by {((hw_mean/sim_mean - 1) * 100):.1f}%")
        
        # Correlation Analysis
        stats_lines.append("\nCORRELATION ANALYSIS")
        stats_lines.append("-" * 30)
        
        correlation_metrics = ['num_qubits', 'depth', 'fidelity', 'robust_fidelity', 'kl_divergence', 'entanglement']
        available_metrics = [m for m in correlation_metrics if m in self.df.columns]
        
        if len(available_metrics) > 1:
            corr_matrix = self.df[available_metrics].corr()
            
            stats_lines.append("Strong correlations (|r| > 0.5):")
            for i, metric1 in enumerate(available_metrics):
                for j, metric2 in enumerate(available_metrics):
                    if i < j:  # Avoid duplicates
                        corr_val = corr_matrix.loc[metric1, metric2]
                        if abs(corr_val) > 0.5:
                            stats_lines.append(f"  {metric1} ↔ {metric2}: r = {corr_val:.3f}")
        
        # Data Quality Assessment
        stats_lines.append("\nDATA QUALITY ASSESSMENT")
        stats_lines.append("-" * 35)
        
        for metric in ['fidelity', 'kl_divergence', 'entanglement']:
            if metric in self.df.columns:
                total_count = len(self.df)
                zero_count = len(self.df[self.df[metric] == 0])
                valid_count = total_count - zero_count
                
                stats_lines.append(f"{metric.replace('_', ' ').title()}:")
                stats_lines.append(f"  Valid data: {valid_count}/{total_count} ({(valid_count/total_count)*100:.1f}%)")
                stats_lines.append(f"  Zero values: {zero_count} ({(zero_count/total_count)*100:.1f}%)")
        
        # Save to file
        stats_file_path = self.output_dir / "comprehensive_statistics.txt"
        with open(stats_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(stats_lines))
        
        print(f"Comprehensive statistics saved to: {stats_file_path}")
    
    def plot_metrics_by_qubit_with_depth_colormap(self):
        """Create visualizations with qubits on x-axis, metrics on y-axis, and depth as colormap"""
        # Define the metrics we want to visualize
        metrics = ['fidelity', 'kl_divergence', 'entanglement']
        metric_labels = ['Fidelity', 'Expressibility', 'Entanglement']
            
        # Create a separate plot for each metric
        for metric, label in zip(metrics, metric_labels):
            plt.figure(figsize=(12, 8))
                
            # Create separate plots for each source type (Simulator vs Hardware)
            for source in self.df['source_type'].unique():
                if source == 'Unknown':
                    continue
                        
                data = self.df[self.df['source_type'] == source]
                if len(data) > 0:
                    # Create scatter plot with qubits on x-axis, metric on y-axis, and depth as color
                    scatter = plt.scatter(data['num_qubits'], data[metric], 
                                        c=data['depth'], cmap='viridis', 
                                        s=80, alpha=0.7, 
                                        label=source, edgecolors='black')
            
            plt.colorbar(label='Circuit Depth')
            plt.xlabel('Number of Qubits')
            plt.ylabel(label)
            plt.title(f'{label} vs Number of Qubits (Depth as Color)')
            plt.grid(True, alpha=0.3)
            plt.legend()
                
            # Save the plot
            self.save_plot(plt.gcf(), f'13_{metrics.index(metric)+1}_{metric}_vs_qubits_depth_colormap')
    
def generate_all_plots(self):
    """Generate all visualization plots"""
    print("Generating comprehensive quantum circuit data visualizations...")
    print(f"Output directory: {self.output_dir.absolute()}")
        
    self.plot_qubit_scaling_analysis()
    self.plot_depth_analysis()
    self.plot_correlation_matrix()
    self.plot_expressibility_analysis()
    self.plot_fidelity_analysis()
    self.plot_entanglement_analysis()
    self.plot_hardware_vs_simulator()
    self.plot_divergence_relationships()
    self.plot_normalized_distributions()
    self.plot_summary_statistics()
    self.plot_average_metrics()
    self.plot_metrics_by_qubit_with_depth_colormap()
        
    # Generate summary report
    self.generate_summary_report()
        
    # Generate comprehensive statistics file
    self.generate_statistics_file()
        
    print(f"\nVisualization complete! Generated {len(list(self.output_dir.glob('*.png')))} plots.")
        
def generate_summary_report(self):
    """Generate a summary report of the analysis"""
    report = []
    report.append("# Quantum Circuit Data Analysis Summary\n")
        
    # Dataset overview
    report.append("## Dataset Overview")
    report.append(f"- Total circuits analyzed: {len(self.df)}")
    report.append(f"- Data sources: {', '.join(self.df['source_type'].unique())}")
    report.append(f"- Qubit range: {self.df['num_qubits'].min()} - {self.df['num_qubits'].max()}")
    report.append(f"- Depth range: {self.df['depth'].min()} - {self.df['depth'].max()}")
    report.append("")
        
    # Source breakdown
    report.append("## Source Breakdown")
    for source in self.df['source_type'].unique():
        count = len(self.df[self.df['source_type'] == source])
        report.append(f"- {source}: {count} circuits ({count/len(self.df)*100:.1f}%)")
    report.append("")
        
    # Metric statistics
    report.append("## Metric Statistics")
    metrics = ['fidelity', 'expressibility', 'entanglement', 'kl_divergence']
    for metric in metrics:
        mean_val = self.df[metric].mean()
        std_val = self.df[metric].std()
        min_val = self.df[metric].min()
        max_val = self.df[metric].max()
        report.append(f"- {metric.title()}: μ={mean_val:.3f}, σ={std_val:.3f}, range=[{min_val:.3f}, {max_val:.3f}]")
        
    # Save report
    report_path = self.output_dir / "analysis_summary.md"
    with open(self.output_dir / "analysis_summary.md", 'w') as f:
        f.write('\n'.join(report))
        
    print(f"Summary report saved to: {self.output_dir / 'analysis_summary.md'}")
    
def generate_statistics_file(self):
    """Generate comprehensive statistics summary text file"""
    stats_lines = []
    stats_lines.append("=" * 80)
    stats_lines.append("QUANTUM CIRCUIT DATA ANALYSIS - COMPREHENSIVE STATISTICS")
    stats_lines.append("=" * 80)
    stats_lines.append("")
        
    # Dataset Overview
    stats_lines.append("DATASET OVERVIEW")
    stats_lines.append("-" * 40)
    stats_lines.append(f"Total circuits analyzed: {len(self.df)}")
        
    # Source type breakdown
    source_counts = self.df['source_type'].value_counts()
    for source, count in source_counts.items():
        percentage = (count / len(self.df)) * 100
        stats_lines.append(f"  - {source}: {count} circuits ({percentage:.1f}%)")
        
    # Qubit range
    if len(self.df) > 0:
        min_qubits = self.df['num_qubits'].min()
        max_qubits = self.df['num_qubits'].max()
        stats_lines.append(f"Qubit range: {min_qubits} - {max_qubits}")
            
        # Depth range
        min_depth = self.df['depth'].min()
        max_depth = self.df['depth'].max()
        stats_lines.append(f"Depth range: {min_depth} - {max_depth}")
        
    stats_lines.append("")
        
    # Detailed Statistics by Source Type
    for source in ['Simulator', 'Hardware']:
        source_data = self.df[self.df['source_type'] == source]
        if len(source_data) == 0:
            continue
                
        stats_lines.append(f"{source.upper()} DATA STATISTICS")
        stats_lines.append("-" * 40)
        stats_lines.append(f"Number of circuits: {len(source_data)}")
            
        # Metrics statistics
        metrics = {
            'fidelity': 'Fidelity',
            'robust_fidelity': 'Robust Fidelity', 
            'kl_divergence': 'KL Divergence (Expressibility)',
            'entanglement': 'Entanglement',
            'js_divergence': 'JS Divergence',
            'l2_norm': 'L2 Norm'
        }
            
        for metric, label in metrics.items():
            if metric in source_data.columns:
                data = source_data[metric]
                valid_data = data[data != 0]  # Exclude zero values
                    
                if len(valid_data) > 0:
                    stats_lines.append(f"\n{label}:")
                    stats_lines.append(f"  Mean: {valid_data.mean():.6f}")
                    stats_lines.append(f"  Std:  {valid_data.std():.6f}")
                    stats_lines.append(f"  Min:  {valid_data.min():.6f}")
                    stats_lines.append(f"  Max:  {valid_data.max():.6f}")
                    stats_lines.append(f"  Median: {valid_data.median():.6f}")
                    stats_lines.append(f"  Valid samples: {len(valid_data)}/{len(data)}")
                        
                    # Percentiles
                    q25 = valid_data.quantile(0.25)
                    q75 = valid_data.quantile(0.75)
                    stats_lines.append(f"  Q1 (25%): {q25:.6f}")
                    stats_lines.append(f"  Q3 (75%): {q75:.6f}")
                    stats_lines.append(f"  IQR: {q75 - q25:.6f}")
                else:
                    stats_lines.append(f"\n{label}: No valid data (all zeros)")
            
        # Circuit complexity statistics
        stats_lines.append(f"\nCircuit Complexity:")
        stats_lines.append(f"  Qubits - Mean: {source_data['num_qubits'].mean():.2f}, Range: {source_data['num_qubits'].min()}-{source_data['num_qubits'].max()}")
        stats_lines.append(f"  Depth  - Mean: {source_data['depth'].mean():.2f}, Range: {source_data['depth'].min()}-{source_data['depth'].max()}")
            
        stats_lines.append("")
        
    # Comparative Statistics
    if len(self.df[self.df['source_type'] == 'Simulator']) > 0 and len(self.df[self.df['source_type'] == 'Hardware']) > 0:
        stats_lines.append("COMPARATIVE ANALYSIS (SIMULATOR vs HARDWARE)")
        stats_lines.append("-" * 50)
            
        for metric, label in metrics.items():
            if metric in self.df.columns:
                sim_data = self.df[self.df['source_type'] == 'Simulator'][metric]
                hw_data = self.df[self.df['source_type'] == 'Hardware'][metric]
                    
                sim_valid = sim_data[sim_data != 0]
                hw_valid = hw_data[hw_data != 0]
                    
                if len(sim_valid) > 0 and len(hw_valid) > 0:
                    sim_mean = sim_valid.mean()
                    hw_mean = hw_valid.mean()
                    difference = abs(sim_mean - hw_mean)
                    relative_diff = (difference / max(sim_mean, hw_mean)) * 100 if max(sim_mean, hw_mean) > 0 else 0
                        
                    stats_lines.append(f"\n{label}:")
                    stats_lines.append(f"  Simulator mean: {sim_mean:.6f}")
                    stats_lines.append(f"  Hardware mean:  {hw_mean:.6f}")
                    stats_lines.append(f"  Absolute diff:  {difference:.6f}")
                    stats_lines.append(f"  Relative diff:  {relative_diff:.2f}%")
                        
                    # Statistical significance (simple comparison)
                    if sim_mean > hw_mean:
                        stats_lines.append(f"  Simulator > Hardware by {((sim_mean/hw_mean - 1) * 100):.1f}%")
                    else:
                        stats_lines.append(f"  Hardware > Simulator by {((hw_mean/sim_mean - 1) * 100):.1f}%")
        
    # Correlation Analysis
    stats_lines.append("\nCORRELATION ANALYSIS")
    stats_lines.append("-" * 30)
        
    correlation_metrics = ['num_qubits', 'depth', 'fidelity', 'robust_fidelity', 'kl_divergence', 'entanglement']
    available_metrics = [m for m in correlation_metrics if m in self.df.columns]
        
    if len(available_metrics) > 1:
        corr_matrix = self.df[available_metrics].corr()
            
        stats_lines.append("Strong correlations (|r| > 0.5):")
        for i, metric1 in enumerate(available_metrics):
            for j, metric2 in enumerate(available_metrics):
                if i < j:  # Avoid duplicates
                    corr_val = corr_matrix.loc[metric1, metric2]
                    if abs(corr_val) > 0.5:
                        stats_lines.append(f"  {metric1} ↔ {metric2}: r = {corr_val:.3f}")
        
    # Data Quality Assessment
    stats_lines.append("\nDATA QUALITY ASSESSMENT")
    stats_lines.append("-" * 35)
        
    for metric in ['fidelity', 'kl_divergence', 'entanglement']:
        if metric in self.df.columns:
            total_count = len(self.df)
            zero_count = len(self.df[self.df[metric] == 0])
            valid_count = total_count - zero_count
                
            stats_lines.append(f"{metric.replace('_', ' ').title()}:")
            stats_lines.append(f"  Valid data: {valid_count}/{total_count} ({(valid_count/total_count)*100:.1f}%)")
            stats_lines.append(f"  Zero values: {zero_count} ({(zero_count/total_count)*100:.1f}%)")
        
    # Save to file
    stats_file_path = self.output_dir / "comprehensive_statistics.txt"
    with open(stats_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(stats_lines))
        
    print(f"Comprehensive statistics saved to: {stats_file_path}")

def main():
    """Main execution function"""
    # Path to the merged data file
    data_path = r"C:\Users\jungh\Documents\GitHub\Kaist\OAT_Model\raw_data\merged_data.json"
    
    # Create visualizer and generate all plots
    visualizer = QuantumDataVisualizer(data_path)
    visualizer.generate_all_plots()
    
    print("\nAll visualizations completed successfully!")
    print("Check the 'quantum_analysis_plots' directory for all generated plots.")

if __name__ == "__main__":
    main()
