import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_json_data(file_path):
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None

def extract_circuit_data(results_data):
    """Extracts circuit data including expressibility values, depth, and n_qubits."""
    # Initialize data containers
    entropy_values = []
    shadow_values = []
    depths = []
    n_qubits_list = []
    two_qubit_ratios = []
    circuit_ids = []
    
    # Print the first few keys of the data to understand its structure
    if isinstance(results_data, dict):
        print("Top-level keys in the JSON data:")
        print(list(results_data.keys())[:5])
    elif isinstance(results_data, list):
        print(f"Data is a list with {len(results_data)} items")
        if len(results_data) > 0:
            print("First item keys:")
            if isinstance(results_data[0], dict):
                print(list(results_data[0].keys())[:5])
    else:
        print(f"Unexpected data type: {type(results_data)}")
        return {}, 0
    
    # Process the data based on its structure
    if isinstance(results_data, list):
        # Process list of circuit results
        for i, entry in enumerate(results_data):
            try:
                # Print detailed info for the first few entries to debug
                if i < 3:
                    print(f"\nProcessing entry {i}:")
                    print(f"Keys: {list(entry.keys())}")
                    if 'execution_result' in entry and 'expressibility' in entry['execution_result']:
                        print(f"Expressibility keys: {list(entry['execution_result']['expressibility'].keys())}")
                
                # Extract circuit info
                circuit_info = entry['circuit_info']
                depth = circuit_info.get('depth', 0)
                n_qubits = circuit_info.get('n_qubits', 0)
                two_qubit_ratio = circuit_info.get('two_qubit_ratio', 0)
                circuit_id = circuit_info.get('circuit_id', i)
                
                # Extract expressibility values
                execution_result = entry['execution_result']
                expressibility = execution_result['expressibility']
                
                # Check if both required metrics exist
                has_entropy = 'entropy_based' in expressibility
                has_shadow = 'classical_shadow' in expressibility
                
                if has_entropy and has_shadow:
                    entropy_val = expressibility['entropy_based']['expressibility_value']
                    shadow_val = expressibility['classical_shadow']['normalized_distance']
                    
                    if entropy_val is not None and shadow_val is not None:
                        # Append all data
                        entropy_values.append(float(entropy_val))
                        shadow_values.append(float(shadow_val))
                        depths.append(depth)
                        n_qubits_list.append(n_qubits)
                        two_qubit_ratios.append(two_qubit_ratio)
                        circuit_ids.append(circuit_id)
                        
                        if i < 3:  # Debug output for first few entries
                            print(f"Added values: entropy={entropy_val}, shadow={shadow_val}, depth={depth}, n_qubits={n_qubits}")
                
            except (KeyError, TypeError, ValueError) as e:
                if i < 3:  # Only print errors for first few entries to avoid spam
                    print(f"Error processing entry {i}: {type(e).__name__} - {e}")
                continue
    
    # Create a dictionary with all the extracted data
    circuit_data = {
        'entropy_values': entropy_values,
        'shadow_values': shadow_values,
        'depths': depths,
        'n_qubits_list': n_qubits_list,
        'two_qubit_ratios': two_qubit_ratios,
        'circuit_ids': circuit_ids
    }
    
    return circuit_data, len(entropy_values)

def normalize_values(values):
    """Normalize values to [0, 1] range."""
    if not values:
        return []
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.5] * len(values)  # If all values are the same, return 0.5
    return [(val - min_val) / (max_val - min_val) for val in values]

def plot_correlation(entropy_values, shadow_values, output_path):
    """Creates and saves a scatter plot of the expressibility values."""
    if not entropy_values or not shadow_values:
        print("No data available to plot.")
        return
    
    # Store original values for reference
    original_entropy = entropy_values.copy()
    original_shadow = shadow_values.copy()
    
    # Normalize both metrics to [0, 1] range
    normalized_entropy = normalize_values(entropy_values)
    
    # For shadow values, we invert the normalization since lower distance means better expressibility
    # This way, for both metrics, higher values = better expressibility
    normalized_shadow = [1 - val for val in normalize_values(shadow_values)]
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Original values
    scatter1 = ax1.scatter(original_entropy, original_shadow, alpha=0.7, 
                          edgecolors='w', linewidth=0.5, c=original_entropy, 
                          cmap='viridis', s=50)
    
    ax1.set_title('Original Metrics (Different Scales)', fontsize=14)
    ax1.set_xlabel('Entropy-based Expressibility (bits)\n(Higher values → Better expressibility)', fontsize=11)
    ax1.set_ylabel('Classical Shadow Normalized Distance\n(Higher values → Further from Haar randomness)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add colorbar to the first plot
    cbar1 = fig.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Entropy Value (bits)', rotation=270, labelpad=20)
    
    # Calculate correlation for original values
    corr_original = np.corrcoef(original_entropy, original_shadow)[0, 1]
    ax1.text(0.05, 0.95, f'Pearson Correlation: {corr_original:.3f}', 
             transform=ax1.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.7))
    
    # Plot 2: Normalized values
    scatter2 = ax2.scatter(normalized_entropy, normalized_shadow, alpha=0.7, 
                          edgecolors='w', linewidth=0.5, c=normalized_entropy, 
                          cmap='viridis', s=50)
    
    ax2.set_title('Normalized Metrics (0-1 Scale)', fontsize=14)
    ax2.set_xlabel('Normalized Entropy-based Expressibility\n(Higher values → Better expressibility)', fontsize=11)
    ax2.set_ylabel('Inverted Normalized Classical Shadow Distance\n(Higher values → Better expressibility)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Set both axes to the same range for better comparison
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # Add colorbar to the second plot
    cbar2 = fig.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Normalized Entropy Value', rotation=270, labelpad=20)
    
    # Calculate correlation for normalized values
    corr_normalized = np.corrcoef(normalized_entropy, normalized_shadow)[0, 1]
    ax2.text(0.05, 0.95, f'Pearson Correlation: {corr_normalized:.3f}', 
             transform=ax2.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.7))
    
    # Add diagonal line on normalized plot to show perfect correlation
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    
    # Add an explanation note
    note_text = (
        "Note: These plots compare two different expressibility metrics:\n"
        "• Left plot: Original metrics with different scales and units\n"
        "• Right plot: Both metrics normalized to [0,1] range with inverted shadow distance\n"
        "In the normalized plot, higher values indicate better expressibility for both metrics."
    )
    plt.figtext(0.5, 0.01, note_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', fc='whitesmoke', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    plt.suptitle('Comparison of Entropy-Based and Classical Shadow Expressibility Metrics', 
                fontsize=16, y=0.98)
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    # plt.show() # Uncomment to display the plot directly

def plot_by_depth_and_qubits(circuit_data, output_dir):
    """Creates visualizations of expressibility metrics by depth and number of qubits."""
    if not circuit_data or not circuit_data.get('entropy_values'):
        print("No data available to plot.")
        return
    
    # Extract data
    entropy_values = circuit_data['entropy_values']
    shadow_values = circuit_data['shadow_values']
    depths = circuit_data['depths']
    n_qubits_list = circuit_data['n_qubits_list']
    two_qubit_ratios = circuit_data['two_qubit_ratios']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Depth vs. Expressibility (Entropy-based)
    plt.figure(figsize=(12, 8))
    
    # Use a colormap to represent different numbers of qubits
    unique_qubits = sorted(set(n_qubits_list))
    qubit_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_qubits)))
    qubit_color_map = {q: qubit_colors[i] for i, q in enumerate(unique_qubits)}
    
    # Create scatter plot with different colors for different qubit counts
    for q in unique_qubits:
        indices = [i for i, qubits in enumerate(n_qubits_list) if qubits == q]
        q_depths = [depths[i] for i in indices]
        q_entropy = [entropy_values[i] for i in indices]
        plt.scatter(q_depths, q_entropy, c=[qubit_color_map[q]], label=f'{q} qubits', 
                    alpha=0.7, s=50, edgecolors='w', linewidth=0.5)
    
    plt.title('Circuit Depth vs. Entropy-based Expressibility', fontsize=16)
    plt.xlabel('Circuit Depth', fontsize=14)
    plt.ylabel('Entropy-based Expressibility (bits)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Number of Qubits')
    plt.tight_layout()
    
    # Save the figure
    depth_entropy_path = os.path.join(output_dir, 'depth_vs_entropy.png')
    plt.savefig(depth_entropy_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {depth_entropy_path}")
    
    # 2. Depth vs. Expressibility (Classical Shadow)
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with different colors for different qubit counts
    for q in unique_qubits:
        indices = [i for i, qubits in enumerate(n_qubits_list) if qubits == q]
        q_depths = [depths[i] for i in indices]
        q_shadow = [shadow_values[i] for i in indices]
        plt.scatter(q_depths, q_shadow, c=[qubit_color_map[q]], label=f'{q} qubits', 
                    alpha=0.7, s=50, edgecolors='w', linewidth=0.5)
    
    plt.title('Circuit Depth vs. Classical Shadow Normalized Distance', fontsize=16)
    plt.xlabel('Circuit Depth', fontsize=14)
    plt.ylabel('Classical Shadow Normalized Distance', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Number of Qubits')
    plt.tight_layout()
    
    # Save the figure
    depth_shadow_path = os.path.join(output_dir, 'depth_vs_shadow.png')
    plt.savefig(depth_shadow_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {depth_shadow_path}")
    
    # 3. Number of Qubits vs. Expressibility (Entropy-based)
    plt.figure(figsize=(12, 8))
    
    # Use a colormap to represent different depths
    unique_depths = sorted(set(depths))
    depth_colors = plt.cm.plasma(np.linspace(0, 1, len(unique_depths)))
    depth_color_map = {d: depth_colors[i] for i, d in enumerate(unique_depths)}
    
    # Create scatter plot with different colors for different depths
    for d in unique_depths:
        indices = [i for i, depth in enumerate(depths) if depth == d]
        d_qubits = [n_qubits_list[i] for i in indices]
        d_entropy = [entropy_values[i] for i in indices]
        plt.scatter(d_qubits, d_entropy, c=[depth_color_map[d]], label=f'Depth {d}', 
                    alpha=0.7, s=50, edgecolors='w', linewidth=0.5)
    
    plt.title('Number of Qubits vs. Entropy-based Expressibility', fontsize=16)
    plt.xlabel('Number of Qubits', fontsize=14)
    plt.ylabel('Entropy-based Expressibility (bits)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Circuit Depth')
    plt.tight_layout()
    
    # Save the figure
    qubits_entropy_path = os.path.join(output_dir, 'qubits_vs_entropy.png')
    plt.savefig(qubits_entropy_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {qubits_entropy_path}")
    
    # 4. Number of Qubits vs. Expressibility (Classical Shadow)
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with different colors for different depths
    for d in unique_depths:
        indices = [i for i, depth in enumerate(depths) if depth == d]
        d_qubits = [n_qubits_list[i] for i in indices]
        d_shadow = [shadow_values[i] for i in indices]
        plt.scatter(d_qubits, d_shadow, c=[depth_color_map[d]], label=f'Depth {d}', 
                    alpha=0.7, s=50, edgecolors='w', linewidth=0.5)
    
    plt.title('Number of Qubits vs. Classical Shadow Normalized Distance', fontsize=16)
    plt.xlabel('Number of Qubits', fontsize=14)
    plt.ylabel('Classical Shadow Normalized Distance', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Circuit Depth')
    plt.tight_layout()
    
    # Save the figure
    qubits_shadow_path = os.path.join(output_dir, 'qubits_vs_shadow.png')
    plt.savefig(qubits_shadow_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {qubits_shadow_path}")
    
    # 5. 3D Plot: Depth, Qubits, and Expressibility (Entropy-based)
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create 3D scatter plot
    scatter = ax.scatter(depths, n_qubits_list, entropy_values, 
                        c=entropy_values, cmap='viridis', 
                        s=50, alpha=0.7, edgecolors='w', linewidth=0.5)
    
    ax.set_title('3D Visualization: Depth, Qubits, and Entropy-based Expressibility', fontsize=16)
    ax.set_xlabel('Circuit Depth', fontsize=14)
    ax.set_ylabel('Number of Qubits', fontsize=14)
    ax.set_zlabel('Entropy-based Expressibility (bits)', fontsize=14)
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Entropy-based Expressibility (bits)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    # Save the figure
    plot_3d_path = os.path.join(output_dir, '3d_depth_qubits_entropy.png')
    plt.savefig(plot_3d_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_3d_path}")
    
    # 6. Heatmap: Average Entropy-based Expressibility by Depth and Qubits
    plt.figure(figsize=(14, 10))
    
    # Create a grid for the heatmap
    depth_qubit_pairs = [(d, q) for d in unique_depths for q in unique_qubits]
    avg_entropy_values = []
    
    for d, q in depth_qubit_pairs:
        indices = [i for i, (depth, qubits) in enumerate(zip(depths, n_qubits_list)) 
                  if depth == d and qubits == q]
        if indices:
            avg_entropy = sum(entropy_values[i] for i in indices) / len(indices)
            avg_entropy_values.append(avg_entropy)
        else:
            avg_entropy_values.append(np.nan)  # No data for this combination
    
    # Reshape data for heatmap
    heatmap_data = np.full((len(unique_depths), len(unique_qubits)), np.nan)
    for idx, (d, q) in enumerate(depth_qubit_pairs):
        d_idx = unique_depths.index(d)
        q_idx = unique_qubits.index(q)
        heatmap_data[d_idx, q_idx] = avg_entropy_values[idx]
    
    # Create heatmap
    plt.imshow(heatmap_data, cmap='viridis', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Average Entropy-based Expressibility')
    
    # Set ticks and labels
    plt.xticks(range(len(unique_qubits)), unique_qubits)
    plt.yticks(range(len(unique_depths)), unique_depths)
    plt.xlabel('Number of Qubits', fontsize=14)
    plt.ylabel('Circuit Depth', fontsize=14)
    plt.title('Average Entropy-based Expressibility by Depth and Qubits', fontsize=16)
    
    # Add text annotations with the actual values
    for i in range(len(unique_depths)):
        for j in range(len(unique_qubits)):
            if not np.isnan(heatmap_data[i, j]):
                plt.text(j, i, f"{heatmap_data[i, j]:.2f}", 
                         ha="center", va="center", color="white" if heatmap_data[i, j] < np.nanmean(heatmap_data) else "black")
    
    plt.tight_layout()
    
    # Save the figure
    heatmap_path = os.path.join(output_dir, 'heatmap_depth_qubits_entropy.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {heatmap_path}")

def main():
    # Determine the base directory of the script to locate the JSON file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the JSON file relative to the script's location
    json_file_path = os.path.join(script_dir, 'grid_circuits', 'mega_results', 'batch_1_results_20250529_101750.json')
    output_plot_path = os.path.join(script_dir, 'expressibility_correlation.png')
    output_dir = os.path.join(script_dir, 'expressibility_visualizations')

    print(f"Loading data from: {json_file_path}")
    results_data = load_json_data(json_file_path)
    
    if results_data:
        print(f"Extracting circuit data...")
        circuit_data, data_count = extract_circuit_data(results_data)
        print(f"Found {data_count} data points with both expressibility values.")
        
        if data_count > 0:
            # Plot correlation between the two expressibility metrics
            plot_correlation(circuit_data['entropy_values'], circuit_data['shadow_values'], output_plot_path)
            
            # Create additional visualizations by depth and number of qubits
            plot_by_depth_and_qubits(circuit_data, output_dir)
        else:
            print("Not enough data to create plots.")

if __name__ == '__main__':
    main()
