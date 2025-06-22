import json
import gzip
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class QuantumMegaJobAnalyzer:
    """Class to analyze mega job result files"""
    
    def __init__(self, mega_results_dir="grid_circuits/mega_results"):
        """
        Initialize quantum mega job analyzer
        
        Args:
            mega_results_dir (str): Path to the mega job result directory
        """
        self.mega_results_dir = mega_results_dir
        self.circuits_data = []
        self.summary_stats = {}
        self.df = None
        
        # Set visualization style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_mega_job_data(self):
        """Load mega job result files"""
        print("📁 Loading mega job result files...")
        
        if not os.path.exists(self.mega_results_dir):
            print(f"❌ Unable to find the mega job result directory: {self.mega_results_dir}")
            return
        
        # Find result files
        result_files = list(Path(self.mega_results_dir).glob("*results*.json"))
        summary_files = list(Path(self.mega_results_dir).glob("*summary*.json"))
        
        print(f"Found files:")
        print(f"  Result files: {len(result_files)}")
        print(f"  Summary files: {len(summary_files)}")
        
        # Load summary statistics
        if summary_files:
            with open(summary_files[0], 'r', encoding='utf-8') as f:
                self.summary_stats = json.load(f)
            print(f"✅ Summary statistics loaded: {summary_files[0].name}")
        
        # Load detailed results
        for file_path in result_files:
            try:
                print(f"  📂 {file_path.name} loading...")
                    with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract data based on file structure
                if 'detailed_results' in data:
                    # mega_job_results file format
                    circuits = data['detailed_results']
                    print(f"    📊 Loaded {len(circuits)} circuits from detailed_results")
                elif isinstance(data, list):
                    # batch_results file format
                    circuits = data
                    print(f"    📊 Loaded {len(circuits)} circuits from batch data")
                else:
                    print(f"    ⚠️ Unknown file format: {file_path.name}")
                    continue
                
                self.circuits_data.extend(circuits)
                    
            except Exception as e:
                print(f"    ❌ Failed to load file {file_path.name}: {str(e)}")
        
        print(f"\n✅ Total {len(self.circuits_data)} circuits data loaded")
        
        # Create DataFrame
        self._create_mega_dataframe()
        
    def _create_mega_dataframe(self):
        """Convert mega job data to pandas DataFrame"""
        print("\n🔄 Converting mega job data to DataFrame...")
        
        data_rows = []
        
        for i, circuit in enumerate(self.circuits_data):
            try:
                # Extract circuit information
                circuit_info = circuit.get('circuit_info', {})
                execution_result = circuit.get('execution_result', {})
                
                # Basic circuit parameters
                n_qubits = circuit_info.get('n_qubits', 0)
                depth = circuit_info.get('depth', 0)
                gate_count = circuit_info.get('gate_count', 0)
                two_qubit_ratio = circuit_info.get('two_qubit_ratio', 0)
                
                # Performance metrics
                fidelity = execution_result.get('zero_state_probability', 0)
                robust_fidelity = execution_result.get('robust_fidelity', 0)
                
                # Error rate information
                error_rates = execution_result.get('error_rates', {})
                total_error_rate = error_rates.get('total_error_rate', 0)
                
                # 🎯 **수정된 Expressibility 정보 파싱**
                expressibility_info = execution_result.get('expressibility', {})
                
                # Classical Shadow 표현력 데이터
                classical_shadow_data = expressibility_info.get('classical_shadow', {})
                if classical_shadow_data:
                    classical_shadow_expr = classical_shadow_data.get('distance', 0)
                else:
                    # 구버전 호환성 (직접 expressibility_value)
                    classical_shadow_expr = expressibility_info.get('expressibility_value', 0)
                
                # 엔트로피 기반 표현력 데이터  
                entropy_based_data = expressibility_info.get('entropy_based', {})
                if entropy_based_data:
                    entropy_expressibility = entropy_based_data.get('expressibility_value', 0)
                    angle_entropy = entropy_based_data.get('angle_entropy', 0)
                else:
                    # 데이터가 없는 경우 0으로 설정
                    entropy_expressibility = 0
                    angle_entropy = 0
                
                # 🆕 **features 섹션에서도 expressibility 데이터 찾기 (백업 경로)**
                features_data = circuit.get('features', {})
                
                # Classical Shadow 백업 경로
                if classical_shadow_expr == 0:
                    classical_shadow_expr = features_data.get('expressibility_distance', 0)
                    if classical_shadow_expr == 0:
                        classical_shadow_expr = features_data.get('normalized_expressibility', 0)
                
                # 엔트로피 표현력 백업 경로
                if entropy_expressibility == 0:
                    entropy_expressibility = features_data.get('entropy_expressibility', 0)
                
                if angle_entropy == 0:
                    angle_entropy = features_data.get('angle_entropy', 0)
                
                if distance_entropy == 0:
                    distance_entropy = features_data.get('distance_entropy', 0)
                
                # Measurement statistics
                measurement_stats = execution_result.get('measurement_statistics', {})
                entropy = measurement_stats.get('entropy', 0)
                concentration = measurement_stats.get('concentration', 0)
                measured_states = measurement_stats.get('measured_states', 0)
                
                # Execution information (원래)
                original_execution_time = circuit.get('execution_time', 0)
                total_shots = execution_result.get('total_shots', 0)
                
                # 🎯 **더 의미있는 복잡도 기반 메트릭들 계산**
                
                # 1. 회로 복잡도 (게이트 수 × 깊이)
                circuit_complexity = gate_count * depth
                
                # 2. 양자 볼륨 추정 (큐빗수와 깊이 기반)
                quantum_volume = min(n_qubits, depth) ** 2
                
                # 3. 2큐빗 게이트 수 (실제 얽힘 연산)
                two_qubit_gates = int(gate_count * two_qubit_ratio)
                
                # 4. 추정 실행 시간 (복잡도 기반)
                # 기본 시간 + 큐빗당 시간 + 게이트당 시간 + 깊이당 시간
                estimated_execution_time = (
                    0.01 +  # 기본 오버헤드
                    n_qubits * 0.001 +  # 큐빗당 0.001초
                    gate_count * 0.0001 +  # 게이트당 0.0001초
                    depth * 0.002 +  # 깊이당 0.002초
                    two_qubit_gates * 0.0005  # 2큐빗 게이트 추가 시간
                )
                
                # 5. 자원 효율성 메트릭들
                performance_per_complexity = robust_fidelity / (circuit_complexity + 1e-6)
                performance_per_shot = robust_fidelity / (total_shots + 1e-6)
                fidelity_per_qubit = fidelity / (n_qubits + 1e-6)
                
                # 6. 샷 효율성 (성능 대비 사용 샷 수)
                shot_efficiency = (robust_fidelity * 1000) / (total_shots + 1)
                
                # Calculate quantum characteristics
                entanglement_potential = two_qubit_ratio * depth
                quantum_coherence = 1.0 - concentration if concentration > 0 else 0
                max_possible_states = min(2**n_qubits, 1000)
                state_exploration = measured_states / max_possible_states if max_possible_states > 0 else 0
                
                # 2Q gate ratio class
                if two_qubit_ratio < 0.2:
                    ratio_class = 'low'
                elif two_qubit_ratio < 0.4:
                    ratio_class = 'medium'
                else:
                    ratio_class = 'high'
                
                # Create data row
                row = {
                    'circuit_id': i,
                    'n_qubits': n_qubits,
                    'depth': depth,
                    'gate_count': gate_count,
                    'two_qubit_ratio': two_qubit_ratio,
                    'two_qubit_ratio_class': ratio_class,
                    
                    # Performance metrics
                    'fidelity': fidelity,
                    'robust_fidelity': robust_fidelity,
                    'total_error_rate': total_error_rate,
                    
                    # Expressibility metrics
                    'classical_shadow_expr': classical_shadow_expr,
                    'entropy_expressibility': entropy_expressibility,
                    
                    # Measurement statistics (상태 분포 엔트로피가 표현력 역할)
                    'entropy': entropy,
                    'concentration': concentration,
                    'entanglement_potential': entanglement_potential,
                    
                    # Derived complexity metrics
                    'circuit_complexity': circuit_complexity,
                    'quantum_volume': quantum_volume,
                    'two_qubit_gates': two_qubit_gates,
                    'estimated_execution_time': estimated_execution_time,
                    'performance_per_complexity': performance_per_complexity,
                    'performance_per_shot': performance_per_shot,
                    'fidelity_per_qubit': fidelity_per_qubit,
                    'shot_efficiency': shot_efficiency,
                    
                    # Original execution info (for reference)
                    'original_execution_time': original_execution_time,
                    'total_shots': total_shots
                }
                
                data_rows.append(row)
                
            except Exception as e:
                print(f"  ⚠️ Circuit {i} processing failed: {str(e)}")
        
        # Create DataFrame
        self.df = pd.DataFrame(data_rows)
        
        print(f"✅ DataFrame created: {len(self.df)} rows, {len(self.df.columns)} columns")
        
        # Print basic statistics
        self._print_mega_statistics()
        
    def _print_mega_statistics(self):
        """Print mega job result statistics"""
        print("\n" + "="*60)
        print("            Mega Job Analysis Statistics")
        print("="*60)
        
        if self.df is None or len(self.df) == 0:
            print("❌ No data to analyze.")
            return
        
        print(f"Total circuit count: {len(self.df)}")
        
        # Circuit configuration statistics
        print(f"\n🔬 Circuit Configuration:")
        print(f"  Qubit range: {self.df['n_qubits'].min()} ~ {self.df['n_qubits'].max()}")
        print(f"  Depth range: {self.df['depth'].min()} ~ {self.df['depth'].max()}")
        print(f"  2Q gate ratio: {self.df['two_qubit_ratio'].min():.1%} ~ {self.df['two_qubit_ratio'].max():.1%}")
        
        # Performance statistics
        print(f"\n⚛️ Performance Metrics:")
        print(f"  Average fidelity: {self.df['fidelity'].mean():.6f} ± {self.df['fidelity'].std():.6f}")
        print(f"  Average Robust fidelity: {self.df['robust_fidelity'].mean():.6f} ± {self.df['robust_fidelity'].std():.6f}")
        print(f"  Average error rate: {self.df['total_error_rate'].mean():.6f} ± {self.df['total_error_rate'].std():.6f}")
        
        # 🔧 Execution time analysis (fixed)
        unique_exec_times = self.df['original_execution_time'].nunique()
        print(f"\n⏱️ Execution Time Analysis:")
        print(f"  Individual circuit time: {self.df['original_execution_time'].iloc[0]:.3f}s (averaged across batch)")
        print(f"  Estimated total batch time: {self.df['original_execution_time'].iloc[0] * len(self.df):.1f}s")
        print(f"  Unique execution times: {unique_exec_times}")
        if unique_exec_times == 1:
            print(f"  ⚠️ Note: All circuits have same execution time (mega job batch average)")
            print(f"  ⚠️ This is expected for mega job execution - represents average per circuit")
        
        # 2Q gate ratio analysis
        print(f"\n🔗 2Q gate ratio analysis:")
        ratio_analysis = self.df.groupby('two_qubit_ratio_class').agg({
            'fidelity': ['count', 'mean', 'std'],
            'robust_fidelity': 'mean',
            'total_error_rate': 'mean'
        }).round(6)
        
        for ratio_class in ['low', 'medium', 'high']:
            if ratio_class in ratio_analysis.index:
                count = ratio_analysis.loc[ratio_class, ('fidelity', 'count')]
                fid_mean = ratio_analysis.loc[ratio_class, ('fidelity', 'mean')]
                robust_mean = ratio_analysis.loc[ratio_class, ('robust_fidelity', 'mean')]
                error_mean = ratio_analysis.loc[ratio_class, ('total_error_rate', 'mean')]
                
                print(f"  {ratio_class.upper()} ({count} circuits):")
                print(f"    fidelity: {fid_mean:.6f}")
                print(f"    Robust fidelity: {robust_mean:.6f}")
                print(f"    error rate: {error_mean:.6f}")
    
    def create_core_expressibility_analysis(self):
        """🎯 Core Expressibility Analysis with Statistical Significance"""
        if self.df is None or len(self.df) == 0:
            print("❌ No data to analyze.")
            return
        
        # Import statistical test functions
        from scipy import stats
        from scipy.stats import pearsonr, spearmanr, kendalltau
        import numpy as np
        
        print("🎯 Creating Core Expressibility Analysis...")
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Classical Shadow vs Entropy-based Expressibility
        plt.subplot(3, 3, 1)
        x = self.df['classical_shadow_expr']
        y = self.df['entropy_expressibility']
        
        # Remove zero values for meaningful correlation
        mask = (x > 0) & (y > 0)
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) > 3:
            plt.scatter(x_clean, y_clean, c=self.df.loc[mask, 'n_qubits'], 
                       cmap='viridis', alpha=0.7, s=60)
            plt.colorbar(label='Number of Qubits')
            
            # Statistical significance test
            pearson_r, pearson_p = pearsonr(x_clean, y_clean)
            spearman_r, spearman_p = spearmanr(x_clean, y_clean)
            
            plt.xlabel('Classical Shadow Expressibility')
            plt.ylabel('Entropy-based Expressibility')
            plt.title(f'Expressibility Methods Comparison\nPearson r={pearson_r:.3f} (p={pearson_p:.3f})\nSpearman ρ={spearman_r:.3f} (p={spearman_p:.3f})')
            
            # Add trend line if significant
            if pearson_p < 0.05:
                z = np.polyfit(x_clean, y_clean, 1)
                p = np.poly1d(z)
                plt.plot(x_clean, p(x_clean), "r--", alpha=0.8, linewidth=2)
        else:
            plt.text(0.5, 0.5, 'Insufficient Non-zero Data', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title('Classical Shadow vs Entropy Expressibility')
        plt.grid(True, alpha=0.3)
        
        # 2. Expressibility vs Number of Qubits
        plt.subplot(3, 3, 2)
        # Use entropy-based expressibility as primary metric
        expr_data = self.df['entropy_expressibility']
        qubit_data = self.df['n_qubits']
        
        # Group by qubit count and calculate statistics
        qubit_groups = self.df.groupby('n_qubits')['entropy_expressibility'].agg(['mean', 'std', 'count'])
        
        # Plot with error bars
        valid_groups = qubit_groups[qubit_groups['count'] >= 3]  # At least 3 samples
        if len(valid_groups) > 0:
            plt.errorbar(valid_groups.index, valid_groups['mean'], 
                        yerr=valid_groups['std'], marker='o', capsize=5, 
                        linewidth=2, markersize=8)
            
            # Statistical test for trend
            if len(valid_groups) > 2:
                correlation_r, correlation_p = pearsonr(valid_groups.index, valid_groups['mean'])
                plt.title(f'Expressibility vs Qubit Count\nCorrelation r={correlation_r:.3f} (p={correlation_p:.3f})')
            else:
                plt.title('Expressibility vs Qubit Count')
        
        plt.xlabel('Number of Qubits')
        plt.ylabel('Entropy-based Expressibility')
        plt.grid(True, alpha=0.3)
        
        # 3. Expressibility vs Circuit Depth
        plt.subplot(3, 3, 3)
        depth_groups = self.df.groupby('depth')['entropy_expressibility'].agg(['mean', 'std', 'count'])
        valid_depth_groups = depth_groups[depth_groups['count'] >= 3]
        
        if len(valid_depth_groups) > 0:
            plt.errorbar(valid_depth_groups.index, valid_depth_groups['mean'], 
                        yerr=valid_depth_groups['std'], marker='s', capsize=5,
                        linewidth=2, markersize=8, color='orange')
            
            # Statistical test
            if len(valid_depth_groups) > 2:
                correlation_r, correlation_p = pearsonr(valid_depth_groups.index, valid_depth_groups['mean'])
                plt.title(f'Expressibility vs Circuit Depth\nCorrelation r={correlation_r:.3f} (p={correlation_p:.3f})')
            else:
                plt.title('Expressibility vs Circuit Depth')
        
        plt.xlabel('Circuit Depth')
        plt.ylabel('Entropy-based Expressibility')
        plt.grid(True, alpha=0.3)
        
        # 4. Qubit-Depth Expressibility Heatmap
        plt.subplot(3, 3, 4)
        pivot_expr = self.df.pivot_table(values='entropy_expressibility', 
                                        index='depth', 
                                        columns='n_qubits', 
                                        aggfunc='mean')
        
        if not pivot_expr.empty:
            sns.heatmap(pivot_expr, annot=True, fmt='.3f', cmap='viridis', 
                       cbar_kws={'label': 'Entropy Expressibility'})
            plt.title('Expressibility Landscape (Qubit-Depth)')
            plt.xlabel('Number of Qubits')
            plt.ylabel('Circuit Depth')
        
        # 5. Statistical Distribution Analysis
        plt.subplot(3, 3, 5)
        # Test for normality
        from scipy.stats import shapiro, normaltest
        
        expr_values = self.df['entropy_expressibility'][self.df['entropy_expressibility'] > 0]
        if len(expr_values) > 3:
            # Histogram with normal distribution overlay
            plt.hist(expr_values, bins=20, density=True, alpha=0.7, color='skyblue', 
                    edgecolor='black', label='Observed')
            
            # Normal distribution test
            shapiro_stat, shapiro_p = shapiro(expr_values)
            normal_stat, normal_p = normaltest(expr_values)
            
            # Overlay normal distribution
            mu, sigma = expr_values.mean(), expr_values.std()
            x = np.linspace(expr_values.min(), expr_values.max(), 100)
            plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                    label=f'Normal (μ={mu:.3f}, σ={sigma:.3f})')
            
            plt.xlabel('Entropy-based Expressibility')
            plt.ylabel('Density')
            plt.title(f'Distribution Analysis\nShapiro p={shapiro_p:.3f}, Normal p={normal_p:.3f}')
            plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. ANOVA Test for Qubit Groups
        plt.subplot(3, 3, 6)
        from scipy.stats import f_oneway
        
        # Group expressibility by qubit count
        qubit_expr_groups = []
        qubit_labels = []
        for qubit_count in sorted(self.df['n_qubits'].unique()):
            group_data = self.df[self.df['n_qubits'] == qubit_count]['entropy_expressibility']
            group_data = group_data[group_data > 0]  # Remove zeros
            if len(group_data) >= 3:  # Minimum sample size
                qubit_expr_groups.append(group_data)
                qubit_labels.append(f'{qubit_count} qubits')
        
        if len(qubit_expr_groups) >= 2:
            # ANOVA test
            f_stat, anova_p = f_oneway(*qubit_expr_groups)
            
            # Box plot
            plt.boxplot(qubit_expr_groups, labels=qubit_labels)
            plt.xlabel('Qubit Groups')
            plt.ylabel('Entropy-based Expressibility')
            plt.title(f'ANOVA Test Across Qubit Groups\nF={f_stat:.3f}, p={anova_p:.3f}')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 7. Correlation Matrix of Key Metrics
        plt.subplot(3, 3, 7)
        key_metrics = ['n_qubits', 'depth', 'entropy_expressibility', 
                      'classical_shadow_expr', 'robust_fidelity']
        
        # Filter out circuits with zero expressibility for meaningful correlations
        mask = self.df['entropy_expressibility'] > 0
        corr_data = self.df.loc[mask, key_metrics]
        
        if len(corr_data) > 10:
            corr_matrix = corr_data.corr()
            
            # Create correlation matrix with significance stars
            mask_matrix = np.zeros_like(corr_matrix, dtype=bool)
            significance_matrix = np.full_like(corr_matrix, '', dtype=object)
            
            for i in range(len(key_metrics)):
                for j in range(len(key_metrics)):
                    if i != j:
                        x = corr_data.iloc[:, i]
                        y = corr_data.iloc[:, j]
                        _, p_val = pearsonr(x, y)
                        if p_val < 0.001:
                            significance_matrix[i, j] = '***'
                        elif p_val < 0.01:
                            significance_matrix[i, j] = '**'
                        elif p_val < 0.05:
                            significance_matrix[i, j] = '*'
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": .8})
            plt.title('Correlation Matrix\n(*p<0.05, **p<0.01, ***p<0.001)')
        
        # 8. Effect Size Analysis
        plt.subplot(3, 3, 8)
        # Calculate Cohen's d between different qubit groups
        if len(qubit_expr_groups) >= 2:
            effect_sizes = []
            comparisons = []
            
            for i in range(len(qubit_expr_groups)):
                for j in range(i+1, len(qubit_expr_groups)):
                    group1 = qubit_expr_groups[i]
                    group2 = qubit_expr_groups[j]
                    
                    # Cohen's d calculation
                    pooled_std = np.sqrt(((len(group1)-1)*np.var(group1, ddof=1) + 
                                         (len(group2)-1)*np.var(group2, ddof=1)) / 
                                        (len(group1) + len(group2) - 2))
                    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
                    
                    effect_sizes.append(abs(cohens_d))
                    comparisons.append(f'{qubit_labels[i]} vs {qubit_labels[j]}')
            
            if effect_sizes:
                plt.barh(range(len(effect_sizes)), effect_sizes)
                plt.yticks(range(len(effect_sizes)), comparisons)
                plt.xlabel("Cohen's d (Effect Size)")
                plt.title('Effect Size Between Qubit Groups\n(>0.2 small, >0.5 medium, >0.8 large)')
                
                # Add vertical lines for effect size interpretation
                plt.axvline(x=0.2, color='yellow', linestyle='--', alpha=0.7, label='Small')
                plt.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium')
                plt.axvline(x=0.8, color='red', linestyle='--', alpha=0.7, label='Large')
                plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 9. Summary Statistics Table
        plt.subplot(3, 3, 9)
        plt.axis('off')
        
        # Create summary statistics
        summary_stats = []
        
        # Overall statistics
        valid_expr = self.df[self.df['entropy_expressibility'] > 0]['entropy_expressibility']
        if len(valid_expr) > 0:
            summary_stats.extend([
                f"Total Circuits: {len(self.df)}",
                f"Valid Expressibility: {len(valid_expr)}",
                f"Mean Expressibility: {valid_expr.mean():.4f}",
                f"Std Expressibility: {valid_expr.std():.4f}",
                f"Expressibility Range: {valid_expr.min():.4f} - {valid_expr.max():.4f}",
                "",
                "Qubit Analysis:",
                f"Qubit Range: {self.df['n_qubits'].min()} - {self.df['n_qubits'].max()}",
                f"Depth Range: {self.df['depth'].min()} - {self.df['depth'].max()}",
                "",
                "Statistical Tests:",
            ])
            
            # Add correlation results
            if len(valid_groups) > 2:
                qubit_corr_r, qubit_corr_p = pearsonr(valid_groups.index, valid_groups['mean'])
                summary_stats.append(f"Qubit-Expr Correlation: r={qubit_corr_r:.3f}, p={qubit_corr_p:.3f}")
            
            if len(valid_depth_groups) > 2:
                depth_corr_r, depth_corr_p = pearsonr(valid_depth_groups.index, valid_depth_groups['mean'])
                summary_stats.append(f"Depth-Expr Correlation: r={depth_corr_r:.3f}, p={depth_corr_p:.3f}")
            
            # Display statistics
            for i, stat in enumerate(summary_stats):
                plt.text(0.05, 0.95 - i*0.08, stat, transform=plt.gca().transAxes, 
                        fontsize=10, fontweight='bold' if stat.endswith(':') else 'normal')
        
        plt.tight_layout()
        plt.savefig('grid_circuits/images/core_expressibility_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Core Expressibility Analysis completed!")
        return self._generate_statistical_report()
    
    def _generate_statistical_report(self):
        """Generate detailed statistical report"""
        from scipy import stats
        from scipy.stats import pearsonr, spearmanr
        
        print("\n" + "="*80)
        print("                    STATISTICAL SIGNIFICANCE REPORT")
        print("="*80)
        
        # Filter valid data
        valid_data = self.df[self.df['entropy_expressibility'] > 0].copy()
        
        if len(valid_data) < 10:
            print("❌ Insufficient data for statistical analysis")
            return
        
        print(f"\n📊 Sample Size: {len(valid_data)} circuits (out of {len(self.df)} total)")
        
        # 1. Expressibility Distribution Analysis
        expr_values = valid_data['entropy_expressibility']
        shapiro_stat, shapiro_p = stats.shapiro(expr_values)
        normal_stat, normal_p = stats.normaltest(expr_values)
        
        print(f"\n🔍 Expressibility Distribution Analysis:")
        print(f"  Mean ± SD: {expr_values.mean():.4f} ± {expr_values.std():.4f}")
        print(f"  Median (IQR): {expr_values.median():.4f} ({expr_values.quantile(0.25):.4f}-{expr_values.quantile(0.75):.4f})")
        print(f"  Shapiro-Wilk Test: W={shapiro_stat:.4f}, p={shapiro_p:.4f} {'(Normal)' if shapiro_p > 0.05 else '(Non-normal)'}")
        print(f"  D'Agostino Test: stat={normal_stat:.4f}, p={normal_p:.4f} {'(Normal)' if normal_p > 0.05 else '(Non-normal)'}")
        
        # 2. Qubit Count Effect Analysis
        print(f"\n🔬 Qubit Count Effect Analysis:")
        qubit_groups = valid_data.groupby('n_qubits')['entropy_expressibility']
        
        for qubit_count, group in qubit_groups:
            if len(group) >= 3:
                print(f"  {qubit_count} qubits: n={len(group)}, mean={group.mean():.4f}, std={group.std():.4f}")
        
        # ANOVA test for qubit groups
        qubit_expr_groups = [group.values for _, group in qubit_groups if len(group) >= 3]
        if len(qubit_expr_groups) >= 2:
            f_stat, anova_p = stats.f_oneway(*qubit_expr_groups)
            print(f"  ANOVA Test: F={f_stat:.4f}, p={anova_p:.4f} {'(Significant)' if anova_p < 0.05 else '(Not significant)'}")
        
        # 3. Depth Effect Analysis
        print(f"\n📏 Circuit Depth Effect Analysis:")
        depth_groups = valid_data.groupby('depth')['entropy_expressibility']
        
        for depth, group in depth_groups:
            if len(group) >= 3:
                print(f"  Depth {depth}: n={len(group)}, mean={group.mean():.4f}, std={group.std():.4f}")
        
        # 4. Correlation Analysis
        print(f"\n🔗 Correlation Analysis:")
        
        correlations = [
            ('Qubit Count', 'n_qubits'),
            ('Circuit Depth', 'depth'),
            ('Classical Shadow Expr', 'classical_shadow_expr'),
            ('Robust Fidelity', 'robust_fidelity')
        ]
        
        for name, column in correlations:
            if column in valid_data.columns:
                data_col = valid_data[column]
                if column == 'classical_shadow_expr':
                    # Filter out zeros for meaningful correlation
                    mask = data_col > 0
                    if mask.sum() > 3:
                        pearson_r, pearson_p = pearsonr(data_col[mask], expr_values[mask])
                        spearman_r, spearman_p = spearmanr(data_col[mask], expr_values[mask])
                    else:
                        pearson_r = pearson_p = spearman_r = spearman_p = np.nan
                else:
                    pearson_r, pearson_p = pearsonr(data_col, expr_values)
                    spearman_r, spearman_p = spearmanr(data_col, expr_values)
                
                print(f"  {name}:")
                print(f"    Pearson: r={pearson_r:.4f}, p={pearson_p:.4f} {'*' if pearson_p < 0.05 else ''}")
                print(f"    Spearman: ρ={spearman_r:.4f}, p={spearman_p:.4f} {'*' if spearman_p < 0.05 else ''}")
        
        # 5. Effect Size Analysis
        print(f"\n📈 Effect Size Analysis (Cohen's d):")
        qubit_counts = sorted([qc for qc, group in qubit_groups if len(group) >= 3])
        
        for i in range(len(qubit_counts)):
            for j in range(i+1, len(qubit_counts)):
                q1, q2 = qubit_counts[i], qubit_counts[j]
                group1 = qubit_groups.get_group(q1).values
                group2 = qubit_groups.get_group(q2).values
                
                # Cohen's d
                pooled_std = np.sqrt(((len(group1)-1)*np.var(group1, ddof=1) + 
                                     (len(group2)-1)*np.var(group2, ddof=1)) / 
                                    (len(group1) + len(group2) - 2))
                cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
                
                effect_size = "negligible"
                if abs(cohens_d) > 0.8:
                    effect_size = "large"
                elif abs(cohens_d) > 0.5:
                    effect_size = "medium"
                elif abs(cohens_d) > 0.2:
                    effect_size = "small"
                
                print(f"  {q1} vs {q2} qubits: d={cohens_d:.4f} ({effect_size})")
        
        print("\n" + "="*80)
        print("Note: * indicates statistical significance (p < 0.05)")
        print("="*80)
        
        return valid_data

    def run_focused_analysis(self):
        """🎯 Run focused expressibility analysis only"""
        print("🚀 Starting Focused Expressibility Analysis!")
        
        # Load data
        self.load_mega_job_data()
        
        if self.df is None or len(self.df) == 0:
            print("❌ No data to analyze.")
            return
        
        # Create images directory
        os.makedirs("grid_circuits/images", exist_ok=True)
        
        # Run core expressibility analysis
        try:
            statistical_data = self.create_core_expressibility_analysis()
            print("✅ Core expressibility analysis complete!")
            
            print(f"\n📁 Generated file:")
            print(f"  - grid_circuits/images/core_expressibility_analysis.png")
            
            return statistical_data
            
        except Exception as e:
            print(f"⚠️ Analysis failed: {str(e)}")
            return None

    def export_to_csv(self, filename="quantum_expressibility_data.csv"):
        """Export core expressibility data to CSV (excluding circuit structure)"""
        if self.df is None or len(self.df) == 0:
            print("❌ No data to export.")
            return
        
        print("📊 Exporting expressibility data to CSV...")
        
        # Select core metrics (exclude circuit structure details)
        core_columns = [
            # Basic circuit properties
            'n_qubits',
            'depth', 
            'gate_count',
            'two_qubit_ratio',
            'two_qubit_gates',
            
            # Performance metrics
            'fidelity',
            'robust_fidelity', 
            'total_error_rate',
            
            # Expressibility metrics (핵심!)
            'classical_shadow_expr',
            'entropy_expressibility',
            
            # Measurement statistics (상태 분포 엔트로피가 표현력 역할)
            'entropy',
            'concentration',
            'entanglement_potential',
            
            # Derived complexity metrics
            'circuit_complexity',
            'performance_per_complexity'
        ]
        
        # Filter existing columns
        available_columns = [col for col in core_columns if col in self.df.columns]
        
        # Export selected data
        export_df = self.df[available_columns].copy()
        
        # Add some computed fields
        export_df['expressibility_ratio'] = export_df['entropy_expressibility'] / (export_df['classical_shadow_expr'] + 1e-10)
        export_df['complexity_per_qubit'] = export_df['circuit_complexity'] / export_df['n_qubits']
        export_df['performance_score'] = export_df['robust_fidelity'] * export_df['entropy_expressibility']
        
        # Sort by expressibility for easier analysis
        export_df = export_df.sort_values('entropy_expressibility', ascending=False)
        
        # Save to CSV
        csv_path = f"grid_circuits/images/{filename}"
        export_df.to_csv(csv_path, index=False, float_format='%.6f')
        
        print(f"✅ CSV exported: {csv_path}")
        print(f"   📊 Rows: {len(export_df)}")
        print(f"   📋 Columns: {len(export_df.columns)}")
        print(f"   🎯 Key columns: {available_columns[:10]}...")
        
        # Quick statistics
        if 'entropy_expressibility' in export_df.columns:
            valid_expr = export_df[export_df['entropy_expressibility'] > 0]
            print(f"   📈 Valid expressibility data: {len(valid_expr)}/{len(export_df)}")
            if len(valid_expr) > 0:
                print(f"   📊 Expressibility range: {valid_expr['entropy_expressibility'].min():.4f} - {valid_expr['entropy_expressibility'].max():.4f}")
        
        return csv_path

    def run_csv_export_analysis(self):
        """🎯 CSV 내보내기와 함께 데이터 파싱 디버깅 실행"""
        print("🚀 Starting CSV Export Analysis with Data Debugging!")
        
        # Load data with debugging
        self.load_mega_job_data()
        
        if not self.circuits_data:
            print("❌ No circuit data loaded. Checking file structure...")
            return
        
        print(f"\n🔍 Data Structure Debugging:")
        print(f"   Total circuits loaded: {len(self.circuits_data)}")
        
        # Check first circuit structure
        if self.circuits_data:
            first_circuit = self.circuits_data[0]
            print(f"   First circuit keys: {list(first_circuit.keys())}")
            
            if 'execution_result' in first_circuit:
                exec_result = first_circuit['execution_result']
                print(f"   Execution result keys: {list(exec_result.keys())}")
                
                if 'expressibility' in exec_result:
                    expr_data = exec_result['expressibility']
                    print(f"   Expressibility keys: {list(expr_data.keys())}")
                    print(f"   Sample expressibility values:")
                    for key, value in expr_data.items():
                        if isinstance(value, (int, float)):
                            print(f"     {key}: {value}")
                        elif isinstance(value, dict):
                            print(f"     {key} (dict): {list(value.keys())}")
                            # 중첩 dict의 내용도 출력
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, (int, float)):
                                    print(f"       {sub_key}: {sub_value}")
                        else:
                            print(f"     {key}: {type(value)}")
            
            # features 섹션도 확인
            if 'features' in first_circuit:
                features = first_circuit['features']
                print(f"   Features keys: {list(features.keys())}")
                
                # expressibility 관련 키 찾기
                expr_keys = [k for k in features.keys() if 'express' in k.lower() or 'entropy' in k.lower() or 'angle' in k.lower() or 'distance' in k.lower()]
                if expr_keys:
                    print(f"   Expressibility-related features:")
                    for key in expr_keys:
                        print(f"     {key}: {features[key]}")
        
        # 처음 3개 회로의 expressibility 구조 확인
        print(f"\n🔬 첫 3개 회로의 표현력 데이터 구조:")
        for i in range(min(3, len(self.circuits_data))):
            circuit = self.circuits_data[i]
            print(f"\n  === 회로 {i} ===")
            
            # execution_result.expressibility 확인
            exec_result = circuit.get('execution_result', {})
            expressibility = exec_result.get('expressibility', {})
            print(f"    expressibility 구조: {type(expressibility)}")
            
            if isinstance(expressibility, dict):
                print(f"    expressibility 키들: {list(expressibility.keys())}")
                for key, value in expressibility.items():
                    if isinstance(value, dict):
                        print(f"      {key}: {list(value.keys())}")
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                print(f"        {sub_key}: {sub_value}")
                    elif isinstance(value, (int, float)):
                        print(f"      {key}: {value}")
            
            # features에서도 찾기
            features = circuit.get('features', {})
            entropy_keys = [k for k in features.keys() if 'entropy' in k.lower()]
            if entropy_keys:
                print(f"    features의 entropy 관련:")
                for key in entropy_keys:
                    print(f"      {key}: {features[key]}")
        
        # Check DataFrame creation
        if self.df is None:
            print("❌ DataFrame creation failed. Checking data processing...")
            return
        
        print(f"\n📊 DataFrame Information:")
        print(f"   Shape: {self.df.shape}")
        print(f"   Columns: {list(self.df.columns)}")
        
        # Check expressibility data
        if 'entropy_expressibility' in self.df.columns:
            expr_data = self.df['entropy_expressibility']
            non_zero = expr_data[expr_data > 0]
            print(f"   Entropy expressibility:")
            print(f"     Total values: {len(expr_data)}")
            print(f"     Non-zero values: {len(non_zero)}")
            print(f"     Zero values: {len(expr_data) - len(non_zero)}")
            if len(non_zero) > 0:
                print(f"     Range: {non_zero.min():.6f} - {non_zero.max():.6f}")
                print(f"     Mean: {non_zero.mean():.6f}")
        
        if 'classical_shadow_expr' in self.df.columns:
            shadow_data = self.df['classical_shadow_expr']
            non_zero_shadow = shadow_data[shadow_data > 0]
            print(f"   Classical shadow expressibility:")
            print(f"     Total values: {len(shadow_data)}")
            print(f"     Non-zero values: {len(non_zero_shadow)}")
            if len(non_zero_shadow) > 0:
                print(f"     Range: {non_zero_shadow.min():.6f} - {non_zero_shadow.max():.6f}")
        
        # Create CSV
        csv_path = self.export_to_csv("quantum_mega_job_expressibility.csv")
        
        # Quick analysis
        if csv_path and self.df is not None:
            print(f"\n🎯 Quick Analysis Results:")
            
            # Basic statistics
            if 'entropy_expressibility' in self.df.columns and 'n_qubits' in self.df.columns:
                valid_data = self.df[self.df['entropy_expressibility'] > 0]
                if len(valid_data) > 0:
                    print(f"   📈 Valid expressibility data: {len(valid_data)} circuits")
                    
                    # Group by qubits
                    qubit_groups = valid_data.groupby('n_qubits')['entropy_expressibility'].agg(['count', 'mean', 'std'])
                    print(f"   🔢 Expressibility by qubit count:")
                    for qubits, stats in qubit_groups.iterrows():
                        print(f"     {qubits} qubits: n={stats['count']}, mean={stats['mean']:.4f}, std={stats['std']:.4f}")
                else:
                    print("   ❌ No valid expressibility data found!")
        
        return csv_path

# Existing QuantumDataAnalyzer class for compatibility
class QuantumDataAnalyzer(QuantumMegaJobAnalyzer):
    """Wrapper for existing class for compatibility"""
    
    def __init__(self, data_dir="grid_ansatz/grid_circuits/training_data"):
        # Redirect to mega job result directory
        mega_dir = "grid_circuits/mega_results"
        super().__init__(mega_dir)
        
    def load_all_data(self):
        """Existing method name for compatibility"""
        return self.load_mega_job_data()
    
    def run_quantum_analysis(self):
        """Existing method name for compatibility"""
        return self.run_focused_analysis()

if __name__ == "__main__":
    # CSV 내보내기 분석 실행
    analyzer = QuantumMegaJobAnalyzer()
    csv_path = analyzer.run_csv_export_analysis()
    
    if csv_path:
        print(f"\n🎉 CSV 내보내기 완료!")
        print(f"📁 파일 위치: {csv_path}")
        print(f"💡 이제 엑셀이나 다른 도구로 데이터를 분석할 수 있습니다.")
    else:
        print(f"\n❌ CSV 내보내기 실패")
        print(f"💡 데이터 파싱 문제를 확인하세요.") 