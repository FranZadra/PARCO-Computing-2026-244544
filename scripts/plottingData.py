import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

if not os.path.exists('results/benchResults.csv'):
    raise FileNotFoundError("File 'results/benchResults.csv' not found")
if not os.path.exists('results/benchResults_perf.csv'):
    raise FileNotFoundError("File 'results/benchResults_perf.csv' not found")

csv_noperf = pd.read_csv('results/benchResults.csv')
csv_perf = pd.read_csv('results/benchResults_perf.csv')

# 90th percentile for non-perf results
percentile_time = csv_noperf.groupby(
    ['matrix', 'mode', 'opt_level', 'schedule', 'chunk_size', 'num_threads'],
    sort=False
)['elapsed_time'].quantile(0.90).reset_index()
percentile_time.columns = [*percentile_time.columns[:-1], 'perc90_elapsed_time']

# 90th percentile for perf results
percentile_perf = csv_perf.groupby(
    ['matrix', 'mode', 'opt_level', 'schedule', 'chunk_size', 'num_threads'],
    sort=False
).agg({
    'elapsed_time': lambda x: x.quantile(0.90),
    'L1_miss_rate': lambda x: x.quantile(0.90),
    'LLC_miss_rate': lambda x: x.quantile(0.90),
}).reset_index()

percentile_time = percentile_time.round(5)
percentile_perf = percentile_perf.round(5)

percentile_time.to_csv('plots/benchResults_p90.csv', index=False)
percentile_perf.to_csv('plots/benchResults_perf_p90.csv', index=False)

print("90th percentile saved to benchResults_p90.csv")
print("Perf analysis saved to benchResults_perf_p90.csv")

# select best chunk sizes for schedule - num_threads
best_chunks = percentile_time.loc[
    percentile_time.groupby(
        ['matrix', 'mode', 'opt_level', 'schedule', 'num_threads']
    )['perc90_elapsed_time'].idxmin()
].reset_index(drop=True)

best_chunks.to_csv('plots/benchResults_best_chunks.csv', index=False)
print("\nBest chunk sizes saved to benchResults_best_chunks.csv")

# plotting data in graphs
matrices = best_chunks['matrix'].unique()

schedule_colors = {
    'sequential': '#1f77b4',
    'static': '#ff7f0e',
    'dynamic': '#2ca02c',
    'guided': '#d62728'
}

schedule_order = ['sequential', 'static', 'dynamic', 'guided']

plot_count = 0
best_configs = []

for matrix in matrices:
    data = best_chunks[best_chunks['matrix'] == matrix]
    
    if data.empty:
        continue
    
    plt.figure(figsize=(12, 7))
    
    all_thread_values = sorted(data['num_threads'].unique())
    
    # retrieve sequential data
    seq_data = best_chunks[
        (best_chunks['matrix'] == matrix) & 
        (best_chunks['mode'] == 'sequential')
    ]

    seq_time = None
    
    if not seq_data.empty:
        seq_time = seq_data['perc90_elapsed_time'].mean()
        
        plt.axhline(
            y=seq_time,
            linewidth=2.5,
            label='sequential',
            color=schedule_colors['sequential'],
            linestyle='--',
            alpha=0.8,
            zorder=1
        )
    
    for schedule in ['static', 'dynamic', 'guided']:
        schedule_data = data[data['schedule'] == schedule]
        
        if schedule_data.empty:
            continue
        
        schedule_data = schedule_data.sort_values('num_threads')
        
        plt.plot(
            schedule_data['num_threads'],
            schedule_data['perc90_elapsed_time'],
            marker='o',
            linewidth=2,
            markersize=8,
            label=schedule,
            color=schedule_colors[schedule],
            zorder=2
        )
    
    # Find best parallel config for speedup calculation
    parallel_data = data[data['mode'] != 'sequential']
    if not parallel_data.empty and seq_time is not None:
        best_parallel = parallel_data.loc[parallel_data['perc90_elapsed_time'].idxmin()]
        speedup = seq_time / best_parallel['perc90_elapsed_time']
        best_configs.append({
            'matrix': matrix,
            'seq_time': seq_time,
            'parallel_time': best_parallel['perc90_elapsed_time'],
            'speedup': speedup,
            'schedule': best_parallel['schedule'],
            'num_threads': best_parallel['num_threads'],
            'chunk_size': best_parallel['chunk_size']
        })

    
    # Graph configuration
    plt.xlabel('Number of Threads', fontsize=12, fontweight='bold')
    plt.ylabel('Time (ms) [90th percentile]', fontsize=12, fontweight='bold')
    plt.title(f'Performance comparison for matrix {matrix}', fontsize=14, fontweight='bold')
    plt.xscale('log', base=2)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(title='Schedule', fontsize=10, title_fontsize=11)
    
    # 2^
    thread_values = sorted(data['num_threads'].unique())
    plt.xticks(thread_values, [str(int(x)) for x in thread_values])
    
    # Save plots
    filename = f'plots/{matrix}_comparison.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")
    plot_count += 1

print("\nAll plots generated successfully!")
print(f"Total plots created: {plot_count}")

# Speedup comparison plot
if best_configs:
    best_configs_df = pd.DataFrame(best_configs)
    
    plt.figure(figsize=(16, 8))
    
    matrices_list = best_configs_df['matrix'].tolist()
    speedups = best_configs_df['speedup'].tolist()
    
    # Create labels with matrix name and configuration details
    labels = []
    for _, config in best_configs_df.iterrows():
        label = f"{config['matrix']}\n{config['schedule']}\nchunk={int(config['chunk_size'])}\nthreads={int(config['num_threads'])}"
        labels.append(label)
    
    # Bar graph
    bars = plt.bar(range(len(matrices_list)), speedups, color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add speedup values on top of bars
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.1f}x',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # add horizontal line for sequential baseline
    plt.axhline(y=1, color='#1f77b4', linestyle='--', linewidth=2.5, 
                label='Sequential baseline', alpha=0.8)
    
    # graph configuration
    plt.xlabel('Best parallel configuration for each matrix', fontsize=12, fontweight='bold')
    plt.ylabel('Speedup', fontsize=12, fontweight='bold')
    plt.title('Speedup: sequential vs best parallel configuration', fontsize=14, fontweight='bold')
    plt.xticks(range(len(labels)), labels, rotation=0, ha='center', fontsize=9)
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.legend(fontsize=10)
    
    # save the plot
    plt.tight_layout()
    plt.savefig('plots/speedup_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nSpeedup comparison plot saved: plots/speedup_comparison.png")
    
    # save best configurations to CSV
    best_configs_df.to_csv('plots/best_parallel_configs.csv', index=False)
    print("Best parallel configurations saved to: plots/best_parallel_configs.csv")
    
    # print summary of best configurations
    print("\n" + "-"*50)
    print("Best parallel configurations for each matrix:")
    print("-"*50)
    for _, config in best_configs_df.iterrows():
        print(f"\n{config['matrix']}:")
        print(f"  Schedule: {config['schedule']}")
        print(f"  Threads: {int(config['num_threads'])}")
        print(f"  Chunk size: {int(config['chunk_size'])}")
        print(f"  Speedup: {config['speedup']:.2f}x")
        print(f"  Sequential time: {config['seq_time']:.5f} ms")
        print(f"  Parallel time: {config['parallel_time']:.5f} ms")

# CACHE analysis
print("\n" + "-"*50)
print("CACHE miss analysis for best configurations:")
print("\n" + "-"*50)

if best_configs:
    cache_analysis = []
    
    for config in best_configs:
        matrix = config['matrix']
        schedule = config['schedule']
        num_threads = config['num_threads']
        chunk_size = config['chunk_size']
        
        # sequential cache data
        seq_cache = percentile_perf[
            (percentile_perf['matrix'] == matrix) &
            (percentile_perf['mode'] == 'sequential')
        ]
        
        # best parallel cache data
        parallel_cache = percentile_perf[
            (percentile_perf['matrix'] == matrix) &
            (percentile_perf['schedule'] == schedule) &
            (percentile_perf['num_threads'] == num_threads) &
            (percentile_perf['chunk_size'] == chunk_size)
        ]
        
        if not seq_cache.empty and not parallel_cache.empty:
            seq_l1 = seq_cache['L1_miss_rate'].iloc[0]
            seq_llc = seq_cache['LLC_miss_rate'].iloc[0]
            par_l1 = parallel_cache['L1_miss_rate'].iloc[0]
            par_llc = parallel_cache['LLC_miss_rate'].iloc[0]
            
            cache_analysis.append({
                'matrix': matrix,
                'schedule': schedule,
                'num_threads': int(num_threads),
                'chunk_size': int(chunk_size),
                'speedup': config['speedup'],
                'seq_L1_miss': seq_l1,
                'par_L1_miss': par_l1,
                'L1_miss_increase': par_l1 - seq_l1,
                'L1_miss_ratio': par_l1 / seq_l1 if seq_l1 > 0 else 0,
                'seq_LLC_miss': seq_llc,
                'par_LLC_miss': par_llc,
                'LLC_miss_increase': par_llc - seq_llc,
                'LLC_miss_ratio': par_llc / seq_llc if seq_llc > 0 else 0
            })
    
    if cache_analysis:
        cache_df = pd.DataFrame(cache_analysis)
        cache_df.to_csv('plots/cache_analysis_best_configs.csv', index=False)
        print("\nCache analysis saved to: plots/cache_analysis_best_configs.csv")
        
        # print cache analysis summary
        print("\n" + "-"*70)
        print("Cache Miss Analysis for Best Configurations:")
        print("-"*70)
        for _, row in cache_df.iterrows():
            print(f"\n{row['matrix']} ({row['schedule']}, {row['num_threads']} threads, chunk={row['chunk_size']}):")
            print(f"  Speedup: {row['speedup']:.2f}x")
            print(f"  L1 Cache Miss Rate:")
            print(f"    Sequential: {row['seq_L1_miss']:.4f}")
            print(f"    Parallel:   {row['par_L1_miss']:.4f}")
            print(f"    Increase:   {row['L1_miss_increase']:.4f} ({row['L1_miss_ratio']:.2f}x)")
            print(f"  LLC Miss Rate:")
            print(f"    Sequential: {row['seq_LLC_miss']:.4f}")
            print(f"    Parallel:   {row['par_LLC_miss']:.4f}")
            print(f"    Increase:   {row['LLC_miss_increase']:.4f} ({row['LLC_miss_ratio']:.2f}x)")
        
        # Graphs
        
        # 1. Graphs L1 vs LLC miss rate for best configurations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        matrices_list = cache_df['matrix'].tolist()
        x_pos = np.arange(len(matrices_list))
        width = 0.35
        
        # L1 Miss Rate
        ax1.bar(x_pos - width/2, cache_df['seq_L1_miss'], width, 
                label='Sequential', color='#1f77b4', alpha=0.8, edgecolor='black')
        ax1.bar(x_pos + width/2, cache_df['par_L1_miss'], width,
                label='Best Parallel', color='#2ca02c', alpha=0.8, edgecolor='black')
        ax1.set_xlabel('Matrix', fontsize=12, fontweight='bold')
        ax1.set_ylabel('L1 Cache Miss Rate', fontsize=12, fontweight='bold')
        ax1.set_title('L1 Cache Miss Rate Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(matrices_list, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # LLC Miss Rate
        ax2.bar(x_pos - width/2, cache_df['seq_LLC_miss'], width,
                label='Sequential', color='#1f77b4', alpha=0.8, edgecolor='black')
        ax2.bar(x_pos + width/2, cache_df['par_LLC_miss'], width,
                label='Best Parallel', color='#2ca02c', alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Matrix', fontsize=12, fontweight='bold')
        ax2.set_ylabel('LLC Miss Rate', fontsize=12, fontweight='bold')
        ax2.set_title('LLC Miss Rate Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(matrices_list, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        plt.tight_layout()
        plt.savefig('plots/cache_miss_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\nCache miss comparison plot saved: plots/cache_miss_comparison.png")
        
        # 2. Scatter plot: Speedup vs Cache Miss Increase
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Speedup vs L1 miss increase
        ax1.scatter(cache_df['L1_miss_increase'], cache_df['speedup'], 
                   s=200, alpha=0.6, c=range(len(cache_df)), cmap='viridis', edgecolors='black', linewidth=1.5)
        for i, row in cache_df.iterrows():
            ax1.annotate(row['matrix'], 
                        (row['L1_miss_increase'], row['speedup']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax1.set_xlabel('L1 Miss Rate Increase', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Speedup', fontsize=12, fontweight='bold')
        ax1.set_title('Speedup vs L1 Cache Miss Increase', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No speedup')
        ax1.axvline(x=0, color='blue', linestyle='--', alpha=0.5, label='No miss increase')
        ax1.legend()
        
        # Speedup vs LLC miss increase
        ax2.scatter(cache_df['LLC_miss_increase'], cache_df['speedup'],
                   s=200, alpha=0.6, c=range(len(cache_df)), cmap='viridis', edgecolors='black', linewidth=1.5)
        for i, row in cache_df.iterrows():
            ax2.annotate(row['matrix'],
                        (row['LLC_miss_increase'], row['speedup']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax2.set_xlabel('LLC Miss Rate Increase', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Speedup', fontsize=12, fontweight='bold')
        ax2.set_title('Speedup vs LLC Miss Increase', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No speedup')
        ax2.axvline(x=0, color='blue', linestyle='--', alpha=0.5, label='No miss increase')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('plots/speedup_vs_cache_miss.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Speedup vs cache miss plot saved: plots/speedup_vs_cache_miss.png")
        
        # 3. Correlation analysis
        print("\n" + "-"*50)
        print("Correlation Analysis:")
        print("\n" + "-"*50)
        
        from scipy.stats import pearsonr, spearmanr
        
        # Speedup vs L1 miss increase correlataion
        if len(cache_df) > 1:
            pearson_l1, p_l1 = pearsonr(cache_df['L1_miss_increase'], cache_df['speedup'])
            spearman_l1, sp_l1 = spearmanr(cache_df['L1_miss_increase'], cache_df['speedup'])
            
            print(f"\nSpeedup vs L1 Miss Increase:")
            print(f"  Pearson correlation:  {pearson_l1:.3f} (p-value: {p_l1:.4f})")
            print(f"  Spearman correlation: {spearman_l1:.3f} (p-value: {sp_l1:.4f})")
            
            # Speedup vs LLC miss increase correlation
            pearson_llc, p_llc = pearsonr(cache_df['LLC_miss_increase'], cache_df['speedup'])
            spearman_llc, sp_llc = spearmanr(cache_df['LLC_miss_increase'], cache_df['speedup'])
            
            print(f"\nSpeedup vs LLC Miss Increase:")
            print(f"  Pearson correlation:  {pearson_llc:.3f} (p-value: {p_llc:.4f})")
            print(f"  Spearman correlation: {spearman_llc:.3f} (p-value: {sp_llc:.4f})")
            
            # Interpretation
            print("\n" + "-"*50)
            print("Interpretation:")
            print("-"*50)
            
            if abs(pearson_l1) < 0.3:
                print("• L1 cache miss increase shows WEAK correlation with speedup")
                print("  → L1 cache behavior is not a strong predictor of parallel performance")
            elif abs(pearson_l1) < 0.7:
                print("• L1 cache miss increase shows MODERATE correlation with speedup")
            else:
                print("• L1 cache miss increase shows STRONG correlation with speedup")
            
            if abs(pearson_llc) < 0.3:
                print("• LLC miss increase shows WEAK correlation with speedup")
                print("  → LLC contention is not a major bottleneck")
            elif abs(pearson_llc) < 0.7:
                print("• LLC miss increase shows MODERATE correlation with speedup")
            else:
                print("• LLC miss increase shows STRONG correlation with speedup")
                print("  → LLC contention significantly impacts parallel performance")

print("\n" + "-"*50)
print("Analysis complete!")
print("\n" + "-"*50)