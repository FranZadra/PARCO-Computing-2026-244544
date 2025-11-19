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

# Colori per gli schedule
schedule_colors = {
    'sequential': '#1f77b4',
    'static': '#ff7f0e',
    'dynamic': '#2ca02c',
    'guided': '#d62728'
}

# Ordine degli schedule per la legenda
schedule_order = ['sequential', 'static', 'dynamic', 'guided']

plot_count = 0
best_configs = []

for matrix in matrices:
    # Filtra i dati per questa matrice (tutti i mode insieme)
    data = best_chunks[best_chunks['matrix'] == matrix]
    
    if data.empty:
        continue
    
    # Crea figura
    plt.figure(figsize=(12, 7))
    
    # Ottieni tutti i valori di thread presenti nel grafico
    all_thread_values = sorted(data['num_threads'].unique())
    
    # Prendi il sequenziale (mode='sequential' o schedule='none')
    seq_data = best_chunks[
        (best_chunks['matrix'] == matrix) & 
        (best_chunks['mode'] == 'sequential')
    ]

    seq_time = None
    
    # Disegna prima il sequenziale se esiste
    if not seq_data.empty:
        # Prendi la media dei tempi sequenziali (dovrebbero essere tutti uguali)
        seq_time = seq_data['perc90_elapsed_time'].mean()
        
        # Crea una linea orizzontale per tutti i valori di thread
        plt.axhline(
            y=seq_time,
            linewidth=2.5,
            label='sequential',
            color=schedule_colors['sequential'],
            linestyle='--',
            alpha=0.8,
            zorder=1
        )
    
    # Plot per gli altri schedule
    for schedule in ['static', 'dynamic', 'guided']:
        schedule_data = data[data['schedule'] == schedule]
        
        if schedule_data.empty:
            continue
        
        # Ordina per numero di thread
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
    
    plt.figure(figsize=(14, 8))
    
    matrices_list = best_configs_df['matrix'].tolist()
    speedups = best_configs_df['speedup'].tolist()
    
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
    plt.xlabel('Matrix', fontsize=12, fontweight='bold')
    plt.ylabel('Speedup', fontsize=12, fontweight='bold')
    plt.title('Speedup: sequential vs best parallel configuration', fontsize=14, fontweight='bold')
    plt.xticks(range(len(matrices_list)), matrices_list, rotation=45, ha='right')
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
    print("\n" + "="*70)
    print("BEST PARALLEL CONFIGURATIONS SUMMARY")
    print("="*70)
    for _, config in best_configs_df.iterrows():
        print(f"\n{config['matrix']}:")
        print(f"  Schedule: {config['schedule']}")
        print(f"  Threads: {int(config['num_threads'])}")
        print(f"  Chunk size: {int(config['chunk_size'])}")
        print(f"  Speedup: {config['speedup']:.2f}x")
        print(f"  Sequential time: {config['seq_time']:.5f} ms")
        print(f"  Parallel time: {config['parallel_time']:.5f} ms")