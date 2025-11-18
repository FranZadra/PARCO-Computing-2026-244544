import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_noperf = pd.read_csv('results/benchResults.csv')
#csv_perf = pd.read_csv('results/benchResults_perf.csv')

# 90th percentile for non-perf results
percentile_time = csv_noperf.groupby(
    ['matrix', 'mode', 'opt_level', 'schedule', 'chunk_size', 'num_threads'],
    sort=False
)['elapsed_time'].quantile(0.90).reset_index()
percentile_time.columns = [*percentile_time.columns[:-1], 'perc90_elapsed_time']

# 90th percentile for perf results
#percentile_perf = csv_perf.groupby(
#    ['matrix', 'mode', 'opt_level', 'schedule', 'chunk_size', 'num_threads'],
#    sort=False
#).agg({
 #   'elapsed_time': lambda x: x.quantile(0.90),
  #  'L1_miss_rate': lambda x: x.quantile(0.90),
   # 'LLC_miss_rate': lambda x: x.quantile(0.90),
#}).reset_index()

percentile_time = percentile_time.round(5)
#percentile_perf = percentile_perf.round(5)

percentile_time.to_csv('plots/benchResults_p90.csv', index=False)
#percentile_perf.to_csv('plots/benchResults_perf_p90.csv', index=False)

print("90th percentile saved to benchResults_p90.csv")
print("Perf analysis saved to benchResults_perf_p90.csv")

# ========== SELEZIONE MIGLIOR CHUNK SIZE ==========
# Per ogni combinazione di matrice, mode, opt_level, schedule, num_threads
# seleziona il chunk_size con il tempo minore
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
    
    # Disegna prima il sequenziale se esiste
    if not seq_data.empty:
        # Prendi la media dei tempi sequenziali (dovrebbero essere tutti uguali)
        seq_time = seq_data['perc90_elapsed_time'].mean()
        
        # Crea una linea orizzontale per tutti i valori di thread
        plt.plot(
            all_thread_values,
            [seq_time] * len(all_thread_values),
            marker='s',
            linewidth=2.5,
            markersize=9,
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
    
    # Configurazione grafico
    plt.xlabel('Number of Threads', fontsize=12, fontweight='bold')
    plt.ylabel('Time (ms)', fontsize=12, fontweight='bold')
    plt.title(f'{matrix}', fontsize=14, fontweight='bold')
    plt.xscale('log', base=2)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(title='Schedule', fontsize=10, title_fontsize=11)
    
    # Imposta i tick sull'asse x per mostrare le potenze di 2
    thread_values = sorted(data['num_threads'].unique())
    plt.xticks(thread_values, [str(int(x)) for x in thread_values])
    
    # Salva il grafico
    filename = f'plots/{matrix}_comparison.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")
    plot_count += 1

print("\n✓ All plots generated successfully!")
print(f"✓ Total plots created: {plot_count}")