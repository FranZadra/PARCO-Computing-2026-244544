import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
from scipy.optimize import curve_fit

# Create organized directory structure
base_dir = 'plots'
subdirs = ['comparison', 'speedup', 'scaling', 'roofline', 'cache']
for subdir in subdirs:
    os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
os.makedirs('results', exist_ok=True)

print("-"*50)
print("SpMV - Data analysis and Plots")
print("-"*50)

# ------ DATA REDUCTION ------
# STEP 1: 90th percentile

if not os.path.exists('results/benchResults.csv'):
    raise FileNotFoundError("File 'results/benchResults.csv' not found")
if not os.path.exists('results/benchResults_perf.csv'):
    raise FileNotFoundError("File 'results/benchResults_perf.csv' not found")

csv_noperf = pd.read_csv('results/benchResults.csv')
csv_perf = pd.read_csv('results/benchResults_perf.csv')

print(f"\n Loaded {len(csv_noperf)} performance measurements")
print(f" Loaded {len(csv_perf)} perf counter measurements")

# 90th percentile for performance metrics
groupby_cols = ['matrix', 'mode', 'opt_level', 'schedule', 'chunk_size', 'num_threads']

percentile_metrics = csv_noperf.groupby(groupby_cols, sort=False).agg({
    'elapsed_time': lambda x: x.quantile(0.90),
    'bandwidth_GB_s': lambda x: x.quantile(0.90),
    'gflops': lambda x: x.quantile(0.90)
}).reset_index()

percentile_metrics.columns = [*groupby_cols, 'p90_time', 'p90_bandwidth', 'p90_gflops']

# 90th percentile for cache metrics
percentile_perf = csv_perf.groupby(groupby_cols, sort=False).agg({
    'elapsed_time': lambda x: x.quantile(0.90),
    'L1_miss_rate': lambda x: x.quantile(0.90),
    'LLC_miss_rate': lambda x: x.quantile(0.90),
}).reset_index()

percentile_perf.columns = [*groupby_cols, 'p90_time_perf', 'p90_L1_miss', 'p90_LLC_miss']

# Round to 6 decimals
percentile_metrics = percentile_metrics.round(6)
percentile_perf = percentile_perf.round(6)

percentile_metrics.to_csv('plots/benchResults_p90.csv', index=False)
percentile_perf.to_csv('plots/benchResults_perf_p90.csv', index=False)

print("\n90th percentile saved to benchResults_p90.csv")
print("Perf analysis saved to benchResults_perf_p90.csv")

# STEP 2: Find best chunk sizes for each schedule-threads combination

best_chunks = percentile_metrics.loc[
    percentile_metrics.groupby(
        ['matrix', 'mode', 'opt_level', 'schedule', 'num_threads']
    )['p90_time'].idxmin()
].reset_index(drop=True)

best_chunks = best_chunks.round(6)
best_chunks.to_csv('plots/benchResults_best_chunks.csv', index=False)
print("\nBest chunk sizes saved to benchResults_best_chunks.csv")

# ------ PLOTTING SECTION ------
# PLOT 1: Performance Comparison (Sequential vs Parallel Schedules)

print("\n" + "-"*80)
print("Generating performance comparison plots...")
print("-"*80)

matrices = best_chunks['matrix'].unique()

schedule_colors = {
    'sequential': '#1f77b4',
    'static': '#ff7f0e',
    'dynamic': '#2ca02c',
    'guided': '#d62728'
}

plot_count = 0

for matrix in matrices:
    data = best_chunks[best_chunks['matrix'] == matrix]
    
    if data.empty:
        continue
    
    plt.figure(figsize=(12, 7))
    
    # Sequential baseline
    seq_data = best_chunks[
        (best_chunks['matrix'] == matrix) & 
        (best_chunks['mode'] == 'sequential')
    ]

    if not seq_data.empty:
        seq_time = seq_data['p90_time'].mean()
        
        plt.axhline(
            y=seq_time,
            linewidth=2.5,
            label='Sequential',
            color=schedule_colors['sequential'],
            linestyle='--',
            alpha=0.8,
            zorder=1
        )
    
    # Plot parallel schedules
    for schedule in ['static', 'dynamic', 'guided']:
        schedule_data = data[data['schedule'] == schedule]
        
        if schedule_data.empty:
            continue
        
        schedule_data = schedule_data.sort_values('num_threads')
        
        plt.plot(
            schedule_data['num_threads'],
            schedule_data['p90_time'],
            marker='o',
            linewidth=2,
            markersize=8,
            label=schedule.capitalize(),
            color=schedule_colors[schedule],
            zorder=2
        )
    
    plt.xlabel('Number of Threads', fontsize=12, fontweight='bold')
    plt.ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    plt.title(f'Performance Comparison: {matrix}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(title='Schedule', fontsize=10, title_fontsize=11)
    
    # Use linear scale with integer thread values (not log scale)
    thread_values = sorted(data['num_threads'].unique())
    plt.xscale('log', base=2)
    plt.xticks(thread_values, [str(int(x)) for x in thread_values])
    plt.xlim(min(thread_values) / 1.5, max(thread_values) * 1.5)
    plt.gca().get_xaxis().set_major_formatter(plt.ScalarFormatter())

    filename = os.path.join('plots', 'comparison', f'{matrix}_comparison.png')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    plot_count += 1

print(f"\nComparison plots created and saved in /comparison subfolder: {plot_count}")

# STEP 3: Calculate metrics for all configurations

print("\n" + "-"*80)
print("Computing scientific metrics (Speedup, Efficiency)...")
print("-"*80)

seq_data = best_chunks[best_chunks['mode'] == 'sequential'].copy()
par_data = best_chunks[best_chunks['mode'] != 'sequential'].copy()

seq_avg = seq_data.groupby('matrix').agg({
    'p90_time': 'mean',
    'p90_bandwidth': 'mean',
    'p90_gflops': 'mean'
}).reset_index()

seq_avg.columns = ['matrix', 'seq_time', 'seq_bandwidth', 'seq_gflops']

metrics_df = par_data.merge(seq_avg, on='matrix', how='left')

metrics_df['par_time'] = metrics_df['p90_time']
metrics_df['par_bandwidth'] = metrics_df['p90_bandwidth']
metrics_df['par_gflops'] = metrics_df['p90_gflops']

metrics_df['speedup'] = metrics_df['seq_time'] / metrics_df['par_time']
metrics_df['efficiency'] = metrics_df['speedup'] / metrics_df['num_threads']
metrics_df['bandwidth_speedup'] = metrics_df['par_bandwidth'] / metrics_df['seq_bandwidth']
metrics_df['gflops_speedup'] = metrics_df['par_gflops'] / metrics_df['seq_gflops']

metrics_df = metrics_df[[
    'matrix', 'schedule', 'num_threads', 'chunk_size',
    'seq_time', 'par_time', 'speedup', 'efficiency',
    'seq_bandwidth', 'par_bandwidth', 'bandwidth_speedup',
    'seq_gflops', 'par_gflops', 'gflops_speedup'
]].round(6)

print(f"Computed metrics for {len(metrics_df)} configurations")

metrics_df.to_csv('plots/scientific_metrics.csv', index=False)

# PLOT 2 - Speedup Comparison (Multiple Views)

print("\n" + "-"*80)
print("Generating speedup comparison plots...")
print("-"*80)

# View 1: Bar plot with MEDIAN ± IQR at BEST thread count
speedup_summary = []

for matrix in matrices:
    matrix_data = metrics_df[metrics_df['matrix'] == matrix]
    
    if matrix_data.empty:
        continue
    
    for schedule in ['static', 'dynamic', 'guided']:
        sched_data = matrix_data[matrix_data['schedule'] == schedule]
        
        if sched_data.empty:
            continue
        
        best_idx = sched_data['speedup'].idxmax()
        best_row = sched_data.loc[best_idx]
        best_threads = best_row['num_threads']
        
        best_thread_data = sched_data[sched_data['num_threads'] == best_threads]
        
        median_speedup = best_thread_data['speedup'].median()
        q1 = best_thread_data['speedup'].quantile(0.25)
        q3 = best_thread_data['speedup'].quantile(0.75)
        iqr = q3 - q1
        
        speedup_summary.append({
            'matrix': matrix,
            'schedule': schedule,
            'num_threads': best_threads,
            'median_speedup': median_speedup,
            'q1': q1,
            'q3': q3,
            'iqr': iqr
        })

speedup_summary_df = pd.DataFrame(speedup_summary).round(6)

if not speedup_summary_df.empty:
    fig, ax = plt.subplots(figsize=(16, 8))
    
    matrices_list = speedup_summary_df['matrix'].unique()
    schedules = ['static', 'dynamic', 'guided']
    x = np.arange(len(matrices_list))
    width = 0.25
    
    for i, schedule in enumerate(schedules):
        sched_data = speedup_summary_df[speedup_summary_df['schedule'] == schedule]
        sched_data = sched_data.set_index('matrix').reindex(matrices_list, fill_value=0)
        
        speedups = sched_data['median_speedup'].values
        errors = sched_data['iqr'].values / 2
        
        bars = ax.bar(x + i*width, speedups, width, 
                     yerr=errors,
                     label=schedule.capitalize(),
                     color=schedule_colors[schedule], 
                     alpha=0.8, edgecolor='black', linewidth=1.5,
                     capsize=5)
        
        for j, (bar, speedup, threads) in enumerate(zip(bars, speedups, sched_data['num_threads'].values)):
            if speedup > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + errors[j],
                       f'{speedup:.1f}x\n@{int(threads)}t',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.axhline(y=1, color='#1f77b4', linestyle='--', linewidth=2.5, 
              label='Sequential baseline', alpha=0.8)
    
    ax.set_xlabel('Matrix', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (median ± IQR at best thread count)', fontsize=12, fontweight='bold')
    ax.set_title('Speedup Comparison: Best Performance by Schedule\n(showing optimal thread count for each schedule)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(matrices_list, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.legend(fontsize=11, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('plots/speedup/speedup_comparison_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved: plots/speedup/speedup_comparison_bar.png")

# View 1b: Grouped bar plot showing speedup per thread count for each matrix
print("\n" + "-"*80)
print("Generating speedup by thread count plots...")
print("-"*80)

all_threads = sorted(metrics_df['num_threads'].unique())

for matrix in matrices:
    matrix_data = metrics_df[metrics_df['matrix'] == matrix]
    
    if matrix_data.empty:
        continue
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    schedules = ['static', 'dynamic', 'guided']
    thread_counts = sorted(matrix_data['num_threads'].unique())
    
    n_schedules = len(schedules)
    n_threads = len(thread_counts)
    
    x = np.arange(n_schedules)
    width = 0.8 / n_threads 
    
    thread_colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_threads))
    
    for i, thread_count in enumerate(thread_counts):
        thread_speedups = []
        
        for schedule in schedules:
            sched_thread_data = matrix_data[
                (matrix_data['schedule'] == schedule) & 
                (matrix_data['num_threads'] == thread_count)
            ]
            
            if not sched_thread_data.empty:
                thread_speedups.append(sched_thread_data['speedup'].median())
            else:
                thread_speedups.append(0)
        
        offset = (i - n_threads/2 + 0.5) * width
        bars = ax.bar(x + offset, thread_speedups, width,
                     label=f'{int(thread_count)} threads',
                     color=thread_colors[i],
                     alpha=0.8, edgecolor='black', linewidth=1)
        
        for bar, speedup in zip(bars, thread_speedups):
            if speedup > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{speedup:.1f}x',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, 
              label='Sequential baseline', alpha=0.6)
    
    ax.set_xlabel('Schedule', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (x)', fontsize=12, fontweight='bold')
    ax.set_title(f'Speedup by Thread Count: {matrix}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in schedules])
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.legend(fontsize=10, loc='upper left', ncol=2)
    
    plt.tight_layout()
    plt.savefig(f'plots/speedup/{matrix}_speedup_by_thread_grouped.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: plots/speedup/{matrix}_speedup_by_thread_grouped.png")

# PLOT 3: Strong Scaling with Amdahl

print("\n" + "-"*80)
print("Generating strong scaling plots with Amdahl's Law...")
print("-"*80)

def amdahl_speedup(p, f_parallel):
    return 1.0 / ((1 - f_parallel) + f_parallel / p)

for matrix in matrices:
    matrix_data = metrics_df[metrics_df['matrix'] == matrix]
    
    if matrix_data.empty:
        continue
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    schedules = ['static', 'dynamic', 'guided']
    
    # Estimate parallel fraction from best schedule
    best_schedule_data = matrix_data.loc[
        matrix_data.groupby('num_threads')['speedup'].idxmax()
    ].sort_values('num_threads')
    
    if len(best_schedule_data) >= 3:
        threads_fit = best_schedule_data['num_threads'].values
        speedup_fit = best_schedule_data['speedup'].values
        
        try:
            popt, _ = curve_fit(amdahl_speedup, threads_fit, speedup_fit, 
                               bounds=(0, 1), p0=[0.9])
            f_parallel_estimated = popt[0]
        except:
            max_speedup = speedup_fit.max()
            max_threads = threads_fit[speedup_fit.argmax()]
            f_parallel_estimated = (max_speedup - 1) / (max_speedup * (1 - 1/max_threads))
            f_parallel_estimated = max(0, min(f_parallel_estimated, 1))
    else:
        f_parallel_estimated = 0.5
    
    # Generate smooth Amdahl curve
    all_threads = sorted(matrix_data['num_threads'].unique())
    amdahl_curve = [amdahl_speedup(p, f_parallel_estimated) for p in all_threads]
    
    # Plot speedup
    for schedule in schedules:
        sched_data = matrix_data[matrix_data['schedule'] == schedule].sort_values('num_threads')
        
        if sched_data.empty:
            continue
        
        threads = sched_data['num_threads']
        speedups = sched_data['speedup']
        
        ax.plot(threads, speedups, 'o-', linewidth=2, markersize=8, 
                label=schedule.capitalize(), color=schedule_colors[schedule])
    
    ax.plot(all_threads, amdahl_curve, 's--', linewidth=2.5, markersize=7,
            label=f"Amdahl (f_par={f_parallel_estimated:.1%})", 
            color='purple', alpha=0.7, zorder=10)
    
    max_threads = matrix_data['num_threads'].max()
    ideal_threads = np.array([1, 2, 4, 8, 16, 32, 64])
    ideal_threads = ideal_threads[ideal_threads <= max_threads]
    ax.plot(ideal_threads, ideal_threads, 'k:', linewidth=2, 
            label='Ideal (linear)', alpha=0.4)
    
    ax.axhline(y=1, color='#1f77b4', linestyle='--', linewidth=1.5,
              label='Sequential baseline', alpha=0.5)
    
    ax.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (x)', fontsize=12, fontweight='bold')
    ax.set_title(f'Strong Scaling - Speedup: {matrix}', fontsize=14, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=9, loc='upper left')
    
    thread_values = sorted(matrix_data['num_threads'].unique())
    ax.set_xticks(thread_values)
    ax.set_xticklabels([str(int(x)) for x in thread_values])
    
    y_ticks = [1, 2, 4, 8, 16, 32, 64, 128]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(y) for y in y_ticks])
    
    plt.tight_layout()
    plt.savefig(f'plots/scaling/{matrix}_strong_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved: plots/scaling/{matrix}_strong_scaling.png")
    
    # Print analysis
    serial_fraction = 1 - f_parallel_estimated
    max_theoretical_speedup = 1 / serial_fraction if serial_fraction > 0 else float('inf')
    
    print(f"\n   {matrix}:")
    print(f"     Parallel fraction (f): {f_parallel_estimated:.1%}")
    print(f"     Serial fraction (1-f): {serial_fraction:.1%}")
    print(f"     Max theoretical speedup: {max_theoretical_speedup:.1f}x")

# PLOT 4 - Roofline Model

print("\n" + "-"*80)
print("Generating Roofline Model plots...")
print("-"*80)

# Hardware specs, official intel for Intel Xeon Gold 6252N
PEAK_BANDWIDTH_GB_s = 131
PEAK_GFLOPS = 1075.2 

print(f"\n Hardware specs:")
print(f"   Peak Memory Bandwidth: {PEAK_BANDWIDTH_GB_s} GB/s")
print(f"   Peak Compute: {PEAK_GFLOPS} GFLOPS\n")

for matrix in matrices:
    matrix_data = metrics_df[metrics_df['matrix'] == matrix]
    matrix_data = matrix_data.copy()
    
    if matrix_data.empty:
        continue
    
    # Calculate Arithmetic Intensity = FLOPS per Byte
    matrix_data['arith_intensity'] = matrix_data['par_gflops'] / matrix_data['par_bandwidth']
    
    # Get best config per schedule
    best_per_schedule = matrix_data.loc[
        matrix_data.groupby('schedule')['par_gflops'].idxmax()
    ]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Roofline: Performance = min(AI × Bandwidth, Peak_Compute)
    ai_range = np.logspace(-2, 2, 1000)
    roofline = np.minimum(ai_range * PEAK_BANDWIDTH_GB_s, PEAK_GFLOPS)
    
    ax.plot(ai_range, roofline, 'k-', linewidth=3, label='Roofline', zorder=1)
    
    ridge_ai = PEAK_GFLOPS / PEAK_BANDWIDTH_GB_s
    ax.axvline(x=ridge_ai, color='r', linestyle='--', linewidth=2, alpha=0.5, zorder=1)
    
    ax.axvspan(ai_range.min(), ridge_ai, alpha=0.15, color='blue')
    ax.axvspan(ridge_ai, ai_range.max(), alpha=0.15, color='green')
    
    ax.text(ridge_ai * 0.1, PEAK_GFLOPS * 0.5, 'MEMORY-BOUND\n(Bottleneck: Bandwidth)',
           fontsize=11, fontweight='bold', ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.text(ridge_ai * 10, PEAK_GFLOPS * 0.5, 'COMPUTE-BOUND\n(Bottleneck: CPU)',
           fontsize=11, fontweight='bold', ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    offset_angles = [0, 45, 90]  
    
    for idx, (_, row) in enumerate(best_per_schedule.iterrows()):
        schedule = row['schedule']
        ai = row['arith_intensity']
        perf = row['par_gflops']
        
        ax.scatter(ai, perf, s=300, alpha=0.8, 
                  color=schedule_colors.get(schedule, '#1f77b4'),
                  edgecolors='black', linewidth=2.5, zorder=5)
        
        angle = offset_angles[idx % len(offset_angles)]
        offset_x = 30 * np.cos(np.radians(angle))
        offset_y = 30 * np.sin(np.radians(angle))
        
        ax.annotate(f"{schedule.upper()}\n{perf:.1f} GFLOPS\nAI={ai:.2f}",
                   xy=(ai, perf),
                   xytext=(offset_x, offset_y), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor=schedule_colors.get(schedule, '#1f77b4'),
                            linewidth=2, alpha=0.9),
                   arrowprops=dict(arrowstyle='->', lw=1.5,
                                  color=schedule_colors.get(schedule, '#1f77b4')))
    
    ax.set_xlabel('Arithmetic Intensity (FLOPS/Byte)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance (GFLOPS)', fontsize=12, fontweight='bold')
    ax.set_title(f'Roofline Model: {matrix}', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    
    explanation = (
        f"Ridge Point: AI = {ridge_ai:.2f} FLOPS/Byte\n"
        f"• Left of ridge → Need MORE bandwidth\n"
        f"• Right of ridge → Need MORE compute\n"
        f"• Below roofline → Can optimize further"
    )
    ax.text(0.02, 0.98, explanation, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'plots/roofline/{matrix}_roofline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: plots/roofline/{matrix}_roofline.png")

# STEP 8: Cache Performance Analysis

print("\n" + "-"*80)
print("Cache performance data analysis...")
print("-"*80)

cache_metrics = []

for _, row in metrics_df.iterrows():
    matrix = row['matrix']
    schedule = row['schedule']
    threads = row['num_threads']
    chunk = row['chunk_size']
    
    seq_cache = percentile_perf[
        (percentile_perf['matrix'] == matrix) &
        (percentile_perf['mode'] == 'sequential')
    ]
    
    par_cache = percentile_perf[
        (percentile_perf['matrix'] == matrix) &
        (percentile_perf['schedule'] == schedule) &
        (percentile_perf['num_threads'] == threads) &
        (percentile_perf['chunk_size'] == chunk)
    ]
    
    if not seq_cache.empty and not par_cache.empty:
        cache_entry = row.to_dict()
        cache_entry['seq_L1_miss'] = seq_cache['p90_L1_miss'].mean()
        cache_entry['par_L1_miss'] = par_cache['p90_L1_miss'].iloc[0]
        cache_entry['seq_LLC_miss'] = seq_cache['p90_LLC_miss'].mean()
        cache_entry['par_LLC_miss'] = par_cache['p90_LLC_miss'].iloc[0]
        cache_entry['L1_miss_increase'] = cache_entry['par_L1_miss'] - cache_entry['seq_L1_miss']
        cache_entry['LLC_miss_increase'] = cache_entry['par_LLC_miss'] - cache_entry['seq_LLC_miss']
        cache_entry['L1_miss_ratio'] = cache_entry['par_L1_miss'] / cache_entry['seq_L1_miss'] if cache_entry['seq_L1_miss'] > 0 else 1
        cache_entry['LLC_miss_ratio'] = cache_entry['par_LLC_miss'] / cache_entry['seq_LLC_miss'] if cache_entry['seq_LLC_miss'] > 0 else 1
        cache_metrics.append(cache_entry)

cache_df = pd.DataFrame(cache_metrics).round(6)
cache_df.to_csv('plots/metrics_with_cache.csv', index=False)
print(f" Merged cache data for {len(cache_df)} configurations")

# PLOT 5 - Cache Analysis

print("\n" + "-"*80)
print("Generating cache analysis plots...")
print("-"*80)

if not cache_df.empty:
    # Get best config per matrix
    best_configs = cache_df.loc[cache_df.groupby('matrix')['speedup'].idxmax()]
    
    # Plot 1: Cache Miss RATIO (Parallel/Sequential)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    matrices_list = best_configs['matrix'].tolist()
    x_pos = np.arange(len(matrices_list))
    
    # L1 Miss RATIO
    l1_ratio = best_configs['L1_miss_ratio'].values
    colors_l1 = ['red' if x > 1.2 else 'orange' if x > 1.0 else 'green' for x in l1_ratio]
    
    bars1 = ax1.bar(x_pos, l1_ratio, color=colors_l1, alpha=0.7, 
                    edgecolor='black', linewidth=1.5)
    ax1.axhline(y=1.0, color='black', linestyle='-', linewidth=2, 
                label='No degradation (1.0x)')
    ax1.axhline(y=1.2, color='orange', linestyle='--', linewidth=1.5, 
                alpha=0.5, label='Warning threshold (1.2x)')
    
    ax1.set_ylabel('L1 Cache Miss Ratio\n(Parallel / Sequential)', 
                   fontsize=11, fontweight='bold')
    ax1.set_xlabel('Matrix', fontsize=11, fontweight='bold')
    ax1.set_title('L1 Cache Degradation Factor\n(>1.0 = worse, <1.0 = better)', 
                 fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(matrices_list, rotation=45, ha='right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(fontsize=10)
    
    for bar, val in zip(bars1, l1_ratio):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}x',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # LLC Miss RATIO
    llc_ratio = best_configs['LLC_miss_ratio'].values
    colors_llc = ['red' if x > 1.2 else 'orange' if x > 1.0 else 'green' for x in llc_ratio]
    
    bars2 = ax2.bar(x_pos, llc_ratio, color=colors_llc, alpha=0.7,
                    edgecolor='black', linewidth=1.5)
    ax2.axhline(y=1.0, color='black', linestyle='-', linewidth=2,
                label='No degradation (1.0x)')
    ax2.axhline(y=1.2, color='orange', linestyle='--', linewidth=1.5,
                alpha=0.5, label='Warning threshold (1.2x)')
    
    ax2.set_ylabel('LLC Miss Ratio\n(Parallel / Sequential)', 
                   fontsize=11, fontweight='bold')
    ax2.set_xlabel('Matrix', fontsize=11, fontweight='bold')
    ax2.set_title('LLC Cache Degradation Factor\n(>1.0 = worse, <1.0 = better)', 
                 fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(matrices_list, rotation=45, ha='right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=10)
    
    for bar, val in zip(bars2, llc_ratio):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}x',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/cache/cache_degradation_ratio.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved: plots/cache/cache_degradation_ratio.png")
    
    # Plot 2: Speedup vs Cache Degradation
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(matrices_list))
    width = 0.3
    
    speedups = best_configs['speedup'].values
    l1_ratios = best_configs['L1_miss_ratio'].values
    llc_ratios = best_configs['LLC_miss_ratio'].values
    
    bars1 = ax.bar(x - width, speedups, width, label='Speedup Achieved',
                  color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x, l1_ratios, width, label='L1 Miss Ratio (par/seq)',
                  color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x + width, llc_ratios, width, label='LLC Miss Ratio (par/seq)',
                  color='#d62728', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.axhline(y=1.0, color='black', linestyle='-', linewidth=2, 
              alpha=0.5, label='Baseline (1.0x)')
    
    # Add values on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Matrix', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ratio (relative to baseline)', fontsize=12, fontweight='bold')
    ax.set_title('Performance vs Cache Degradation\n' + 
                '(Ideal: High speedup + Low cache ratio)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(matrices_list, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('plots/cache/speedup_vs_cache_ratio.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved: plots/cache/speedup_vs_cache_ratio.png")
    
    # Correlation analysis (text only, no plot)
    if len(best_configs) > 2:
        print("\n" + "-"*50)
        print("CORRELATION ANALYSIS: Cache Impact vs Speedup")
        print("-"*50)
        
        # Negative correlation expected: higher miss ratio → lower speedup
        corr_l1, p_l1 = pearsonr(best_configs['L1_miss_ratio'], best_configs['speedup'])
        corr_llc, p_llc = pearsonr(best_configs['LLC_miss_ratio'], best_configs['speedup'])
        
        print(f"\nSpeedup vs L1 Miss Ratio:")
        print(f"  Pearson r = {corr_l1:.3f} (p-value = {p_l1:.4f})")
        if corr_l1 < -0.5:
            print("  STRONG NEGATIVE correlation")
            print("  -> L1 cache degradation SIGNIFICANTLY limits speedup")
            print("  -> Consider: better data locality, cache-aware scheduling")
        elif corr_l1 < -0.3:
            print("  MODERATE NEGATIVE correlation")
            print("  -> L1 cache has SOME impact on speedup")
        else:
            print("  WEAK correlation")
            print("  -> L1 cache is NOT a major bottleneck")
        
        print(f"\nSpeedup vs LLC Miss Ratio:")
        print(f"  Pearson r = {corr_llc:.3f} (p-value = {p_llc:.4f})")
        if corr_llc < -0.5:
            print("  STRONG NEGATIVE correlation")
            print("  -> LLC contention SIGNIFICANTLY limits speedup")
            print("  -> Consider: reducing memory footprint, NUMA-aware allocation")
        elif corr_llc < -0.3:
            print("  MODERATE NEGATIVE correlation")
            print("  -> LLC has SOME impact on speedup")
        else:
            print("  WEAK correlation")
            print("  -> LLC is NOT a major bottleneck")
        
        print("\n" + "-"*50)
        print("Cache Summary (Best Configurations):")
        print("-"*50)
        print(f"{'Matrix':<20} {'Speedup':>10} {'L1 Ratio':>12} {'LLC Ratio':>12}")
        print("-"*50)
        for _, row in best_configs.iterrows():
            print(f"{row['matrix']:<20} {row['speedup']:>9.2f}x {row['L1_miss_ratio']:>11.2f}x {row['LLC_miss_ratio']:>11.2f}x")

# Summary

print("\n" + "-"*50)
print("Performance Summary")
print("-"*50)

best_overall = metrics_df.loc[metrics_df.groupby('matrix')['speedup'].idxmax()]
summary_table = best_overall[['matrix', 'schedule', 'num_threads', 'chunk_size', 
                              'speedup', 'efficiency', 'gflops_speedup', 
                              'bandwidth_speedup']].copy()

print("\nBest Configuration per Matrix:")
print(summary_table.to_string(index=False))

summary_table.to_csv('plots/best_overall_performance_summary.csv', index=False)

print("\n" + "-"*50)
print("Data analysis completed!")
print("-"*50)