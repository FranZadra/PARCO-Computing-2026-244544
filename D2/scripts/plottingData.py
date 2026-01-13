import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

results_dir = 'results'
plots_dir = 'plots'

# Results subfolders
subdirs = {
    'data': os.path.join(plots_dir, 'data_reduction'),
    'strong_speedup': os.path.join(plots_dir, 'speedup_strong'),
    'strong_comp_comm': os.path.join(plots_dir, 'comp_vs_comm_strong'),
    'strong_comm_volume': os.path.join(plots_dir, 'comm_volume_strong'),
    'weak': os.path.join(plots_dir, 'weak_scaling'),
    'comparison': os.path.join(plots_dir, 'comparison')
}

for subdir in subdirs.values():
    os.makedirs(subdir, exist_ok=True)

# Data reduction: 90th percentile
def calculate_90th_percentile(df):
    group_cols = ['num_procs', 'rank']
    if 'matrix' in df.columns:
        group_cols.append('matrix')
    
    percentile_data = df.groupby(group_cols).agg({
        'elapsed_time': lambda x: np.percentile(x, 90),
        'comm_time': lambda x: np.percentile(x, 90),
        'local_nz': 'first',
        'ghost_entries': 'first',
        'local_flops': 'first'
    }).reset_index()
    
    # Computation time
    percentile_data['comp_time'] = percentile_data['elapsed_time'] - percentile_data['comm_time']
    
    return percentile_data

# Amdahl law for strong scaling
def amdahl_law(p, parallel_fraction):
    serial_fraction = 1 - parallel_fraction
    speedup = 1 / (serial_fraction + parallel_fraction / p)
    return speedup

def estimate_parallel_fraction(procs, times):
    if len(procs) < 2:
        return 0.95  
    
    best_fraction = 0.95
    best_error = float('inf')
    
    baseline = times[0]
    actual_speedup = baseline / times
    
    for frac in np.linspace(0.5, 0.99, 50):
        predicted_speedup = np.array([amdahl_law(p, frac) for p in procs])
        error = np.sum((actual_speedup - predicted_speedup) ** 2)
        if error < best_error:
            best_error = error
            best_fraction = frac
    
    return best_fraction

# Gustavson law for weak scaling
def gustafson_law(p, serial_fraction):
    speedup = serial_fraction + p * (1 - serial_fraction)
    return speedup

def estimate_serial_fraction_weak(procs, times):
    if len(procs) < 2:
        return 0.05 
    
    baseline = times[0]
    
    best_fraction = 0.05
    best_error = float('inf')
    
    for s_frac in np.linspace(0.001, 0.3, 100):
        predicted_speedup = np.array([gustafson_law(p, s_frac) for p in procs])
        actual_scaled_speedup = procs * (baseline / times)
        error = np.sum((actual_scaled_speedup - predicted_speedup) ** 2)
        if error < best_error:
            best_error = error
            best_fraction = s_frac
    
    return best_fraction

# Load CSV results
print("Loading results...")
strong_df = pd.read_csv(os.path.join(results_dir, 'strong_scaling_all.csv'))
weak_df = pd.read_csv(os.path.join(results_dir, 'weak_scaling_all.csv'))

# STEP 1: 90th percentile
strong_90th = calculate_90th_percentile(strong_df)
weak_90th = calculate_90th_percentile(weak_df)

strong_90th.to_csv(os.path.join(subdirs['data'], 'strong_scaling_90th.csv'), index=False)
weak_90th.to_csv(os.path.join(subdirs['data'], 'weak_scaling_90th.csv'), index=False)

print(f"MPI Data reduction OK, 90 percentile calculated")

# Loading MPI+OpenMP
strong_hybrid_df = pd.read_csv(os.path.join(results_dir, 'strong_scaling_hybrid.csv'))
weak_hybrid_df   = pd.read_csv(os.path.join(results_dir, 'weak_scaling_hybrid.csv'))

strong_hybrid_90th = calculate_90th_percentile(strong_hybrid_df)
weak_hybrid_90th   = calculate_90th_percentile(weak_hybrid_df)

# Salva anche questi
strong_hybrid_90th.to_csv(os.path.join(subdirs['data'], 'strong_scaling_hybrid_90th.csv'), index=False)
weak_hybrid_90th.to_csv(os.path.join(subdirs['data'], 'weak_scaling_hybrid_90th.csv'), index=False)

print(f"MPI+OpenMP Data reduction OK, 90 percentile calculated")


# Aggregatin data
strong_agg = strong_90th.groupby(['num_procs', 'matrix']).agg({
    'elapsed_time': 'max',
    'comm_time': 'mean',
    'comp_time': 'mean',
    'local_nz': ['min', 'mean', 'max'],
    'ghost_entries': ['min', 'mean', 'max']
}).reset_index()

strong_agg.columns = ['num_procs', 'matrix', 'elapsed_time', 'comm_time', 'comp_time',
                      'local_nz_min', 'local_nz_mean', 'local_nz_max',
                      'ghost_entries_min', 'ghost_entries_mean', 'ghost_entries_max']

strong_hybrid_agg = strong_hybrid_90th.groupby(['num_procs', 'matrix']).agg({
    'elapsed_time': 'max',
    'comm_time':   'mean',
    'comp_time':   'mean',
    'local_nz':    ['min', 'mean', 'max'],
    'ghost_entries': ['min', 'mean', 'max']
}).reset_index()

strong_hybrid_agg.columns = ['num_procs', 'matrix', 'elapsed_time', 'comm_time', 'comp_time',
                             'local_nz_min', 'local_nz_mean', 'local_nz_max',
                             'ghost_entries_min', 'ghost_entries_mean', 'ghost_entries_max']

weak_agg = weak_90th.groupby(['num_procs']).agg({
    'elapsed_time': 'max',
    'comm_time': 'mean',
    'comp_time': 'mean',
    'local_nz': ['min', 'mean', 'max'],
    'ghost_entries': ['min', 'mean', 'max']
}).reset_index()

weak_agg.columns = ['num_procs', 'elapsed_time', 'comm_time', 'comp_time',
                    'local_nz_min', 'local_nz_mean', 'local_nz_max',
                    'ghost_entries_min', 'ghost_entries_mean', 'ghost_entries_max']

weak_hybrid_agg = weak_hybrid_90th.groupby(['num_procs']).agg({
    'elapsed_time': 'max',
    'comm_time':   'mean',
    'comp_time':   'mean',
    'local_nz':    ['min', 'mean', 'max'],
    'ghost_entries': ['min', 'mean', 'max']
}).reset_index()

weak_hybrid_agg.columns = ['num_procs', 'elapsed_time', 'comm_time', 'comp_time',
                           'local_nz_min', 'local_nz_mean', 'local_nz_max',
                           'ghost_entries_min', 'ghost_entries_mean', 'ghost_entries_max']

matrices = strong_agg['matrix'].unique()
print(f"\nMatrici trovate: {matrices}")

# ------ PLOTTING SECTION ------
# PLOT 1: Strong Scaling - Speedup with Amdahl's Law
print("\n" + "-"*80)
print("Generating Strong Scaling Speedup Plots with Amdahl's Law")
print("-"*80)

for matrix in matrices:
    fig, ax = plt.subplots(figsize=(10, 7))
    
    matrix_data = strong_agg[strong_agg['matrix'] == matrix].sort_values('num_procs')
    procs = matrix_data['num_procs'].values
    times = matrix_data['elapsed_time'].values
    
    if len(times) == 0:
        continue
    
    baseline = times[0]
    speedup = baseline / times
    
    parallel_frac = estimate_parallel_fraction(procs, times)
    
    # Amdahl calculation
    procs_continuous = np.logspace(0, np.log2(procs[-1]), 100, base=2)
    amdahl_speedup = np.array([amdahl_law(p, parallel_frac) for p in procs_continuous])
    
    # Plot
    ax.plot(procs, speedup, marker='o', label='Actual Speedup', linewidth=2.5, markersize=10, color='#2E86AB')
    ax.plot(procs, procs / procs[0], 'k--', label='Ideal (Linear)', linewidth=2)
    ax.plot(procs_continuous, amdahl_speedup, '-.', label=f'Amdahl (p={parallel_frac:.2f})', 
            linewidth=2, color='#E63946')
    
    ax.set_xlabel('Number of Processes', fontsize=13, fontweight='bold')
    ax.set_ylabel('Speedup', fontsize=13, fontweight='bold')
    ax.set_title(f'Strong Scaling - Speedup with Amdahl\'s Law ({matrix})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    
    plt.tight_layout()
    filename = os.path.join(subdirs['strong_speedup'], f'speedup_{matrix}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

print(f"\nStrong scaling plots created and saved in /speedup_strong subfolder")

# PLOT 2: Strong Scaling - Computation vs Communication
print("\n" + "-"*80)
print("Generating Strong Scaling Plots - Computation vs Communication")
print("-"*80)

for matrix in matrices:
    fig, ax = plt.subplots(figsize=(10, 7))
    
    matrix_data = strong_agg[strong_agg['matrix'] == matrix].sort_values('num_procs')
    procs = matrix_data['num_procs'].values
    comp_times = matrix_data['comp_time'].values
    comm_times = matrix_data['comm_time'].values
    
    x = np.arange(len(procs))
    width = 0.6
    ax.bar(x, comp_times, width, label='Computation Time', color='steelblue')
    ax.bar(x, comm_times, width, bottom=comp_times, label='Communication Time', color='coral')
    ax.set_xlabel('Number of Processes', fontsize=13, fontweight='bold')
    ax.set_ylabel('Execution Time (s)', fontsize=13, fontweight='bold')
    ax.set_title(f'Computation vs Communication Time ({matrix})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(procs)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filename = os.path.join(subdirs['strong_comp_comm'], f'comp_vs_comm_{matrix}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

print(f"\nStrong scaling plots created and saved in /comp_vs_comm_strong subfolder")

print("\n" + "-"*80)
print("Generating Strong Scaling Plots - Computation vs Communication with MPI+OpenMP")
print("-"*80)

for matrix in matrices:
    # MPI data
    mpi_data = strong_agg[strong_agg['matrix'] == matrix].sort_values('num_procs')
    # MPI+OpenMP data
    hyb_data = strong_hybrid_agg[strong_hybrid_agg['matrix'] == matrix].sort_values('num_procs')

    common = np.intersect1d(mpi_data['num_procs'].values,
                            hyb_data['num_procs'].values)
    if len(common) == 0:
        continue

    mpi_data = mpi_data[mpi_data['num_procs'].isin(common)].sort_values('num_procs')
    hyb_data = hyb_data[hyb_data['num_procs'].isin(common)].sort_values('num_procs')

    procs = common

    comp_mpi = mpi_data['comp_time'].values
    comm_mpi = mpi_data['comm_time'].values

    comp_hyb = hyb_data['comp_time'].values
    comm_hyb = hyb_data['comm_time'].values

    x = np.arange(len(procs))
    width = 0.35 

    fig, ax = plt.subplots(figsize=(10, 7))

    # MPI Bars
    ax.bar(x - width/2, comp_mpi, width,
           label='MPI - Computation', color='steelblue')
    ax.bar(x - width/2, comm_mpi, width,
           bottom=comp_mpi, label='MPI - Communication', color='coral')

    # MPI+OpenMP Bars
    ax.bar(x + width/2, comp_hyb, width, label='MPI+OpenMP - Computation', color='seagreen')
    ax.bar(x + width/2, comm_hyb, width, bottom=comp_hyb, label='MPI+OpenMP - Communication', color='goldenrod')

    ax.set_xlabel('Number of Processes', fontsize=13, fontweight='bold')
    ax.set_ylabel('Execution Time (s)', fontsize=13, fontweight='bold')
    ax.set_title(f'Computation vs Communication Time - MPI vs MPI+OpenMP ({matrix})', fontsize=14, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(procs)

    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filename = os.path.join(subdirs['strong_comp_comm'],
                            f'comp_vs_comm_mpi_vs_hybrid_{matrix}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

print(f"\nStrong scaling plots created and saved in /comp_vs_comm_strong subfolder")

# PLOT 3: Strong Scaling - Communication Volume Tables
print("\n" + "-"*80)
print("Generating Strong Scaling Plots - Communication Volume Tables")
print("-"*80)

for matrix in matrices:
    matrix_data = strong_agg[strong_agg['matrix'] == matrix].sort_values('num_procs')
    
    comm_table = pd.DataFrame({
        'Num_Processes': matrix_data['num_procs'],
        'Ghost_Entries_Min': matrix_data['ghost_entries_min'].astype(int),
        'Ghost_Entries_Avg': matrix_data['ghost_entries_mean'].round(2),
        'Ghost_Entries_Max': matrix_data['ghost_entries_max'].astype(int),
        'Load_Imbalance_Ratio': (matrix_data['ghost_entries_max'] / matrix_data['ghost_entries_mean']).round(3)
    })
    
    filename = os.path.join(subdirs['strong_comm_volume'], f'comm_volume_{matrix}.csv')
    comm_table.to_csv(filename, index=False)

print(f"\nStrong scaling plots created and saved in /comm_volume_strong subfolder")


# PLOT 4: Strong Scaling - Matrix Comparison
print("\n" + "-"*80)
print("Generating Strong Scaling Plots - Matrix Comparison")
print("-"*80)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax1 = axes[0]
for matrix in matrices:
    matrix_data = strong_agg[strong_agg['matrix'] == matrix].sort_values('num_procs')
    procs = matrix_data['num_procs'].values
    times = matrix_data['elapsed_time'].values
    if len(times) > 0:
        baseline = times[0]
        speedup = baseline / times
        ax1.plot(procs, speedup, marker='o', label=matrix, linewidth=2.5, markersize=8)

if len(procs) > 0:
    ax1.plot(procs, procs / procs[0], 'k--', label='Ideal', linewidth=2)

ax1.set_xlabel('Number of Processes', fontsize=13, fontweight='bold')
ax1.set_ylabel('Speedup', fontsize=13, fontweight='bold')
ax1.set_title('Strong Scaling - Speedup Comparison', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log', base=2)
ax1.set_yscale('log', base=2)

ax2 = axes[1]
for matrix in matrices:
    matrix_data = strong_agg[strong_agg['matrix'] == matrix].sort_values('num_procs')
    procs = matrix_data['num_procs'].values
    times = matrix_data['elapsed_time'].values
    if len(times) > 0:
        baseline = times[0]
        speedup = baseline / times
        efficiency = (speedup / procs) * 100
        ax2.plot(procs, efficiency, marker='s', label=matrix, linewidth=2.5, markersize=8)

ax2.axhline(y=100, color='k', linestyle='--', label='Ideal', linewidth=2)
ax2.set_xlabel('Number of Processes', fontsize=13, fontweight='bold')
ax2.set_ylabel('Efficiency (%)', fontsize=13, fontweight='bold')
ax2.set_title('Strong Scaling - Efficiency Comparison', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log', base=2)
ax2.set_ylim([0, 110])

plt.tight_layout()
filename = os.path.join(subdirs['comparison'], 'strong_scaling_comparison.png')
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"\nStrong scaling plots created and saved in /comparison subfolder")
plt.close()

# PLOT 5: Strong Scaling - MPI vs MPI+OpenMP
print("\n" + "-"*80)
print("Generating Strong Scaling Plots - MPI vs MPI+OpenMP")
print("-"*80)


for matrix in matrices:
    mpi_data = strong_agg[strong_agg['matrix'] == matrix].sort_values('num_procs')
    hyb_data = strong_hybrid_agg[strong_hybrid_agg['matrix'] == matrix].sort_values('num_procs')
    if len(mpi_data) == 0 or len(hyb_data) == 0:
        continue

    procs = mpi_data['num_procs'].values
    common = np.intersect1d(mpi_data['num_procs'].values,
                            hyb_data['num_procs'].values)
    mpi_data = mpi_data[mpi_data['num_procs'].isin(common)].sort_values('num_procs')
    hyb_data = hyb_data[hyb_data['num_procs'].isin(common)].sort_values('num_procs')
    if len(common) == 0:
        continue

    procs = common
    # Speedup
    fig, ax = plt.subplots(figsize=(10, 7))

    times_mpi = mpi_data['elapsed_time'].values
    times_hyb = hyb_data['elapsed_time'].values

    baseline_mpi = times_mpi[0]
    baseline_hyb = times_hyb[0]

    speedup_mpi = baseline_mpi / times_mpi
    speedup_hyb = baseline_hyb / times_hyb

    ax.plot(procs, speedup_mpi, marker='o', linewidth=2.5, markersize=8,
            label='MPI', color='#1f77b4')
    ax.plot(procs, speedup_hyb, marker='s', linewidth=2.5, markersize=8,
            label='MPI+OpenMP', color='#ff7f0e')
    ax.plot(procs, procs / procs[0], 'k--', label='Ideal', linewidth=2)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xlabel('Number of Processes', fontsize=13, fontweight='bold')
    ax.set_ylabel('Speedup (90th percentile)', fontsize=13, fontweight='bold')
    ax.set_title(f'Strong Scaling - MPI vs MPI+OpenMP ({matrix})',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    filename = os.path.join(subdirs['comparison'],
                            f'strong_mpi_vs_hybrid_{matrix}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

print(f"\nStrong scaling plots created and saved in /comparison subfolder")

# PLOT 6: Weak Scaling - Speedup with Gustavson's Law
print("\n" + "-"*80)
print("Generating weak Scaling Speedup Plots with Gustavson's Law")
print("-"*80)

fig, ax = plt.subplots(figsize=(10, 7))

procs_weak = weak_agg['num_procs'].values
times_weak = weak_agg['elapsed_time'].values

if len(times_weak) > 0:
    baseline_weak = times_weak[0]
    scaled_speedup = procs_weak * (baseline_weak / times_weak)
    
    serial_frac = estimate_serial_fraction_weak(procs_weak, times_weak)
    
    # Gustafson
    procs_continuous = np.logspace(0, np.log2(procs_weak[-1]), 100, base=2)
    gustafson_speedup = np.array([gustafson_law(p, serial_frac) for p in procs_continuous])
    
    # Plot
    ax.plot(procs_weak, scaled_speedup, marker='s', label='Actual Scaled Speedup', 
            linewidth=3, markersize=12, color='#06A77D')
    ax.plot(procs_weak, procs_weak, 'k--', label='Ideal (Linear)', linewidth=2)
    ax.plot(procs_continuous, gustafson_speedup, '-.', 
            label=f'Gustafson (s={serial_frac:.3f})', linewidth=2, color='#E63946')
    
    ax.set_xlabel('Number of Processes', fontsize=13, fontweight='bold')
    ax.set_ylabel('Scaled Speedup', fontsize=13, fontweight='bold')
    ax.set_title('Weak Scaling - Scaled Speedup with Gustafson\'s Law', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    
    plt.tight_layout()
    filename = os.path.join(subdirs['weak'], 'weak_scaling_speedup.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nWeak scaling plots created and saved in /weak_scaling subfolder")
    plt.close()

# PLOT 7: Weak Scaling - Efficiency 
print("\n" + "-"*80)
print("Generating weak scaling efficiency Plots")
print("-"*80)

fig, ax = plt.subplots(figsize=(10, 7))

if len(times_weak) > 0:
    efficiency_weak = (baseline_weak / times_weak) * 100
    
    gustafson_efficiency = (gustafson_speedup / procs_continuous) * 100
    
    ax.plot(procs_weak, efficiency_weak, marker='s', color='#06A77D', 
             linewidth=3, markersize=12, label='Actual')
    ax.axhline(y=100, color='k', linestyle='--', label='Ideal', linewidth=2)
    ax.plot(procs_continuous, gustafson_efficiency, '-.', 
            label=f'Gustafson (s={serial_frac:.3f})', linewidth=2, color='#E63946')
    
    ax.set_xlabel('Number of Processes', fontsize=13, fontweight='bold')
    ax.set_ylabel('Efficiency (%)', fontsize=13, fontweight='bold')
    ax.set_title('Weak Scaling - Efficiency', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_ylim([0, 110])
    
    plt.tight_layout()
    filename = os.path.join(subdirs['weak'], 'weak_scaling_efficiency.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nWeak scaling efficiency plots created and saved in /weak_scaling subfolder")
    plt.close()

# PLOT 8: Weak Scaling - Execution Time + Computation vs Communication (COMBINED)
print("\n" + "-"*80)
print("Generating weak scaling execution time + computation vs communication (combined)")
print("-"*80)

fig, ax = plt.subplots(figsize=(10, 7))

comp_times_weak = weak_agg['comp_time'].values
comm_times_weak = weak_agg['comm_time'].values

x_weak = np.arange(len(procs_weak))
width = 0.6

ax.bar(x_weak, comp_times_weak, width, label='Computation Time', color='steelblue', alpha=0.8)
ax.bar(x_weak, comm_times_weak, width, bottom=comp_times_weak,
       label='Communication Time', color='coral', alpha=0.8)

ax.plot(x_weak, times_weak, marker='o', color='#6A4C93',
        linewidth=3, markersize=12, label='Total Execution Time', zorder=10)

ax.axhline(y=baseline_weak, color='gray', linestyle='--',
           label='Baseline (1 proc)', linewidth=2, alpha=0.7, zorder=5)

ax.set_xlabel('Number of Processes', fontsize=13, fontweight='bold')
ax.set_ylabel('Execution Time (s)', fontsize=13, fontweight='bold')
ax.set_title('Weak Scaling - Execution Time & Computation vs Communication',
             fontsize=14, fontweight='bold')
ax.set_xticks(x_weak)
ax.set_xticklabels(procs_weak)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, axis='y', zorder=0)

plt.tight_layout()
filename = os.path.join(subdirs['weak'], 'weak_scaling_time_and_comp_vs_comm.png')
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"\nWeak scaling combined plot created and saved in /weak_scaling subfolder")
plt.close()

# PLOT 9: Weak Scaling - MPI vs MPI+OpenMP
print("\n" + "-"*80)
print("Generating MPI vs MPI+OpenMP Weak Scaling Plots")
print("-"*80)


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

common_weak = np.intersect1d(weak_agg['num_procs'].values,
                             weak_hybrid_agg['num_procs'].values)
weak_mpi_cmp = weak_agg[weak_agg['num_procs'].isin(common_weak)].sort_values('num_procs')
weak_hyb_cmp = weak_hybrid_agg[weak_hybrid_agg['num_procs'].isin(common_weak)].sort_values('num_procs')

if len(weak_mpi_cmp) > 0 and len(weak_hyb_cmp) > 0:
    procs_w = weak_mpi_cmp['num_procs'].values

    # Scaled speedup
    ax1 = axes[0]
    times_mpi_w = weak_mpi_cmp['elapsed_time'].values
    times_hyb_w = weak_hyb_cmp['elapsed_time'].values

    baseline_mpi_w = times_mpi_w[0]
    baseline_hyb_w = times_hyb_w[0]

    scaled_speedup_mpi = procs_w * (baseline_mpi_w / times_mpi_w)
    scaled_speedup_hyb = procs_w * (baseline_hyb_w / times_hyb_w)

    ax1.plot(procs_w, scaled_speedup_mpi, marker='o', linewidth=2.5, markersize=8,
             label='MPI', color='#1f77b4')
    ax1.plot(procs_w, scaled_speedup_hyb, marker='s', linewidth=2.5, markersize=8,
             label='MPI+OpenMP', color='#ff7f0e')
    ax1.plot(procs_w, procs_w, 'k--', label='Ideal', linewidth=2)

    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=2)
    ax1.set_xlabel('Number of Processes', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Scaled Speedup (90th percentile)', fontsize=13, fontweight='bold')
    ax1.set_title('Weak Scaling - Scaled Speedup MPI vs MPI+OpenMP',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # Efficiency
    ax2 = axes[1]
    eff_mpi_w = (baseline_mpi_w / times_mpi_w) * 100
    eff_hyb_w = (baseline_hyb_w / times_hyb_w) * 100

    ax2.plot(procs_w, eff_mpi_w, marker='o', linewidth=2.5, markersize=8,
             label='MPI', color='#1f77b4')
    ax2.plot(procs_w, eff_hyb_w, marker='s', linewidth=2.5, markersize=8,
             label='MPI+OpenMP', color='#ff7f0e')
    ax2.axhline(y=100, color='k', linestyle='--', label='Ideal', linewidth=2)

    ax2.set_xscale('log', base=2)
    ax2.set_ylim([0, 110])
    ax2.set_xlabel('Number of Processes', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Efficiency (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Weak Scaling - Efficiency MPI vs MPI+OpenMP',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    filename = os.path.join(subdirs['comparison'],
                            'weak_mpi_vs_hybrid.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nWeak scaling plots created and saved in /comparison subfolder")
    plt.close()

# PLOT 10: Weak Scaling - Computation vs Communication
print("\n" + "-"*80)
print("Generating weak scaling Computation vs Communication Plots")
print("-"*80)

fig, ax = plt.subplots(figsize=(10, 7))

comp_times_weak = weak_agg['comp_time'].values
comm_times_weak = weak_agg['comm_time'].values
x_weak = np.arange(len(procs_weak))
width = 0.6
ax.bar(x_weak, comp_times_weak, width, label='Computation Time', color='steelblue')
ax.bar(x_weak, comm_times_weak, width, bottom=comp_times_weak, 
        label='Communication Time', color='coral')
ax.set_xlabel('Number of Processes', fontsize=13, fontweight='bold')
ax.set_ylabel('Execution Time (s)', fontsize=13, fontweight='bold')
ax.set_title('Weak Scaling - Computation vs Communication Time', fontsize=14, fontweight='bold')
ax.set_xticks(x_weak)
ax.set_xticklabels(procs_weak)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
filename = os.path.join(subdirs['weak'], 'weak_scaling_comp_vs_comm.png')
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"\nWeak scaling execution timeplots created and saved in /comparison subfolder")
plt.close()

# PLOT 11: Weak Scaling - Load Balance
print("\n" + "-"*80)
print("Generating weak scaling load balance plots")
print("-"*80)

fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(procs_weak, weak_agg['local_nz_min'], marker='v', label='Min NNZ', 
         linewidth=2.5, markersize=10, color='#E63946')
ax.plot(procs_weak, weak_agg['local_nz_mean'], marker='o', label='Avg NNZ', 
         linewidth=2.5, markersize=10, color='#06FFA5')
ax.plot(procs_weak, weak_agg['local_nz_max'], marker='^', label='Max NNZ', 
         linewidth=2.5, markersize=10, color='#F77F00')
ax.set_xlabel('Number of Processes', fontsize=13, fontweight='bold')
ax.set_ylabel('Non-zeros per Rank', fontsize=13, fontweight='bold')
ax.set_title('Weak Scaling - Load Balance (NNZ per Rank)', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xscale('log', base=2)

plt.tight_layout()
filename = os.path.join(subdirs['weak'], 'weak_scaling_load_balance.png')
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"\nWeak scaling load balance plots created and saved in /weak_scaling subfolder")
plt.close()

# PLOT 12: Weak Scaling - Communication Volume Table
weak_comm_table = pd.DataFrame({
    'Num_Processes': procs_weak,
    'Ghost_Entries_Min': weak_agg['ghost_entries_min'].astype(int),
    'Ghost_Entries_Avg': weak_agg['ghost_entries_mean'].round(2),
    'Ghost_Entries_Max': weak_agg['ghost_entries_max'].astype(int),
    'Load_Imbalance_Ratio': (weak_agg['ghost_entries_max'] / weak_agg['ghost_entries_mean']).round(3)
})

filename = os.path.join(subdirs['weak'], 'comm_volume_weak_scaling.csv')
weak_comm_table.to_csv(filename, index=False)
print(f"\nWeak scaling communication volume table created and saved in /weak_scaling subfolder")

# Summary
print("\n" + "-"*50)
print("Data analysis completed!")
print("-"*50)
