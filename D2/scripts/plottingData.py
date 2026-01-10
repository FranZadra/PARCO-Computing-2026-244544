import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plots_dir = os.path.join('..', 'plots')
os.makedirs(plots_dir, exist_ok=True)

# 90 percentile and Data Preparation
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

    percentile_data['comp_time'] = percentile_data['elapsed_time'] - percentile_data['comm_time']

    return percentile_data

# Leggi i file CSV dalla cartella results (percorso relativo da scripts/)
strong_df = pd.read_csv('../results/strong_scaling_all.csv')
weak_df = pd.read_csv('../results/weak_scaling_all.csv')

strong_90th = calculate_90th_percentile(strong_df)
strong_90th.to_csv(os.path.join(plots_dir, 'strong_scaling_90th.csv'), index=False)
print(f"Created: plots/strong_scaling_90th.csv ({len(strong_90th)} rows)")

weak_90th = calculate_90th_percentile(weak_df)
weak_90th.to_csv(os.path.join(plots_dir, 'weak_scaling_90th.csv'), index=False)
print(f"Created: plots/weak_scaling_90th.csv ({len(weak_90th)} rows)")

strong_90th = pd.read_csv(os.path.join(plots_dir, 'strong_scaling_90th.csv'))
weak_90th = pd.read_csv(os.path.join(plots_dir, 'weak_scaling_90th.csv'))

strong_rank0 = strong_90th[strong_90th['rank'] == 0].copy()
matrices = strong_rank0['matrix'].unique()

print(f"\nGraphs for {len(matrices)} matrices...")

#GRAPHS
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Strong Scaling - Speedup
ax1 = axes[0, 0]
for matrix in matrices:
    matrix_data = strong_rank0[strong_rank0['matrix'] == matrix].sort_values('num_procs')
    procs = matrix_data['num_procs'].values
    times = matrix_data['elapsed_time'].values

    if len(times) > 0:
        baseline = times[0]
        speedup = baseline / times
        ax1.plot(procs, speedup, marker='o', label=matrix, linewidth=2, markersize=6)

# Ideal line
if len(procs) > 0:
    ax1.plot(procs, procs / procs[0], 'k--', label='Ideal', linewidth=2)

ax1.set_xlabel('Number of Processes', fontsize=11, fontweight='bold')
ax1.set_ylabel('Speedup', fontsize=11, fontweight='bold')
ax1.set_title('Strong Scaling - Speedup', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log', base=2)
ax1.set_yscale('log', base=2)

# Subplot 2: Strong Scaling - Efficiency
ax2 = axes[0, 1]
for matrix in matrices:
    matrix_data = strong_rank0[strong_rank0['matrix'] == matrix].sort_values('num_procs')
    procs = matrix_data['num_procs'].values
    times = matrix_data['elapsed_time'].values

    if len(times) > 0:
        baseline = times[0]
        speedup = baseline / times
        efficiency = speedup / procs * procs[0]
        ax2.plot(procs, efficiency * 100, marker='o', label=matrix, linewidth=2, markersize=6)

ax2.axhline(y=100, color='k', linestyle='--', label='Ideal', linewidth=2)
ax2.set_xlabel('Number of Processes', fontsize=11, fontweight='bold')
ax2.set_ylabel('Efficiency (%)', fontsize=11, fontweight='bold')
ax2.set_title('Strong Scaling - Efficiency', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log', base=2)

# Subplot 3: Weak Scaling
ax3 = axes[1, 0]
weak_rank0 = weak_90th[weak_90th['rank'] == 0].sort_values('num_procs')
procs_weak = weak_rank0['num_procs'].values
times_weak = weak_rank0['elapsed_time'].values

if len(times_weak) > 0:
    baseline_weak = times_weak[0]
    efficiency_weak = (baseline_weak / times_weak) * 100
    ax3.plot(procs_weak, efficiency_weak, marker='s', color='green', 
             linewidth=2.5, markersize=10, label='Actual')
    ax3.axhline(y=100, color='k', linestyle='--', label='Ideal', linewidth=2)

ax3.set_xlabel('Number of Processes', fontsize=11, fontweight='bold')
ax3.set_ylabel('Efficiency (%)', fontsize=11, fontweight='bold')
ax3.set_title('Weak Scaling - Efficiency', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
if len(procs_weak) > 2:
    ax3.set_xscale('log', base=2)

# Subplot 4: Computation vs Communication Time (stacked bar chart)
ax4 = axes[1, 1]

# Uses first matrix for stacked bar chart
matrix_for_stacked = matrices[0]
stacked_data = strong_rank0[strong_rank0['matrix'] == matrix_for_stacked].sort_values('num_procs')
procs_stacked = stacked_data['num_procs'].values
comp_times = stacked_data['comp_time'].values
comm_times = stacked_data['comm_time'].values

x = np.arange(len(procs_stacked))
width = 0.6

p1 = ax4.bar(x, comp_times, width, label='Computation Time', color='steelblue')
p2 = ax4.bar(x, comm_times, width, bottom=comp_times, label='Communication Time', color='coral')

ax4.set_xlabel('Number of Processes', fontsize=11, fontweight='bold')
ax4.set_ylabel('Execution Time (s)', fontsize=11, fontweight='bold')
ax4.set_title(f'Computation vs Communication ({matrix_for_stacked})', 
              fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(procs_stacked)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'spmv_analysis.png'), dpi=300, bbox_inches='tight')
print("Graph saved: plots/spmv_analysis.png")
plt.close()

# GRAPH 2: Computation vs Communication detailed for all matrices
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
axes2 = axes2.flatten()

for idx, matrix in enumerate(matrices):
    if idx < len(axes2):
        ax = axes2[idx]
        matrix_data = strong_rank0[strong_rank0['matrix'] == matrix].sort_values('num_procs')
        procs_mat = matrix_data['num_procs'].values
        comp_times_mat = matrix_data['comp_time'].values
        comm_times_mat = matrix_data['comm_time'].values

        x_mat = np.arange(len(procs_mat))
        width_mat = 0.6

        ax.bar(x_mat, comp_times_mat, width_mat, label='Computation', color='steelblue')
        ax.bar(x_mat, comm_times_mat, width_mat, bottom=comp_times_mat, 
               label='Communication', color='coral')

        ax.set_xlabel('Number of Processes', fontsize=10, fontweight='bold')
        ax.set_ylabel('Time (s)', fontsize=10, fontweight='bold')
        ax.set_title(matrix, fontsize=11, fontweight='bold')
        ax.set_xticks(x_mat)
        ax.set_xticklabels(procs_mat, rotation=45)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

# Nascondi subplot inutilizzati
for idx in range(len(matrices), len(axes2)):
    axes2[idx].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'computation_vs_communication_detailed.png'), dpi=300, bbox_inches='tight')
print("Graph saved: plots/computation_vs_communication_detailed.png")
plt.close()

# GRAPH 3: Communication Time Percentage
fig3, ax = plt.subplots(figsize=(10, 6))

for matrix in matrices:
    matrix_data = strong_rank0[strong_rank0['matrix'] == matrix].sort_values('num_procs')
    procs_mat = matrix_data['num_procs'].values
    comm_percentage = (matrix_data['comm_time'] / matrix_data['elapsed_time']) * 100

    ax.plot(procs_mat, comm_percentage, marker='o', label=matrix, linewidth=2, markersize=6)

ax.set_xlabel('Number of Processes', fontsize=11, fontweight='bold')
ax.set_ylabel('Communication Time (%)', fontsize=11, fontweight='bold')
ax.set_title('Communication Overhead vs Number of Processes', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xscale('log', base=2)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'communication_overhead.png'), dpi=300, bbox_inches='tight')
print("Graph saved: plots/communication_overhead.png")
plt.close()

print("\nAnalysis completed successfully!")
print("\nGenerated files in '../plots/' directory:")
print("  - strong_scaling_90th.csv")
print("  - weak_scaling_90th.csv")
print("  - spmv_analysis.png")
print("  - computation_vs_communication_detailed.png")
print("  - communication_overhead.png")