#!/usr/bin/env python3
"""
MPI SpMV Scaling Analysis and Plotting
Generates performance plots from benchmark CSV files
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Configuration
RESULTS_DIR = "../results"
PLOTS_DIR = "../plots"
STRONG_SCALING_CSV = f"{RESULTS_DIR}/strong_scaling.csv"
WEAK_SCALING_CSV = f"{RESULTS_DIR}/weak_scaling.csv"

# Ensure plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def load_data():
    """Load CSV data files"""
    try:
        strong_df = pd.read_csv(STRONG_SCALING_CSV)
        print(f"Loaded strong scaling data: {len(strong_df)} rows")
    except FileNotFoundError:
        print(f"Warning: {STRONG_SCALING_CSV} not found")
        strong_df = None
    
    try:
        weak_df = pd.read_csv(WEAK_SCALING_CSV)
        print(f"Loaded weak scaling data: {len(weak_df)} rows")
    except FileNotFoundError:
        print(f"Warning: {WEAK_SCALING_CSV} not found")
        weak_df = None
    
    return strong_df, weak_df


def plot_strong_scaling_speedup(df):
    """Plot speedup for strong scaling"""
    print("\nGenerating strong scaling speedup plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    matrices = df['matrix'].unique()
    
    for matrix in matrices:
        matrix_df = df[df['matrix'] == matrix]
        
        # Group by processes and calculate mean time
        grouped = matrix_df.groupby('processes')['time_ms'].mean().reset_index()
        grouped = grouped.sort_values('processes')
        
        processes = grouped['processes'].values
        times = grouped['time_ms'].values
        
        # Calculate speedup (baseline = 1 process)
        baseline_time = times[0] if len(times) > 0 else 1
        speedup = baseline_time / times
        
        # Plot
        ax.plot(processes, speedup, marker='o', linewidth=2, 
                markersize=8, label=matrix)
    
    # Ideal speedup line
    max_procs = df['processes'].max()
    ideal_procs = np.array([1, 2, 4, 8, 16, 32, 64, 96])
    ideal_procs = ideal_procs[ideal_procs <= max_procs]
    ax.plot(ideal_procs, ideal_procs, 'k--', linewidth=2, 
            label='Ideal Speedup', alpha=0.7)
    
    ax.set_xlabel('Number of Processes', fontsize=14, fontweight='bold')
    ax.set_ylabel('Speedup', fontsize=14, fontweight='bold')
    ax.set_title('Strong Scaling: Speedup vs Number of Processes', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/strong_scaling_speedup.png", dpi=300)
    print(f"Saved: {PLOTS_DIR}/strong_scaling_speedup.png")
    plt.close()


def plot_strong_scaling_efficiency(df):
    """Plot efficiency for strong scaling"""
    print("Generating strong scaling efficiency plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    matrices = df['matrix'].unique()
    
    for matrix in matrices:
        matrix_df = df[df['matrix'] == matrix]
        
        # Group by processes and calculate mean time
        grouped = matrix_df.groupby('processes')['time_ms'].mean().reset_index()
        grouped = grouped.sort_values('processes')
        
        processes = grouped['processes'].values
        times = grouped['time_ms'].values
        
        # Calculate efficiency
        baseline_time = times[0] if len(times) > 0 else 1
        speedup = baseline_time / times
        efficiency = (speedup / processes) * 100
        
        # Plot
        ax.plot(processes, efficiency, marker='s', linewidth=2, 
                markersize=8, label=matrix)
    
    # 100% efficiency line
    ax.axhline(y=100, color='k', linestyle='--', linewidth=2, 
               label='Ideal (100%)', alpha=0.7)
    
    ax.set_xlabel('Number of Processes', fontsize=14, fontweight='bold')
    ax.set_ylabel('Parallel Efficiency (%)', fontsize=14, fontweight='bold')
    ax.set_title('Strong Scaling: Parallel Efficiency', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/strong_scaling_efficiency.png", dpi=300)
    print(f"Saved: {PLOTS_DIR}/strong_scaling_efficiency.png")
    plt.close()


def plot_strong_scaling_gflops(df):
    """Plot GFLOPS for strong scaling"""
    print("Generating strong scaling GFLOPS plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    matrices = df['matrix'].unique()
    
    for matrix in matrices:
        matrix_df = df[df['matrix'] == matrix]
        
        # Group by processes and calculate mean GFLOPS
        grouped = matrix_df.groupby('processes')['gflops'].mean().reset_index()
        grouped = grouped.sort_values('processes')
        
        processes = grouped['processes'].values
        gflops = grouped['gflops'].values
        
        # Plot
        ax.plot(processes, gflops, marker='^', linewidth=2, 
                markersize=8, label=matrix)
    
    ax.set_xlabel('Number of Processes', fontsize=14, fontweight='bold')
    ax.set_ylabel('GFLOPS', fontsize=14, fontweight='bold')
    ax.set_title('Strong Scaling: Computational Performance', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/strong_scaling_gflops.png", dpi=300)
    print(f"Saved: {PLOTS_DIR}/strong_scaling_gflops.png")
    plt.close()


def plot_communication_breakdown(df):
    """Plot communication vs computation time breakdown"""
    print("Generating communication breakdown plot...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    matrices = df['matrix'].unique()
    
    # Use first matrix for detailed analysis
    matrix = matrices[0] if len(matrices) > 0 else None
    if matrix is None:
        print("No matrix data available")
        return
    
    matrix_df = df[df['matrix'] == matrix]
    grouped = matrix_df.groupby('processes').agg({
        'comm_ms': 'mean',
        'comp_ms': 'mean'
    }).reset_index()
    grouped = grouped.sort_values('processes')
    
    processes = grouped['processes'].values
    comm_time = grouped['comm_ms'].values
    comp_time = grouped['comp_ms'].values
    
    # Stacked bar chart
    width = 0.5
    x_pos = np.arange(len(processes))
    
    p1 = ax.bar(x_pos, comp_time, width, label='Computation', color='#2E86AB')
    p2 = ax.bar(x_pos, comm_time, width, bottom=comp_time, 
                label='Communication', color='#A23B72')
    
    ax.set_xlabel('Number of Processes', fontsize=14, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=14, fontweight='bold')
    ax.set_title(f'Communication vs Computation Breakdown: {matrix}', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(p) for p in processes])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for i, (c, co) in enumerate(zip(comm_time, comp_time)):
        total = c + co
        comm_pct = (c / total) * 100
        ax.text(i, total + 0.5, f'{comm_pct:.1f}%', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/communication_breakdown.png", dpi=300)
    print(f"Saved: {PLOTS_DIR}/communication_breakdown.png")
    plt.close()


def plot_load_balance(df):
    """Plot load balance (NNZ distribution)"""
    print("Generating load balance plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    matrices = df['matrix'].unique()
    matrix = matrices[0] if len(matrices) > 0 else None
    
    if matrix is None:
        print("No matrix data available")
        return
    
    matrix_df = df[df['matrix'] == matrix]
    grouped = matrix_df.groupby('processes').agg({
        'min_nz': 'mean',
        'avg_nz': 'mean',
        'max_nz': 'mean',
        'imbalance_pct': 'mean'
    }).reset_index()
    grouped = grouped.sort_values('processes')
    
    processes = grouped['processes'].values
    
    # Plot 1: Min/Avg/Max NNZ
    ax1.plot(processes, grouped['min_nz'], marker='v', linewidth=2, 
             label='Min NNZ', color='green')
    ax1.plot(processes, grouped['avg_nz'], marker='o', linewidth=2, 
             label='Avg NNZ', color='blue')
    ax1.plot(processes, grouped['max_nz'], marker='^', linewidth=2, 
             label='Max NNZ', color='red')
    
    ax1.set_xlabel('Number of Processes', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Non-zeros per Process', fontsize=13, fontweight='bold')
    ax1.set_title(f'Load Balance: NNZ Distribution - {matrix}', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # Plot 2: Imbalance percentage
    ax2.bar(range(len(processes)), grouped['imbalance_pct'], 
            color='coral', alpha=0.7)
    ax2.set_xlabel('Number of Processes', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Imbalance (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Load Imbalance Percentage', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(processes)))
    ax2.set_xticklabels([str(p) for p in processes])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=10, color='red', linestyle='--', alpha=0.5, 
                label='10% threshold')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/load_balance.png", dpi=300)
    print(f"Saved: {PLOTS_DIR}/load_balance.png")
    plt.close()


def plot_weak_scaling(df):
    """Plot weak scaling efficiency"""
    print("\nGenerating weak scaling plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    grouped = df.groupby('processes').agg({
        'time_ms': 'mean',
        'efficiency_pct': 'mean',
        'gflops': 'mean'
    }).reset_index()
    grouped = grouped.sort_values('processes')
    
    processes = grouped['processes'].values
    time_ms = grouped['time_ms'].values
    efficiency = grouped['efficiency_pct'].values
    gflops = grouped['gflops'].values
    
    # Plot 1: Execution time (should stay constant for ideal weak scaling)
    ax1.plot(processes, time_ms, marker='o', linewidth=2, 
             markersize=8, color='#2E86AB', label='Execution Time')
    baseline_time = time_ms[0] if len(time_ms) > 0 else 1
    ax1.axhline(y=baseline_time, color='k', linestyle='--', 
                linewidth=2, label='Ideal (constant)', alpha=0.7)
    
    ax1.set_xlabel('Number of Processes', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Time (ms)', fontsize=13, fontweight='bold')
    ax1.set_title('Weak Scaling: Execution Time', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # Plot 2: Efficiency
    ax2.plot(processes, efficiency, marker='s', linewidth=2, 
             markersize=8, color='#A23B72', label='Efficiency')
    ax2.axhline(y=100, color='k', linestyle='--', linewidth=2, 
                label='Ideal (100%)', alpha=0.7)
    
    ax2.set_xlabel('Number of Processes', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Efficiency (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Weak Scaling: Parallel Efficiency', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    ax2.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/weak_scaling.png", dpi=300)
    print(f"Saved: {PLOTS_DIR}/weak_scaling.png")
    plt.close()


def generate_summary_table(strong_df, weak_df):
    """Generate summary statistics table"""
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    if strong_df is not None:
        print("\nStrong Scaling Summary:")
        for matrix in strong_df['matrix'].unique():
            matrix_df = strong_df[strong_df['matrix'] == matrix]
            
            # Calculate metrics for max processes
            max_procs = matrix_df['processes'].max()
            max_procs_df = matrix_df[matrix_df['processes'] == max_procs]
            baseline_df = matrix_df[matrix_df['processes'] == 1]
            
            if len(baseline_df) > 0 and len(max_procs_df) > 0:
                baseline_time = baseline_df['time_ms'].mean()
                max_time = max_procs_df['time_ms'].mean()
                speedup = baseline_time / max_time
                efficiency = (speedup / max_procs) * 100
                
                print(f"\n  {matrix}:")
                print(f"    Max processes: {max_procs}")
                print(f"    Speedup: {speedup:.2f}x")
                print(f"    Efficiency: {efficiency:.2f}%")
                print(f"    Best GFLOPS: {max_procs_df['gflops'].mean():.2f}")
    
    if weak_df is not None:
        print("\nWeak Scaling Summary:")
        max_procs = weak_df['processes'].max()
        max_procs_df = weak_df[weak_df['processes'] == max_procs]
        
        if len(max_procs_df) > 0:
            efficiency = max_procs_df['efficiency_pct'].mean()
            gflops = max_procs_df['gflops'].mean()
            
            print(f"  Max processes: {max_procs}")
            print(f"  Efficiency at {max_procs}p: {efficiency:.2f}%")
            print(f"  GFLOPS at {max_procs}p: {gflops:.2f}")
    
    print("\n" + "="*60)


def main():
    """Main execution function"""
    print("="*60)
    print("MPI SpMV Performance Analysis")
    print("="*60)
    
    # Load data
    strong_df, weak_df = load_data()
    
    if strong_df is None and weak_df is None:
        print("Error: No data files found!")
        sys.exit(1)
    
    # Generate plots
    print("\nGenerating plots...")
    
    if strong_df is not None:
        plot_strong_scaling_speedup(strong_df)
        plot_strong_scaling_efficiency(strong_df)
        plot_strong_scaling_gflops(strong_df)
        plot_communication_breakdown(strong_df)
        plot_load_balance(strong_df)
    
    if weak_df is not None:
        plot_weak_scaling(weak_df)
    
    # Generate summary
    generate_summary_table(strong_df, weak_df)
    
    print("\n" + "="*60)
    print("Analysis complete! Plots saved in:", PLOTS_DIR)
    print("="*60)


if __name__ == "__main__":
    main()