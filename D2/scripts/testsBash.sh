#!/bin/bash

# Configuration
SRC_DIR="../src"
INCLUDE_DIR="../include"
EXEC="../results/spmv_mpi.out"
DATA_DIR="../data"
RESULTS_DIR="../results"

# MPI Configuration
PROCESSES_PER_NODE=24
MAX_PROCESSES=128
PROCESSES=(1 2 4 8 16 32 64 128)

# Test parameters
REPEATS=10
ROWS_PER_PROC=10000
NNZ_PER_ROW=50

# Compiler
MPICC="mpicc"
CFLAGS="-O3 -Wall -I$INCLUDE_DIR"

# Output files
STRONG_SCALING_CSV="$RESULTS_DIR/strong_scaling_all.csv"
WEAK_SCALING_CSV="$RESULTS_DIR/weak_scaling_all.csv"

find_matrices() {
    MATRICES=($(ls "$DATA_DIR"/*.mtx 2>/dev/null | xargs -n 1 basename))
    if [ ${#MATRICES[@]} -eq 0 ]; then
        echo "ERROR: No .mtx files found in $DATA_DIR"
        exit 1
    fi
    echo "Found ${#MATRICES[@]} matrices: ${MATRICES[@]}"
}

compile_code() {
    rm -f "$EXEC"
    mkdir -p "$RESULTS_DIR"

    $MPICC $CFLAGS "$SRC_DIR"/*.c -o "$EXEC"

    if [ $? -ne 0 ]; then
        echo "ERROR: Compilation failed!"
        exit 1
    fi
    
    echo "Compilation successful"
}

run_strong_scaling() {
    local matrix="$1"
    local num_procs="$2"
    
    echo ""
    echo "STRONG SCALING: $matrix | $num_procs processes"
    echo ""
    
    local log_file="$RESULTS_DIR/logs/strong_${matrix%.mtx}_np${num_procs}.log"
    mkdir -p "$RESULTS_DIR/logs"
    
    mpirun -np "$num_procs" "$EXEC" "$DATA_DIR/$matrix" "$REPEATS" 2>&1 | tee "$log_file" | \
        grep "^\[RESULT\]" | sed 's/\[RESULT\] //' | \
        awk -v matrix="${matrix%.mtx}" '{print $0","matrix}' >> "$STRONG_SCALING_CSV"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: MPI execution failed"
        return 1
    fi
    
    echo " -> Data appended to: $STRONG_SCALING_CSV"
    echo " -> Full log saved to: $log_file"
}

run_weak_scaling() {
    local num_procs="$1"
    local total_rows=$((ROWS_PER_PROC * num_procs))
    
    echo ""
    echo "WEAK SCALING: $num_procs processes | ${total_rows} total rows"
    echo ""
    
    local log_file="$RESULTS_DIR/logs/weak_np${num_procs}.log"
    mkdir -p "$RESULTS_DIR/logs"
    
    mpirun -np "$num_procs" "$EXEC" synthetic "$REPEATS" "$ROWS_PER_PROC" "$NNZ_PER_ROW" 2>&1 | \
        tee "$log_file" | grep "^\[RESULT\]" | sed 's/\[RESULT\] //' >> "$WEAK_SCALING_CSV"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: MPI execution failed"
        return 1
    fi
    
    echo " Data appended to: $WEAK_SCALING_CSV"
    echo " Full log saved to: $log_file"
}

initialize_csv_files() {
    # Header per strong scaling (con colonna matrix name)
    echo "rank,num_procs,run,elapsed_time,comm_time,local_nz,ghost_entries,local_flops,matrix" > "$STRONG_SCALING_CSV"
    
    # Header per weak scaling
    echo "rank,num_procs,run,elapsed_time,comm_time,local_nz,ghost_entries,local_flops" > "$WEAK_SCALING_CSV"
    
    echo "CSV files initialized:"
    echo "  - $STRONG_SCALING_CSV"
    echo "  - $WEAK_SCALING_CSV"
}

rm -rf "$RESULTS_DIR"/logs
rm -f "$STRONG_SCALING_CSV" "$WEAK_SCALING_CSV"
mkdir -p "$RESULTS_DIR"

find_matrices
compile_code
initialize_csv_files

echo ""
echo "Strong Scaling Tests:"
echo ""

for matrix in "${MATRICES[@]}"; do
    echo ""
    echo "Testing matrix: $matrix"
    echo "-----------------------------------------"
    for num_procs in "${PROCESSES[@]}"; do
        if [ "$num_procs" -le "$MAX_PROCESSES" ]; then
            run_strong_scaling "$matrix" "$num_procs"
            sleep 1 
        fi
    done
done

echo ""
echo "Weak Scaling Tests:"
echo ""

for num_procs in "${PROCESSES[@]}"; do
    if [ "$num_procs" -le "$MAX_PROCESSES" ]; then
        run_weak_scaling "$num_procs"
        sleep 1
    fi
done

echo ""
echo "Benchmark Summary:"
echo ""
echo "Results Summary:"
echo "  Strong Scaling: $STRONG_SCALING_CSV"
echo "  Weak Scaling:   $WEAK_SCALING_CSV"
echo "  Full Logs:      $RESULTS_DIR/logs/"
echo ""
echo "Total data points collected:"
echo "  Strong: $(tail -n +2 "$STRONG_SCALING_CSV" | wc -l) measurements"
echo "  Weak:   $(tail -n +2 "$WEAK_SCALING_CSV" | wc -l) measurements"
echo ""
