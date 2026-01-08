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
    
    cd "$RESULTS_DIR" || exit 1
    
    mpirun -np "$num_procs" "$EXEC" "$DATA_DIR/$matrix" "$REPEATS"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: MPI execution failed"
        return 1
    fi
    
    mkdir -p "strong_${matrix%.mtx}"
    mv timings_rank*_np${num_procs}.csv "strong_${matrix%.mtx}/"
    
    cd - > /dev/null
    
    echo "Data saved to: $RESULTS_DIR/strong_${matrix%.mtx}/"
}

run_weak_scaling() {
    local num_procs="$1"
    local total_rows=$((ROWS_PER_PROC * num_procs))
    
    echo ""
    echo "WEAK SCALING: $num_procs processes | ${total_rows} total rows"
    
    cd "$RESULTS_DIR" || exit 1
    
    mpirun -np "$num_procs" "$EXEC" synthetic "$REPEATS" "$ROWS_PER_PROC" "$NNZ_PER_ROW"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: MPI execution failed"
        return 1
    fi
    
    # Organizza output
    mkdir -p "weak_scaling"
    mv timings_rank*_np${num_procs}.csv "weak_scaling/"
    
    cd - > /dev/null
    
    echo "Data saved to: $RESULTS_DIR/weak_scaling/"
}


echo "MPI SpMV Benchmark"
echo "  Max processes: $MAX_PROCESSES"
echo "  Repeats: $REPEATS"
echo "  Data directory: $DATA_DIR"

rm -rf "$RESULTS_DIR"/strong_*
rm -rf "$RESULTS_DIR"/weak_scaling
mkdir -p "$RESULTS_DIR"

find_matrices
compile_code

echo ""
echo "Strong Scaling Tests"
echo ""

for matrix in "${MATRICES[@]}"; do
    for num_procs in "${PROCESSES[@]}"; do
        if [ "$num_procs" -le "$MAX_PROCESSES" ]; then
            run_strong_scaling "$matrix" "$num_procs"
            sleep 2 
        fi
    done
done

echo ""
echo "Weak Scaling Tests"
echo ""

for num_procs in "${PROCESSES[@]}"; do
    if [ "$num_procs" -le "$MAX_PROCESSES" ]; then
        run_weak_scaling "$num_procs"
        sleep 2
    fi
done

echo ""
echo "Data Analysis"
echo ""
echo "Raw CSV files generated in: $RESULTS_DIR"
echo ""

echo "---------------------"
echo "Benchmark Completed"
echo "---------------------"
