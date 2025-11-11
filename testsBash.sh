#!/bin/bash

# Configuration
SRC_DIR="src"
INCLUDE_DIR="include"
EXEC="./build/testResult.out"
MATRICES=("1138_bus.mtx" "ash85.mtx" "bcsstk13.mtx" "bcsstk18.mtx" "bcsstk32.mtx")
OUTPUT="benchResults.csv"

CC="gcc-15"

REPEATS=10
OPT_FLAGS=("" "-O1" "-O2" "-O3" "-Ofast")

SCHEDULES=("static" "dynamic" "guided")
CHUNKSIZES=(1 10 100 1000)
THREADS=(1 2 4 8 16 32 64)

# Initialize CSV
echo "matrix,mode,opt_level,schedule,chunk_size,num_threads,perf,run,value" > "$OUTPUT"

# Compilation function
compile_code() {
    local flags="$1"
    local parallel="$2"
    echo "Compilation with flags: $flags (OpenMP: $parallel)"

    rm -f $EXEC
    mkdir -p build

    if [ "$parallel" = "yes" ]; then
        $CC -g -Wall $flags -fopenmp -I"$INCLUDE_DIR" "$SRC_DIR"/*.c -o "$EXEC"
    else
        $CC -g -Wall $flags -I"$INCLUDE_DIR" "$SRC_DIR"/*.c -o "$EXEC"
    fi

    if [ $? -ne 0 ]; then
        echo "ERROR: Compilation failed!"
        exit 1
    fi
}

# Execution function
run_and_record() {
    local matrix="$1"
    local mode="$2"
    local opt="$3"
    local schedule="$4"
    local chunk="$5"
    local threads="$6"
    local use_perf="$7"

    for ((r=1; r<=REPEATS; r++)); do
        if [ "$use_perf" = "yes" ]; then
            if [ "$mode" = "sequential" ]; then
                RESULT=$(perf stat -x, -e cache-misses "$EXEC" "data/$matrix" 2>&1 \
                         | grep "cache-misses" | awk -F',' '{print $1}')
            else
                RESULT=$(perf stat -x, -e cache-misses "$EXEC" "data/$matrix" "$threads" "$schedule" "$chunk" 2>&1 \
                         | grep "cache-misses" | awk -F',' '{print $1}')
            fi
            echo "Perf measurement not implemented for this test. Skipping..."
            perf_flag="yes"
        else
            if [ "$mode" = "sequential" ]; then
                RESULT=$("$EXEC" "data/$matrix" 2>&1 \
                        | grep "Result_time:" | awk '{print $2}')
            else
                RESULT=$("$EXEC" "data/$matrix" "$threads" "$schedule" "$chunk" 2>&1 \
                        | grep "Result_time:" | awk '{print $2}')
fi
perf_flag="no"
        fi

        if [ -z "$RESULT" ]; then
            echo "ERROR: failed execution/output not found!"
            RESULT="ERROR"
        fi

        echo "${matrix},${mode},${opt},${schedule},${chunk},${threads},${perf_flag},${r},${RESULT}" >> "$OUTPUT"
        echo "→ ${matrix} | ${mode} ${opt} | sched=${schedule} chunk=${chunk} threads=${threads} | perf=${perf_flag} | run=${r} | result=${RESULT}"
    done
}

# SEQUENTIAL
echo ">>> Parte SEQUENZIALE..."
for matrix in "${MATRICES[@]}"; do
  for opt in "${OPT_FLAGS[@]}"; do
    compile_code "$opt" "no"
    for perf_mode in no yes; do
      run_and_record "$matrix" "sequential" "$opt" "none" "none" 1 "$perf_mode"
    done
  done
done

# PARALLEL
echo ">>> Parte PARALLELA..."
compile_code "-O3" "yes"

for matrix in "${MATRICES[@]}"; do
  for schedule in "${SCHEDULES[@]}"; do
    for chunk in "${CHUNKSIZES[@]}"; do
      for threads in "${THREADS[@]}"; do
        for perf_mode in no yes; do
          run_and_record "$matrix" "parallel" "-O3" "$schedule" "$chunk" "$threads" "$perf_mode"
        done
      done
    done
  done
done

echo "Done testing! Results are saved in: $OUTPUT"