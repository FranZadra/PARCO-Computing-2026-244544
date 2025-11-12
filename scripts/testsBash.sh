#!/bin/bash

# Configuration
SRC_DIR="../src"
INCLUDE_DIR="../include"
EXEC="../results/testResult.out"
MATRICES=("Andrews.mtx" "bcsstk27.mtx" "bcsstk32.mtx" "hood.mtx" "msdoor.mtx")
OUTPUT_TIME="../plots/benchResults.csv"
OUTPUT_PERF="../plots/benchResults_perf.csv"

CC="gcc"
REPEATS=10
OPT_FLAGS=("" "-O1" "-O2" "-O3" "-Ofast")

SCHEDULES=("static" "dynamic" "guided")
CHUNKSIZES=(1 10 100 1000)
THREADS=(1 2 4 8 16 32 64)

# Initialize CSV
echo "matrix,mode,opt_level,schedule,chunk_size,num_threads,run,elapsed_time" > "$OUTPUT_TIME"
echo "matrix,mode,opt_level,schedule,chunk_size,num_threads,run,elapsed_time,L1_loads,L1_misses,L1_miss_rate,LLC_loads,LLC_misses,LLC_miss_rate" > "$OUTPUT_PERF"

# Compilation function
compile_code() {
    local flags="$1"
    local parallel="$2"
    echo -e "\nCompilation with flags: $flags (OpenMP: $parallel)"

    rm -f $EXEC
    mkdir -p results

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
                PERF_OUTPUT=$(perf stat -x, -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
                    "$EXEC" "../data/$matrix" 2>&1)
            else
                PERF_OUTPUT=$(perf stat -x, -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
                    "$EXEC" "../data/$matrix" "$threads" "$schedule" "$chunk" 2>&1)
            fi

            L1_LOADS=$(echo "$PERF_OUTPUT" | grep "L1-dcache-loads" | awk -F',' '{print $1}')
            L1_MISSES=$(echo "$PERF_OUTPUT" | grep "L1-dcache-load-misses" | awk -F',' '{print $1}')

            LLC_LOADS=$(echo "$PERF_OUTPUT" | grep "LLC-loads" | awk -F',' '{print $1}')
            LLC_MISSES=$(echo "$PERF_OUTPUT" | grep "LLC-load-misses" | awk -F',' '{print $1}')

            if [ ! -z "$L1_LOADS" ] && [ ! -z "$L1_MISSES" ] && [ "$L1_LOADS" != "0" ]; then
                L1_MISS_RATE=$(awk "BEGIN {printf \"%.2f\", ($L1_MISSES / $L1_LOADS) * 100}")
            else
                L1_MISS_RATE="N/A"
            fi

            if [ ! -z "$LLC_LOADS" ] && [ ! -z "$LLC_MISSES" ] && [ "$LLC_LOADS" != "0" ]; then
                LLC_MISS_RATE=$(awk "BEGIN {printf \"%.2f\", ($LLC_MISSES / $LLC_LOADS) * 100}")
            else
                LLC_MISS_RATE="N/A"
            fi

            elapsed_time=$(echo "$PERF_OUTPUT" | grep -E "Result_time:" | tail -1 | awk '{print $2}')

            if [ -z "$elapsed_time" ]; then
                elapsed_time="ERROR"
            fi

            # CSV perf
            echo "${matrix},${mode},${opt},${schedule},${chunk},${threads},${r},${elapsed_time},${L1_LOADS},${L1_MISSES},${L1_MISS_RATE},${LLC_LOADS},${LLC_MISSES},${LLC_MISS_RATE}" >> "$OUTPUT_PERF"
            # Terminal perf
            echo "→ ${matrix} | ${mode} ${opt} | sched=${schedule} chunk=${chunk} threads=${threads} | run=${r} | time=${elapsed_time}s | L1miss=${L1_MISSES}/${L1_LOADS} (${L1_MISS_RATE}%) | LLCmiss=${LLC_MISSES}/${LLC_LOADS} (${LLC_MISS_RATE}%)"
        else
            if [ "$mode" = "sequential" ]; then
                elapsed_time=$("$EXEC" "../data/$matrix" 2>&1 | grep "Result_time:" | awk '{print $2}')
            else
                elapsed_time=$("$EXEC" "../data/$matrix" "$threads" "$schedule" "$chunk" 2>&1 | grep "Result_time:" | awk '{print $2}')
            fi

            if [ -z "$elapsed_time" ]; then
                elapsed_time="ERROR"
            fi

            # CSV without perf
            echo "${matrix},${mode},${opt},${schedule},${chunk},${threads},${r},${elapsed_time}" >> "$OUTPUT_TIME"
            # Terminal
            echo "→ ${matrix} | ${mode} ${opt} | sched=${schedule} chunk=${chunk} threads=${threads} | run=${r} | time=${elapsed_time}s"
        fi
    done
}

# SEQUENTIAL
echo -e "\n\n>>> Parte SEQUENZIALE...\n\n"
for matrix in "${MATRICES[@]}"; do
  for opt in "${OPT_FLAGS[@]}"; do
    compile_code "$opt" "no"
    for perf_mode in no yes; do
      run_and_record "$matrix" "sequential" "$opt" "none" "none" 1 "$perf_mode"
    done
  done
done

# PARALLEL
echo -e "\n\n>>> Parte PARALLELA...\n\n"
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

echo -e "\nDone testing!"
echo "Time results saved in: $OUTPUT_TIME"
echo "Perf results saved in: $OUTPUT_PERF"
