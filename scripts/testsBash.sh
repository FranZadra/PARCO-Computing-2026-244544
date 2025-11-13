#!/bin/bash

# Configuration
SRC_DIR="../src"
INCLUDE_DIR="../include"
EXEC="../results/testResult.out"
MATRICES=("Andrews.mtx" "bcsstk27.mtx" "bcsstk32.mtx" "hood.mtx" "msdoor.mtx")
OUTPUT_TIME="../results/benchResults.csv"
OUTPUT_PERF="../results/benchResults_perf.csv"

CC="gcc"
REPEATS=10
OPT_FLAGS=("" "-O1" "-O2" "-O3" "-Ofast")

SCHEDULES=("static" "dynamic" "guided")
CHUNKSIZES=(1 10 100 1000 10000)
THREADS=(1 2 4 8 16 32 64)

USE_PERF_MODE="both"
if [ $# -ge 1 ]; then
    USE_PERF_MODE="$1"
fi


# Initialize CSV
if [ "$USE_PERF_MODE" = "time" ] || [ "$USE_PERF_MODE" = "both" ]; then
    echo "matrix,mode,opt_level,schedule,chunk_size,num_threads,run,elapsed_time" > "$OUTPUT_TIME"
fi

if [ "$USE_PERF_MODE" = "perf" ] || [ "$USE_PERF_MODE" = "both" ]; then
    echo "matrix,mode,opt_level,schedule,chunk_size,num_threads,run,elapsed_time,L1_loads,L1_misses,L1_miss_rate,LLC_loads,LLC_misses,LLC_miss_rate" > "$OUTPUT_PERF"
fi

# Compilation function
compile_code() {
    local flags="$1"
    local parallel="$2"
    echo -e "\nCompilation with flags: $flags (OpenMP: $parallel)"

    rm -f $EXEC
    mkdir -p ../results

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


    if [ "$use_perf" = "yes" ]; then
        # CON PERF: esegui 10 volte l'eseguibile con repeats=1
        for ((r=1; r<=REPEATS; r++)); do
            if [ "$mode" = "sequential" ]; then
                PERF_OUTPUT=$(perf stat -x, -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
                    "$EXEC" "../data/$matrix" 1 2>&1)
            else
                PERF_OUTPUT=$(perf stat -x, -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
                    "$EXEC" "../data/$matrix" "$threads" "$schedule" "$chunk" 1 2>&1)
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

            elapsed_time=$(echo "$PERF_OUTPUT" | grep -E "Result_time:" | awk '{print $2}')

            if [ -z "$elapsed_time" ]; then
                elapsed_time="ERROR"
            fi

            # CSV perf
            echo "${matrix},${mode},${opt},${schedule},${chunk},${threads},${r},${elapsed_time},${L1_LOADS},${L1_MISSES},${L1_MISS_RATE},${LLC_LOADS},${LLC_MISSES},${LLC_MISS_RATE}" >> "$OUTPUT_PERF"
            # Terminal perf
            echo "${matrix} | ${mode} ${opt} | sched=${schedule} chunk=${chunk} threads=${threads} | run=${r} | time=${elapsed_time}s | L1miss=${L1_MISSES}/${L1_LOADS} (${L1_MISS_RATE}%) | LLCmiss=${LLC_MISSES}/${LLC_LOADS} (${LLC_MISS_RATE}%)"
        done
    else
        if [ "$mode" = "sequential" ]; then
            OUTPUT=$("$EXEC" "../data/$matrix" "$REPEATS" 2>&1)
        else
            OUTPUT=$("$EXEC" "../data/$matrix" "$threads" "$schedule" "$chunk" "$REPEATS" 2>&1)
        fi
        
        TIMES=($(echo "$OUTPUT" | grep "Result_time:" | awk '{print $2}'))
        
        if [ ${#TIMES[@]} -ne $REPEATS ]; then
            echo "WARNING: Expected $REPEATS times but got ${#TIMES[@]} for ${matrix}"
        fi
        
        for ((r=1; r<=REPEATS; r++)); do
            elapsed_time="${TIMES[$((r-1))]}"
            
            if [ -z "$elapsed_time" ]; then
                elapsed_time="ERROR"
            fi
            
            echo "${matrix},${mode},${opt},${schedule},${chunk},${threads},${r},${elapsed_time}" >> "$OUTPUT_TIME"
            echo "${matrix} | ${mode} ${opt} | sched=${schedule} chunk=${chunk} threads=${threads} | run=${r} | time=${elapsed_time}s"
        done
    fi
}

# SEQUENTIAL
echo -e "\n\n>>> Parte SEQUENZIALE...\n\n"
for matrix in "${MATRICES[@]}"; do
  for opt in "${OPT_FLAGS[@]}"; do
    compile_code "$opt" "no"
    if [ "$USE_PERF_MODE" = "time" ]; then
      run_and_record "$matrix" "sequential" "$opt" "none" "none" 1 "no"
    elif [ "$USE_PERF_MODE" = "perf" ]; then
      run_and_record "$matrix" "sequential" "$opt" "none" "none" 1 "yes"
    else
      run_and_record "$matrix" "sequential" "$opt" "none" "none" 1 "no"
      run_and_record "$matrix" "sequential" "$opt" "none" "none" 1 "yes"
    fi
  done
done

# PARALLEL
echo -e "\n\n>>> Parte PARALLELA...\n\n"
compile_code "-O3" "yes"

for matrix in "${MATRICES[@]}"; do
  for schedule in "${SCHEDULES[@]}"; do
    for chunk in "${CHUNKSIZES[@]}"; do
      for threads in "${THREADS[@]}"; do
        if [ "$USE_PERF_MODE" = "time" ]; then
          run_and_record "$matrix" "parallel" "-O3" "$schedule" "$chunk" "$threads" "no"
        elif [ "$USE_PERF_MODE" = "perf" ]; then
          run_and_record "$matrix" "parallel" "-O3" "$schedule" "$chunk" "$threads" "yes"
        else
          run_and_record "$matrix" "parallel" "-O3" "$schedule" "$chunk" "$threads" "no"
          run_and_record "$matrix" "parallel" "-O3" "$schedule" "$chunk" "$threads" "yes"
        fi
      done
    done
  done
done

echo -e "\nDone testing!"
if [ "$USE_PERF_MODE" = "time" ] || [ "$USE_PERF_MODE" = "both" ]; then
    echo "Time results saved in: $OUTPUT_TIME"
fi
if [ "$USE_PERF_MODE" = "perf" ] || [ "$USE_PERF_MODE" = "both" ]; then
    echo "Perf results saved in: $OUTPUT_PERF"
fi