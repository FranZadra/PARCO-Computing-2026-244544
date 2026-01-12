#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#ifdef _OPENMP
    #include <omp.h>
#endif
#include "utils.h"
#include "distribute.h"
#include "communication.h"


int main(int argc, char *argv[]) {
    int rank, comm_size;
    int provided;
    #ifdef _OPENMP
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        if (provided < MPI_THREAD_FUNNELED) {
            if (rank == 0) {
                fprintf(stderr, "ERROR: MPI implementation does not support MPI_THREAD_FUNNELED\n");
                fprintf(stderr, "Provided level: %d, Required: %d\n", provided, MPI_THREAD_FUNNELED);
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (rank == 0) {
            printf("MPI thread support: MPI_THREAD_FUNNELED (provided=%d)\n", provided);
        }
    #else
        MPI_Init(&argc, &argv);
    #endif
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    
    srand(time(NULL) + rank * 1000);
    
    double start_total = MPI_Wtime();
    
    if (argc < 3) {
        if (rank == 0) {
            printMPIUsage(argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    
    char* matrixFile = argv[1];
    int repeats = atoi(argv[2]);
    int synthetic = (strcmp(matrixFile, "synthetic") == 0);
    
    int rows_per_proc = 0;
    int nnz_per_row = 0;
    if (synthetic) {
        if (argc < 5) {
            if (rank == 0) {
                fprintf(stderr, "Error: Synthetic mode requires rows_per_proc and nnz_per_row\n");
                printMPIUsage(argv[0]);
            }
            MPI_Finalize();
            return EXIT_FAILURE;
        }
        rows_per_proc = atoi(argv[3]);
        nnz_per_row = atoi(argv[4]);
    }

    // MATRIX LOADING
    SparseMatrix matrix;
    int matrixRows, matrixCols, matrixNz;
    
    if (!synthetic) {
        if (rank == 0) {
            if (loadMatrixMarket(matrixFile, &matrix) != 0) {
                fprintf(stderr, "Failed to load matrix\n");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            
            matrix.row_ptr = malloc((matrix.rows + 1) * sizeof(int));
            matrix.col_ind = malloc(matrix.nz * sizeof(int));
            matrix.vals = malloc(matrix.nz * sizeof(double));

            if (!matrix.row_ptr || !matrix.col_ind || !matrix.vals) {
                fprintf(stderr, "Memory allocation failed\n");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            
            COOtoCSR(&matrix);
            
            matrixRows = matrix.rows;
            matrixCols = matrix.cols;
            matrixNz = matrix.nz;

            printf("Real Matrix (Strong Scaling): Rows=%d | Cols=%d | NNZ=%d | Procs=%d\n", matrixRows, matrixCols, matrixNz, comm_size);
        }
        
        MPI_Bcast(&matrixRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&matrixCols, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&matrixNz, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            matrix.rows = matrixRows;
            matrix.cols = matrixCols;
            matrix.nz = matrixNz;
            
            matrix.Arow = malloc(matrix.nz * sizeof(int));
            matrix.Acol = malloc(matrix.nz * sizeof(int));
            matrix.Aval = malloc(matrix.nz * sizeof(double));
        }
        MPI_Bcast(matrix.Arow, matrixNz, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(matrix.Acol, matrixNz, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(matrix.Aval, matrixNz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
    } else {
        matrixRows = rows_per_proc * comm_size;
        matrixCols = matrixRows;
        matrixNz = 0;
        
        if (rank == 0) {
            printf("Synthetic Matrix (Weak Scaling): Rows/proc=%d | Total rows=%d | NNZ/row=~%d | Procs=%d\n", rows_per_proc, matrixRows, nnz_per_row, comm_size);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // MATRIX DISTRIBUTION AMONG PROCESSES
    SparseMatrix localMatrix;

    if (!synthetic) {
        distributeMatrix(&matrix, &localMatrix, rank, comm_size);
    } else {
        generateLocalMatrix(&localMatrix, rows_per_proc, matrixCols, nnz_per_row, rank, comm_size);
        
        int local_nz = localMatrix.nz;

        MPI_Barrier(MPI_COMM_WORLD);
        
        MPI_Allreduce(&local_nz, &matrixNz, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }
        
    if (rank == 0) {
        printf("Matrix distributed and converted to CSR\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    // GHOST ELEMENTS IDENTIFICATION
    CommPattern commPattern;
    identifyGhostColumns(&localMatrix, &commPattern, rank, comm_size, matrixCols);
    
    if (rank == 0) {
        printf("Ghost columns identified: %d total ghost values needed\n", commPattern.num_ghost_cols);
    }
    
    // RANDOM VECTOR DISTRIBUTION - CYCLIC
    // Owner of element i is: rank = i % comm_size
    int local_vec_size = matrixCols / comm_size;
    if (rank < (matrixCols % comm_size)) {
        local_vec_size++;
    }

    double *localRandVec = (double*)malloc(local_vec_size * sizeof(double));
    double *localResult = (double*)malloc(localMatrix.rows * sizeof(double));

    if (localRandVec == NULL || localResult == NULL) {
        fprintf(stderr, "[Rank %d] ERROR: Failed to allocate vectors\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Initialize with cyclic ownership
    for (int i = 0; i < local_vec_size; i++) {
        // Global index with cyclic distribution
        int global_idx = rank + i * comm_size;
        localRandVec[i] = ((double)rand() / RAND_MAX) * 8.0 - 4.0;
    }
    
    if (rank == 0) {
        printf("Local vector initialized (size=%d per process)\n", local_vec_size);
    }
    
    // CACHE WARMUP
    parallelSpMV(&localMatrix, localRandVec, localResult, &commPattern, rank, comm_size);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // SPMV BENCHMARK and DATA COLLECTION
    PerformanceMetrics metrics;
    metrics.local_nz = localMatrix.nz;
    metrics.ghost_entries = commPattern.num_ghost_cols;
    metrics.local_flops = 2LL * localMatrix.nz;
    metrics.num_repeats = repeats;

    metrics.elapsed_times = (double*)malloc(repeats * sizeof(double));
    metrics.comm_times = (double*)malloc(repeats * sizeof(double));

    for (int r = 0; r < repeats; r++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double iter_start = MPI_Wtime();
        
        double comm_start = MPI_Wtime();
        exchangeGhostValues(localRandVec, &commPattern, rank, comm_size);
        double comm_end = MPI_Wtime();
        metrics.comm_times[r] = comm_end - comm_start;
        
        localSpMV(&localMatrix, localRandVec, localResult, &commPattern, rank, comm_size);
        
        double iter_end = MPI_Wtime();
        metrics.elapsed_times[r] = iter_end - iter_start;
    }


    // RAW OUTPUT DATA
    for (int r = 0; r < repeats; r++) {
        printf("[RESULT] %d,%d,%d,%.9f,%.9f,%d,%d,%lld\n",
            rank, comm_size, r,
            metrics.elapsed_times[r], 
            metrics.comm_times[r],
            metrics.local_nz,
            metrics.ghost_entries,
            metrics.local_flops);
        fflush(stdout); 
    }

    if (rank == 0) {
        printf("\nBenchmark Summary\n");
        printf("Processes: %d\n", comm_size);
        printf("Iterations: %d\n", repeats);
        printf("Matrix NNZ: %d\n", matrixNz);
    }

    double end_total = MPI_Wtime();
    if (rank == 0) {
        printf("Total execution time: %.3f seconds\n", end_total - start_total);
    }

    free(metrics.elapsed_times);
    free(metrics.comm_times);
    freeSparseMatrix(&localMatrix);
    freeCommPattern(&commPattern);
    free(localRandVec);
    free(localResult);
    if (!synthetic) {
        if (matrix.Arow) free(matrix.Arow);
        if (matrix.Acol) free(matrix.Acol);
        if (matrix.Aval) free(matrix.Aval);
        
        if (rank == 0) {
            if (matrix.row_ptr) free(matrix.row_ptr);
            if (matrix.col_ind) free(matrix.col_ind);
            if (matrix.vals) free(matrix.vals);
        }
    }
    MPI_Finalize();
    return EXIT_SUCCESS;
}
