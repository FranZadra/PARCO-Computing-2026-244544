#include "distribute.h"
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

// STRONG SCALING: Real matrix distribution and conversion COO to CSR
void distributeMatrix(SparseMatrix *matrix, SparseMatrix *localMatrix, int rank, int comm_size) {
    int i;
    // Cyclic row distribution: owner(i) = i mod P
    int local_rows = matrix->rows / comm_size;
    if (rank < (matrix->rows % comm_size)) {
        local_rows++; 
    }
    
    localMatrix->rows = local_rows;
    localMatrix->cols = matrix->cols;
    
    // Local mapping -> global
    localMatrix->global_row_indices = (int*)malloc(local_rows * sizeof(int));
    for (i = 0; i < local_rows; i++) {
        localMatrix->global_row_indices[i] = rank + i * comm_size;
    }
    
    // Local NNZ count
    localMatrix->nz = 0;
    for (i = 0; i < matrix->nz; i++) {
        int row_owner = matrix->Arow[i] % comm_size; 
        if (row_owner == rank) {
            localMatrix->nz++;
        }
    }
    
    // Local COO
    if (localMatrix->nz > 0) {
        localMatrix->Arow = (int*)malloc(localMatrix->nz * sizeof(int));
        localMatrix->Acol = (int*)malloc(localMatrix->nz * sizeof(int));
        localMatrix->Aval = (double*)malloc(localMatrix->nz * sizeof(double));
    } else {
        localMatrix->Arow = NULL;
        localMatrix->Acol = NULL;
        localMatrix->Aval = NULL;
    }

    // NNZ copy with global -> local conversion
    int index = 0;
    for (i = 0; i < matrix->nz; i++) {
        int global_row = matrix->Arow[i];
        int row_owner = global_row % comm_size;
        
        if (row_owner == rank) {
            localMatrix->Arow[index] = global_row / comm_size;  
            localMatrix->Acol[index] = matrix->Acol[i];        
            localMatrix->Aval[index] = matrix->Aval[i];
            index++;
        }
    }
    
    // Local CSR
    if (localMatrix->nz > 0) {
        localMatrix->row_ptr = (int*)malloc((localMatrix->rows + 1) * sizeof(int));
        localMatrix->col_ind = (int*)malloc(localMatrix->nz * sizeof(int));
        localMatrix->vals = (double*)malloc(localMatrix->nz * sizeof(double));
        
        COOtoCSR(localMatrix);
    } else {
        localMatrix->row_ptr = (int*)calloc(localMatrix->rows + 1, sizeof(int));
        localMatrix->col_ind = NULL;
        localMatrix->vals = NULL;
    }
}

// WEAK SCALING: Synthetic matrix generation, distribution and conversion COO to CSR
void generateLocalMatrix(SparseMatrix *local_matrix, int rows_per_proc, int total_cols, int nnz_per_row, int rank, int size) {
    int i, j;
    
    local_matrix->rows = rows_per_proc;
    local_matrix->cols = total_cols;
    
    // Allocate global row indices
    local_matrix->global_row_indices = (int*)malloc(rows_per_proc * sizeof(int));
    if (local_matrix->global_row_indices == NULL) {
        fprintf(stderr, "[Rank %d] ERROR: Failed to allocate global_row_indices\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    for (i = 0; i < rows_per_proc; i++) {
        local_matrix->global_row_indices[i] = rank + i * size;
    }
    
    local_matrix->row_ptr = (int*)malloc((rows_per_proc + 1) * sizeof(int));
    if (local_matrix->row_ptr == NULL) {
        fprintf(stderr, "[Rank %d] ERROR: Failed to allocate row_ptr\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Determine nnz per row and total nnz
    srand(rank * 12345 + 789);
    int total_nz = 0;
    local_matrix->row_ptr[0] = 0;
    
    for (i = 0; i < rows_per_proc; i++) {
        int actual_nnz = nnz_per_row + (rand() % 5) - 2;
        if (actual_nnz < 1) actual_nnz = 1;
        if (actual_nnz > total_cols) actual_nnz = total_cols;
        
        total_nz += actual_nnz;
        local_matrix->row_ptr[i + 1] = total_nz;
    }
    
    local_matrix->nz = total_nz;
    
    // CSR arrays
    local_matrix->col_ind = (int*)malloc(total_nz * sizeof(int));
    local_matrix->vals = (double*)malloc(total_nz * sizeof(double));
    
    if (local_matrix->col_ind == NULL || local_matrix->vals == NULL) {
        fprintf(stderr, "[Rank %d] ERROR: Failed to allocate CSR arrays (nz=%d)\n", rank, total_nz);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Generate column indices and values
    srand(rank * 12345 + 789);
    
    for (i = 0; i < rows_per_proc; i++) {
        int row_start = local_matrix->row_ptr[i];
        int row_end = local_matrix->row_ptr[i + 1];
        int row_nnz = row_end - row_start;
        
        // Generate unique random columns
        for (j = 0; j < row_nnz; j++) {
            int col = rand() % total_cols;
            
            int duplicate = 0;
            for (int k = row_start; k < row_start + j; k++) {
                if (local_matrix->col_ind[k] == col) {
                    duplicate = 1;
                    col = (col + 1) % total_cols;
                    k = row_start - 1; 
                }
            }
            
            local_matrix->col_ind[row_start + j] = col;
            local_matrix->vals[row_start + j] = ((double)rand() / RAND_MAX) * 10.0;
        }
    }

    local_matrix->Arow = NULL;
    local_matrix->Acol = NULL;
    local_matrix->Aval = NULL;
}