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
    // Local matrix dimensions
    local_matrix->rows = rows_per_proc;
    local_matrix->cols = total_cols;
    
    // Local -> global mapping with cyclic distribution
    local_matrix->global_row_indices = (int*)malloc(rows_per_proc * sizeof(int));
    for (i = 0; i < rows_per_proc; i++) {
        local_matrix->global_row_indices[i] = rank + i * size;
    }
    
    int total_nz = 0;
    for (i = 0; i < rows_per_proc; i++) {
        int actual_nnz = nnz_per_row + (rand() % 5) - 2;
        if (actual_nnz < 1) actual_nnz = 1;
        if (actual_nnz > total_cols) actual_nnz = total_cols;
        total_nz += actual_nnz;
    }

    // Local COO
    local_matrix->nz = total_nz;
    local_matrix->Arow = (int*)malloc(total_nz * sizeof(int));
    local_matrix->Acol = (int*)malloc(total_nz * sizeof(int));
    local_matrix->Aval = (double*)malloc(total_nz * sizeof(double));
    
    // Random NNZ generation
    srand(rank * 12345);
    int index = 0;
    for (i = 0; i < rows_per_proc; i++) {
        int actual_nnz = nnz_per_row + (rand() % 5) - 2;
        if (actual_nnz < 1) actual_nnz = 1;
        if (actual_nnz > total_cols) actual_nnz = total_cols;
        
        int *used_cols = (int*)calloc(total_cols, sizeof(int));
        
        for (j = 0; j < actual_nnz; j++) {
            int col;
            do {
                col = rand() % total_cols;
            } while (used_cols[col]);
            
            used_cols[col] = 1;
            
            local_matrix->Arow[index] = i;
            local_matrix->Acol[index] = col;
            local_matrix->Aval[index] = ((double)rand() / RAND_MAX) * 10.0;
            index++;
        }
        
        free(used_cols);
    }

    // Local CSR
    local_matrix->row_ptr = (int*)malloc((local_matrix->rows + 1) * sizeof(int));
    local_matrix->col_ind = (int*)malloc(local_matrix->nz * sizeof(int));
    local_matrix->vals = (double*)malloc(local_matrix->nz * sizeof(double));
    
    COOtoCSR(local_matrix);
}