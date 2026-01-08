#ifndef _COMMUNICATION_H_
#define _COMMUNICATION_H_

#include "utils.h"
#include <mpi.h>

typedef struct {
    // Ghost data
    int num_ghost_cols;           // ghost columns count
    int *ghost_col_indices;       // ghost global indices
    double *ghost_values;         
    int *ghost_to_local;          // global->local mapping (size = total_cols)
    int total_cols;
    
    int *send_counts;             
    int *recv_counts;             
    int *send_displs;             
    int *recv_displs;             

    int *send_indices;      
    int total_to_send;
} CommPattern;

void identifyGhostColumns(SparseMatrix *local_matrix, CommPattern *comm_pattern, int rank, int comm_size, int total_cols);
void exchangeGhostValues(double *local_x, CommPattern *comm_pattern, int rank, int comm_size);
void localSpMV(SparseMatrix *local_matrix, double *local_x, double *local_y, CommPattern *comm_pattern, int rank, int comm_size);
void parallelSpMV(SparseMatrix *local_matrix, double *local_x, double *local_y, CommPattern *comm_pattern, int rank, int comm_size);
void freeCommPattern(CommPattern *comm_pattern);

#endif
