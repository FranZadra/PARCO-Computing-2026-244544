#include <stdlib.h>
#include <string.h>
#include "communication.h"

void identifyGhostColumns(SparseMatrix *local_matrix, CommPattern *comm_pattern, int rank, int comm_size, int total_cols) {
    int i, col;
    
    int *col_present = (int*)calloc(total_cols, sizeof(int));
    for (i = 0; i < local_matrix->nz; i++) {
        col_present[local_matrix->col_ind[i]] = 1;
    }

    comm_pattern->send_counts = (int*)calloc(comm_size, sizeof(int));
    comm_pattern->recv_counts = (int*)calloc(comm_size, sizeof(int));
    comm_pattern->send_displs = (int*)calloc(comm_size, sizeof(int));
    comm_pattern->recv_displs = (int*)calloc(comm_size, sizeof(int));

    int ghost_count = 0;
    for (col = 0; col < total_cols; col++) {
        if (col_present[col] && (col % comm_size != rank)) {
            int owner = col % comm_size;
            comm_pattern->recv_counts[owner]++;
            ghost_count++;
        }
    }
    comm_pattern->num_ghost_cols = ghost_count;
    comm_pattern->total_cols = total_cols;

    comm_pattern->recv_displs[0] = 0;
    for (i = 1; i < comm_size; i++) {
        comm_pattern->recv_displs[i] = comm_pattern->recv_displs[i-1] + comm_pattern->recv_counts[i-1];
    }

    comm_pattern->ghost_col_indices = (int*)malloc(ghost_count * sizeof(int));
    comm_pattern->ghost_values = (double*)malloc(ghost_count * sizeof(double));
    
    int *current_pos = (int*)malloc(comm_size * sizeof(int));
    for(i = 0; i < comm_size; i++) {
        current_pos[i] = comm_pattern->recv_displs[i];
    }

    comm_pattern->ghost_to_local = (int*)malloc(total_cols * sizeof(int));
    for (i = 0; i < total_cols; i++) {
        comm_pattern->ghost_to_local[i] = -1;
    }

    for (col = 0; col < total_cols; col++) {
        if (col_present[col] && (col % comm_size != rank)) {
            int owner = col % comm_size;
            int pos = current_pos[owner]++;
            
            comm_pattern->ghost_col_indices[pos] = col;
            comm_pattern->ghost_to_local[col] = pos;
        }
    }
    free(current_pos);
    free(col_present);

    MPI_Alltoall(comm_pattern->recv_counts, 1, MPI_INT, comm_pattern->send_counts, 1, MPI_INT, MPI_COMM_WORLD);
    
    comm_pattern->send_displs[0] = 0;
    for (i = 1; i < comm_size; i++) {
        comm_pattern->send_displs[i] = comm_pattern->send_displs[i-1] + comm_pattern->send_counts[i-1];
    }

    comm_pattern->total_to_send = comm_pattern->send_displs[comm_size-1] + comm_pattern->send_counts[comm_size-1];
    comm_pattern->send_indices = (int*)malloc(comm_pattern->total_to_send * sizeof(int));

    MPI_Alltoallv(comm_pattern->ghost_col_indices, 
                  comm_pattern->recv_counts, comm_pattern->recv_displs, MPI_INT,
                  comm_pattern->send_indices, 
                  comm_pattern->send_counts, comm_pattern->send_displs, MPI_INT,
                  MPI_COMM_WORLD);
    
    for (i = 0; i < comm_pattern->total_to_send; i++) {
        comm_pattern->send_indices[i] /= comm_size;
    }
}

void exchangeGhostValues(double *local_x, CommPattern *comm_pattern, int rank, int comm_size) {
    double *send_buf = (double*)malloc((comm_pattern->total_to_send > 0 ? comm_pattern->total_to_send : 1) * sizeof(double));
    int i;
    for (i = 0; i < comm_pattern->total_to_send; i++) {
        int local_idx = comm_pattern->send_indices[i];
        send_buf[i] = local_x[local_idx];
    }
    
    MPI_Alltoallv(send_buf, 
                  comm_pattern->send_counts, comm_pattern->send_displs, MPI_DOUBLE,
                  comm_pattern->ghost_values, 
                  comm_pattern->recv_counts, comm_pattern->recv_displs, MPI_DOUBLE,
                  MPI_COMM_WORLD);
    
    free(send_buf);
}

void localSpMV(SparseMatrix *local_matrix, double *localRandVec, double *localResult, CommPattern *comm_pattern, int rank, int comm_size) {
    int i, j;
    for (i = 0; i < local_matrix->rows; i++) {
        double sum = 0.0;
        
        for (j = local_matrix->row_ptr[i]; j < local_matrix->row_ptr[i + 1]; j++) {
            int global_col = local_matrix->col_ind[j];
            double vect_val;
            
            int col_owner = global_col % comm_size;
            
            if (col_owner == rank) {
                //local
                int local_idx = global_col / comm_size;
                vect_val = localRandVec[local_idx];
            } else {
                //ghost
                int ghost_idx = comm_pattern->ghost_to_local[global_col];
                vect_val = comm_pattern->ghost_values[ghost_idx];
            }
            
            sum += local_matrix->vals[j] * vect_val;
        }

        localResult[i] = sum;
    }
}

void parallelSpMV(SparseMatrix *local_matrix, double *localRandVec, double *localResult, CommPattern *comm_pattern, int rank, int comm_size) {
    exchangeGhostValues(localRandVec, comm_pattern, rank, comm_size);
    localSpMV(local_matrix, localRandVec, localResult, comm_pattern, rank, comm_size);
}

void freeCommPattern(CommPattern *comm_pattern) {
    free(comm_pattern->ghost_col_indices);
    free(comm_pattern->ghost_values);
    free(comm_pattern->ghost_to_local);
    free(comm_pattern->send_counts);
    free(comm_pattern->recv_counts);
    free(comm_pattern->send_displs);
    free(comm_pattern->recv_displs);
    free(comm_pattern->send_indices); 
}
