#ifndef _DISTRIBUTE_H_
#define _DISTRIBUTE_H_

#include "utils.h"

void distributeMatrix(SparseMatrix *global_matrix, SparseMatrix *local_matrix, int rank, int size);
void generateLocalMatrix(SparseMatrix *local_matrix, int rows_per_proc, int total_cols, int nnz_per_row, int rank, int size);

#endif
