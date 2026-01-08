#ifndef _UTILS_H_
#define _UTILS_H_

#include "mmio.h"

typedef struct {
    int rows;
    int cols;
    int nz;
    
    int *Arow;
    int *Acol;
    double *Aval;
    
    int *row_ptr;
    int *col_ind;
    double *vals;

    int *global_row_indices;
} SparseMatrix;

typedef struct {
    double *elapsed_times;
    double *comm_times;
    int num_repeats;

    long long local_flops;
    int local_nz;
    int ghost_entries;
} PerformanceMetrics;

#include "distribute.h"
#include "communication.h"

void printMPIUsage(char* prog_name);
void spVM(SparseMatrix* matrix, double* rvec, double *res);

int loadMatrixMarket(const char *filename, SparseMatrix* matrix);
void COOtoCSR(SparseMatrix* matrix );
void sortCSRRows(SparseMatrix* matrix);

double* randVect(double* rvec, int COLS);

void freeSparseMatrix(SparseMatrix *matrix);

void printMatrixInfo(SparseMatrix* matrix);
void printVectorInt(char* name, int* v, int size);
void printVectorDouble(char* name, double* v, int size);
void printCOO(SparseMatrix* matrix);
void printCSR(SparseMatrix* matrix);

#endif
