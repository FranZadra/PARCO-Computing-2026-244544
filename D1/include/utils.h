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
} SparseMatrix;

void spVM(SparseMatrix* matrix, double* rvec, double *res);
int loadMatrixMarket(const char *filename, SparseMatrix* matrix);
void printVectorInt(char* name, int* v, int size);
void printVectorDouble(char* name, double* v, int size);
void sortCSRRows(SparseMatrix* matrix);
void printCOO(SparseMatrix* matrix);
void printCSR(SparseMatrix* matrix);
void COOtoCSR(SparseMatrix* matrix );
double* randVect(double* rvec, int COLS);
void freeSparseMatrix(SparseMatrix *matrix);
#endif
