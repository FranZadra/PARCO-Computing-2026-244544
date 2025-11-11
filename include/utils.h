#ifndef _UTILS_H_
#define _UTILS_H_

#include "mmio.h"

void spVM(int *row_ptr, int *col_ind, double* vals , double* rvec, int ROWS);
int loadMatrixMarket(const char *filename, int *ROWS, int *COLS, int *nz, int **Arow, int **Acol, double **Aval);
void printVectorInt(char* name, int* v, int size);
void printVectorDouble(char* name, double* v, int size);
void sortCSRRows(int ROWS, int* row_ptr, int* col_ind, double* vals);
void printCOO(int nz, int* Arow, int* Acol, double* Aval);
void printCSR(int ROWS, int* row_ptr, int* col_ind, double* vals);
void COOtoCSR(int ROWS, int nz, int* Arow, int* Acol, double* Aval, int* row_ptr, int* col_ind, double* vals);
double* randVect(double* rvec, int COLS);

#endif
