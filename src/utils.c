
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "mmio.h"

int i,j;

void spVM(SparseMatrix* matrix, double* rvec) {
    double* res = (double*)malloc(matrix->rows*sizeof(double));
    #ifdef _OPENMP
        #pragma omp parallel for schedule(runtime)
    #endif
    for (i = 0; i < matrix->rows; ++i) {
        res[i] = 0;
        for (j = matrix->row_ptr[i]; j < matrix->row_ptr[i+1]; ++j) {
            res[i] += matrix->vals[j] * rvec[matrix->col_ind[j]];
        }
    }
    free(res);
}

int loadMatrixMarket(const char *filename, SparseMatrix* matrix)
{
    FILE *f;
    MM_typecode matcode;
    int ret_code;
    int i;

    printf("Retrieving the matrix from '%s'...\n", filename);

    if ((f = fopen(filename, "r")) == NULL)
    {
        fprintf(stderr, "Error: could not open file %s\n", filename);
        return -1;
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        fclose(f);
        return -1;
    }

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode))
    {
        printf("Sorry, this application does not support type: [%s]\n", mm_typecode_to_str(matcode));
        fclose(f);
        return -1;
    }

    if ((ret_code = mm_read_mtx_crd_size(f, &matrix->rows, &matrix->cols, &matrix->nz)) != 0)
    {
        fprintf(stderr, "Error reading matrix size.\n");
        fclose(f);
        return -1;
    }

    matrix->Arow = malloc((matrix->nz) * sizeof(int));
    matrix->Acol = malloc((matrix->nz) * sizeof(int));
    matrix->Aval = malloc((matrix->nz) * sizeof(double));

    if (!matrix->Arow || !matrix->Acol || !matrix->Aval)
    {
        fprintf(stderr, "Memory allocation failed.\n");
        fclose(f);
        return -1;
    }

    for (i = 0; i < matrix->nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &matrix->Arow[i], &matrix->Acol[i], &matrix->Aval[i]);
        matrix->Arow[i]--;  // from 1-based to 0-based
        matrix->Acol[i]--;
    }

    fclose(f);

    return 0;
}

void printVectorInt(char* name, int *v, int size) {
    printf("%s[", name);
    for (i = 0; i < size; i++) {
        printf(" %d ", v[i]);
    }
    printf("]\n");
}

void printVectorDouble(char* name, double *v, int size) {
    printf("%s[", name);
    for (i = 0; i < size; i++) {
        printf(" %f ", v[i]);
    }
    printf("]\n");
}

void printCOO(SparseMatrix* matrix) {
    printf("\nMatrix in COO:\n");
    printVectorInt("Arow", matrix->Arow, matrix->nz);
    printVectorInt("Acol", matrix->Acol, matrix->nz);
    printVectorDouble("Values", matrix->Aval, matrix->nz);
}

void printCSR(SparseMatrix* matrix) {
    printf("\nMatrix in CSR:\n");
    printVectorInt("row_ptr", matrix->row_ptr, matrix->rows + 1);
    printVectorInt("col_ind", matrix->col_ind, matrix->row_ptr[matrix->rows]);
    printVectorDouble("vals", matrix->vals, matrix->row_ptr[matrix->rows]);
}

void sortCSRRows(SparseMatrix* matrix) {
    for (i = 0; i < matrix->rows; i++) {
        int start = matrix->row_ptr[i];
        int end = matrix->row_ptr[i+1];

        for (j = start + 1; j < end; j++) {
            int key_col = matrix->col_ind[j];
            double key_val = matrix->vals[j];
            int k = j - 1;

            while (k >= start && matrix->col_ind[k] > key_col) {
                matrix->col_ind[k+1] = matrix->col_ind[k];
                matrix->vals[k+1] = matrix->vals[k];
                k--;
            }
            matrix->col_ind[k+1] = key_col;
            matrix->vals[k+1] = key_val;
        }
    }
}

void COOtoCSR(SparseMatrix* matrix) {
    //printf("\nConverting the matrix from COO to CSR format...\n");
        for (i = 0; i <= matrix->rows; i++) {
            matrix->row_ptr[i] = 0;
        }

        for (i = 0; i < matrix->nz; i++) {
            matrix->row_ptr[matrix->Arow[i] + 1]++;
        }

        for (i = 1; i <= matrix->rows; i++) {
            matrix->row_ptr[i] += matrix->row_ptr[i - 1];
        }

        int* current_pos = (int *)malloc(matrix->rows * sizeof(int));
        for (i = 0; i < matrix->rows; i++) {
            current_pos[i] = matrix->row_ptr[i];
        }

        for (i = 0; i < matrix->nz; i++) {
            int row = matrix->Arow[i];
            int dest = current_pos[row];

            matrix->col_ind[dest] = matrix->Acol[i];
            matrix->vals[dest] = matrix->Aval[i];

            current_pos[row]++;
        }
        free(current_pos);

        //printf("Sorting columns within each row...\n");
        sortCSRRows(matrix);

        //printf("Job done, ready for SpMV Matrix-Vector multiplication...\n");
}

double* randVect(double* rvec, int COLS){
    for (i = 0; i < COLS; i++) {
        rvec[i] = ((double)rand() / RAND_MAX) * 8.0 - 4.0;
    }
    return rvec;
}

void freeSparseMatrix(SparseMatrix *matrix) {
    free(matrix->Arow);
    free(matrix->Acol);
    free(matrix->Aval);
    free(matrix->row_ptr);
    free(matrix->col_ind);
    free(matrix->vals);
}