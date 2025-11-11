
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "mmio.h"

int i,j;

void spVM(int *row_ptr, int *col_ind, double *vals , double* rvec, int ROWS) {
    double* res = (double*)malloc(ROWS*sizeof(double));
    #ifdef _OPENMP
        #pragma omp parallel for schedule(runtime)
    #endif
    for (i = 0; i < ROWS; ++i) {
        res[i] = 0;
        for (j = row_ptr[i]; j < row_ptr[i+1]; ++j) {
            res[i] += vals[j] * rvec[col_ind[j]];
        }
    }
}

int loadMatrixMarket(const char *filename,int *ROWS, int *COLS, int *nz, int **Arow, int **Acol, double **Aval)
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

    if ((ret_code = mm_read_mtx_crd_size(f, ROWS, COLS, nz)) != 0)
    {
        fprintf(stderr, "Error reading matrix size.\n");
        fclose(f);
        return -1;
    }

    *Arow = malloc((*nz) * sizeof(int));
    *Acol = malloc((*nz) * sizeof(int));
    *Aval = malloc((*nz) * sizeof(double));

    if (!*Arow || !*Acol || !*Aval)
    {
        fprintf(stderr, "Memory allocation failed.\n");
        fclose(f);
        return -1;
    }

    for (i = 0; i < *nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &(*Arow)[i], &(*Acol)[i], &(*Aval)[i]);
        (*Arow)[i]--;  // from 1-based to 0-based
        (*Acol)[i]--;
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

void printCOO(int nz, int* Arow, int* Acol, double* Aval) {
    printf("\nMatrix in COO:\n");
    printVectorInt("Arow", Arow, nz);
    printVectorInt("Acol", Acol, nz);
    printVectorDouble("Values", Aval, nz);
}

void printCSR(int ROWS, int* row_ptr, int* col_ind, double* vals) {
    printf("\nMatrix in CSR:\n");
    printVectorInt("row_ptr", row_ptr, ROWS + 1);
    printVectorInt("col_ind", col_ind, row_ptr[ROWS]);
    printVectorDouble("vals", vals, row_ptr[ROWS]);
}

void sortCSRRows(int ROWS, int* row_ptr, int* col_ind, double* vals) {
    for (i = 0; i < ROWS; i++) {
        int start = row_ptr[i];
        int end = row_ptr[i+1];

        for (j = start + 1; j < end; j++) {
            int key_col = col_ind[j];
            double key_val = vals[j];
            int k = j - 1;

            while (k >= start && col_ind[k] > key_col) {
                col_ind[k+1] = col_ind[k];
                vals[k+1] = vals[k];
                k--;
            }
            col_ind[k+1] = key_col;
            vals[k+1] = key_val;
        }
    }
}

void COOtoCSR(int ROWS, int nz, int* Arow, int* Acol, double* Aval, int* row_ptr, int* col_ind, double* vals) {
    //printf("\nConverting the matrix from COO to CSR format...\n");
        for (i = 0; i <= ROWS; i++) {
            row_ptr[i] = 0;
        }

        for (i = 0; i < nz; i++) {
            row_ptr[Arow[i] + 1]++;
        }

        for (i = 1; i <= ROWS; i++) {
            row_ptr[i] += row_ptr[i - 1];
        }

        int* current_pos = (int *)malloc(ROWS * sizeof(int));
        for (i = 0; i < ROWS; i++) {
            current_pos[i] = row_ptr[i];
        }

        for (i = 0; i < nz; i++) {
            int row = Arow[i];
            int dest = current_pos[row];

            col_ind[dest] = Acol[i];
            vals[dest] = Aval[i];

            current_pos[row]++;
        }
        free(current_pos);

        //printf("Sorting columns within each row...\n");
        sortCSRRows(ROWS, row_ptr, col_ind, vals);

        //printf("Job done, ready for SpMV Matrix-Vector multiplication...\n");
}

double* randVect(double* rvec, int COLS){
    for (i = 0; i < COLS; i++) {
        rvec[i] = ((double)rand() / RAND_MAX) * 8.0 - 4.0;
    }
    return rvec;
}
