#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "timer.h"
#include "utils.h"
#include "ompconfig.h"
#ifdef _OPENMP
    #include <omp.h>
#endif


int main(int argc, char *argv[])
{
    srand(time(NULL));
    double start, finish, elapsed; 
    int repeats, i;
    SparseMatrix matrix;

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <matrix_market_file>\n", argv[0]);
        exit(EXIT_FAILURE);
    } else {
        //set the compiler flags
        #ifdef _OPENMP
        if(argc != 6) {
            fprintf(stderr, "Usage with OpenMP: %s <matrix_market_file> <num_threads> <schedule_type> <chunk_size> <repeats>\n", argv[0]);
            exit(EXIT_FAILURE);
        } else {
            OmpConfig ompConf;
            ompConf.num_threads = atoi(argv[2]);
            ompConf.schedule_type = parseSchedule(argv[3]);
            ompConf.chunk_size = atoi(argv[4]);
            repeats = atoi(argv[5]);

            setOmpConfig(ompConf);
            printOmpConfig(ompConf);
        }
        #else
        if(argc != 3) {
            fprintf(stderr, "Usage: %s <matrix_market_file>\n", argv[0]);
            exit(EXIT_FAILURE);
        } 
        repeats = atoi(argv[2]);
        //printf("OpenMP not enabled. Running in sequential mode.\n");
        #endif
    }

    if (loadMatrixMarket(argv[1], &matrix) != 0)
    {
        fprintf(stderr, "Failed to load matrix.\n");
        exit(EXIT_FAILURE);
    }
    //printf("Matrix loaded: %d x %d, non-zeros: %d\n", matrix.rows, matrix.cols, matrix.nz);

    matrix.row_ptr = malloc((matrix.rows + 1) * sizeof(int));
    matrix.col_ind = malloc(matrix.nz * sizeof(int));
    matrix.vals = malloc(matrix.nz * sizeof(double));
    
    COOtoCSR(&matrix);

        double* rvec = (double*)malloc(matrix.cols * sizeof(double));
        rvec = randVect(rvec, matrix.cols);

        for(i = 0; i < repeats; i++) {
            GET_TIME(start)
            spVM(&matrix, rvec);
            GET_TIME(finish)
            elapsed = finish - start;
            printf("Result_time: %f\n", elapsed);
        }
        
    
        freeSparseMatrix(&matrix);
        free(rvec);

	return 0;
}




