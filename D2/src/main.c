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
        }
        #else
        if(argc != 3) {
            fprintf(stderr, "Usage: %s <matrix_market_file>\n", argv[0]);
            exit(EXIT_FAILURE);
        } 
        repeats = atoi(argv[2]);
        #endif
    }

    if (loadMatrixMarket(argv[1], &matrix) != 0)
    {
        fprintf(stderr, "Failed to load matrix.\n");
        exit(EXIT_FAILURE);
    }

    matrix.row_ptr = malloc((matrix.rows + 1) * sizeof(int));
    matrix.col_ind = malloc(matrix.nz * sizeof(int));
    matrix.vals = malloc(matrix.nz * sizeof(double));
    
    COOtoCSR(&matrix);

    double* rvec = (double*)malloc(matrix.cols * sizeof(double));
    double* res = (double*)malloc(matrix.rows*sizeof(double));
    rvec = randVect(rvec, matrix.cols);

    // Calculate total bytes transferred (bandwidth)
    long long bytes_transferred = 
        (long long)matrix.nz * sizeof(double) +      
        (long long)matrix.nz * sizeof(int) +         
        (long long)(matrix.rows + 1) * sizeof(int) + 
        (long long)matrix.cols * sizeof(double) +  
        (long long)matrix.rows * sizeof(double);     
    
    double bytes_gb = bytes_transferred / (1024.0 * 1024.0 * 1024.0);

    // Calculate flops
    long long flops = 2LL * matrix.nz;

    // cache warm-up
    spVM(&matrix, rvec, res);

    for(i = 0; i < repeats; i++) {
        GET_TIME(start)
        spVM(&matrix, rvec, res);
        GET_TIME(finish)
        elapsed = finish - start;

        double bandwidth_gbs = bytes_gb / elapsed;
        double gflops = (flops / elapsed) / 1e9;  // GFLOPS

        printf("Result_time: %f\n", elapsed * 1000.0);
        printf("Result_bandwidth: %.6f\n", bandwidth_gbs);
        printf("Result_gflops: %.6f\n", gflops);
    }
        
    
    freeSparseMatrix(&matrix);
    free(rvec);
    free(res);

	return 0;
}




