#ifndef OMP_CONFIG_H
#define OMP_CONFIG_H

#include <omp.h>

typedef struct {
    omp_sched_t schedule_type;
    int chunk_size;
    int num_threads;
} OmpConfig;

void setOmpConfig(OmpConfig ompConf);
void printOmpConfig(OmpConfig ompConf);
omp_sched_t parseSchedule(const char *name);

#endif