#include "ompconfig.h"
#include <stdio.h>
#include <string.h>

omp_sched_t parseSchedule(const char *name) {
#ifdef _OPENMP
    if (strcmp(name, "static") == 0)
        return omp_sched_static;
    else if (strcmp(name, "dynamic") == 0)
        return omp_sched_dynamic;
    else if (strcmp(name, "guided") == 0)
        return omp_sched_guided;
    else{
        fprintf(stderr, "Unknown schedule type '%s', default to 'auto'.\n", name);
        return omp_sched_auto;
    }
#else
    (void)name;  // Evita warning unused parameter
    return omp_sched_auto;
#endif
}

void setOmpConfig(OmpConfig ompConf) {
#ifdef _OPENMP
    omp_set_num_threads(ompConf.num_threads);
    omp_set_schedule(ompConf.schedule_type, ompConf.chunk_size);
#endif
}

void printOmpConfig(OmpConfig ompConf) {
#ifdef _OPENMP
    const char *schedName =
        ompConf.schedule_type == omp_sched_static ? "static" :
        ompConf.schedule_type == omp_sched_dynamic ? "dynamic" :
        ompConf.schedule_type == omp_sched_guided ? "guided" :
        "auto";
    printf("OMP configuration -> schedule: %s | chunk: %d | threads: %d\n",
           schedName, ompConf.chunk_size, ompConf.num_threads);
#endif
}