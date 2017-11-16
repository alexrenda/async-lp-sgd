#ifndef TIMING_H
#define TIMING_H

#include <omp.h>
#include <cassert>

class timing_t {
  double total_time;
  int steps;
  double start_time;

public:
  timing_t(): total_time(0), steps(0), start_time(-1) {}

  void start_timing_round(){
    start_time = omp_get_wtime();
  }

  void end_timing_round(int stepsTaken){
    assert(start_time != -1);

    steps += stepsTaken;
    total_time += omp_get_wtime() - start_time;
    start_time = -1;
  }

  double time_per_step(){
    return total_time/steps;
  }
};

#endif /* TIMING_H */
