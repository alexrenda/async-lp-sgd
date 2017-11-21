#ifndef TIMING_H
#define TIMING_H

#include <omp.h>
#include <cassert>

class timing_t {
  double _total_time;
  int _steps;
  double _start_time;

public:
  timing_t(): _total_time(0), _steps(0), _start_time(-1) {}

  void start_timing_round() {
    _start_time = omp_get_wtime();
  }

  void end_timing_round(int _stepsTaken) {
    assert(_start_time >= 0);

    _total_time += omp_get_wtime() - _start_time;
    _steps += _stepsTaken;
    _start_time = -1;
  }

  double time_per_step() {
    return _total_time / _steps;
  }

  double total_time() {
    return _total_time;
  }
};

#endif /* TIMING_H */
