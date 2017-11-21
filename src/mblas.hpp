#ifndef MBLAS_H
#define MBLAS_H

#ifdef OSX_ACCELERATE
#  include <Accelerate/Accelerate.h>
#elif defined(__GNUC__) || defined(__GNUG__)
#  include <cblas.h>
#else
#  error you gotta have some blas
#endif /* OSX_ACCELERATE */

#ifdef OSX_ACCELERATE
#  define SAXPBY catlas_saxpby
#else
void inline saxby(const int n, const float a, const float *x, const int incx, const float b, float *y, const int incy){
  int xa = 0;
  int ya = 0;
  for(int i = 0; i < n; i++, xa += incx, ya += incy){
    y[ya] = a * x[xa] + b * y[ya];
  }
}
#  define SAXPBY saxby
#endif /* OSX_ACCELERATE */

#endif /* MBLAS_H */
