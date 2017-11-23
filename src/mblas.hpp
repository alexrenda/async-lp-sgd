#ifndef MBLAS_H
#define MBLAS_H

#define ALIGNMENT 64

#define CEIL_DIV(n, d) (((n) + (d) - 1) / (d))
#define ALIGN_ABOVE(n) (CEIL_DIV(n, ALIGNMENT) * ALIGNMENT)

#ifdef __INTEL_COMPILER
#  define ALIGNED_MALLOC(size) _mm_malloc(size, ALIGNMENT)
#  define ALIGNED_FREE(ptr) _mm_free(ptr)
#else
#  include <malloc.h>
#  define ALIGNED_MALLOC(size) aligned_alloc(ALIGNMENT, size)
#  define ALIGNED_FREE(ptr) free(ptr)
#endif /* __INTEL_COMPILER */

#ifdef OSX_ACCELERATE
#  include <Accelerate/Accelerate.h>
#elif defined(__INTEL_COMPILER)
#  include <mkl_cblas.h>
#  include <mkl.h>
#elif defined(__GNUC__) || defined(__GNUG__)
#  include <cblas.h>
#else
#  error you gotta have some blas
#endif /* OSX_ACCELERATE */

#ifdef OSX_ACCELERATE
#  define SAXPBY catlas_saxpby
#elif defined(__INTEL_COMPILER)
#  define SAXPBY cblas_saxpby
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
