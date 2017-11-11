#ifdef OSX_ACCELERATE
#  include <Accelerate/Accelerate.h>
#elif defined(__GNUC__) || defined(__GNUG__)
#  include <cblas.h>
#endif

#include <stdio.h>
#include <stdlib.h>

int main() {
  srand(1234);



  printf("Hello, world!\n");
}

// https://stackoverflow.com/a/6852396
// Assumes 0 <= max <= RAND_MAX
// Returns in the closed interval [0, max]
long random_at_most(long max) {
  unsigned long
    // max <= RAND_MAX < ULONG_MAX, so this is okay.
    num_bins = (unsigned long) max + 1,
    num_rand = (unsigned long) RAND_MAX + 1,
    bin_size = num_rand / num_bins,
    defect   = num_rand % num_bins;

  long x;
  do {
    x = random();
  }
  // This is carefully written not to overflow
  while (num_rand - defect <= (unsigned long)x);

  // Truncated division is intentional
  return x/bin_size;
}

void sgd(float* __restrict__ w,
         const float* __restrict__ xs,
         const char* __restrict__ ys,
         const size_t n,
         const size_t d,
         const unsigned int niter,
         const float alpha,
         const float lambda
         ) {

  memset(w, 0, sizeof(float) * d);

  for (int i = 0; i < niter; i++) {
    const int idx = random_at_most(n);
    const float *x = &xs[idx * d];
    const char y = ys[idx];

    float wTx = cblas_sdot(d, x, 1, w, 1);
    const float scale = -y / (1 + expf(y * wTx));

    catlas_saxpby(d, -alpha * scale, x, 1, -2 * lambda * alpha, w, 1);
  }
}
