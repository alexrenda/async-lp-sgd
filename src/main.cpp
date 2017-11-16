#ifdef OSX_ACCELERATE
#  include <Accelerate/Accelerate.h>
#elif defined(__GNUC__) || defined(__GNUG__)
#  include <cblas.h>
#else
#  error you gotta have some blas cmon
#endif

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
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <random>
#include <functional>
#include <cassert>

#include "mnist.h"

#define CEIL_DIV(n, d) (((n) + (d) - 1) / (d))

void sgd_impl
(float* __restrict__ w,
 float* __restrict__ grad,
 const float* __restrict__ xs,
 const float* __restrict__ ys,
 const size_t n,
 const size_t d,
 const unsigned int niter,
 const float alpha,
 const float beta,
 const float lambda,
 std::mt19937 &gen,
 std::uniform_int_distribution<int> &dist
 ) {
  for (unsigned int iter = 0; iter < niter; iter++) {
    const int idx = dist(gen);
    const float *x = &xs[idx * d];
    const char y = ys[idx];

    float wTx = cblas_sdot(d, x, 1, w, 1);
    const float scale = -y / (1 + expf(y * wTx));
    SAXPBY(d, -alpha * scale, x, 1, 1 - 2 * lambda * alpha, w, 1);
  }
}

float loss
(const float* __restrict__ w,
 const float* __restrict__ xs,
 const float* __restrict__ ys,
 const size_t n,
 const size_t d,
 const float lambda
 ) {
  float loss = 0;

  for (unsigned int idx = 0; idx < n; idx++) {
    const float *x = &xs[idx * d];
    const float y = ys[idx];

    float wTx = cblas_sdot(d, x, 1, w, 1);
    float wTw = cblas_sdot(d, w, 1, w, 1);

    loss += log2f(1 + expf(-y * wTx)) + lambda * wTw;
  }

  return loss;
}

void sgd
(
 float* __restrict__ w,
 const float* __restrict__ xs,
 const float* __restrict__ ys,
 const size_t n,
 const size_t d,
 const unsigned int niter,
 const float alpha,
 const float beta,
 const float lambda,
 const unsigned int seed,
 const unsigned int report_loss_every,
 float * __restrict__ losses
 ) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> normal_dist(0, 1);
  std::uniform_int_distribution<int> uniform_dist(1,n-1);

  float* __restrict__ grad = (float*) calloc(d, sizeof(float));

  for (int j = 0; j < d; j++) {
    w[j] = normal_dist(gen);
  }

  if (report_loss_every == 0) {
    sgd_impl(w, grad, xs, ys, n, d, niter, alpha, beta, lambda, gen, uniform_dist);
  } else {
    for (int iter = 0; iter < CEIL_DIV(niter, report_loss_every); iter++) {
      int iter_count = std::min(niter - iter*report_loss_every, report_loss_every);
      sgd_impl(w, grad, xs, ys, n, d, iter_count, alpha, beta, lambda, gen, uniform_dist);
      losses[iter] = loss(w, xs, ys, n, d, lambda);
    }
  }

  free(grad);
}


int main() {
  std::mt19937 gen(5);
  std::normal_distribution<float> normal_dist(0, 1);

  /*
  const unsigned int n = 1000;
  const unsigned int d = 2;

  float *w_true = (float*) malloc(sizeof(float) * d);
  for (int j = 0; j < d; j++) {
    w_true[j] = normal_dist(gen);
  }

  printf("w_true = [");
  for (int j = 0; j < d; j++) {
    printf("%.3f,", w_true[j]);
  }
  printf("]\n");


  float *xs = (float*) malloc(sizeof(float) * n * d);
  char *ys = (char*) malloc(sizeof(char) * n);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      xs[i * d + j] = normal_dist(gen);
    }

    ys[i] = cblas_sdot(d, &xs[i * d], 1, w_true, 1) > 0;
  }

  printf("xs = [");
  for (int i = 0; i < n; i++) {
    printf("[");
    for (int j = 0; j < d; j++) {
      printf("%.3f,", xs[i * d + j]);
    }
    printf("],");
  }
  printf("]\n");

  printf("ys = [");
  for (int i = 0; i < n; i++) {
    printf("%u,", ys[i]);
  }
  printf("]\n");
  */

  dataset_t train = get_train_dataset();
  dataset_t test = get_train_dataset();
  assert(train.dim == test.dim);
  float *xs = train.image.data();
  float *ys = train.labels.data();
  const unsigned int n = train.n;
  const unsigned int d = train.dim;

  const unsigned int niter = 100000;
  const unsigned int nloss = 10;
  const unsigned int nprint = niter / nloss;
  float* __restrict__ w = (float*) malloc(sizeof(float) * d);
  float* __restrict__ losses = (float*) malloc(sizeof(float) * nloss);
  sgd(w, xs, ys, n, d, niter, 0.01, 0.9, 0.001, 1234, nprint, losses);

  printf("w_final = ");
  printf("[");
  for (int j = 0; j < d; j++) {
    printf("%.3f,", w[j]);
  }
  printf("]\n");

  for (unsigned int i = 0; i < nloss; i++) {
    printf("Loss: %f\n", losses[i]);
  }
}
