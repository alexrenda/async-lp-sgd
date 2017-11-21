#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <random>
#include <functional>
#include <cassert>

#include "mnist.h"
#include "gd.h"


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
  unsigned int *ys = (unsigned int*) malloc(sizeof(unsigned int) * n);

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
  unsigned int *ys_idx = train.labels_idx.data();
  float *ys_oh = train.labels_oh.data();
  const unsigned int n = train.n;
  const unsigned int d = train.dim;
  const unsigned int c = train.num_labels;

  const unsigned int niter = 10000;
  const unsigned int nloss = 101;
  float* __restrict__ W = (float*) malloc(sizeof(float) * c * d);
  float* __restrict__ losses = (float*) malloc(sizeof(float) * nloss);

  sgd(W, d, xs, d, ys_idx, ys_oh, c, n, d, c, niter, 0.001, 0.99, 1 / n,
      16, 1234, losses, nloss);

  /*
  printf("w_final = ");
  printf("[");
  for (int j = 0; j < c; j++) {
    for (int k = 0; k < d; k++) {
      printf("%.3f,", W[j * d + k]);
    }
  }
  printf("]\n");

  for (unsigned int i = 0; i < nloss; i++) {
    printf("Loss: %f\n", losses[i]);
  }
  */
}
