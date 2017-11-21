#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <random>
#include <functional>
#include <cassert>

#include "mnist.hpp"
#include "gd.hpp"


int main() {
  std::mt19937 gen(5);
  std::normal_distribution<float> normal_dist(0, 1);

  dataset_t train = get_train_dataset();
  dataset_t test = get_train_dataset();
  assert(train.dim == test.dim);
  assert(train.num_labels == test.num_labels);

  const unsigned int d = train.dim;
  const unsigned int c = train.num_labels;

  const unsigned int n_train = train.n;
  float *X_train = train.image.data();
  unsigned int *ys_idx_train = train.labels_idx.data();
  float *ys_oh_train = train.labels_oh.data();

  const unsigned int n_test = test.n;
  float *X_test = test.image.data();
  unsigned int *ys_idx_test = test.labels_idx.data();
  float *ys_oh_test = test.labels_oh.data();

  const unsigned int niter = 100;
  float* __restrict__ W = (float*) malloc(sizeof(float) * c * d);

  sgd(W, d, d, c,
      X_train, ys_idx_train, ys_oh_train, n_train,
      X_test, ys_idx_test, ys_oh_test, n_test,
      d, c, niter, 0.001, 0.99, 1 / d,
      16, 1234);
}
