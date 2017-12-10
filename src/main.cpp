#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <random>
#include <functional>
#include <cassert>

#include "data.hpp"
#include "gd.hpp"

int main(int argc, char **argv) {
  init_data();

  if (argc < 2) {
    fprintf(stderr, "Expected an alpha\n");
    exit(1);
  }

  float alpha = atof(argv[1]);

  if (alpha <= 0) {
    fprintf(stderr, "alpha must be > 0, instead found %f\n", alpha);
    exit(1);
  }

  dataset_t train = get_train_dataset();
  dataset_t test = get_test_dataset();
  assert(train.dim == test.dim);
  assert(train.num_labels == test.num_labels);

  const unsigned int d = train.dim;
  const unsigned int c = train.num_labels;

  const unsigned int n_train = train.n;
  float *X_train = train.image;
  int *ys_idx_train = train.labels_idx;
  float *ys_oh_train = train.labels_oh;

  const unsigned int n_test = test.n;
  float *X_test = test.image;
  int *ys_idx_test = test.labels_idx;
  float *ys_oh_test = test.labels_oh;

  const unsigned int niter = 10000;

  sgd(X_train, ys_idx_train, ys_oh_train, n_train,
      X_test, ys_idx_test, ys_oh_test,
      n_test, d, c, niter, alpha, 0.0002, 0.9, 0.999,
      32, 1234);
}
