#ifndef MNIST_H
#define MNIST_H

#define RANDOM 0
#define MNIST 1
#define DATA_TYPE RANDOM

#include <vector>

struct dataset_t {
  const int n;                  // Number
  const int dim;                // Dimensions of an image (28 * 28)
  const int num_labels;         // Number of labels (10)
  float* labels_oh;             // One-hot labels: byte[n][num_labels]
  int* labels_idx;              // Index labels: byte[n]
  float* image;                 // Images: byte[n][dim]
  const float* w_opt;           // optimal weights

  dataset_t(int n, int dim, int num_labels)
    : n(n), dim(dim), num_labels(num_labels) {}
};

void init_data();
dataset_t get_train_dataset();
dataset_t get_test_dataset();

#endif /* MNIST_H */
