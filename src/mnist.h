#ifndef MNIST_H
#define MNIST_H

#include <vector>

struct dataset_t {
  const int n;
  const int dim;
  std::vector<char> labels;
  std::vector<float> image;

  dataset_t(int n, int dim)
    : n(n), dim(dim), labels(n), image(n * dim) {}
};

dataset_t get_train_dataset();
dataset_t get_test_dataset();

#endif /* MNIST_H */
