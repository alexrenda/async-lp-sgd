#ifndef MNIST_H
#define MNIST_H

#include <vector>

struct dataset_t {
  const int n;                  // Number
  const int dim;                // Dimensions of an image (28 * 28)
  const int num_labels;         // Number of labels (10)
  std::vector<float> labels_oh; // One-hot labels: byte[n][num_labels]
  std::vector<char> labels_idx; // Index labels: byte[n]
  std::vector<float> image;     // Images: byte[n][dim]

  dataset_t(int n, int dim, int num_labels)
    : n(n), dim(dim), num_labels(num_labels), labels_oh(n * num_labels), labels_idx(n), image(n * dim) {}
};

dataset_t get_train_dataset();
dataset_t get_test_dataset();

#endif /* MNIST_H */
