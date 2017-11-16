#ifndef MNIST_H
#define MNIST_H

struct dataset_t {
  const int N;                     // Number
  const int dim;                   // Dimensions of an image (28 * 28)
  const int num_labels;                   // Number of labels (10)
  float* const __restrict__ labels; // Labels: byte[n][10]
  float* const __restrict__ image; // Images: byte[n][dim*dim]
};

dataset_t get_train_dataset();
dataset_t get_test_dataset();

void free_dataset(dataset_t* dataset);

#endif /* MNIST_H */
