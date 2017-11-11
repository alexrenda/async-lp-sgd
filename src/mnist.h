#ifndef MNIST_H
#define MNIST_H

typedef struct dataset_t {
  const int N;                  // Number
  const int dim;                // Dimensions of an image
  char* const __restrict__ labels; // Labels: byte[n]
  float* const __restrict__ image;  // Images: byte[n][dim*dim]
} dataset_t;

dataset_t* get_train_dataset();
dataset_t* get_test_dataset();

void free_dataset(dataset_t* dataset);

#endif /* MNIST_H */
