#ifndef MNIST_LOW_H
#define MNIST_LOW_H

#include <vector>
#include <cmath>
#include "mnist.h"

struct dataset_low_t {
  const int n;                  // Number
  const int dim;                // Dimensions of an image (28 * 28)
  const int num_labels;         // Number of labels (10)
  std::vector<unsigned int> labels;     // Labels: byte[n][num_labels]
  std::vector<char> image;      // Images: byte[n][dim]

  dataset_low_t(dataset_t dataset) :
      n(dataset.n), dim(dataset.dim), num_labels(dataset.num_labels),
      labels(n * num_labels), image(n * dim) {
        for(int i = 0; i < n * num_labels; i++){
          labels[i] = (char) dataset.labels[i];
        }

        for(int i = 0; i < n * dim; i++){
          image[i] = (char) lround(dataset.image[i] * 255);
        }
      }
};

dataset_low_t get_train_dataset_low(){
  return dataset_low_t(get_train_dataset());
}

dataset_low_t get_test_dataset_low(){
  return dataset_low_t(get_test_dataset());
}

#endif /* MNIST_LOW_H */
