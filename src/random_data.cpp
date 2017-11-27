#include <stdlib.h>
#include <random>

#include "data.hpp"
#include "mblas.hpp"

#if DATA_TYPE == RANDOM

float *data_W_true;
const int data_n_train = 1000;
const int data_n_test  = 100;
const int data_d = 100;

const int data_W_seed = 1239013;
const int data_train_seed = 943851;
const int data_test_seed = 87231;

std::normal_distribution<float> data_normal_dist(0, 1);
std::uniform_real_distribution<float> data_uniform_dist(0, 1);

void init_data() {
  std::mt19937 gen(data_W_seed);

  data_W_true = (float*) malloc(sizeof(float) * data_d);
  for (unsigned int i = 0; i < data_d; i++) {
    data_W_true[i] = data_normal_dist(gen);
  }
}

dataset_t data_get_dataset(const int n, const int seed) {
  dataset_t res(n, data_d, 2);

  std::mt19937 gen(seed);

  for (int i = 0; i < n; i++) {
    float dot_res = 0;
    for (int j = 0; j < data_d; j++) {
      float datum = data_normal_dist(gen);
      dot_res += data_W_true[j] * datum;
      res.image[i * data_d + j] = datum;
    }
    float p = 1 / (1 + expf(-dot_res));
    if (data_uniform_dist(gen) < p) {
      res.labels_oh[i * 2] = 0;
      res.labels_oh[i * 2 + 1] = 1;
      res.labels_idx[i] = 1;
    } else {
      res.labels_oh[i * 2] = 1;
      res.labels_oh[i * 2 + 1] = 0;
      res.labels_idx[i] = -1;
    }
  }

  return res;
}

dataset_t get_train_dataset() {
  return data_get_dataset(data_n_train, data_train_seed);
}
dataset_t get_test_dataset() {
  return data_get_dataset(data_n_test, data_test_seed);
}

#endif /* DATA_TYPE */
