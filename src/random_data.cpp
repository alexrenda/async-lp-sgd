#include <stdlib.h>
#include <random>

#include "data.hpp"
#include "mblas.hpp"

#if DATA_TYPE == RANDOM

float *data_W_true;
const int data_n_train = 10000;
const int data_n_test  = 1000;
const int data_d = 64;

const int data_W_seed = 1239013;
const int data_train_seed = 943851;
const int data_test_seed = 87231;

float data_w_opt[64] = {-0.179545, 0.716161, 0.088537, 0.279593, 0.724652, 0.231110, -0.308080, -0.220804, -0.055388, -1.044265, 0.092021, 0.140935, 0.133860, 0.887708, 0.098904, -0.262281, -0.204640, 0.173512, -0.421641, 0.296451, -0.038988, 0.130729, 0.116526, -0.404306, 1.576270, -0.455838, -0.414096, 0.189506, -0.044801, -0.019440, -0.205070, 0.041816, -0.135568, 0.554743, 0.561596, 0.445931, 0.404573, 0.124280, -0.023788, -0.875212, -0.791725, 0.167050, 0.333376, -0.675848, -0.345740, -0.295259, 0.458078, 0.250816, 0.161586, 0.376586, 0.061365, 0.118915, 0.189889, -0.112699, -0.077394, -0.145247, 0.056836, -0.373764, 0.045236, -0.514315, -0.294290, -0.621477, -0.210478, 0.727010};

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
  dataset_t res(n, data_d, 1);

  res.w_opt = data_w_opt;
  res.labels_oh = (float*) malloc(sizeof(float) * n * 2);
  res.labels_idx = (int*) malloc(sizeof(int) * n);
  res.image = (float*) malloc(sizeof(float) * n * data_d);

  std::mt19937 gen(seed);

  for (int i = 0; i < n; i++) {
    float dot_res = 0;
    for (int j = 0; j < data_d; j++) {
      float datum = data_normal_dist(gen);
      dot_res += data_W_true[j] * datum;
      res.image[i * data_d + j] = datum;
    }
    float p = 1 / (1 + expf(-dot_res));
    if (p < data_uniform_dist(gen)) {
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
