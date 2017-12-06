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

float data_w_opt[64] = {-0.0138092882676, -0.00821719539232, -0.000375650872344, 0.0135199604386, -0.0365724227532, -0.0202630628383, 0.00485014140576, -0.0309749170382, 0.0190457362618, 0.00795978575664, 0.00364350319312, 0.0267562597881, 0.00299804810799, -0.0118379032164, -0.000710764686635, 0.0245587650911, 0.00905398124303, -0.034756055283, -0.0154160872738, -0.023245741177, 0.0016837676782, 0.0145730687017, -0.0122485559625, -0.0144608537234, -0.0407876948403, -0.00864920754231, -0.00576077561667, -0.0258275369611, 0.0201978758353, -0.0163282083933, 0.0459476781641, 0.0193380834658, -0.00827845786664, -0.0205427322618, -0.0129899476169, -0.0223369249832, -0.0183254514674, 0.00245264626451, 0.00148435244413, 0.0402834108636, -0.012573832796, 0.0021650171453, -0.0302594032706, 0.032614072927, 0.0466877833839, 0.00734802017328, -0.046928280831, -0.0339081194318, 0.00788375119152, -0.00996085871859, 0.0243822285252, 0.00264181083633, -0.0131723745567, -0.0139918375042, -0.00895191357344, 0.0035127228767, 0.00355641577044, 0.00237982573149, -0.0246308375878, -0.00320457130973, -0.0088710151371, 0.00380854632522, 0.0512997301514, -0.0118878372084};

std::normal_distribution<float> data_normal_dist(0, 1);
std::uniform_real_distribution<float> data_uniform_dist(0, 1);

void init_data() {
  std::mt19937 gen(data_W_seed);

  data_W_true = (float*) malloc(sizeof(float) * data_d);
  for (unsigned int i = 0; i < data_d; i++) {
    data_W_true[i] = data_normal_dist(gen) / 100;
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
    float p = 1 / (1 + expf(-dot_res)) + data_normal_dist(gen) / 5;
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
