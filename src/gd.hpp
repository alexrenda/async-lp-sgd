#ifndef GD_H
#define GD_HPP

#include <vector>

typedef struct gd_losses {
  std::vector<float> train_losses;
  std::vector<float> train_errors;
  std::vector<float> test_losses;
  std::vector<float> test_errors;
  std::vector<float> times;
} gd_losses_t;

gd_losses_t sgd
(
 float* __restrict__ W,             // c x d
 const size_t W_lda,                // lda (axis 1 stride) of W
 const size_t X_lda,                // lda (axis 1 stride) of X
 const size_t ys_oh_lda,            // lda of ys
 const float* __restrict__ X_train, // n x d
 const unsigned int* __restrict__ ys_idx_train, // n x 1
 const float* __restrict__ ys_oh_train,         // n x 1
 const size_t n_train,                         // num training samples
 const float* __restrict__ X_test,             // n x d
 const unsigned int* __restrict__ ys_idx_test, // n x 1
 const float* __restrict__ ys_oh_test,         // n x 1
 const size_t n_test,                          // num training samples
 const size_t d,                               // data dimensionality
 const size_t c,                               // num classes
 const unsigned int niter,      // number of iterations to run
 const float alpha,             // step size
 const float beta,              // parameter of momentum
 const float lambda,            // regularization parameter
 const size_t batch_size,       // parameter of momentum
 const unsigned int seed        // random seed
 );

#endif /* GD_H */
