#ifndef GD_H
#define GD_HPP


// #define PROGRESS
#define LOSSES
#define RAW_OUTPUT

enum gd_type {
  ADAM_SERIAL,
  ADAM_SHARED,
  ADAM_PRIVATE,
  SVRG,
  SGD
};

#define GD_TYPE ADAM_SERIAL

void sgd
(
 const float* __restrict__ X_train_in,     // n x d
 const int* __restrict__ ys_idx_train_in,  // n x 1
 const float* __restrict__ ys_oh_train_in, // n x 1
 const size_t n_train,                     // num training samples
 const float* __restrict__ X_test_in,      // n x d
 const int* __restrict__ ys_idx_test_in,   // n x 1
 const float* __restrict__ ys_oh_test_in,  // n x 1
 const size_t n_test,                      // num training samples
 const size_t d,                           // data dimensionality
 const size_t c,                           // num classes
 const unsigned int niter,      // number of iterations to run
 const float alpha,             // step size
 const float lambda,            // regularization parameter
 const float beta_1,            // ema parameter of 1st moment
 const float beta_2,            // ema parameter of 2nd moment
 const size_t batch_size,       // batch size
 const unsigned int seed        // random seed
 );

#endif /* GD_H */
