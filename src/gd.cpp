#include <stdlib.h>
#include <random>
#include <functional>
#include <vector>
#include "string.h"

#include "gd.hpp"
#include "loss.hpp"
#include "mblas.hpp"
#include "timing.hpp"

#define PROGRESS
#define LOSSES

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
 ) {
  gd_losses_t losses;

  // initialize randoms
  std::mt19937 gen(seed);
  std::normal_distribution<float> normal_dist(0, 1);
  std::uniform_int_distribution<int> uniform_dist(0, n_train-1);

  // gradient
  float* __restrict__ G = (float*) calloc(c * W_lda, sizeof(float));
  // tmp array for holding batch X
  float *batch_X = (float*) malloc(sizeof(float) * batch_size * X_lda);
  // tmp array for holding one-hot batch ys
  float *batch_ys = (float*) malloc(sizeof(float) * batch_size * ys_oh_lda);
  // vector used for fisher-yates-esque batch selection w/out replacement
  unsigned int *batch_idx = (unsigned int*) malloc(sizeof(unsigned int) * n_train);
  // collection of uniform distributions for batch selection
  std::vector<std::uniform_int_distribution<int>> batch_dists;
  // scratch space
  float* __restrict__ scratch = (float*) malloc(scratch_size(n_train + n_test,d,c));

  // initialize the batch selection vector (invariant is that it's an unordered set)
  for (unsigned int i = 0; i < n_train; i++) {
    batch_idx[i] = i;
  }

  // initialize each distribution (this is ugly... TODO: any better way?)
  for (unsigned int i = 0; i < batch_size; i++) {
    batch_dists.push_back(std::uniform_int_distribution<int>(0, n_train - i - 1));
  }

  // initialize weight vector
  for (unsigned int j = 0; j < c; j++) {
    for (unsigned int k = 0; k < d; k++) {
      W[j * W_lda + k] = normal_dist(gen);
    }
  }

  loss_t loss = multinomial_loss(W, W_lda, X_train, X_lda, ys_idx_train, n_train,
                                 d, c, lambda, scratch);
  losses.train_losses.push_back(loss.loss);
  losses.train_errors.push_back(loss.error);

  loss = multinomial_loss(W, W_lda, X_test, X_lda, ys_idx_test, n_test,
                          d, c, lambda, scratch);
  losses.test_losses.push_back(loss.loss);
  losses.test_errors.push_back(loss.error);

  timing_t loss_timer = timing_t();
  timing_t grad_timer = timing_t();

#ifdef PROGRESS
  printf("TOTAL ITERS | ITER NUM | TRAIN LOSS | TRAIN ERROR | TEST LOSS | TEST ERROR | WC TIME\n");
#endif /* PROGRESS */

  for (unsigned int iter = 0; iter < niter; iter++) {

#ifdef PROGRESS
    printf("%11d | %8d | %10.2f | %11.3f | %9.2f | %10.3f | %7.3f\r", niter, iter,
           losses.train_losses.back(), losses.train_errors.back(),
           losses.test_losses.back(), losses.test_errors.back(),
           grad_timer.total_time()
           );
    fflush(stdout);
    if (iter % (niter / 10) == 0) printf("\n");
#endif /* PROGRESS */

    grad_timer.start_timing_round();
    for (unsigned int bidx = 0; bidx < batch_size; bidx++) {
      const int rand_idx = batch_dists[bidx](gen);
      const int idx = batch_idx[rand_idx];
      batch_idx[rand_idx] = batch_idx[n_train - 1 - bidx];
      batch_idx[n_train - 1 - bidx] = idx;

      memcpy(&batch_X[bidx * X_lda], &X_train[idx * X_lda], sizeof(float) * d);
      memcpy(&batch_ys[bidx * ys_oh_lda],
             &ys_oh_train[idx * ys_oh_lda], sizeof(float) * c);
    }

    multinomial_gradient_batch(G, W, W_lda, batch_X, X_lda,
                               batch_ys, ys_oh_lda,
                               batch_size, d, c, beta, lambda, scratch);

    SAXPBY(c * W_lda, -alpha * batch_size, G, 1, 1, W, 1);

    grad_timer.end_timing_round(batch_size);

#ifdef LOSSES
    loss_timer.start_timing_round();

    loss = multinomial_loss(W, W_lda, X_train, X_lda, ys_idx_train, n_train,
                            d, c, lambda, scratch);
    losses.train_losses.push_back(loss.loss);
    losses.train_errors.push_back(loss.error);

    loss = multinomial_loss(W, W_lda, X_test, X_lda, ys_idx_test, n_test,
                            d, c, lambda, scratch);
    losses.test_losses.push_back(loss.loss);
    losses.test_errors.push_back(loss.error);

    loss_timer.end_timing_round(1);
#endif /* LOSSES */
  }

  printf("Grad time per step: %f\n", grad_timer.time_per_step());
  printf("Loss time per step: %f\n", loss_timer.time_per_step());

#ifdef LOSSES
  printf("Final training loss: %f\n", losses.train_losses.back());
  printf("Final training error: %f\n", losses.train_errors.back());

  printf("Final testing loss: %f\n", losses.test_losses.back());
  printf("Final testing error: %f\n", losses.test_errors.back());
#endif /* LOSSES */

  free(G);
  free(batch_idx);
  free(batch_X);
  free(batch_ys);

  return losses;
}
