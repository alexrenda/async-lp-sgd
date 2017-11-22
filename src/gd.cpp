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

#define HOGWILD

gd_losses_t sgd
(
 const float* __restrict__ X_train_in,             // n x d
 const unsigned int* __restrict__ ys_idx_train_in, // n x 1
 const float* __restrict__ ys_oh_train_in,         // n x 1
 const size_t n_train,                // num training samples
 const float* __restrict__ X_test_in, // n x d
 const unsigned int* __restrict__ ys_idx_test_in, // n x 1
 const float* __restrict__ ys_oh_test_in,         // n x 1
 const size_t n_test,           // num training samples
 const size_t d,                // data dimensionality
 const size_t c,                // num classes
 const unsigned int niter,      // number of iterations to run
 const float alpha,             // step size
 const float beta,              // parameter of momentum
 const float lambda,            // regularization parameter
 const size_t batch_size,       // parameter of momentum
 const unsigned int seed        // random seed
 ) {
#ifdef HOGWILD
  // omp_set_num_threads(16);
#else
  omp_set_num_threads(1);
#endif

  gd_losses_t losses;

  // initialize randoms
  std::mt19937 gen(seed);
  std::normal_distribution<float> normal_dist(0, 1);
  std::uniform_int_distribution<unsigned int> uniform_dist(0, n_train-1);

  const size_t W_lda = ALIGN_ABOVE(d);
  assert(W_lda % ALIGNMENT == 0);
  __assume(W_lda % ALIGNMENT == 0);
  const size_t X_lda = ALIGN_ABOVE(d);
  assert(X_lda % ALIGNMENT == 0);
  __assume(X_lda % ALIGNMENT == 0);
  const size_t ys_oh_lda = ALIGN_ABOVE(c);
  assert(ys_oh_lda % ALIGNMENT == 0);
  __assume(ys_oh_lda % ALIGNMENT == 0);

  float* __restrict__ W = (float*) ALIGNED_MALLOC(c * W_lda * sizeof(float));
  __assume_aligned(W, ALIGNMENT);

  const float* __restrict__ X_train = (float*) ALIGNED_MALLOC(n_train * X_lda * sizeof(float));
  __assume_aligned(X_train, ALIGNMENT);
  for (int i = 0; i < n_train; i++) {
    memcpy((float*) &X_train[i * X_lda], &X_train_in[i * d], d * sizeof(float));
  }
  __assume_aligned(X_train, ALIGNMENT);
  const unsigned int* __restrict__ ys_idx_train = (unsigned int*) ALIGNED_MALLOC(n_train * sizeof(unsigned int));
  __assume_aligned(ys_idx_train, ALIGNMENT);
  memcpy((unsigned int*) ys_idx_train, ys_idx_train_in, n_train * sizeof(unsigned int));
  const float* __restrict__ ys_oh_train = (float*) ALIGNED_MALLOC(n_train * ys_oh_lda * sizeof(float));
  __assume_aligned(ys_oh_train, ALIGNMENT);
  for (int i = 0; i < n_train; i++) {
    memcpy((float*) &ys_oh_train[i * ys_oh_lda], &ys_oh_train_in[i * c], c * sizeof(float));
  }

  const float* __restrict__ X_test = (float*) ALIGNED_MALLOC(n_test * X_lda * sizeof(float));
  for (int i = 0; i < n_test; i++) {
    memcpy((float*) &X_test[i * X_lda], &X_test_in[i * d], d * sizeof(float));
  }
  const unsigned int* __restrict__ ys_idx_test = (unsigned int*) ALIGNED_MALLOC(n_test * sizeof(unsigned int));
  __assume_aligned(ys_idx_test, ALIGNMENT);
  memcpy((unsigned int*) ys_idx_test, ys_idx_test_in, n_test * sizeof(unsigned int));
  const float* __restrict__ ys_oh_test = (float*) ALIGNED_MALLOC(n_test * ys_oh_lda * sizeof(float));
  __assume_aligned(ys_oh_test, ALIGNMENT);
  for (int i = 0; i < n_test; i++) {
    memcpy((float*) &ys_oh_test[i * ys_oh_lda], &ys_oh_test_in[i * c], c * sizeof(float));
  }


  // gradient
  float* __restrict__ G_all = (float*) ALIGNED_MALLOC(c * W_lda * omp_get_max_threads() * sizeof(float));
  __assume_aligned(G_all, ALIGNMENT);
  memset(G_all, 0, c * W_lda * omp_get_max_threads() * sizeof(float));

  // tmp array for holding batch X
  float *batch_X = (float*) ALIGNED_MALLOC(sizeof(float) * batch_size * X_lda);
  __assume_aligned(batch_X, ALIGNMENT);
  // tmp array for holding one-hot batch ys
  float *batch_ys = (float*) ALIGNED_MALLOC(sizeof(float) * batch_size * ys_oh_lda);
  __assume_aligned(batch_ys, ALIGNMENT);
  // vector used for fisher-yates-esque batch selection w/out replacement
  unsigned int *batch_idx = (unsigned int*) ALIGNED_MALLOC(sizeof(unsigned int) * n_train);
  __assume_aligned(batch_idx, ALIGNMENT);

  // collection of uniform distributions for batch selection
  std::vector< std::uniform_int_distribution<unsigned int> > batch_dists;
  // scratch space
  const size_t scratch_size_per_thread = scratch_size(n_train + n_test,d,c);
  assert(scratch_size_per_thread % ALIGNMENT == 0);
  float* __restrict__ scratch_all = (float*) ALIGNED_MALLOC(scratch_size_per_thread * omp_get_max_threads() * sizeof(float));
  __assume_aligned(scratch_all, ALIGNMENT);

  // initialize the batch selection vector (invariant is that it's an unordered set)
  for (unsigned int i = 0; i < n_train; i++) {
    batch_idx[i] = i;
  }

  // initialize each distribution (this is ugly... TODO: any better way?)
  for (unsigned int i = 0; i < batch_size; i++) {
    batch_dists.push_back(std::uniform_int_distribution<unsigned int>(0, n_train - i - 1));
  }

  // initialize weight vector
  for (unsigned int j = 0; j < c; j++) {
    for (unsigned int k = 0; k < d; k++) {
      W[j * W_lda + k] = normal_dist(gen);
    }
  }

  losses.times.push_back(0);
  loss_t loss = multinomial_loss(W, W_lda, X_train, X_lda, ys_idx_train, n_train,
                                 d, c, lambda, scratch_all);
  losses.train_losses.push_back(loss.loss);
  losses.train_errors.push_back(loss.error);

  loss = multinomial_loss(W, W_lda, X_test, X_lda, ys_idx_test, n_test,
                          d, c, lambda, scratch_all);
  losses.test_losses.push_back(loss.loss);
  losses.test_errors.push_back(loss.error);

  timing_t loss_timer = timing_t();
  timing_t grad_timer = timing_t();

#ifdef PROGRESS
  fprintf(stderr, "TOTAL ITERS | ITER NUM | TRAIN LOSS | TRAIN ERROR | TEST LOSS | TEST ERROR | WC TIME\n");
  fflush(stderr);
#endif /* PROGRESS */

#pragma omp parallel for
  for (unsigned int _iter = 0; _iter < niter; _iter++) {
    unsigned int tno = omp_get_thread_num();
    float* __restrict__ scratch = &scratch_all[scratch_size_per_thread * tno];
    __assume_aligned(scratch, ALIGNMENT);

    float* __restrict__ G = &G_all[c * W_lda * tno];
    __assume_aligned(G, ALIGNMENT);

#ifdef PROGRESS
#pragma omp critical
    {
      unsigned int it = losses.times.size();

      fprintf(stderr,
              "%11d | %8d | %10.2f | %11.3f | %9.2f | %10.3f | %7.3f\r", niter, it,
              losses.train_losses.back(), losses.train_errors.back(),
              losses.test_losses.back(), losses.test_errors.back(),
              grad_timer.total_time()
              );
      if (it % (niter / 10) == 0) {
        fprintf(stderr, "\n");
      }
    }
    fflush(stderr);
#endif /* PROGRESS */

#pragma omp critical
    grad_timer.start_timing_round();

    for (unsigned int bidx = 0; bidx < batch_size; bidx++) {
      const unsigned int rand_idx = batch_dists[bidx](gen);
      const unsigned int idx = batch_idx[rand_idx];
      batch_idx[rand_idx] = batch_idx[n_train - 1 - bidx];
      batch_idx[n_train - 1 - bidx] = idx;

      float *x_dst = &batch_X[bidx * X_lda];
      const float *x_src = &X_train[idx * X_lda];

#pragma vector aligned
      for (unsigned int j = 0; j < d; j++) {
        x_dst[j] = x_src[j];
      }

      float *ys_dst = &batch_ys[bidx * ys_oh_lda];
      const float *ys_src = &ys_oh_train[idx * ys_oh_lda];

#pragma vector aligned
      for (unsigned int k = 0; k < c; k++) {
        ys_dst[k] = ys_src[k];
      }
    }

    multinomial_gradient_batch(G, W, W_lda, batch_X, X_lda,
                               batch_ys, ys_oh_lda,
                               batch_size, d, c, beta, lambda, scratch);

    SAXPBY(c * W_lda, -alpha * batch_size, G, 1, 1, W, 1);

#pragma omp critical
    grad_timer.end_timing_round(batch_size);

#ifdef LOSSES
    loss_timer.start_timing_round();
    loss_t train_loss = multinomial_loss(W, W_lda, X_train, X_lda, ys_idx_train,
                                         n_train, d, c, lambda, scratch);
    loss_t test_loss = multinomial_loss(W, W_lda, X_test, X_lda, ys_idx_test,
                                        n_test, d, c, lambda, scratch);
    loss_timer.end_timing_round(1);

#pragma omp critical
    {
      losses.times.push_back(grad_timer.total_time());
      losses.train_losses.push_back(train_loss.loss);
      losses.train_errors.push_back(train_loss.error);
      losses.test_losses.push_back(test_loss.loss);
      losses.test_errors.push_back(test_loss.error);
    }

#endif /* LOSSES */
  }

#ifdef PROGRESS
  fprintf(stderr, "\n");
#endif /* PROGRESS */

  fprintf(stderr, "Grad time per step: %f\n", grad_timer.time_per_step());
  fprintf(stderr, "Loss time per step: %f\n", loss_timer.time_per_step());

#ifdef LOSSES
  fprintf(stderr, "Final training loss: %f\n", losses.train_losses.back());
  fprintf(stderr, "Final training error: %f\n", losses.train_errors.back());

  fprintf(stderr, "Final testing loss: %f\n", losses.test_losses.back());
  fprintf(stderr, "Final testing error: %f\n", losses.test_errors.back());
#endif /* LOSSES */

  ALIGNED_FREE(W);
  ALIGNED_FREE((float*) X_train);
  ALIGNED_FREE((unsigned int*) ys_idx_train);
  ALIGNED_FREE((float*) ys_oh_train);
  ALIGNED_FREE((float*) X_test);
  ALIGNED_FREE((unsigned int*) ys_idx_test);
  ALIGNED_FREE((float*) ys_oh_test);
  ALIGNED_FREE((float*) scratch_all);
  ALIGNED_FREE(G_all);
  ALIGNED_FREE(batch_idx);
  ALIGNED_FREE(batch_X);
  ALIGNED_FREE(batch_ys);

  return losses;
}
