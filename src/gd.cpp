#include <stdlib.h>
#include <float.h>
#include <atomic>
#include <random>
#include <functional>
#include <vector>
#include "string.h"

#include "gd.hpp"
#include "loss.hpp"
#include "mblas.hpp"
#include "timing.hpp"

gd_losses_t sgd
(
 const float* __restrict__ X_train_in,     // n x d
 const int* __restrict__ ys_idx_train_in,  // n x 1
 const float* __restrict__ ys_oh_train_in, // n x 1
 const size_t n_train,                     // num training samples
 const float* __restrict__ X_test_in,      // n x d
 const int* __restrict__ ys_idx_test_in,   // n x 1
 const float* __restrict__ ys_oh_test_in,  // n x 1
 const float* __restrict__ W_opt_in,       // optimum weight value
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
 ) {
#ifndef HOGWILD
  omp_set_num_threads(1);
#endif

  gd_losses_t losses;

  // initialize randoms
  std::mt19937 gen_main(seed);
  std::vector<std::mt19937> gen_all;
  for (int t = 0; t < omp_get_max_threads(); t++) {
    gen_all.push_back(std::mt19937(seed + t));
  }
  std::normal_distribution<float> normal_dist(0, 1);
  std::uniform_int_distribution<unsigned int> uniform_dist(0, n_train-1);
  std::uniform_int_distribution<int> thread_dist(0, omp_get_max_threads() - 1);

  const size_t X_lda = ALIGN_ABOVE(d);
  assert(X_lda % ALIGNMENT == 0);
  __assume(X_lda % ALIGNMENT == 0);
  const size_t ys_oh_lda = ALIGN_ABOVE(c);
  assert(ys_oh_lda % ALIGNMENT == 0);
  __assume(ys_oh_lda % ALIGNMENT == 0);

  float* __restrict__ W = (float*) ALIGNED_MALLOC(d * sizeof(float));
  __assume_aligned(W, ALIGNMENT);

  const float* __restrict__ W_opt = (float*) ALIGNED_MALLOC(d * sizeof(float));
  __assume_aligned(W_opt, ALIGNMENT);
  memcpy((float*) W_opt, W_opt_in, d * sizeof(float));

  const float* __restrict__ X_train = (float*) ALIGNED_MALLOC(n_train * X_lda * sizeof(float));
  __assume_aligned(X_train, ALIGNMENT);
  for (int i = 0; i < n_train; i++) {
    for (int j = 0; j < d; j++) {
      ((float*)X_train)[i * X_lda + j] = X_train_in[i * d + j];
    }
  }

  const int* __restrict__ ys_idx_train = (int*) ALIGNED_MALLOC(n_train * sizeof(int));
  __assume_aligned(ys_idx_train, ALIGNMENT);
  memcpy((int*) ys_idx_train, ys_idx_train_in, n_train * sizeof(int));
  const float* __restrict__ ys_oh_train = (float*) ALIGNED_MALLOC(n_train * ys_oh_lda * sizeof(float));
  __assume_aligned(ys_oh_train, ALIGNMENT);
  for (int i = 0; i < n_train; i++) {
    memcpy((float*) &ys_oh_train[i * ys_oh_lda], &ys_oh_train_in[i * c], c * sizeof(float));
  }

  const float* __restrict__ X_test = (float*) ALIGNED_MALLOC(n_test * X_lda * sizeof(float));
  for (int i = 0; i < n_test; i++) {
    memcpy((float*) &X_test[i * X_lda], &X_test_in[i * d], d * sizeof(float));
  }
  const int* __restrict__ ys_idx_test = (int*) ALIGNED_MALLOC(n_test * sizeof(int));
  __assume_aligned(ys_idx_test, ALIGNMENT);
  memcpy((int*) ys_idx_test, ys_idx_test_in, n_test * sizeof(int));
  const float* __restrict__ ys_oh_test = (float*) ALIGNED_MALLOC(n_test * ys_oh_lda * sizeof(float));
  __assume_aligned(ys_oh_test, ALIGNMENT);
  for (int i = 0; i < n_test; i++) {
    memcpy((float*) &ys_oh_test[i * ys_oh_lda], &ys_oh_test_in[i * c], c * sizeof(float));
  }


  // gradient
  float* __restrict__ G_all = (float*) ALIGNED_MALLOC(ALIGN_ABOVE(d) * omp_get_max_threads() * sizeof(float));
  __assume_aligned(G_all, ALIGNMENT);
  memset(G_all, 0, ALIGN_ABOVE(d) * omp_get_max_threads() * sizeof(float));
  for (int t = 0; t < omp_get_max_threads(); t++) {
    for (int j = 0; j < d; j++) {
      G_all[t * ALIGN_ABOVE(d) + j] = 0;
    }
  }

  // timing
  unsigned int* __restrict__ t_all = (unsigned int*) ALIGNED_MALLOC(omp_get_max_threads() * sizeof(unsigned int));
  __assume_aligned(t_all, ALIGNMENT);
  for (int t = 0; t < omp_get_max_threads(); t++) {
    t_all[t] = 1;
  }


#ifdef ADAM_SHARED
  float* __restrict__ m = (float*) ALIGNED_MALLOC(d * sizeof(float));
  memset(m, 0, d * sizeof(float));
  __assume_aligned(m, ALIGNMENT);
  float* __restrict__ v = (float*) ALIGNED_MALLOC(d * sizeof(float));
  __assume_aligned(v, ALIGNMENT);
  memset(v, 0, d * sizeof(float));
#else
  float* __restrict__ m_all = (float*) ALIGNED_MALLOC(ALIGN_ABOVE(d) * omp_get_max_threads() * sizeof(float));
  __assume_aligned(m_all, ALIGNMENT);
  memset(m_all, 0, ALIGN_ABOVE(d) * omp_get_max_threads() * sizeof(float));
  float* __restrict__ v_all = (float*) ALIGNED_MALLOC(ALIGN_ABOVE(d) * omp_get_max_threads() * sizeof(float));
  __assume_aligned(v_all, ALIGNMENT);
  memset(v_all, 0, ALIGN_ABOVE(d) * omp_get_max_threads() * sizeof(float));
#endif /* ADAM_SHARED */

  // tmp array for holding batch X
  float *batch_X = (float*) ALIGNED_MALLOC(sizeof(float) * batch_size * X_lda);
  __assume_aligned(batch_X, ALIGNMENT);
  // tmp array for holding one-hot batch ys
  float *batch_ys_oh = (float*) ALIGNED_MALLOC(sizeof(float) * batch_size * ys_oh_lda);
  __assume_aligned(batch_ys_oh, ALIGNMENT);
  // tmp array for holding idx batch ys
  int *batch_ys_idx = (int*) ALIGNED_MALLOC(sizeof(int) * batch_size);
  __assume_aligned(batch_ys_idx, ALIGNMENT);
  // vector used for fisher-yates-esque batch selection w/out replacement
  unsigned int *batch_idx_all = (unsigned int*) ALIGNED_MALLOC(sizeof(unsigned int) * ALIGN_ABOVE(n_train) * omp_get_max_threads());
  __assume_aligned(batch_idx_all, ALIGNMENT);

  // collection of uniform distributions for batch selection
  std::vector< std::uniform_int_distribution<unsigned int> > batch_dists;
  // scratch space
  const size_t scratch_size_per_thread = scratch_size(n_train + n_test,d,c);
  assert(scratch_size_per_thread % ALIGNMENT == 0);
  float* __restrict__ scratch_all = (float*) ALIGNED_MALLOC(scratch_size_per_thread * omp_get_max_threads() * sizeof(float));
  __assume_aligned(scratch_all, ALIGNMENT);

  // initialize the batch selection vector (invariant is that it's an unordered set)
  for (int t = 0; t < omp_get_max_threads(); t++) {
    for (unsigned int i = 0; i < n_train; i++) {
      batch_idx_all[t * ALIGN_ABOVE(n_train) + i] = i;
    }
  }

  // initialize each distribution (this is ugly... TODO: any better way?)
  for (unsigned int i = 0; i < batch_size; i++) {
    batch_dists.push_back(std::uniform_int_distribution<unsigned int>(0, n_train - i - 1));
  }

  // initialize weight vector
#pragma vector aligned
  for (unsigned int j = 0; j < d; j++) {
    W[j] = normal_dist(gen_main);
  }

  losses.times.push_back(0);
  loss_t loss;

  loss = logistic_loss(W, X_train, X_lda, ys_idx_train, n_train,
                       d, lambda, scratch_all);

  losses.train_losses.push_back(loss.loss);
  losses.train_errors.push_back(loss.error);
  losses.train_pos.push_back(loss.pos);

  loss = logistic_loss(W, X_test, X_lda, ys_idx_test, n_test,
                       d, lambda, scratch_all);

  losses.test_errors.push_back(loss.error);

  logistic_gradient_batch(G_all, W, X_train, X_lda,
                          ys_idx_train, n_train, d, lambda, scratch_all);

  float nrm = cblas_snrm2(d, G_all, 1);
  losses.grad_sizes.push_back(nrm);

  timing_t full_timer = timing_t();
  full_timer.start_timing_round();


#ifdef PROGRESS
  fprintf(stderr, "#ITER | NORM | DIST TO OPT"
#ifdef LOSS
          "| TRAIN LOSS | TRAIN ERR"
#endif /* LOSS */
          "\n");
  fflush(stderr);
#endif /* PROGRESS */

#pragma omp parallel for schedule(guided)
  for (unsigned int _iter = 0; _iter < niter; _iter++) {
    unsigned int tno = omp_get_thread_num();

    const int m_t = t_all[tno]++;

    float t_exp;
#ifdef ADAM_SHARED
    t_exp = m_t * omp_get_max_threads() + thread_dist(gen_all[tno]);

    float* __restrict__ m_m = m;
    float* __restrict__ m_v = v;
#else
    t_exp = m_t;
    float* __restrict__ m_m = &m_all[ALIGN_ABOVE(d) * tno];
    float* __restrict__ m_v = &v_all[ALIGN_ABOVE(d) * tno];
#endif /* ADAM_SHARED */

    __assume_aligned(m_m, ALIGNMENT);
    __assume_aligned(m_v, ALIGNMENT);

    float beta_1_t = powf(beta_1, t_exp);
    float beta_2_t = powf(beta_2, t_exp);

    float alpha_t = alpha * sqrtf(1 - beta_2_t) / (1 - beta_1_t);

    float* __restrict__ scratch = &scratch_all[scratch_size_per_thread * tno];
    __assume_aligned(scratch, ALIGNMENT);

    float* __restrict__ G = &G_all[ALIGN_ABOVE(d) * tno];
    __assume_aligned(G, ALIGNMENT);

    unsigned int* __restrict__ batch_idx = &batch_idx_all[ALIGN_ABOVE(n_train) * tno];
    __assume_aligned(batch_idx, ALIGNMENT);

    for (unsigned int bidx = 0; bidx < batch_size; bidx++) {
      const unsigned int rand_idx = batch_dists[bidx](gen_all[tno]);
      const unsigned int idx = batch_idx[rand_idx];
      batch_idx[rand_idx] = batch_idx[n_train - 1 - bidx];
      batch_idx[n_train - 1 - bidx] = idx;

      float *x_dst = &batch_X[bidx * X_lda];
      const float *x_src = &X_train[idx * X_lda];

#pragma vector aligned
      for (unsigned int j = 0; j < d; j++) {
        x_dst[j] = x_src[j];
      }

      batch_ys_idx[bidx] = ys_idx_train[idx];
      float *ys_dst = &batch_ys_oh[bidx * ys_oh_lda];
      const float *ys_src = &ys_oh_train[idx * ys_oh_lda];

#pragma vector aligned
      for (unsigned int k = 0; k < c; k++) {
        ys_dst[k] = ys_src[k];
      }
    }

    memset(G, 0, d * sizeof(float));
    logistic_gradient_batch(G, W, batch_X, X_lda,
                            batch_ys_idx,
                            batch_size, d, lambda, scratch);


#pragma vector aligned
    for (unsigned int j = 0; j < d; j++) {
      m_m[j] = beta_1 * m_m[j] + (1 - beta_1) * G[j];
      m_v[j] = beta_2 * m_v[j] + (1 - beta_2) * G[j] * G[j];

      W[j] -= alpha_t * m_m[j] / (sqrtf(m_v[j]) + 1e-8);
      // W[j] -= alpha * G[j];
    }

    nrm = cblas_snrm2(d, G, 1);

    float dto = 0;
    for (unsigned int j = 0; j < d; j++) {
      float dst = W[j] - W_opt[j];
      dto += dst * dst;
    }
    dto = sqrtf(dto);

#ifdef LOSSES
    loss_t train_loss, test_loss;
    train_loss = logistic_loss(W, X_train, X_lda, ys_idx_train, n_train,
                               d, lambda, scratch);
    test_loss = logistic_loss(W, X_test, X_lda, ys_idx_test, n_test,
                              d, lambda, scratch);

#endif /* LOSSES */

#ifdef RAW_OUTPUT
    printf(
#ifdef LOSSES
           "%f %f %f %f %f %f\n",
#else
           "%f %f %f\n",
#endif /* LOSSES */
           full_timer.total_time(),
           nrm,
           dto
#ifdef LOSSES
           , train_loss.loss,
           train_loss.error,
           test_loss.error
#endif /* LOSSES */
           );
    // fflush(stdout);
#endif  /* RAW_OUTPUT */

#ifdef PROGRESS
    unsigned int it = m_t * omp_get_max_threads();
    fprintf(stderr,
            "%5d | %4.2f | %11.4f"
#ifdef LOSSES
            " | %10.2f | %9.3f"
#endif /* LOSSES */
            "\r",
            it, nrm, dto
#ifdef LOSSES
            ,train_loss.loss, train_loss.error
#endif /* LOSSES */
            );
    if (it % (niter / 10) == 0) {
      fprintf(stderr, "\n");
    }
    fflush(stderr);
#endif /* PROGRESS */
  }

#ifdef PROGRESS
  fprintf(stderr, "\n");
#endif /* PROGRESS */

  losses.times.push_back(full_timer.total_time());


  loss = logistic_loss(W, X_train, X_lda, ys_idx_train, n_train,
                       d, lambda, scratch_all);

  losses.train_losses.push_back(loss.loss);
  losses.train_errors.push_back(loss.error);

  loss = logistic_loss(W, X_test, X_lda, ys_idx_test, n_test,
                       d, lambda, scratch_all);

  losses.test_errors.push_back(loss.error);

  logistic_gradient_batch(G_all, W, X_train, X_lda,
                          ys_idx_train, n_train, d, lambda, scratch_all);

  nrm = cblas_snrm2(d, G_all, 1);
  losses.grad_sizes.push_back(nrm);


  fprintf(stderr, "Final training loss: %f\n", losses.train_losses.back());
  fprintf(stderr, "Final training error: %f\n", losses.train_errors.back());
  fprintf(stderr, "Final testing error: %f\n", losses.test_errors.back());

  ALIGNED_FREE(W);
  ALIGNED_FREE((float*) X_train);
  ALIGNED_FREE((int*) ys_idx_train);
  ALIGNED_FREE((float*) ys_oh_train);
  ALIGNED_FREE((float*) X_test);
  ALIGNED_FREE((int*) ys_idx_test);
  ALIGNED_FREE((float*) ys_oh_test);
  ALIGNED_FREE((float*) scratch_all);
  ALIGNED_FREE(G_all);
  ALIGNED_FREE(batch_idx_all);
  ALIGNED_FREE(batch_X);
  ALIGNED_FREE(batch_ys_oh);
  ALIGNED_FREE(batch_ys_idx);
  ALIGNED_FREE(t_all);

#ifdef ADAM_SHARED
  ALIGNED_FREE(m);
  ALIGNED_FREE(v);
#else
  ALIGNED_FREE(m_all);
  ALIGNED_FREE(v_all);
#endif /* ADAM_SHARED */


  return losses;
}
