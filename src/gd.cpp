#include <stdlib.h>
#include <random>
#include <functional>
#include <vector>
#include "string.h"

#include "gd.hpp"
#include "loss.hpp"
#include "mblas.hpp"
#include "timing.hpp"

void sgd
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
 const unsigned int nepoch,     // number of epochs to run (zero for SGD)
 const unsigned int niter,      // number of iterations to run per epoch
 const float alpha,             // step size
 const float lambda,            // regularization parameter
 const size_t batch_size,       // batch size
 const unsigned int seed        // random seed
 ) {
#ifndef HOGWILD
  omp_set_num_threads(1);
#endif

  fprintf(stderr, "Number of threads: %d\n", omp_get_max_threads());

  // gd_losses_t losses;

  // initialize randoms
  std::mt19937 gen(seed);
  std::normal_distribution<float> normal_dist(0, 1);
  std::uniform_int_distribution<unsigned int> uniform_dist(0, n_train-1);

  const size_t W_lda = ALIGN_ABOVE(d);
  assert(W_lda % ALIGNMENT == 0);
  __assume(W_lda % ALIGNMENT == 0);
  assert(W_lda >= d);
  __assume(W_lda >= d);
  const size_t X_lda = ALIGN_ABOVE(d);
  assert(X_lda % ALIGNMENT == 0);
  __assume(X_lda % ALIGNMENT == 0);
  const size_t ys_oh_lda = ALIGN_ABOVE(c);
  assert(ys_oh_lda % ALIGNMENT == 0);
  __assume(ys_oh_lda % ALIGNMENT == 0);

  auto max_threads = omp_get_max_threads();

  float* __restrict__ W_shared = (float*) ALIGNED_MALLOC(c * W_lda * sizeof(float));
  __assume_aligned(W_shared, ALIGNMENT);
  float* __restrict__ W_private_scratch = (float*) ALIGNED_MALLOC(max_threads * c * W_lda * sizeof(float));
  __assume_aligned(W_private_scratch, ALIGNMENT);

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
  float* __restrict__ G_all = (float*) ALIGNED_MALLOC(c * W_lda * max_threads * sizeof(float));
  __assume_aligned(G_all, ALIGNMENT);

  // tmp array for holding batch X
  int batch_X_size = batch_size * X_lda;
  float *batch_X_scratch = (float*) ALIGNED_MALLOC(sizeof(float) * batch_X_size * max_threads);
  __assume_aligned(batch_X_scratch, ALIGNMENT);

  // tmp array for holding one-hot batch ys
  int batch_ys_size = batch_size * ys_oh_lda;
  float *batch_ys_scratch = (float*) ALIGNED_MALLOC(sizeof(float) * batch_ys_size * max_threads);
  __assume_aligned(batch_ys_scratch, ALIGNMENT);

  // vector used for fisher-yates-esque batch selection w/out replacement
  int batch_idx_size = n_train;
  unsigned int *batch_idx_scratch = (unsigned int*) ALIGNED_MALLOC(sizeof(unsigned int) * batch_idx_size * max_threads);
  __assume_aligned(batch_idx_scratch, ALIGNMENT);

  // collection of uniform distributions for batch selection
  std::vector< std::uniform_int_distribution<unsigned int> > batch_dists;
  // scratch space
  const size_t scratch_size_per_thread = scratch_size(n_train + n_test,d,c);
  assert(scratch_size_per_thread % ALIGNMENT == 0);
  float* __restrict__ scratch_all = (float*) ALIGNED_MALLOC(scratch_size_per_thread * omp_get_max_threads() * sizeof(float));
  __assume_aligned(scratch_all, ALIGNMENT);

  // initialize the batch selection vector (invariant is that it's an unordered set)
  for (int tno = 0; tno < max_threads; ++tno) {
    for (unsigned int i = 0; i < n_train; i++) {
      batch_idx_scratch[tno * batch_idx_size + i] = i;
    }
  }

  // initialize each distribution (this is ugly... TODO: any better way?)
  for (unsigned int i = 0; i < batch_size; i++) {
    batch_dists.push_back(std::uniform_int_distribution<unsigned int>(0, n_train - i - 1));
  }

  // initialize weight vector
  #pragma vector aligned
  for (unsigned int j = 0; j < c; j++) {
    float* __restrict__ Wj = &W_shared[j * W_lda];

    for (unsigned int k = 0; k < d; k++) {
      Wj[k] = normal_dist(gen);
    }
  }

  timing_t grad_timer = timing_t();

  printf("time niter trainloss trainerror testloss\n");

  fprintf(stderr, "#/# EPOCH | #/# ITERATION | TRAIN LOSS | TRAIN ERR | NORM | TEST ERR | WC TIME\n");
  fflush(stderr);

  for (unsigned int _epoch = 0; _epoch < nepoch; _epoch++) {

    grad_timer.start_timing_round();

    #pragma omp parallel for schedule(guided)
    for (unsigned int _iter = 0; _iter < niter; _iter++) {
      unsigned int tno = omp_get_thread_num();
      float* __restrict__ scratch = &scratch_all[scratch_size_per_thread * tno];
      __assume_aligned(scratch, ALIGNMENT);

    #ifdef MEM_COPY
      float* __restrict__ W_private = &W_private_scratch[tno * c * W_lda];
      memcpy(W_private, W_shared, c * W_lda * sizeof(float));
    #else
      float* W_private = W_shared;
      __assume_aligned(W_private, ALIGNMENT);
    #endif

      float* __restrict__ G = &G_all[c * W_lda * tno];
      __assume_aligned(G, ALIGNMENT);

      float* __restrict__ batch_X = &batch_X_scratch[tno * batch_X_size];
      float* __restrict__ batch_ys = &batch_ys_scratch[tno * batch_ys_size];
      unsigned int* __restrict__ batch_idx = &batch_idx_scratch[tno * batch_idx_size];

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

      multinomial_gradient_batch(G, W_private, W_lda, batch_X, X_lda,
                                 batch_ys, ys_oh_lda,
                                 batch_size, d, c, 1, lambda, scratch);

      // #pragma omp atomic weight_vector
      #pragma vector aligned
      for (unsigned int j = 0; j < c; j++) {
        float* __restrict__ Wj = &W_shared[j * W_lda];
        float* __restrict__ Gj = &G[j * W_lda];

        #pragma vector aligned
        for (unsigned int k = 0; k < d; k++) {
          #ifdef ATOMIC_UPDATES
            #pragma omp atomic
            Wj[k] -= alpha * Gj[k];
          #else
            Wj[k] -= alpha * Gj[k];
          #endif
        }
      }
    }

    grad_timer.end_timing_round(niter);

    float* __restrict__ scratch = scratch_all;
    __assume_aligned(scratch, ALIGNMENT);

    loss_t train_loss = multinomial_loss(W_shared, W_lda, X_train, X_lda, ys_idx_train,
                                         n_train, d, c, lambda, scratch);
    loss_t test_loss = multinomial_loss(W_shared, W_lda, X_test, X_lda, ys_idx_test,
                                        n_test, d, c, lambda, scratch);


    printf(
           "%f %d %f %f %f\n",
           grad_timer.total_time(),
           niter*(_epoch+1),
           train_loss.loss,
           train_loss.error,
           test_loss.error
           );
    fflush(stdout);

    {
      float nrm = 0;
      fprintf(stderr,
              "%4d %4d | %6d %6d | %10.2f | %9.3f | %4.2f | %8.3f | %7.3f\n",
              _epoch, nepoch, niter*(_epoch+1), niter,
              train_loss.loss, train_loss.error,
              nrm, test_loss.error,
              grad_timer.total_time()
              );
    }
    fflush(stderr);
  }

  fprintf(stderr, "\n");

  fprintf(stderr, "Grad time per step: %f\n", grad_timer.time_per_step());

  auto train_loss = multinomial_loss(W_shared, W_lda, X_train, X_lda, ys_idx_train, n_train,
                                     d, c, lambda, scratch_all);

  auto test_loss = multinomial_loss(W_shared, W_lda, X_test, X_lda, ys_idx_test, n_test,
                                    d, c, lambda, scratch_all);

  fprintf(stderr, "Final training loss: %f\n", train_loss.loss);
  fprintf(stderr, "Final training error: %f\n", train_loss.error);
  fprintf(stderr, "Final testing error: %f\n", test_loss.error);

  ALIGNED_FREE(W_shared);
  ALIGNED_FREE(W_private_scratch);
  ALIGNED_FREE((float*) X_train);
  ALIGNED_FREE((unsigned int*) ys_idx_train);
  ALIGNED_FREE((float*) ys_oh_train);
  ALIGNED_FREE((float*) X_test);
  ALIGNED_FREE((unsigned int*) ys_idx_test);
  ALIGNED_FREE((float*) ys_oh_test);
  ALIGNED_FREE((float*) scratch_all);
  ALIGNED_FREE(G_all);
  ALIGNED_FREE(batch_idx_scratch);
  ALIGNED_FREE(batch_X_scratch);
  ALIGNED_FREE(batch_ys_scratch);
}
