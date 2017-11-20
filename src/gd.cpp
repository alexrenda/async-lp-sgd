#include <stdlib.h>
#include <random>
#include <functional>
#include <vector>

#include "loss.h"
#include "mblas.h"
#include "string.h"
#include "timing.h"

void sgd
(
 float* __restrict__ W,       // c x d
 const size_t W_lda,          // lda (axis 1 stride) of W
 const float* __restrict__ X, // n x d
 const size_t X_lda,          // lda (axis 1 stride) of X
 const unsigned int* __restrict__ ys,  // n x 1
 const size_t n,              // num training samples
 const size_t d,              // data dimensionality
 const size_t c,              // num classes
 const unsigned int niter,    // number of iterations to run
 const float alpha,           // step size
 const float beta,            // parameter of momentum
 const size_t batch_size,     // parameter of momentum
 const unsigned int seed,     // random seed
 float* __restrict losses,    // intermediate losses
 const size_t nlosses         // how many intermediate losses to write (2 <= nlosses <= niter)
 ) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> normal_dist(0, 1);
  std::uniform_int_distribution<int> uniform_dist(0, n-1);

  float* __restrict__ G = (float*) calloc(c * W_lda, sizeof(float));
  float* __restrict__ scratch = (float*) malloc(scratch_size(n,d,c));

  for (unsigned int j = 0; j < c; j++) {
    for (unsigned int k = 0; k < d; k++) {
      W[j * W_lda + k] = normal_dist(gen);
    }
  }

  // TODO: nlosses must be well formed
  unsigned int n_outer_iter = (nlosses > 1) ? (nlosses - 1) : 1;
  unsigned int inner_count_per_outer = niter / n_outer_iter;

  loss_t initial_loss = multinomial_loss
    (W,
     W_lda,
     X,
     X_lda,
     ys,
     n,
     d,
     c,
     scratch
     );

  losses[0] = initial_loss.loss;

  printf("Initial loss: %.1f, error: %.3f\n",
         initial_loss.loss, initial_loss.error);

  timing_t loss_timer = timing_t();
  timing_t grad_timer = timing_t();


  unsigned int *batch_idx = (unsigned int*) malloc(sizeof(unsigned int) * n);
  for (int i = 0; i < n; i++) {
    batch_idx[i] = i;
  }

  std::vector<std::uniform_int_distribution<int>> batch_dists;
  for (int i = 0; i < batch_size; i++) {
    batch_dists.push_back(std::uniform_int_distribution<int>(0, n - i - 1));
  }

  float *batch_X = (float*) malloc(sizeof(float) * batch_size * X_lda);
  unsigned int *batch_ys = (unsigned int*) malloc(sizeof(unsigned int) * batch_size);

  for (unsigned int outer_i = 0; outer_i < n_outer_iter; outer_i++) {
    unsigned int inner_niter;
    if (outer_i == n_outer_iter - 1) {
      inner_niter = niter - outer_i * inner_count_per_outer;
    } else {
      inner_niter = inner_count_per_outer;
    }

    for (unsigned int iter = 0; iter < inner_niter; iter++) {
      grad_timer.start_timing_round();
      if (batch_size == 1) {
        const int idx = uniform_dist(gen);
        multinomial_gradient_batch(G, W, W_lda, &X[idx * X_lda], X_lda,
                                   &ys[idx], 4, d, c, beta, scratch);
      } else {
        for (unsigned int bidx = 0; bidx < batch_size; bidx++) {
          const int rand_idx = batch_dists[bidx](gen);
          const int idx = batch_idx[rand_idx];
          batch_idx[rand_idx] = batch_idx[n - 1 - bidx];
          batch_idx[n - 1 - bidx] = idx;

          memcpy(&batch_X[bidx * X_lda], &X[idx * X_lda], sizeof(float) * d);
          batch_ys[bidx] = ys[idx];
        }
        multinomial_gradient_batch(G, W, W_lda, batch_X, X_lda, batch_ys,
                                   batch_size, d, c, beta, scratch);
      }
      grad_timer.end_timing_round(batch_size);
      SAXPBY(c * W_lda, -alpha * batch_size, G, 1, 1, W, 1);
    }

    loss_timer.start_timing_round();
    loss_t loss = multinomial_loss
      (W,
       W_lda,
       X,
       X_lda,
       ys,
       n,
       d,
       c,
       scratch
       );
    loss_timer.end_timing_round(1);

    printf("Iter %d: ran for %d steps (loss: %.1f, error: %.3f)\n",
           outer_i, inner_niter, loss.loss, loss.error);

    losses[outer_i + 1] = loss.loss;

  }

  printf("Grad time per step: %f\n", grad_timer.time_per_step());
  printf("Loss time per step: %f\n", loss_timer.time_per_step());

  free(G);
  free(batch_idx);
  free(batch_X);
  free(batch_ys);
}
