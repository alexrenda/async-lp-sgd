#include <stdlib.h>
#include <random>
#include <functional>

#include "loss.h"
#include "mblas.h"

void sgd
(
 float* __restrict__ W,       // c x d
 const size_t W_lda,          // lda (axis 1 stride) of W
 const float* __restrict__ X, // n x d
 const size_t X_lda,          // lda (axis 1 stride) of X
 const char* __restrict__ ys,  // n x 1
 const size_t n,              // num training samples
 const size_t d,              // data dimensionality
 const size_t c,              // num classes
 const unsigned int niter,    // number of iterations to run
 const float alpha,           // step size
 const float beta,            // parameter of momentum
 const unsigned int seed,     // random seed
 float* __restrict losses,    // intermediate losses
 const size_t nlosses         // how many intermediate losses to write (2 <= nlosses <= niter)
 ) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> normal_dist(0, 1);
  std::uniform_int_distribution<int> uniform_dist(0, n-1);

  float* __restrict__ G = (float*) calloc(c * W_lda, sizeof(float));
  float* __restrict__ scratch = (float*) malloc(scratch_size(n,d,c));

  for (int j = 0; j < c; j++) {
    for (int k = 0; k < d; k++) {
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

  for (unsigned int outer_i = 0; outer_i < n_outer_iter; outer_i++) {
    unsigned int inner_niter;
    if (outer_i == n_outer_iter - 1) {
      inner_niter = niter - outer_i * inner_count_per_outer;
    } else {
      inner_niter = inner_count_per_outer;
    }

    for (unsigned int iter = 0; iter < inner_niter; iter++) {
      const int idx = uniform_dist(gen);
      const float *x = &X[idx * X_lda];
      const char y = ys[idx];

      multinomial_gradient(G, W, W_lda, x, y, d, c, beta, scratch);
      SAXPBY(c * W_lda, -alpha, G, 1, 1, W, 1);
    }
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

    printf("Iter %d: ran for %d steps (loss: %.1f, error: %.3f)\n",
           outer_i, inner_niter, loss.loss, loss.error);

    losses[outer_i + 1] = loss.loss;

  }

  free(G);

}
