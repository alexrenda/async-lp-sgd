#include <string.h>
#include <stdio.h>
#include <math.h>

#include "mblas.hpp"
#include "loss.hpp"

size_t
scratch_size
(
 size_t n,
 size_t d,
 size_t c
 ) {
  return sizeof(float) * (ALIGN_ABOVE(n) * ALIGN_ABOVE(c) +
                          ALIGN_ABOVE(c) * ALIGN_ABOVE(d));
}

loss_t multinomial_loss
(
 const float* __restrict__ W, // c x d
 const size_t W_lda,          // lda (axis 1 stride) of W
 const float* __restrict__ X, // n x d
 const size_t X_lda,          // lda (axis 1 stride) of X
 const unsigned int* __restrict__ y,  // n x 1
 const size_t n,              // num training samples
 const size_t d,              // data dimensionality
 const size_t c,              // num classes
 const float lambda,          // regularization parameter
 float* __restrict__ scratch  // scratch space
 ) {
  float *XWT   /* n x c */ = scratch;
  const size_t XWT_lda = ALIGN_ABOVE(c);

  __assume_aligned(XWT, ALIGNMENT);
  __assume_aligned(W, ALIGNMENT);
  __assume_aligned(X, ALIGNMENT);
  __assume_aligned(y, ALIGNMENT);

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              n, c, d,
              1, X, X_lda, W, W_lda,
              0, XWT, XWT_lda);

  float loss = 0;
  unsigned int correct = 0;

  for (unsigned int i = 0; i < n; i++) {
    const float* Wx = &XWT[i * XWT_lda];
    __assume_aligned(Wx, ALIGNMENT);

    const int maxidx = cblas_isamax(c, Wx, 1);
    const float maxval = Wx[maxidx];

    float sum = 0;

    for (unsigned int j = 0; j < c; j++) {
      sum += expf(Wx[j] - maxval);
    }

    loss -= Wx[y[i]] - maxval - logf(sum);
    correct += maxidx == y[i];
  }

  float reg = 0;
  for (unsigned int k = 0; k < c; k++) {
    for (unsigned int j = 0; j < d; j++) {
      reg += W[k * W_lda + j];
    }
  }

  loss_t ret;
  ret.loss = loss / n + 2 * lambda * reg;
  ret.error = 1 - ((float)correct) / n;

  return ret;
}

void multinomial_gradient_batch
(
 float* __restrict__ G,       // c x d
 const float* __restrict__ W, // c x d
 const size_t WG_lda,         // lda (axis 1 stride) of W and G
 const float* __restrict__ X, // n x d data
 const size_t X_lda,          // lda (axis 1 stride) of X
 const float* __restrict__ y_oh,  // n x c classes
 const size_t ys_lda,
 const size_t n,              // number of losses
 const size_t d,              // data dimensionality
 const size_t c,              // num classes
 const float lambda,          // regularization parameter
 float* __restrict__ scratch  // scratch space
 ) {
  float *XWT   /* n x c */ = scratch;
  const size_t XWT_lda = ALIGN_ABOVE(c);

  __assume_aligned(XWT, ALIGNMENT);
  __assume_aligned(G, ALIGNMENT);
  __assume_aligned(W, ALIGNMENT);
  __assume_aligned(X, ALIGNMENT);
  __assume_aligned(y_oh, ALIGNMENT);

  memset(G, 0, sizeof(float) * c * WG_lda);

  cblas_saxpy(c * WG_lda, lambda, W, 1, G, 1);

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              n, c, d,
              1, X, X_lda, W, WG_lda,
              0, XWT, XWT_lda);

  for (unsigned int i = 0; i < n; i++) {
    float *Wx = &XWT[i * XWT_lda];
    const float *x = &X[i * X_lda];

    vsExp(c, Wx, Wx);

    float sum = cblas_sasum(c, Wx, 1);

    SAXPBY(c, -1, &y_oh[i * ys_lda], 1, (1 / sum), Wx, 1);
    cblas_sger(CblasRowMajor, c, d, 1, Wx, 1, x, 1, G, WG_lda);
  }

  SAXPBY(c * WG_lda, lambda, W, 1, 1, G, 1);
}
