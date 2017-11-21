#include <string.h>
#include <stdio.h>
#include <math.h>

#include "mblas.hpp"
#include "loss.hpp"

#define CEIL_DIV(n, d) (((n) + (d) - 1) / (d))
#define ALIGN_ABOVE(n) (CEIL_DIV(n, 16) * 16)

size_t
scratch_size
(
 size_t n,
 size_t d,
 size_t c
 ) {
  return sizeof(float) * (ALIGN_ABOVE(n) * ALIGN_ABOVE(c) + ALIGN_ABOVE(d) * ALIGN_ABOVE(c));
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

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              n, c, d,
              1, X, X_lda, W, W_lda,
              0, XWT, XWT_lda);

  float loss = 0;
  unsigned int correct = 0;

  for (unsigned int i = 0; i < n; i++) {
    float sum = 0;
    unsigned int max_idx = -1;
    float max_val = -1;

    for (unsigned int j = 0; j < c; j++) {
      float curr = XWT[i * XWT_lda + j];
      float exped = expf(curr);
      if (exped > max_val) {
        max_idx = j;
        max_val = exped;
      }
      sum += exped;
    }

    loss += logf(sum) - XWT[i * XWT_lda + y[i]];
    correct += max_idx == y[i];
  }

  loss_t ret;
  ret.loss = loss / n;
  ret.error = 1 - ((float)correct) / n;

  return ret;
}

void multinomial_gradient_single
(
 float* __restrict__ G,       // c x d
 const float* __restrict__ W, // c x d
 const size_t WG_lda,         // lda (axis 1 stride) of W and G
 const float* __restrict__ x, // d x 1
 const float* y_oh,           // c x 1
 const size_t d,              // data dimensionality
 const size_t c,              // num classes
 const float beta,            // momentum parameter
 const float lambda,          // regularization parameter
 float* __restrict__ scratch  // scratch space
 ) {
  float *Wx   /* c x 1 */ = scratch;
  float *G_tmp   /* c x d */ = Wx + c;
  memset(G_tmp, 0, sizeof(float) * c * WG_lda);

  cblas_sgemv(CblasRowMajor, CblasNoTrans,
              c, d,
              1, W, WG_lda, x, 1,
              0, Wx, 1);

  vsExp(c, Wx, Wx);

  float sum = cblas_sasum(c, Wx, 1);

  SAXPBY(c, -1, y_oh, 1, (1 / sum), Wx, 1);
  cblas_sger(CblasRowMajor, c, d, 1, Wx, 1, x, 1, G_tmp, WG_lda);

  for (unsigned int i = 0; i < c; i++) {
    SAXPBY(d, beta, &G_tmp[i * WG_lda], 1, (1 - beta), &G[i * WG_lda], 1);
  }
}

void multinomial_gradient_batch
(
 float* __restrict__ G,       // c x d
 const float* __restrict__ W, // c x d
 const size_t WG_lda,         // lda (axis 1 stride) of W and G
 const float* __restrict__ X, // n x d data
 const size_t X_lda,          // lda (axis 1 stride) of X
 const float* __restrict__ y_oh,  // n x 1 classes
 const size_t ys_lda,
 const size_t n,              // number of losses
 const size_t d,              // data dimensionality
 const size_t c,              // num classes
 const float beta,            // momentum parameter
 const float lambda,          // regularization parameter
 float* __restrict__ scratch  // scratch space
 ) {
  float *XWT   /* n x c */ = scratch;
  const size_t XWT_lda = ALIGN_ABOVE(c);

  float *G_tmp   /* c x d */ = XWT + n * XWT_lda;
  memset(G_tmp, 0, sizeof(float) * c * WG_lda);

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
    cblas_sger(CblasRowMajor, c, d, 1, Wx, 1, x, 1, G_tmp, WG_lda);
  }

  SAXPBY(c * WG_lda, beta, G_tmp, 1, (1 - beta), G, 1);
}
