#include <string.h>
#include <stdio.h>

#include "mblas.h"
#include "loss.h"

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
    const int vvexp_len = c;
    vvexp2f(&XWT[i * XWT_lda], &XWT[i * XWT_lda], &vvexp_len);

    float sum = cblas_sasum(c, &XWT[i * XWT_lda], 1);
    unsigned int max_idx = cblas_isamax(c, &XWT[i * XWT_lda], 1);

    loss -= log2f(XWT[i * XWT_lda + y[i]] / sum);
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
 const unsigned int y,        // class
 const size_t d,              // data dimensionality
 const size_t c,              // num classes
 const float beta,            // momentum parameter
 float* __restrict__ scratch  // scratch space
 ) {
  float *Wx   /* c x 1 */ = scratch;

  cblas_sgemv(CblasRowMajor, CblasNoTrans,
              c, d,
              1, W, WG_lda, x, 1,
              0, Wx, 1);

  const int length = c;
  vvexp2f(Wx, Wx, &length);
  float sum = cblas_sasum(c, Wx, 1);

  // this is a rank 1 update?

  for (unsigned int j = 0; j < c; j++) {
    float grad = -beta * ((y == j) - Wx[j] / sum);
    SAXPBY(d, grad, x, 1, (1 - beta), &G[j * WG_lda], 1);
  }
}

void multinomial_gradient_batch
(
 float* __restrict__ G,       // c x d
 const float* __restrict__ W, // c x d
 const size_t WG_lda,         // lda (axis 1 stride) of W and G
 const float* __restrict__ X, // n x d data
 const size_t X_lda,          // lda (axis 1 stride) of X
 const unsigned int* __restrict__ y,  // n x 1 classes
 const size_t n,              // number of losses
 const size_t d,              // data dimensionality
 const size_t c,              // num classes
 const float beta,            // momentum parameter
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

    const int length = c;
    vvexp2f(Wx, Wx, &length);

    float sum = cblas_sasum(c, Wx, 1);

    // this is a rank 1 update?
    for (unsigned int j = 0; j < c; j++) {
      float grad = -beta * ((y[i] == j) - Wx[j] / sum);
      SAXPBY(d, grad, x, 1, (1 - beta), &G_tmp[j * WG_lda], 1);
    }
  }

  SAXPBY(c * WG_lda, beta, G_tmp, 1, (1 - beta), G, 1);
}
