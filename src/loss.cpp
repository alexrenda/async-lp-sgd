#include <string.h>

#include "mblas.h"
#include "loss.h"

size_t
scratch_size
(
 size_t n,
 size_t d,
 size_t c
 ) {
  return sizeof(float) * (n * c);
}

loss_t multinomial_loss
(
 const float* __restrict__ W, // c x d
 const size_t W_lda,          // lda (axis 1 stride) of W
 const float* __restrict__ X, // n x d
 const size_t X_lda,          // lda (axis 1 stride) of X
 const char* __restrict__ y,  // n x 1
 const size_t n,              // num training samples
 const size_t d,              // data dimensionality
 const size_t c,              // num classes
 float* __restrict__ scratch  // scratch space
 ) {
  float *XWT   /* n x c */ = scratch;
  const size_t XWT_lda = c;

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              n, c, d,
              1, X, X_lda, W, W_lda,
              0, XWT, XWT_lda);

  float loss = 0;
  unsigned int correct = 0;

  for (int i = 0; i < n; i++) {
    const int vvexp_len = c;
    vvexp2f(&XWT[i * XWT_lda], &XWT[i * XWT_lda], &vvexp_len);

    float sum = cblas_sasum(c, &XWT[i * XWT_lda], 1);
    int max_idx = cblas_isamax(c, &XWT[i * XWT_lda], 1);

    loss -= log2f(XWT[i * XWT_lda + y[i]] / sum);
    correct += max_idx == y[i];
  }

  loss_t ret;
  ret.loss = loss / n;
  ret.error = 1 - ((float)correct) / n;

  return ret;
}

void multinomial_gradient
(
 float* __restrict__ G,       // c x d
 const float* __restrict__ W, // c x d
 const size_t WG_lda,         // lda (axis 1 stride) of W and G
 const float* __restrict__ x, // d x 1
 const char y,                // class
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
