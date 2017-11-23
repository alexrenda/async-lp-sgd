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

    float maxval = Wx[0];
    float maxidx = 0;
    float minval = Wx[0];
    float minidx = 0;

    for (unsigned int j = 1; j < c; j++) {
      const float m_val = Wx[j];
      if (m_val > maxval) {
        maxidx = j;
        maxval = m_val;
      } else if (m_val < minval) {
        minidx = j;
        minval = m_val;
      }
    }

    float sum = 0;

    for (unsigned int j = 0; j < c; j++) {
      sum += expf(Wx[j] - minval);
    }

    float numerator = expf(Wx[y[i]] - minval);
    loss -= logf(numerator / sum);
    correct += maxidx == y[i];
  }

  float reg = 0;
  for (unsigned int k = 0; k < c; k++) {
    for (unsigned int j = 0; j < d; j++) {
      reg += W[k * W_lda + j];
    }
  }

  loss_t ret;
  ret.loss = loss / n + lambda * reg / 2;
  ret.error = 1 - ((float)correct) / n;

  return ret;
}

void multinomial_gradient_batch
(
 float* __restrict__ V,       // c x d
 const float* __restrict__ W, // c x d
 const size_t WG_lda,         // lda (axis 1 stride) of W and G
 const float* __restrict__ X, // n x d data
 const size_t X_lda,          // lda (axis 1 stride) of X
 const float* __restrict__ y_oh,  // n x c classes
 const size_t ys_lda,
 const size_t n,              // number of losses
 const size_t d,              // data dimensionality
 const size_t c,              // num classes
 const float alpha,           // learning rate
 const float beta,            // momentum parameter
 const float lambda,          // regularization parameter
 float* __restrict__ scratch  // scratch space
 ) {
  float *XWT   /* n x c */ = scratch;
  const size_t XWT_lda = ALIGN_ABOVE(c);

  float *G_tmp   /* c x d */ = XWT + n * XWT_lda;

  __assume_aligned(XWT, ALIGNMENT);
  __assume_aligned(V, ALIGNMENT);
  __assume_aligned(G_tmp, ALIGNMENT);
  __assume_aligned(W, ALIGNMENT);
  __assume_aligned(X, ALIGNMENT);
  __assume_aligned(y_oh, ALIGNMENT);

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

  for (unsigned int k = 0; k < c; k++) {
    for (unsigned int j = 0; j < d; j++) {

      if (G_tmp[k * WG_lda + j] > 0) {
        V[k * WG_lda + j] =
          beta * V[k * WG_lda + j]
          - alpha * (G_tmp[k * WG_lda + j]/n + lambda * W[k * WG_lda + j]);
      }
    }
  }
}
