size_t
scratch_size
(
 size_t n,
 size_t d,
 size_t c
 );

typedef struct loss_t {
  float loss;
  float error;
} loss_t;

loss_t multinomial_loss
(
 const float* __restrict__ W, // c x d
 const size_t W_lda,          // lda (axis 1 stride) of W
 const float* __restrict__ X, // n x d
 const size_t X_lda,          // lda (axis 1 stride) of X
 const unsigned int* __restrict__ y,  // n x 1
 const size_t n,              // num samples
 const size_t d,              // data dimensionality
 const size_t c,              // num classes
 const float lambda,          // regularization parameter
 float* __restrict__ scratch  // scratch space
 );

void multinomial_gradient_single
(
 float* __restrict__ G,       // c x d
 const float* __restrict__ W, // c x d
 const size_t WG_lda,         // lda (axis 1 stride) of W and G
 const float* __restrict__ x, // 1 x d
 const float *ys_oh,          // class ( 1 x c)
 const size_t d,              // data dimensionality
 const size_t c,              // num classes
 const float beta,            // momentum parameter
 const float lambda,          // regularization parameter
 float* __restrict__ scratch  // scratch space
 );

void multinomial_gradient_batch
(
 float* __restrict__ G,       // c x d
 const float* __restrict__ W, // c x d
 const size_t WG_lda,         // lda (axis 1 stride) of W and G
 const float* __restrict__ X, // n x d
 const size_t X_lda,          // lda (axis 1 stride) of X
 const float * __restrict__ y, // class
 const size_t ys_lda,
 const size_t n,              // number of samples
 const size_t d,              // data dimensionality
 const size_t c,              // num classes
 const float beta,            // momentum parameter
 const float lambda,          // regularization parameter
 float* __restrict__ scratch  // scratch space
 );
