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
 const char* __restrict__ y,  // n x 1
 const size_t n,              // num training samples
 const size_t d,              // data dimensionality
 const size_t c,              // num classes
 float* __restrict__ scratch  // scratch space
 );

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
 );
