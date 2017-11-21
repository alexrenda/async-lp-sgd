void sgd
(
 float* __restrict__ W,       // c x d
 const size_t W_lda,          // lda (axis 1 stride) of W
 const float* __restrict__ X, // n x d
 const size_t X_lda,          // lda (axis 1 stride) of X
 const unsigned int* __restrict__ ys_idx,  // n x 1
 const float* __restrict__ ys_oh,  // n x 1
 const size_t ys_oh_lda,
 const size_t n,              // num training samples
 const size_t d,              // data dimensionality
 const size_t c,              // num classes
 const unsigned int niter,    // number of iterations to run
 const float alpha,           // step size
 const float beta,            // parameter of momentum
 const float lambda,          // regularization parameter
 const size_t batch_size,     // batch size
 const unsigned int seed,     // random seed
 float* __restrict losses,    // intermediate losses
 const size_t nlosses         // how many intermediate losses to write (if possible)
 );
