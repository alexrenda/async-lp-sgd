#ifdef OSX_ACCELERATE
#  include <Accelerate/Accelerate.h>
#elif defined(__GNUC__) || defined(__GNUG__)
#  include <cblas.h>
#endif

#include <stdio.h>

int main() {
  printf("Hello, world!\n");
}
