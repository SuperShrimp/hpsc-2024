#include <cstdio>
#include <omp.h>

int main() {
  int n = 10;
  double dx = 1. / n;
  double pi = 0;
#pragma omp parallel for num_threads(10) reduction(+:pi)
  for (int i=0; i<n; i++) {
    // 好几个dx同时写x，会造成冲突，导致结果有问题

  double x = (i + 0.5) * dx;

  pi += 4.0 / (1.0 + x * x) * dx;
  }
  printf("%17.15f\n",pi);
}
