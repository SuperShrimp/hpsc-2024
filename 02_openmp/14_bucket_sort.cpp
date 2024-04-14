#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include <time.h>
#include <iostream>
int main() {
  clock_t start, end;
  start = clock();

  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");
  std::vector<int> bucket(range,0); 
  for (int i=0; i<n; i++)
    bucket[key[i]]++;
  std::vector<int> offset(range,0);
  for (int i=1; i<range; i++) 
    offset[i] = offset[i-1] + bucket[i-1];
#pragma omp paralell 
  for (int i=0; i<range; i++) {
    int j = offset[i];
    int q = bucket[i];
    #pragma omp for
    for (int m=q; m>0; m--) {
      key[j++] = i;
    }
  }
  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  end = clock();
  std::cout<<"time = "<<double(end-start)/CLOCKS_PER_SEC<<"s"<<std::endl;
}
