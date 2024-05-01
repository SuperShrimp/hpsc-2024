#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  float a[N], b[N];
  for(int i=0; i<N; i++) {
    __m256 fx_vec = _mm256_setzero_ps();
    __m256 fy_vec = _mm256_setzero_ps();
      float xi = x[i];
      float yi = y[i];
      float mi = m[i];
      float xj[N], yj[N];
    for(int j=0; j<N; j++) {

      __m512 y_j_vec = _mm512_load_ps(y);
      __m512 x_j_vec = _mm512_load_ps(x);
       __m256 alli = _mm256_set1_ps(i);
       __m256 allj = _mm256_set1_ps(j);
      __mmask8 mask = _mm256_cmp_ps_mask(alli, allj, _MM_CMPINT_NE);
        __m256 xi_vec = _mm256_set1_ps(xi);
        __m256 yi_vec = _mm256_set1_ps(yi);
        __m256 zero_vec = _mm256_setzero_ps();
        __m256 xj_vec = _mm256_mask_blend_ps(mask, zero_vec, _mm256_load_ps(x));
        __m256 yj_vec = _mm256_mask_blend_ps(mask, zero_vec, _mm256_load_ps(y));
        __m256 mj_vec = _mm256_mask_blend_ps(mask, zero_vec, _mm256_load_ps(m));
        __m256 rx = _mm256_sub_ps(xi_vec, xj_vec);
        __m256 ry = _mm256_sub_ps(yi_vec, yj_vec);
        __m256 rxy = _mm256_add_ps(_mm256_mul_ps(rx, rx), _mm256_mul_ps(ry, ry));
        __m256 r1 = _mm256_rsqrt14_ps(rxy);
        __m256 r3 = _mm256_mul_ps(r1, _mm256_mul_ps(r1, r1));
        
        __m256 fxi_vec = _mm256_div_ps(_mm256_mul_ps(rx, mj_vec), r3);
        __m256 fyi_vec = _mm256_div_ps(_mm256_mul_ps(ry, mj_vec), r3);

        fx_vec = _mm256_sub_ps(fx_vec, fxi_vec);
        fy_vec = _mm256_sub_ps(fy_vec, fyi_vec);
        }
      float fx_temp[N], fy_temp[N];
      _mm256_store_ps(fx_temp, fx_vec);
      _mm256_store_ps(fy_temp, fx_vec);
      for(int m=0; m<N; m++)
      {
        fx[i] += fx_temp[m];
        fy[i] += fy_temp[m];
      }
      
    printf("%d %g %g\n",i,fx[i],fy[i]);

  }
}
