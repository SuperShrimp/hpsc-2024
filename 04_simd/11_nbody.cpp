#include <cstdio>
#include <cstdlib>
#include <cmath>
#include<x86intrin.h>
#include <avx512vlintrin.h>
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

for (int i = 0; i < N; i++) {
        __m256 fx_simd = _mm256_setzero_ps();
        __m256 fy_simd = _mm256_setzero_ps();
        
        for (int j = 0; j < N; j ++) {
            // __m256 x_i_values = _mm256_broadcast_ss(&x[i]);
            // __m256 y_i_values = _mm256_broadcast_ss(&y[i]);
             __m256 x_i_values = _mm256_set1_ps(x[i]);
            __m256 y_i_values = _mm256_set1_ps(y[i]);
            __m256 x_j_values = _mm256_load_ps(x);
            __m256 y_j_values = _mm256_load_ps(x);
            // Load m values for j to j+7
            __m256 m_values = _mm256_load_ps(m);
            
            // Compute rx and ry
            __m256 rx = _mm256_sub_ps(x_i_values, x_j_values);
            __m256 ry = _mm256_sub_ps(y_i_values, y_j_values);
            
            // Compute r^2
            __m256 r_squared = _mm256_add_ps(_mm256_mul_ps(rx, rx), _mm256_mul_ps(ry, ry));
            
            // Compute r^3
            __m256 r_cubed = _mm256_mul_ps(r_squared, _mm256_sqrt_ps(r_squared));
            
            // Compute m[j] / (r^3)
            __m256 m_over_r_cubed = _mm256_div_ps(m_values, r_cubed);
            
            // Accumulate fx and fy
            fx_simd = _mm256_sub_ps(fx_simd, _mm256_mul_ps(rx, m_over_r_cubed));
            fy_simd = _mm256_sub_ps(fy_simd, _mm256_mul_ps(ry, m_over_r_cubed));
        }
        
        // Reduce fx_simd and fy_simd horizontally
        float fx_temp[8], fy_temp[8];
        _mm256_storeu_ps(fx_temp, fx_simd);
        _mm256_storeu_ps(fy_temp, fy_simd);
        float fx_sum = 0, fy_sum = 0;
        for (int k = 0; k < 8; ++k) {
            fx_sum += fx_temp[k];
            fy_sum += fy_temp[k];
        }
        
        // Update fx[i] and fy[i]
        fx[i] = fx_sum;
        fy[i] = fy_sum;

        // Print results
        printf("%d %g %g\n", i, fx[i], fy[i]);
    }
}
