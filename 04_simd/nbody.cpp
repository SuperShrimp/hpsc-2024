#include <iostream>
#include <cmath>
#include <immintrin.h>

const int N = 100; // 假设 N 为100

void compute_forces_avx512(float* fx, float* fy, const float* x, const float* y, const float* m) {
    // 计算力
    for (int i = 0; i < N; ++i) {
        // 初始化力为0
        __m512 fx_vec = _mm512_setzero_ps();
        __m512 fy_vec = _mm512_setzero_ps();
        float xi = x[i];
        float yi = y[i];
        float mi = m[i];

        // 计算距离并累加力
        for (int j = 0; j < N; j += 16) {
            // 设置掩码
            __mmask16 mask = _mm512_cmp_epi32_mask(_mm512_set1_epi32(i), _mm512_set_epi32(j+15, j+14, j+13, j+12, j+11, j+10, j+9, j+8, j+7, j+6, j+5, j+4, j+3, j+2, j+1, j), _MM_CMPINT_NE);

            // 加载 x, y, m 向量
            __m512 xi_vec = _mm512_set1_ps(xi);
            __m512 yi_vec = _mm512_set1_ps(yi);
            __m512 xj_vec = _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, &x[j]);
            __m512 yj_vec = _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, &y[j]);
            __m512 mj_vec = _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, &m[j]);

            // 计算 rx, ry
            __m512 rx_vec = _mm512_sub_ps(xi_vec, xj_vec);
            __m512 ry_vec = _mm512_sub_ps(yi_vec, yj_vec);

            // 计算 r^2, r^3
            __m512 r_sq_vec = _mm512_add_ps(_mm512_mul_ps(rx_vec, rx_vec), _mm512_mul_ps(ry_vec, ry_vec));
            __m512 r_cu_vec = _mm512_mul_ps(r_sq_vec, _mm512_sqrt_ps(r_sq_vec));

            // 计算力并累加到 fx, fy 向量
            __m512 inv_r_cube = _mm512_rcp28_ps(r_cu_vec);
            __m512 fx_contribution = _mm512_mul_ps(_mm512_div_ps(rx_vec, r_cu_vec), mj_vec);
            __m512 fy_contribution = _mm512_mul_ps(_mm512_div_ps(ry_vec, r_cu_vec), mj_vec);

            fx_vec = _mm512_sub_ps(fx_vec, fx_contribution);
            fy_vec = _mm512_sub_ps(fy_vec, fy_contribution);
        }

        // 存储最终结果
        alignas(64) float fx_tmp[16];
        alignas(64) float fy_tmp[16];
        _mm512_store_ps(fx_tmp, fx_vec);
        _mm512_store_ps(fy_tmp, fy_vec);
        for (int k = 0; k < 16; ++k) {
            fx[i] += fx_tmp[k];
            fy[i] += fy_tmp[k];
        }

        printf("%d %g %g\n", i, fx[i], fy[i]);
    }
}

int main() {
    float fx[N], fy[N];
    float x[N], y[N], m[N];

    // 初始化 x, y, m
    for (int i = 0; i < N; ++i) {
        x[i] = i;
        y[i] = i;
        m[i] = i;
    }

    // 计算力并输出
    compute_forces_avx512(fx, fy, x, y, m);

    return 0;
}
