/*
 * KVTC CUDA Kernel Test Harness
 * 
 * Tests individual kernels for correctness:
 * 1. PCA transform roundtrip (transform -> inverse = identity)
 * 2. Quantize/dequantize roundtrip
 * 3. RoPE inverse/forward roundtrip
 * 4. Bit allocation sanity check
 * 
 * Build: nvcc -o test_kvtc test_kvtc_kernels.cu kvtc_kernels.cu -lcudart -lm
 * Run:   ./test_kvtc
 */

#include "kvtc.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define DIM 128
#define NUM_ROWS 64
#define EPSILON 1e-3f

/* ─── CUDA error checking ────────────────────────────────────────── */

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "  CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return 0; /* test fails */ \
    } \
} while(0)

#define CUDA_CHECK_SYNC() do { \
    CUDA_CHECK(cudaGetLastError()); \
    CUDA_CHECK(cudaDeviceSynchronize()); \
} while(0)

/* ─── Helpers ────────────────────────────────────────────────────── */

float rand_float() {
    return ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

float max_abs_error(const float *a, const float *b, int n) {
    float maxe = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(a[i] - b[i]);
        if (e > maxe) maxe = e;
    }
    return maxe;
}

float cosine_similarity(const float *a, const float *b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    return (float)(dot / (sqrt(na) * sqrt(nb) + 1e-10));
}

/* ─── Test: PCA roundtrip ────────────────────────────────────────── */

int test_pca_roundtrip() {
    printf("  Test PCA roundtrip... ");
    
    int size = NUM_ROWS * DIM;
    float *h_data = (float*)malloc(size * sizeof(float));
    float *h_eigvec = (float*)malloc(DIM * DIM * sizeof(float));
    float *h_mean = (float*)malloc(DIM * sizeof(float));
    float *h_result = (float*)malloc(size * sizeof(float));
    
    srand(42);
    for (int i = 0; i < size; i++) h_data[i] = rand_float();
    for (int i = 0; i < DIM; i++) h_mean[i] = rand_float() * 0.1f;
    
    /* Create orthogonal eigenvectors (identity for simplicity) */
    memset(h_eigvec, 0, DIM * DIM * sizeof(float));
    for (int i = 0; i < DIM; i++) h_eigvec[i * DIM + i] = 1.0f;
    
    /* GPU buffers */
    float *d_data, *d_eigvec, *d_mean, *d_pca, *d_restored;
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_eigvec, DIM * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pca, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_restored, size * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_eigvec, h_eigvec, DIM * DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mean, h_mean, DIM * sizeof(float), cudaMemcpyHostToDevice));
    
    /* Transform then inverse */
    kvtc_pca_transform(d_data, d_eigvec, d_mean, d_pca, NUM_ROWS, DIM, NULL);
    kvtc_pca_inverse(d_pca, d_eigvec, d_mean, d_restored, NUM_ROWS, DIM, NULL);
    CUDA_CHECK_SYNC();
    
    CUDA_CHECK(cudaMemcpy(h_result, d_restored, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    float maxe = max_abs_error(h_data, h_result, size);
    float cos = cosine_similarity(h_data, h_result, size);
    
    int pass = (maxe < EPSILON && cos > 0.9999f);
    printf("%s (maxerr=%.6f, cosine=%.6f)\n", pass ? "PASS" : "FAIL", maxe, cos);
    
    free(h_data); free(h_eigvec); free(h_mean); free(h_result);
    cudaFree(d_data); cudaFree(d_eigvec); cudaFree(d_mean);
    cudaFree(d_pca); cudaFree(d_restored);
    
    return pass;
}

/* ─── Test: RoPE roundtrip ───────────────────────────────────────── */

int test_rope_roundtrip() {
    printf("  Test RoPE roundtrip... ");
    
    int size = NUM_ROWS * DIM;
    float *h_data = (float*)malloc(size * sizeof(float));
    int *h_positions = (int*)malloc(NUM_ROWS * sizeof(int));
    float *h_result = (float*)malloc(size * sizeof(float));
    
    srand(123);
    for (int i = 0; i < size; i++) h_data[i] = rand_float();
    for (int i = 0; i < NUM_ROWS; i++) h_positions[i] = i;
    
    float *d_data, *d_buf, *d_result;
    int *d_pos;
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_buf, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos, NUM_ROWS * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos, h_positions, NUM_ROWS * sizeof(int), cudaMemcpyHostToDevice));
    
    /* Forward then inverse */
    kvtc_rope_forward(d_data, d_pos, d_buf, NUM_ROWS, DIM, 10000.0f, NULL);
    kvtc_rope_inverse(d_buf, d_pos, d_result, NUM_ROWS, DIM, 10000.0f, NULL);
    CUDA_CHECK_SYNC();
    
    CUDA_CHECK(cudaMemcpy(h_result, d_result, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    float maxe = max_abs_error(h_data, h_result, size);
    float cos = cosine_similarity(h_data, h_result, size);
    
    int pass = (maxe < EPSILON && cos > 0.9999f);
    printf("%s (maxerr=%.6f, cosine=%.6f)\n", pass ? "PASS" : "FAIL", maxe, cos);
    
    free(h_data); free(h_positions); free(h_result);
    cudaFree(d_data); cudaFree(d_buf); cudaFree(d_result); cudaFree(d_pos);
    
    return pass;
}

/* ─── Test: Quantize/dequantize roundtrip ────────────────────────── */

int test_quantize_roundtrip() {
    printf("  Test quantize roundtrip (8-bit)... ");
    
    int size = NUM_ROWS * DIM;
    float *h_data = (float*)malloc(size * sizeof(float));
    int8_t *h_bw = (int8_t*)malloc(DIM);
    float *h_result = (float*)malloc(size * sizeof(float));
    
    srand(456);
    for (int i = 0; i < size; i++) h_data[i] = rand_float();
    for (int i = 0; i < DIM; i++) h_bw[i] = 8;  /* 8-bit quantization */
    
    float *d_data, *d_dequant;
    int8_t *d_bw;
    float *d_scales, *d_zp;
    int32_t *d_indices;
    
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dequant, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bw, DIM));
    CUDA_CHECK(cudaMalloc(&d_scales, DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_zp, DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_indices, size * sizeof(int32_t)));
    
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bw, h_bw, DIM, cudaMemcpyHostToDevice));
    
    kvtc_quantize(d_data, d_bw, d_scales, d_zp, d_indices, NUM_ROWS, DIM, NULL);
    kvtc_dequantize(d_indices, d_bw, d_scales, d_zp, d_dequant, NUM_ROWS, DIM, NULL);
    CUDA_CHECK_SYNC();
    
    CUDA_CHECK(cudaMemcpy(h_result, d_dequant, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    float maxe = max_abs_error(h_data, h_result, size);
    float cos = cosine_similarity(h_data, h_result, size);
    
    /* 8-bit quant should have max error < 0.01 for data in [-1, 1] */
    int pass = (maxe < 0.01f && cos > 0.999f);
    printf("%s (maxerr=%.6f, cosine=%.6f)\n", pass ? "PASS" : "FAIL", maxe, cos);
    
    free(h_data); free(h_bw); free(h_result);
    cudaFree(d_data); cudaFree(d_dequant); cudaFree(d_bw);
    cudaFree(d_scales); cudaFree(d_zp); cudaFree(d_indices);
    
    return pass;
}

/* ─── Test: Bit allocation ───────────────────────────────────────── */

int test_bit_allocation() {
    printf("  Test bit allocation... ");
    
    float *h_ev = (float*)malloc(DIM * sizeof(float));
    int8_t *h_bw = (int8_t*)malloc(DIM);
    
    /* Eigenvalues: decreasing (first components most important) */
    for (int i = 0; i < DIM; i++) {
        h_ev[i] = 100.0f / (1.0f + i);
    }
    
    float *d_ev;
    int8_t *d_bw;
    CUDA_CHECK(cudaMalloc(&d_ev, DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bw, DIM));
    CUDA_CHECK(cudaMemcpy(d_ev, h_ev, DIM * sizeof(float), cudaMemcpyHostToDevice));
    
    int budget = DIM * 4;  /* 4 bits average */
    kvtc_bit_allocation(d_ev, budget, d_bw, DIM, 16, NULL);
    CUDA_CHECK_SYNC();
    
    CUDA_CHECK(cudaMemcpy(h_bw, d_bw, DIM, cudaMemcpyDeviceToHost));
    
    /* Check: total bits should equal budget */
    int total = 0;
    for (int i = 0; i < DIM; i++) total += h_bw[i];
    
    /* First components should have more bits than last */
    int pass = (total == budget && h_bw[0] >= h_bw[DIM-1]);
    printf("%s (total=%d/%d, first=%d, last=%d)\n",
           pass ? "PASS" : "FAIL", total, budget, h_bw[0], h_bw[DIM-1]);
    
    free(h_ev); free(h_bw);
    cudaFree(d_ev); cudaFree(d_bw);
    
    return pass;
}

/* ─── Main ───────────────────────────────────────────────────────── */

int main() {
    printf("\n  KVTC CUDA Kernel Tests\n");
    printf("  =====================\n\n");
    
    /* Check for CUDA-capable GPU before running any tests */
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        printf("  No CUDA-capable GPU detected (cudaGetDeviceCount: %s)\n",
               cudaGetErrorString(err));
        printf("  Skipping GPU kernel tests — a CUDA GPU is required.\n\n");
        return 0;
    }
    
    int device;
    cudaGetDevice(&device);
    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    printf("  GPU: %s (SM %d.%d, %d MB VRAM)\n\n",
           props.name, props.major, props.minor,
           (int)(props.totalGlobalMem / (1024 * 1024)));
    
    int passed = 0, total = 0;
    
    total++; passed += test_pca_roundtrip();
    total++; passed += test_rope_roundtrip();
    total++; passed += test_quantize_roundtrip();
    total++; passed += test_bit_allocation();
    
    printf("\n  Results: %d/%d passed\n\n", passed, total);
    
    return (passed == total) ? 0 : 1;
}
