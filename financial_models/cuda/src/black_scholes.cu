#include "black_scholes_cuda.h"
#include <cuda_runtime.h>
#include <math.h>

__device__ double d_normalCDF(double x) {
    return 0.5 * erfc(-x / sqrt(2.0));
}

__global__ void blackScholesKernel(double* d_result, double S, double K, double T, double r, double sigma, bool is_call) {
    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    if (is_call) {
        *d_result = S * d_normalCDF(d1) - K * exp(-r * T) * d_normalCDF(d2);
    } else {
        *d_result = K * exp(-r * T) * d_normalCDF(-d2) - S * d_normalCDF(-d1);
    }
}

double blackScholesCUDA(double S, double K, double T, double r, double sigma, bool is_call) {
    double *d_result, h_result;
    cudaMalloc((void**)&d_result, sizeof(double));
    blackScholesKernel<<<1, 1>>>(d_result, S, K, T, r, sigma, is_call);
    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    return h_result;
}
