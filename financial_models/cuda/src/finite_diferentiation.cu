#include "finite_difference_cuda.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ void finiteDifferenceKernel(double* d_prices, int grid_size, int time_steps, double S, double K, double T, double r, double sigma, bool is_call) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= grid_size) return;

    double dt = T / time_steps;
    double dx = sigma * sqrt(3 * dt);
    double discount = exp(-r * dt);

    for (int t = time_steps - 1; t >= 0; t--) {
        d_prices[i] = discount * ((1 - dx) * d_prices[i] + dx * d_prices[min(i + 1, grid_size - 1)]);
    }
}

double finiteDifferenceBlackScholesCUDA(double S, double K, double T, double r, double sigma, bool is_call, int grid_size, int time_steps) {
    double *d_prices, *h_prices;
    h_prices = (double*)malloc(grid_size * sizeof(double));
    cudaMalloc((void**)&d_prices, grid_size * sizeof(double));

    finiteDifferenceKernel<<<(grid_size + 255) / 256, 256>>>(d_prices, grid_size, time_steps, S, K, T, r, sigma, is_call);
    cudaMemcpy(h_prices, d_prices, grid_size * sizeof(double), cudaMemcpyDeviceToHost);

    double result = h_prices[grid_size / 2];

    free(h_prices);
    cudaFree(d_prices);

    return result;
