#include "monte_carlo_cuda.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

__global__ void monteCarloKernel(double* d_results, double S, double K, double T, double r, double sigma, bool is_call, int numSimulations) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    curand_init(1234, idx, 0, &state);

    if (idx < numSimulations) {
        double ST = S * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * curand_normal(&state));
        double payoff = is_call ? fmax(ST - K, 0.0) : fmax(K - ST, 0.0);
        d_results[idx] = exp(-r * T) * payoff;
    }
}

double monteCarloBlackScholesCUDA(double S, double K, double T, double r, double sigma, bool is_call, int numSimulations) {
    double *d_results, *h_results;
    h_results = (double*)malloc(numSimulations * sizeof(double));
    cudaMalloc((void**)&d_results, numSimulations * sizeof(double));

    monteCarloKernel<<<(numSimulations + 255) / 256, 256>>>(d_results, S, K, T, r, sigma, is_call, numSimulations);
    cudaMemcpy(h_results, d_results, numSimulations * sizeof(double), cudaMemcpyDeviceToHost);

    double sum = 0.0;
    for (int i = 0; i < numSimulations; i++) sum += h_results[i];

    free(h_results);
    cudaFree(d_results);

    return sum / numSimulations;
}
