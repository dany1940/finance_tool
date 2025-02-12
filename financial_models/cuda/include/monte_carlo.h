#ifndef MONTE_CARLO_CUDA_H
#define MONTE_CARLO_CUDA_H

#ifdef __CUDACC__
extern "C" {
#endif

double monteCarloBlackScholesCUDA(double S, double K, double T, double r, double sigma, bool is_call, int numSimulations);

#ifdef __CUDACC__
}
#endif

#endif // MONTE_CARLO_CUDA_H
