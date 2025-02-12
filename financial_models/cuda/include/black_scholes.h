#ifndef BLACK_SCHOLES_CUDA_H
#define BLACK_SCHOLES_CUDA_H

#ifdef __CUDACC__
extern "C" {
#endif

double blackScholesCUDA(double S, double K, double T, double r, double sigma, bool is_call);

#ifdef __CUDACC__
}
#endif

#endif // BLACK_SCHOLES_CUDA_H
