#ifndef FINITE_DIFFERENCE_CUDA_H
#define FINITE_DIFFERENCE_CUDA_H

#ifdef __CUDACC__
extern "C" {
#endif

double finiteDifferenceBlackScholesCUDA(double S, double K, double T, double r, double sigma, bool is_call, int grid_size, int time_steps);

#ifdef __CUDACC__
}
#endif

#endif // FINITE_DIFFERENCE_CUDA_H
