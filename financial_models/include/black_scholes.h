#ifndef BLACK_SCHOLES_H
#define BLACK_SCHOLES_H

#include "utils.h"
#include <omp.h>

using namespace std;

double blackScholes(double S, double K, double T, double r, double sigma, bool is_call);

#endif // BLACK_SCHOLES_H
