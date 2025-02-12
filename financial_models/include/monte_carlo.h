#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

#include "utils.h"
#include <omp.h>
#include <vector>

using namespace std;

double monteCarloBlackScholes(double S, double K, double T, double r, double sigma, bool is_call, int numSimulations);

#endif // MONTE_CARLO_H
