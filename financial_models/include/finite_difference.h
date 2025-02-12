#ifndef FINITE_DIFFERENCE_H
#define FINITE_DIFFERENCE_H

#include "utils.h"
#include <vector>
#include <omp.h>

using namespace std;

double finiteDifferenceBlackScholes(double S, double K, double T, double r, double sigma, bool is_call, int gridSize, int timeSteps);

#endif // FINITE_DIFFERENCE_H
