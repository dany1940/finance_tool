#include "utils.h"
#include <cmath>

// Cumulative Normal Distribution Function
double normalCDF(double x) {
    return 0.5 * std::erfc(-x * std::sqrt(0.5));
}

// Probability Density Function for normal distribution
double normalPDF(double x) {
    return (1.0 / std::sqrt(2 * M_PI)) * std::exp(-0.5 * x * x);
}
