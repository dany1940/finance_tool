#include "black_scholes.h"
#include "utils.h"
#include <cmath>

double blackScholes(double S, double K, double T, double r, double sigma, bool isCall) {
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);

    if (isCall) {
        return S * normalCDF(d1) - K * std::exp(-r * T) * normalCDF(d2);
    } else {
        return K * std::exp(-r * T) * normalCDF(-d2) - S * normalCDF(-d1);
    }
}
