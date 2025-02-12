#include "black_scholes.h"

double blackScholes(double S, double K, double T, double r, double sigma, bool is_call) {
    logMessage("Black-Scholes calculation started.");

    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);

    double Nd1 = normalCDF(d1);
    double Nd2 = normalCDF(d2);

    double result;
    if (is_call) {
        result = S * Nd1 - K * exp(-r * T) * Nd2;
    } else {
        result = K * exp(-r * T) * (1 - Nd2) - S * (1 - Nd1);
    }

    logMessage("Black-Scholes calculation completed.");
    return result;
}
