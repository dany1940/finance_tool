#include "black_scholes.h"
#include "utils.h"
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace std;



double blackScholes(double S, double K, double T, double r, double sigma, bool isCall) {
    if (T <= 0 || sigma <= 0 || S <= 0 || K <= 0) {
        cerr << "[ERROR] Invalid input parameters in Black-Scholes.\n";
        return 0.0;
    }

    logMessage("Black-Scholes calculation started.");

    double sqrtT = sqrt(T);
    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    double d2 = d1 - sigma * sqrtT;

    double Nd1 = normalCDF(d1);
    double Nd2 = normalCDF(d2);

    double price = isCall ? (S * Nd1 - K * exp(-r * T) * Nd2)
                          : (K * exp(-r * T) * normalCDF(-d2) - S * normalCDF(-d1));

    logMessage("Black-Scholes calculation completed.");
    return max(price, 0.0);
}
