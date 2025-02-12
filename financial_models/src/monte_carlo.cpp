#include "monte_carlo.h"
#include "utils.h"
#include <cmath>
#include <random>
#include <iostream>
#include <omp.h>

using namespace std;

double monteCarloBlackScholes(double S, double K, double T, double r, double sigma, int numSimulations, bool isCall) {
    if (numSimulations <= 0 || T <= 0 || sigma <= 0 || S <= 0 || K <= 0) {
        cerr << "[ERROR] Invalid input parameters in Monte Carlo.\n";
        return 0.0;
    }

    logMessage("Monte Carlo simulation started.");

    random_device rd;
    mt19937 generator(rd());
    normal_distribution<double> distribution(0.0, 1.0);

    double sumPayoff = 0.0;

    #pragma omp parallel for reduction(+:sumPayoff)
    for (int i = 0; i < numSimulations; i++) {
        double Z = distribution(generator);
        double ST = S * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z);
        double payoff = isCall ? max(ST - K, 0.0) : max(K - ST, 0.0);
        sumPayoff += payoff;
    }

    double price = (sumPayoff / numSimulations) * exp(-r * T);

    logMessage("Monte Carlo simulation completed.");
    return max(price, 0.0);
}
