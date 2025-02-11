#include "monte_carlo.h"
#include <random>
#include <cmath>

double monteCarloBlackScholes(double S, double K, double T, double r, double sigma, bool is_call, int numSimulations) {
    if (numSimulations <= 0) {
        return 0.0; // Ensure valid input
    }

    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<double> distribution(0.0, 1.0);

    double totalPayoff = 0.0;
    for (int i = 0; i < numSimulations; i++) {
        double Z = distribution(generator);
        double ST = S * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z);

        double payoff = is_call ? std::max(ST - K, 0.0) : std::max(K - ST, 0.0);
        totalPayoff += payoff;
    }

    double result = exp(-r * T) * (totalPayoff / numSimulations);

    if (std::isnan(result) || result < 0.0) {
        return 0.0; // Handle NaN cases safely
    }

    return result;
}
