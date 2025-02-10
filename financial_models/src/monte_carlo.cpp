#include "monte_carlo.h"
#include <random>
#include <cmath>
#include <omp.h>

double monteCarloBlackScholes(double S, double K, double T, double r, double sigma, bool isCall, int numSimulations) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);

    double sumPayoff = 0.0;
    #pragma omp parallel for reduction(+:sumPayoff)
    for (int i = 0; i < numSimulations; ++i) {
        double ST = S * std::exp((r - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * d(gen));
        double payoff = isCall ? std::max(0.0, ST - K) : std::max(0.0, K - ST);
        sumPayoff += std::exp(-r * T) * payoff;
    }
    return sumPayoff / numSimulations;
}
