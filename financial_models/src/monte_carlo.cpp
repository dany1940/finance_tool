#include "monte_carlo.h"
#include <random>

double monteCarloBlackScholes(double S, double K, double T, double r, double sigma, bool is_call, int numSimulations) {
    logMessage("Monte Carlo simulation started.");

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> dist(0.0, 1.0);

    double sumPayoff = 0.0;
    double discountFactor = exp(-r * T);

    #pragma omp parallel for reduction(+:sumPayoff)
    for (int i = 0; i < numSimulations; i++) {
        double epsilon = dist(gen);
        double ST = S * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * epsilon);
        double payoff = is_call ? max(ST - K, 0.0) : max(K - ST, 0.0);
        sumPayoff += payoff;
    }

    double price = discountFactor * (sumPayoff / numSimulations);
    logMessage("Monte Carlo simulation completed.");
    return price;
}

