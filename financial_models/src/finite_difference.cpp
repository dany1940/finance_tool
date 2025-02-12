#include "finite_difference.h"
#include "utils.h"
#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

double finiteDifferenceBlackScholes(double S, double K, double T, double r, double sigma, bool isCall, int gridSize, int timeSteps) {
    if (gridSize <= 0 || timeSteps <= 0 || T <= 0 || sigma <= 0 || S <= 0 || K <= 0) {
        cerr << "[ERROR] Invalid input parameters in Finite Difference.\n";
        return 0.0;
    }

    logMessage("Finite Difference Method started.");

    double dt = T / timeSteps;
    double dx = sigma * sqrt(3 * dt);
    double pu = 0.5 * ((sigma * sigma * dt) / (dx * dx) + (r * dt) / dx);
    double pm = 1.0 - (sigma * sigma * dt) / (dx * dx) - r * dt;
    double pd = 0.5 * ((sigma * sigma * dt) / (dx * dx) - (r * dt) / dx);

    vector<double> values(gridSize + 1);
    for (int i = 0; i <= gridSize; ++i) {
        double stockPrice = S * exp((i - gridSize / 2) * dx);
        values[i] = isCall ? max(stockPrice - K, 0.0) : max(K - stockPrice, 0.0);
    }

    for (int t = timeSteps - 1; t >= 0; --t) {
        vector<double> newValues(gridSize + 1);
        for (int i = 1; i < gridSize; ++i) {
            newValues[i] = pu * values[i + 1] + pm * values[i] + pd * values[i - 1];
        }
        values.swap(newValues);
    }

    logMessage("Finite Difference Method completed.");
    return max(values[gridSize / 2], 0.0);
}
