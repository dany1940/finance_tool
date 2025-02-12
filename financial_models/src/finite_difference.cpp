#include "finite_difference.h"

double finiteDifferenceBlackScholes(double S, double K, double T, double r, double sigma, bool is_call, int gridSize, int timeSteps) {
    logMessage("Finite Difference Method started.");

    vector<vector<double>> grid(gridSize + 1, vector<double>(timeSteps + 1, 0.0));
    double dt = T / timeSteps;
    double dx = (2.0 * sigma * sqrt(dt));

    vector<double> stockPrice(gridSize + 1);
    for (int i = 0; i <= gridSize; i++) {
        stockPrice[i] = S * exp((i - gridSize / 2) * dx);
    }

    for (int i = 0; i <= gridSize; i++) {
        grid[i][timeSteps] = is_call ? max(stockPrice[i] - K, 0.0) : max(K - stockPrice[i], 0.0);
    }

    for (int t = timeSteps - 1; t >= 0; t--) {
        #pragma omp parallel for
        for (int i = 1; i < gridSize; i++) {
            double delta = (grid[i + 1][t + 1] - grid[i - 1][t + 1]) / (2.0 * dx);
            double gamma = (grid[i + 1][t + 1] - 2.0 * grid[i][t + 1] + grid[i - 1][t + 1]) / (dx * dx);
            double theta = -0.5 * sigma * sigma * stockPrice[i] * stockPrice[i] * gamma - r * stockPrice[i] * delta + r * grid[i][t + 1];
            grid[i][t] = grid[i][t + 1] - dt * theta;
        }
    }

    logMessage("Finite Difference Method completed.");
    return grid[gridSize / 2][0];
}
