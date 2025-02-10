#include "finite_difference.h"
#include <vector>
#include <cmath>
#include <algorithm>

double finiteDifferenceBlackScholes(double S, double K, double T, double r, double sigma, bool isCall, int N, int M) {
    double dt = T / M;
    double dS = 2 * S / N;
    std::vector<std::vector<double>> grid(N + 1, std::vector<double>(M + 1, 0.0));

    for (int i = 0; i <= N; i++) {
        double stockPrice = i * dS;
        grid[i][M] = isCall ? std::max(0.0, stockPrice - K) : std::max(0.0, K - stockPrice);
    }

    for (int j = M - 1; j >= 0; j--) {
        for (int i = 1; i < N; i++) {
            double stockPrice = i * dS;
            double delta = (grid[i + 1][j + 1] - grid[i - 1][j + 1]) / (2 * dS);
            double gamma = (grid[i + 1][j + 1] - 2 * grid[i][j + 1] + grid[i - 1][j + 1]) / (dS * dS);
            double theta = -0.5 * sigma * sigma * stockPrice * stockPrice * gamma - r * stockPrice * delta + r * grid[i][j + 1];
            grid[i][j] = grid[i][j + 1] - dt * theta;
        }
    }
    return grid[N / 2][0];
}
