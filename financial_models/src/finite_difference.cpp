#include "finite_difference.h"
#include <vector>
#include <cmath>
#include <algorithm>

double finiteDifferenceBlackScholes(double S, double K, double T, double r, double sigma, bool is_call, int grid_size, int time_steps) {
    if (grid_size <= 0 || time_steps <= 0) {
        return 0.0; // Prevent division errors
    }

    double dt = T / time_steps;
    double dx = sigma * sqrt(3 * dt); // Ensures stability
    double discount = exp(-r * dt);

    std::vector<double> prices(grid_size);
    std::vector<double> option_values(grid_size);

    // Initialize stock prices at grid points
    for (int i = 0; i < grid_size; i++) {
        prices[i] = S * exp((i - grid_size / 2) * dx);
        option_values[i] = is_call ? std::max(prices[i] - K, 0.0) : std::max(K - prices[i], 0.0);
    }

    // Backward time-stepping
    for (int t = time_steps - 1; t >= 0; t--) {
        for (int i = 1; i < grid_size - 1; i++) {
            double delta = (option_values[i + 1] - option_values[i - 1]) / (2.0 * dx);
            double gamma = (option_values[i + 1] - 2.0 * option_values[i] + option_values[i - 1]) / (dx * dx);
            double theta = -0.5 * sigma * sigma * prices[i] * prices[i] * gamma - r * prices[i] * delta + r * option_values[i];

            option_values[i] = option_values[i] - dt * theta;
        }

        // Apply boundary conditions
        option_values[0] = is_call ? 0.0 : K * exp(-r * (time_steps - t) * dt);
        option_values[grid_size - 1] = is_call ? prices[grid_size - 1] - K : 0.0;
    }

    double result = option_values[grid_size / 2];

    if (std::isnan(result) || result < 0.0) {
        return 0.0; // Prevent large negative results
    }

    return result;
}
