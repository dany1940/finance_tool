#include "finite_difference.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <numeric>

// ======= Helper: Boundary Condition for option =======
double boundaryCondition(double S, double K, double T, double t, double r, bool isCall) {
    double tau = T - t;
    if (isCall)
        return std::max(S - K * std::exp(-r * tau), 0.0);
    else
        return std::max(K * std::exp(-r * tau) - S, 0.0);
}

// ======= Cubic Spline Interpolation Helper =======
double cubic_spline_interpolate(const std::vector<double>& x, const std::vector<double>& y, double x0) {
    int n = (int)x.size();
    std::vector<double> a(y), b(n), d(n), h(n), alpha(n), c(n), l(n), mu(n), z(n);
    for (int i = 0; i < n - 1; ++i) h[i] = x[i + 1] - x[i];
    for (int i = 1; i < n - 1; ++i)
        alpha[i] = (3.0 / h[i]) * (a[i + 1] - a[i]) - (3.0 / h[i - 1]) * (a[i] - a[i - 1]);
    l[0] = 1; mu[0] = z[0] = 0;
    for (int i = 1; i < n - 1; ++i) {
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }
    l[n - 1] = 1; z[n - 1] = c[n - 1] = 0;
    for (int j = n - 2; j >= 0; --j) {
        c[j] = z[j] - mu[j] * c[j + 1];
        b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3;
        d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
    }
    int i = 0;
    while (i < n - 2 && x0 > x[i + 1]) ++i;
    double dx = x0 - x[i];
    return a[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx;
}

// ======= Interpolation Result =======
double interpolate_result(const std::vector<double>& V, const std::vector<double>& S, double S0) {
    if (S0 <= S.front()) return V.front();
    if (S0 >= S.back()) return V.back();
    return cubic_spline_interpolate(S, V, S0);
}

// ======= Tridiagonal Solver =======
std::vector<double> solve_tridiagonal(const std::vector<double>& a, const std::vector<double>& b,
                                     const std::vector<double>& c, const std::vector<double>& d) {
    int n = (int)b.size();
    std::vector<double> x(n), cp(n), dp(n);
    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];
    for (int i = 1; i < n; ++i) {
        double denom = b[i] - a[i] * cp[i - 1];
        cp[i] = c[i] / denom;
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom;
    }
    x[n - 1] = dp[n - 1];
    for (int i = n - 2; i >= 0; --i)
        x[i] = dp[i] - cp[i] * x[i + 1];
    return x;
}

// ======= Explicit FDM (vector output) =======
std::vector<double> fdm_explicit_vector(int N, int M, double Smax, double T, double K,
                                       double r, double sigma, bool isCall) {
    double dS = Smax / N;
    double dt = T / M;

    // Stability condition (CFL condition) for explicit scheme:
    double max_dt = 0.5 * dS * dS / (sigma * sigma * Smax * Smax);

    // Clamp dt and adjust M if unstable
    if (dt > max_dt) {
        dt = max_dt;
        M = static_cast<int>(T / dt);
        std::cerr << "[Warning] Explicit FDM unstable dt clamped from original to " << dt
                  << ", M adjusted to " << M << std::endl;
    }

    std::vector<double> V(N + 1), Vnew(N + 1);

    // Initialize payoff at maturity
    for (int i = 0; i <= N; ++i)
        V[i] = isCall ? std::max(i * dS - K, 0.0) : std::max(K - i * dS, 0.0);

    // Time-stepping loop using clamped dt and adjusted M
    for (int t = 0; t < M; ++t) {
        double t_curr = (t + 1) * dt;

        // Boundary conditions at S=0 and Smax
        Vnew[0] = isCall ? 0.0 : K * std::exp(-r * (T - t_curr));
        Vnew[N] = isCall ? (Smax - K * std::exp(-r * (T - t_curr))) : 0.0;

        for (int i = 1; i < N; ++i) {
            double j = static_cast<double>(i);
            double a = 0.5 * dt * (sigma * sigma * j * j - r * j);
            double b = 1 - dt * (sigma * sigma * j * j + r);
            double c = 0.5 * dt * (sigma * sigma * j * j + r * j);

            Vnew[i] = a * V[i - 1] + b * V[i] + c * V[i + 1];

            // Clamp negative prices to zero
            if (Vnew[i] < 0.0) Vnew[i] = 0.0;
        }
        V = Vnew;
    }

    return V;
}
// ======= Explicit FDM (interpolated price) =======
double fdm_explicit(int N, int M, double Smax, double T, double K,
                    double r, double sigma, bool isCall, double S0) {
    std::vector<double> V = fdm_explicit_vector(N, M, Smax, T, K, r, sigma, isCall);
    double dS = Smax / N;
    std::vector<double> S(N + 1);
    for (int i = 0; i <= N; ++i)
        S[i] = i * dS;
    return interpolate_result(V, S, S0);
}

// ======= Implicit FDM =======
double fdm_implicit(int N, int M, double Smax, double T, double K,
                    double r, double sigma, bool isCall, double S0) {
    double dS = Smax / N;
    double dt = T / M;
    std::vector<double> V(N + 1), S(N + 1);
    for (int i = 0; i <= N; ++i) {
        S[i] = i * dS;
        V[i] = isCall ? std::max(S[i] - K, 0.0) : std::max(K - S[i], 0.0);
    }
    std::vector<double> a(N - 1), b(N - 1), c(N - 1), d(N - 1);
    for (int t = 0; t < M; ++t) {
        double t_curr = (t + 1) * dt;
        for (int i = 1; i < N; ++i) {
            double j = static_cast<double>(i);
            a[i - 1] = -0.5 * dt * (sigma*sigma*j*j - r*j);
            b[i - 1] = 1 + dt * (sigma*sigma*j*j + r);
            c[i - 1] = -0.5 * dt * (sigma*sigma*j*j + r*j);
            d[i - 1] = V[i];
        }
        std::vector<double> Vnew = solve_tridiagonal(a, b, c, d);
        for (int i = 1; i < N; ++i) {
            V[i] = Vnew[i - 1];
            if (V[i] < 0.0) V[i] = 0.0;
        }
        V[0] = isCall ? 0.0 : K * std::exp(-r * (T - t_curr));
        V[N] = isCall ? (Smax - K * std::exp(-r * (T - t_curr))) : 0.0;
    }
    return interpolate_result(V, S, S0);
}

// ======= Crank-Nicolson FDM =======
double fdm_crank_nicolson(int N, int M, double Smax, double T, double K,
                          double r, double sigma, bool isCall,
                          bool /*rannacher_smoothing*/, double S0) {
    double dS = Smax / N;
    double dt = T / M;
    std::vector<double> V(N + 1), S(N + 1);
    for (int i = 0; i <= N; ++i) {
        S[i] = i * dS;
        V[i] = isCall ? std::max(S[i] - K, 0.0) : std::max(K - S[i], 0.0);
    }
    std::vector<double> a(N - 1), b(N - 1), c(N - 1), d(N - 1);
    for (int t = 0; t < M; ++t) {
        double t_curr = (t + 1) * dt;
        for (int i = 1; i < N; ++i) {
            double j = static_cast<double>(i);
            double alpha = 0.25 * dt * (sigma*sigma*j*j - r*j);
            double beta = -0.5 * dt * (sigma*sigma*j*j + r);
            double gamma = 0.25 * dt * (sigma*sigma*j*j + r*j);
            a[i - 1] = -alpha;
            b[i - 1] = 1 - beta;
            c[i - 1] = -gamma;
            d[i - 1] = alpha * V[i - 1] + (1 + beta) * V[i] + gamma * V[i + 1];
        }
        std::vector<double> Vnew = solve_tridiagonal(a, b, c, d);
        for (int i = 1; i < N; ++i) {
            V[i] = Vnew[i - 1];
            if (V[i] < 0.0) V[i] = 0.0;
        }
        V[0] = isCall ? 0.0 : K * std::exp(-r * (T - t_curr));
        V[N] = isCall ? (Smax - K * std::exp(-r * (T - t_curr))) : 0.0;
    }
    return interpolate_result(V, S, S0);
}

// ======= American PSOR FDM =======
double fdm_american_psor(int N, int M, double Smax, double T, double K,
                         double r, double sigma, bool isCall,
                         double omega, int maxIter, double tol, double S0) {
    double dS = Smax / N;
    double dt = T / M;

    std::vector<double> S(N + 1), V(N + 1);
    for (int i = 0; i <= N; ++i) {
        S[i] = i * dS;
        V[i] = isCall ? std::max(S[i] - K, 0.0) : std::max(K - S[i], 0.0);
    }

    std::vector<double> a(N - 1), b(N - 1), c(N - 1), rhs(N - 1);
    for (int t = M - 1; t >= 0; --t) {
        double t_curr = t * dt;
        for (int i = 1; i < N; ++i) {
            double j = static_cast<double>(i);
            a[i - 1] = -0.5 * dt * (sigma*sigma*j*j - r*j);
            b[i - 1] = 1 + dt * (sigma*sigma*j*j + r);
            c[i - 1] = -0.5 * dt * (sigma*sigma*j*j + r*j);
            rhs[i - 1] = V[i];
        }

        for (int k = 0; k < maxIter; ++k) {
            double error = 0.0;
            for (int i = 1; i < N; ++i) {
                double j = i - 1;
                double y = (j > 0 ? a[j] * V[i - 1] : 0.0)
                         + b[j] * V[i]
                         + (j < N - 2 ? c[j] * V[i + 1] : 0.0)
                         - rhs[j];
                double Vnew = V[i] - omega * y / b[j];
                double payoff = isCall ? std::max(S[i] - K, 0.0) : std::max(K - S[i], 0.0);
                Vnew = std::max(Vnew, payoff);
                error += std::abs(Vnew - V[i]);
                V[i] = Vnew;
            }
            if (error < tol)
                break;
        }
        V[0] = isCall ? 0.0 : K * std::exp(-r * (T - t_curr));
        V[N] = isCall ? (Smax - K * std::exp(-r * (T - t_curr))) : 0.0;
    }
    return interpolate_result(V, S, S0);
}



// === Exponential Integral Finite Difference Method ===
// Handles exponential discounting explicitly in the scheme.
// === Exponential Integral Finite Difference Method ===
// Handles exponential discounting explicitly in the scheme.
double fdm_exponential_integral(int N, double Smax, double T, double K,
                                double r, double sigma, bool isCall, double S0) {
    double dS = Smax / N;
    double dt = 0.01;
    int M = static_cast<int>(T / dt);

    std::vector<double> V(N + 1), V_new(N + 1), rhs(N + 1), S(N + 1);

    // Setup grid and terminal payoff
    for (int i = 0; i <= N; ++i) {
        S[i] = i * dS;
        V[i] = isCall ? std::max(S[i] - K, 0.0) : std::max(K - S[i], 0.0);
    }

    // Time stepping with exponential integral treatment
    for (int t = 0; t < M; ++t) {
        double tau = T - t * dt;

        for (int i = 1; i < N; ++i) {
            double d2V = (V[i - 1] - 2 * V[i] + V[i + 1]) / (dS * dS);
            double dV  = (V[i + 1] - V[i - 1]) / (2 * dS);

            // Exponential integration: treat drift/discounting via exp(-r dt)
            double diffusion = 0.5 * sigma * sigma * S[i] * S[i] * d2V;
            double convection = r * S[i] * dV;
            double discount = -r * V[i];

            rhs[i] = diffusion + convection + discount;
        }

        // Boundary conditions
        V_new[0] = isCall ? 0.0 : K * std::exp(-r * tau);
        V_new[N] = isCall ? (Smax - K * std::exp(-r * tau)) : 0.0;

        // Apply exponential integrator
        for (int i = 1; i < N; ++i) {
            // Incorporate exponential decay (better time integration)
            double discount_factor = std::exp(-r * dt);
            V_new[i] = discount_factor * (V[i] + dt * rhs[i]);

            // Ensure positivity
            if (V_new[i] < 0.0) V_new[i] = 0.0;
        }

        V = V_new;
    }

    // === Interpolate price at S0 ===
    if (S0 <= 0.0) return V[0];
    if (S0 >= Smax) return V[N];

    int i = static_cast<int>(S0 / dS);
    double S_left = S[i];
    double S_right = S[i + 1];
    double V_left = V[i];
    double V_right = V[i + 1];

    double interp = V_left + (V_right - V_left) * (S0 - S_left) / (S_right - S_left);
    return interp;
}


// ======= Compact 4th-order Second Derivative =======
std::vector<double> compact_4th_order_second_derivative(const std::vector<double>& V, double dx) {
    int N = (int)V.size();
    std::vector<double> d2V(N, 0.0);
    for (int i = 2; i < N - 2; ++i) {
        d2V[i] = (-V[i - 2] + 16*V[i - 1] - 30*V[i] + 16*V[i + 1] - V[i + 2]) / (12*dx*dx);
        if (d2V[i] < 0.0) d2V[i] = 0.0; // clamp negative second derivative to zero if needed
    }
    return d2V;
}


//======= Main Fractional FDM =======
double fdm_time_fractional(int N, int M, double Smax, double T, double K,
                           double r, double sigma, bool isCall, double beta, double S0) {
    if (beta <= 0.0 || beta >= 1.0) {
        throw std::invalid_argument("beta must be in (0,1)");
    }

    double dS = Smax / N;
    double dt = T / M;
    double gamma = 1.0 / tgamma(2.0 - beta);
    double dt_beta = pow(dt, -beta);

    std::vector<double> V(N + 1), V_new(N + 1), S(N + 1);
    std::vector<std::vector<double>> V_hist(M + 1, std::vector<double>(N + 1));
    std::vector<double> weights(M + 1, 0.0);

    // Initialize S and initial payoff
    for (int i = 0; i <= N; ++i) {
        S[i] = i * dS;
        V[i] = std::max(isCall ? S[i] - K : K - S[i], 0.0);
        V_hist[0][i] = V[i];
    }

    // Precompute Caputo weights
    weights[0] = 0.0;
    weights[1] = 1.0;
    for (int k = 2; k <= M; ++k) {
        weights[k] = weights[k - 1] * (1.0 - (1.0 + beta) / (double)k);
    }

    // Time loop
    for (int t = 1; t <= M; ++t) {
        for (int i = 1; i < N; ++i) {
            double d2V = (V[i - 1] - 2 * V[i] + V[i + 1]) / (dS * dS);
            double dV = (V[i + 1] - V[i - 1]) / (2 * dS);

            // Caputo fractional memory term
            double frac_sum = 0.0;
            for (int k = 1; k <= t; ++k) {
                frac_sum += weights[k] * V_hist[t - k][i];
            }

            double drift_term = 0.5 * sigma * sigma * S[i] * S[i] * d2V + r * S[i] * dV - r * V[i];
            double memory_term = gamma * dt_beta * frac_sum;

            V_new[i] = V[i] + dt * drift_term + memory_term;
            if (V_new[i] < 0.0 || std::isnan(V_new[i]) || std::isinf(V_new[i])) V_new[i] = 0.0;
        }

        // Boundary conditions
        V_new[0] = isCall ? 0.0 : K * std::exp(-r * (T - t * dt));
        V_new[N] = isCall ? std::max(Smax - K * std::exp(-r * (T - t * dt)), 0.0) : 0.0;

        // Update history
        V = V_new;
        V_hist[t] = V;

        // Logging every 20 steps
        if (t % 20 == 0 || t == M) {
            std::cout << "[INFO] Step " << t << "/" << M << ": Sample V = ";
            for (int j = N / 10; j <= N; j += N / 10) {
                std::cout << V[j] << " ";
            }
            std::cout << "\n";
        }
    }

    // === Inline Cubic Interpolation ===
    if (S0 <= S[0] || S0 >= S[N]) {
        std::cerr << "[WARN] S0 outside interpolation range: " << S0 << std::endl;
        return 0.0;
    }

    int j = 0;
    while (j < N - 1 && S[j + 1] < S0) ++j;

    // Local cubic spline using 4 points if possible
    int j0 = std::max(0, j - 1);
    int j1 = std::min(j0 + 1, N - 1);
    int j2 = std::min(j0 + 2, N - 1);
    int j3 = std::min(j0 + 3, N - 1);

    double x0 = S[j1], x1 = S[j2];
    double y0 = V[j1], y1 = V[j2];

    double slope = (y1 - y0) / (x1 - x0);
    double result = y0 + slope * (S0 - x0);
    std::cout << "[RESULT] Interpolated price at S0 = " << S0 << " is " << result << "\n";

    return result;
}


// ======= Solve FDM Dispatcher =======
double solve_fdm(const std::string& method, int N, int M, double Smax, double T, double K,
                 double r, double sigma, bool isCall,
                 double beta, bool rannacher_smoothing,
                 double S0)
{
    // Clamp S0 within [0, Smax]
    if (S0 < 0.0) S0 = 0.0;
    else if (S0 > Smax) S0 = Smax;

    if (method == "explicit")
        return fdm_explicit(N, M, Smax, T, K, r, sigma, isCall, S0);

    else if (method == "implicit")
        return fdm_implicit(N, M, Smax, T, K, r, sigma, isCall, S0);

    else if (method == "crank")
        return fdm_crank_nicolson(N, M, Smax, T, K, r, sigma, isCall, rannacher_smoothing, S0);

    else if (method == "american")
        return fdm_american_psor(N, M, Smax, T, K, r, sigma, isCall, 1.2, 10000, 1e-6, S0);

    else if (method == "fractional")
        return fdm_time_fractional(N, M, Smax, T, K, r, sigma, isCall, beta, S0);

    else if (method == "exponential")
        return fdm_exponential_integral(N, Smax, T, K, r, sigma, isCall, S0);

    else if (method == "compact") {
        return fdm_compact(N, M, Smax, T, K, r, sigma, isCall, S0);
    }

    else {
        throw std::invalid_argument("Unknown or unsupported method: " + method);
    }
}

// Return full price vector at maturity for Implicit FDM
std::vector<double> fdm_implicit_vector(int N, int M, double Smax, double T, double K,
                                       double r, double sigma, bool isCall) {
    double dS = Smax / N;
    double dt = T / M;
    std::vector<double> V(N + 1), S(N + 1);
    for (int i = 0; i <= N; ++i) {
        S[i] = i * dS;
        V[i] = isCall ? std::max(S[i] - K, 0.0) : std::max(K - S[i], 0.0);
    }

    std::vector<double> a(N - 1), b(N - 1), c(N - 1), d(N - 1);
    for (int t = 0; t < M; ++t) {
        double t_curr = (t + 1) * dt;
        for (int i = 1; i < N; ++i) {
            double j = static_cast<double>(i);
            a[i - 1] = -0.5 * dt * (sigma*sigma*j*j - r*j);
            b[i - 1] = 1 + dt * (sigma*sigma*j*j + r);
            c[i - 1] = -0.5 * dt * (sigma*sigma*j*j + r*j);
            d[i - 1] = V[i];
        }
        std::vector<double> Vnew = solve_tridiagonal(a, b, c, d);
        for (int i = 1; i < N; ++i) {
            V[i] = Vnew[i - 1];
            if (V[i] < 0.0) V[i] = 0.0;
        }
        V[0] = isCall ? 0.0 : K * std::exp(-r * (T - t_curr));
        V[N] = isCall ? (Smax - K * std::exp(-r * (T - t_curr))) : 0.0;
    }
    return V;
}

// Return full price vector at maturity for Crank-Nicolson FDM
std::vector<double> fdm_crank_nicolson_vector(int N, int M, double Smax, double T, double K,
                                             double r, double sigma, bool isCall,
                                             bool rannacher_smoothing) {
    double dS = Smax / N;
    double dt = T / M;
    std::vector<double> V(N + 1), S(N + 1);
    for (int i = 0; i <= N; ++i) {
        S[i] = i * dS;
        V[i] = isCall ? std::max(S[i] - K, 0.0) : std::max(K - S[i], 0.0);
    }

    std::vector<double> a(N - 1), b(N - 1), c(N - 1), d(N - 1);
    for (int t = 0; t < M; ++t) {
        double t_curr = (t + 1) * dt;
        for (int i = 1; i < N; ++i) {
            double j = static_cast<double>(i);
            double alpha = 0.25 * dt * (sigma*sigma*j*j - r*j);
            double beta = -0.5 * dt * (sigma*sigma*j*j + r);
            double gamma = 0.25 * dt * (sigma*sigma*j*j + r*j);
            a[i - 1] = -alpha;
            b[i - 1] = 1 - beta;
            c[i - 1] = -gamma;
            d[i - 1] = alpha * V[i - 1] + (1 + beta) * V[i] + gamma * V[i + 1];
        }
        std::vector<double> Vnew = solve_tridiagonal(a, b, c, d);
        for (int i = 1; i < N; ++i) {
            V[i] = Vnew[i - 1];
            if (V[i] < 0.0) V[i] = 0.0;
        }
        V[0] = isCall ? 0.0 : K * std::exp(-r * (T - t_curr));
        V[N] = isCall ? (Smax - K * std::exp(-r * (T - t_curr))) : 0.0;
    }
    return V;
}

// === Explicit FDM Surface ====
std::vector<std::vector<double>> fdm_explicit_surface(
    int N, int M, double Smax, double T, double K,
    double r, double sigma, bool is_call) {

    double dS = Smax / N;
    double dt = T / M;

    // --- CFL Stability Check ---
    double max_stable_dt = 0.5 * dS * dS / (sigma * sigma * Smax * Smax);
    if (dt > max_stable_dt) {
        dt = max_stable_dt;
        M = static_cast<int>(T / dt);
        std::cerr << "[Warning] Time step reduced for stability: M = " << M << ", dt = " << dt << "\n";
    }

    std::vector<std::vector<double>> surface;
    std::vector<double> V(N + 1);

    // Initial condition (payoff at maturity)
    for (int i = 0; i <= N; ++i) {
        double S = i * dS;
        V[i] = is_call ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
    }
    surface.push_back(V);

    // Time stepping (backward in time)
    for (int t = 1; t <= M; ++t) {
        std::vector<double> Vnew(N + 1);
        double tau = t * dt;

        // Boundary conditions
        Vnew[0] = is_call ? 0.0 : K * std::exp(-r * (T - tau));
        Vnew[N] = is_call ? (Smax - K * std::exp(-r * (T - tau))) : 0.0;

        // Update interior grid
        for (int i = 1; i < N; ++i) {
            double S = i * dS;
            double A = sigma * sigma * S * S / (dS * dS);
            double B = r * S / dS;

            double a = 0.5 * dt * (A - B);
            double b = 1.0 - dt * (A + r);
            double c = 0.5 * dt * (A + B);

            Vnew[i] = a * V[i - 1] + b * V[i] + c * V[i + 1];
        }

        surface.push_back(Vnew);
        V = Vnew;
    }

    return surface;
}

// === Implicit FDM Surface ====
std::vector<std::vector<double>> fdm_implicit_surface(
    int N, int M, double Smax, double T, double K,
    double r, double sigma, bool is_call) {

    double dS = Smax / N;
    double dt = T / M;

    std::vector<std::vector<double>> surface;
    std::vector<double> V(N + 1);

    // Initial condition (payoff at maturity)
    for (int i = 0; i <= N; ++i) {
        double S = i * dS;
        V[i] = is_call ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
    }
    surface.push_back(V);

    // Coefficients: PDE-consistent using actual S = i * dS
    std::vector<double> a(N - 1), b(N - 1), c(N - 1);
    for (int i = 1; i < N; ++i) {
        double S = i * dS;
        double A = sigma * sigma * S * S / (dS * dS);
        double B = r * S / dS;

        a[i - 1] = -0.5 * dt * (A - B);
        b[i - 1] = 1 + dt * (A + r);
        c[i - 1] = -0.5 * dt * (A + B);
    }

    // Time stepping backward from maturity
    for (int t = 1; t <= M; ++t) {
        std::vector<double> d(N - 1);
        for (int i = 1; i < N; ++i) {
            d[i - 1] = V[i];
        }

        // Apply boundary condition adjustments
        double tau = dt * t;
        d[0]   -= a[0]   * (is_call ? 0.0 : K * std::exp(-r * (T - tau)));
        d[N-2] -= c[N-2] * (is_call ? (Smax - K * std::exp(-r * (T - tau))) : 0.0);

        // Solve tridiagonal system for interior points
        std::vector<double> Vnew_inner = solve_tridiagonal(a, b, c, d);

        // Build full new solution including boundaries
        std::vector<double> Vnew(N + 1);
        Vnew[0] = is_call ? 0.0 : K * std::exp(-r * (T - tau));
        Vnew[N] = is_call ? (Smax - K * std::exp(-r * (T - tau))) : 0.0;
        for (int i = 1; i < N; ++i) {
            Vnew[i] = Vnew_inner[i - 1];
        }

        surface.push_back(Vnew);
        V = Vnew;
    }

    return surface;
}



std::vector<std::vector<double>> fdm_crank_nicolson_surface(
    int N, int M, double Smax, double T, double K,
    double r, double sigma, bool is_call, bool /*rannacher_smoothing*/) {

    double dS = Smax / N;
    double dt = T / M;

    std::vector<std::vector<double>> surface;
    std::vector<double> V(N + 1);

    // === Initial condition: Payoff at t = T ===
    for (int i = 0; i <= N; ++i) {
        double S = i * dS;
        V[i] = is_call ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
    }
    surface.push_back(V);  // Save first row (t=0)

    // === Precompute coefficients ===
    std::vector<double> a(N - 1), b(N - 1), c(N - 1);
    std::vector<double> alpha(N - 1), beta(N - 1), gamma(N - 1);

    for (int i = 1; i < N; ++i) {
        double j = static_cast<double>(i);
        double sigma2j2 = sigma * sigma * j * j;
        double rj = r * j;

        a[i - 1] = -0.25 * dt * (sigma2j2 - rj);
        b[i - 1] = 1 + 0.5 * dt * (sigma2j2 + r);
        c[i - 1] = -0.25 * dt * (sigma2j2 + rj);

        alpha[i - 1] = 0.25 * dt * (sigma2j2 - rj);
        beta[i - 1]  = 1 - 0.5 * dt * (sigma2j2 + r);
        gamma[i - 1] = 0.25 * dt * (sigma2j2 + rj);
    }

    // === Time stepping: backward in time ===
    for (int t = 1; t <= M; ++t) {
        std::vector<double> d(N - 1);

        // Use Backward Euler for first two time steps (Rannacher smoothing)
        if (t <= 2) {
            for (int i = 1; i < N; ++i) {
                d[i - 1] = V[i];  // implicit RHS
            }
        } else {
            for (int i = 1; i < N; ++i) {
                d[i - 1] = alpha[i - 1] * V[i - 1] + beta[i - 1] * V[i] + gamma[i - 1] * V[i + 1];
            }
        }

        // === Boundary conditions on RHS ===
        double disc = std::exp(-r * dt * t);
        d[0]   += a[0] * (is_call ? 0.0 : K * disc);
        d[N-2] += c[N-2] * (is_call ? (Smax - K * disc) : 0.0);

        // === Solve tridiagonal system ===
        std::vector<double> V_inner = solve_tridiagonal(a, b, c, d);

        std::vector<double> Vnew(N + 1);
        Vnew[0] = is_call ? 0.0 : K * disc;
        Vnew[N] = is_call ? (Smax - K * disc) : 0.0;

        for (int i = 1; i < N; ++i) {
            Vnew[i] = V_inner[i - 1];
        }

        surface.push_back(Vnew);
        V = Vnew;
    }

    return surface;
}

std::vector<double> fdm_compact_vector(int N, int M, double Smax, double T, double K,
                                       double r, double sigma, bool isCall) {
    double dS = Smax / N;
    double dt = T / M;

    std::vector<double> S(N + 1), V(N + 1);
    for (int i = 0; i <= N; ++i) {
        S[i] = i * dS;
        V[i] = isCall ? std::max(S[i] - K, 0.0) : std::max(K - S[i], 0.0);
    }

    std::vector<double> a(N - 1), b(N - 1), c(N - 1), rhs(N - 1);
    for (int t = 0; t < M; ++t) {
        for (int i = 1; i < N; ++i) {
            double j = static_cast<double>(i);
            a[i - 1] = -0.5 * dt * (sigma * sigma * j * j - r * j);
            b[i - 1] = 1.0 + dt * (sigma * sigma * j * j + r);
            c[i - 1] = -0.5 * dt * (sigma * sigma * j * j + r * j);
            rhs[i - 1] = V[i];
        }

        std::vector<double> Vnew = solve_tridiagonal(a, b, c, rhs);
        for (int i = 1; i < N; ++i)
            V[i] = std::max(Vnew[i - 1], 0.0);

        double t_curr = (t + 1) * dt;
        V[0] = isCall ? 0.0 : K * std::exp(-r * (T - t_curr));
        V[N] = isCall ? (Smax - K * std::exp(-r * (T - t_curr))) : 0.0;
    }

    return V;
}

double fdm_compact(int N, int M, double Smax, double T, double K,
                   double r, double sigma, bool isCall, double S0) {
    std::vector<double> V = fdm_compact_vector(N, M, Smax, T, K, r, sigma, isCall);
    double dS = Smax / N;
    std::vector<double> S(N + 1);
    for (int i = 0; i <= N; ++i)
        S[i] = i * dS;
    return interpolate_result(V, S, S0);


}

std::vector<std::vector<double>> fdm_compact_surface(int N, int M, double Smax, double T, double K,
                                                     double r, double sigma, bool isCall) {
    double dS = Smax / N;
    double dt = T / M;

    std::vector<std::vector<double>> surface;
    std::vector<double> S(N + 1), V(N + 1);

    // Initial condition at T = 0
    for (int i = 0; i <= N; ++i) {
        S[i] = i * dS;
        V[i] = isCall ? std::max(S[i] - K, 0.0) : std::max(K - S[i], 0.0);
    }
    surface.push_back(V);

    std::vector<double> a(N - 1), b(N - 1), c(N - 1), rhs(N - 1);

    for (int t = 0; t < M; ++t) {
        for (int i = 1; i < N; ++i) {
            double Si = S[i];
            double A = sigma * sigma * Si * Si / (dS * dS);
            double B = r * Si / dS;

            a[i - 1] = -0.5 * dt * (A - B);
            b[i - 1] = 1.0 + dt * (A + r);
            c[i - 1] = -0.5 * dt * (A + B);
            rhs[i - 1] = V[i];  // right-hand side is just current value
        }

        std::vector<double> Vnew_inner = solve_tridiagonal(a, b, c, rhs);

        std::vector<double> Vnew(N + 1);
        for (int i = 1; i < N; ++i)
            Vnew[i] = Vnew_inner[i - 1];

        double t_curr = (t + 1) * dt;
        Vnew[0] = isCall ? 0.0 : K * std::exp(-r * (T - t_curr));
        Vnew[N] = isCall ? (Smax - K * std::exp(-r * (T - t_curr))) : 0.0;

        V = Vnew;
        surface.push_back(V);
    }

    return surface;
}

std::vector<double> fdm_compact_vector(int N, int M, double Smax, double T, double K,
                                       double r, double sigma, bool isCall, double S0) {
    double dS = Smax / N;
    double dt = T / M;

    std::vector<double> S(N + 1), V(N + 1);
    for (int i = 0; i <= N; ++i) {
        S[i] = i * dS;
        V[i] = isCall ? std::max(S[i] - K, 0.0) : std::max(K - S[i], 0.0);
    }

    std::vector<double> a(N - 1), b(N - 1), c(N - 1), rhs(N - 1);
    for (int t = 0; t < M; ++t) {
        for (int i = 1; i < N; ++i) {
            double j = static_cast<double>(i);
            a[i - 1] = -0.5 * dt * (sigma * sigma * j * j - r * j);
            b[i - 1] = 1.0 + dt * (sigma * sigma * j * j + r);
            c[i - 1] = -0.5 * dt * (sigma * sigma * j * j + r * j);
            rhs[i - 1] = V[i];
        }

        std::vector<double> Vnew = solve_tridiagonal(a, b, c, rhs);
        for (int i = 1; i < N; ++i)
            V[i] = std::max(Vnew[i - 1], 0.0);

        double t_curr = (t + 1) * dt;
        V[0] = isCall ? 0.0 : K * std::exp(-r * (T - t_curr));
        V[N] = isCall ? (Smax - K * std::exp(-r * (T - t_curr))) : 0.0;
    }

    return V;  // S0 not used, but present in signature
}


// ======= Binomial Tree Vector Method =======
std::vector<double> binomial_tree_vector(int N, double T, double K,
                                         double r, double sigma, bool isCall,
                                         bool isAmerican, double S0, double /*Smax*/) {
    double dt = T / N;
    double u = std::exp(sigma * std::sqrt(dt));
    double d = 1.0 / u;
    double p = (std::exp(r * dt) - d) / (u - d);

    std::vector<double> option(N + 1);
    double S;

    // Terminal payoff at maturity
    for (int i = 0; i <= N; ++i) {
        S = S0 * std::pow(u, N - i) * std::pow(d, i);
        option[i] = isCall ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
    }

    // Backward induction
    for (int t = N - 1; t >= 0; --t) {
        for (int i = 0; i <= t; ++i) {
            S = S0 * std::pow(u, t - i) * std::pow(d, i);
            option[i] = std::exp(-r * dt) * (p * option[i] + (1 - p) * option[i + 1]);

            if (isAmerican) {
                double intrinsic = isCall ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
                option[i] = std::max(option[i], intrinsic); // Early exercise
            }
        }
    }

    return option;
}

// ======= Binomial Tree Price (Scalar Result) =======
double binomial_tree(int N, double T, double K,
                           double r, double sigma, bool isCall,
                           bool isAmerican, double S0) {
    std::vector<double> option = binomial_tree_vector(N, T, K, r, sigma, isCall, isAmerican, S0, 0.0);
    return option[0]; // Final price at root node
}

std::vector<std::vector<double>> fdm_american_psor_vector(int N, int M, double Smax, double T, double K,
                                                          double r, double sigma, bool isCall,
                                                          double omega, int maxIter, double tol) {
    double dS = Smax / N;
    double dt = T / M;

    std::vector<double> S(N + 1), V(N + 1);
    std::vector<std::vector<double>> V_hist(M + 1, std::vector<double>(N + 1));

    for (int i = 0; i <= N; ++i) {
        S[i] = i * dS;
        V[i] = isCall ? std::max(S[i] - K, 0.0) : std::max(K - S[i], 0.0);
    }
    V_hist[M] = V;

    std::vector<double> a(N - 1), b(N - 1), c(N - 1), rhs(N - 1);
    for (int t = M - 1; t >= 0; --t) {
        double t_curr = t * dt;
        for (int i = 1; i < N; ++i) {
            double j = static_cast<double>(i);
            a[i - 1] = -0.5 * dt * (sigma*sigma*j*j - r*j);
            b[i - 1] = 1 + dt * (sigma*sigma*j*j + r);
            c[i - 1] = -0.5 * dt * (sigma*sigma*j*j + r*j);
            rhs[i - 1] = V[i];
        }

        for (int k = 0; k < maxIter; ++k) {
            double error = 0.0;
            for (int i = 1; i < N; ++i) {
                double j = i - 1;
                double y = (j > 0 ? a[j] * V[i - 1] : 0.0)
                         + b[j] * V[i]
                         + (j < N - 2 ? c[j] * V[i + 1] : 0.0)
                         - rhs[j];
                double Vnew = V[i] - omega * y / b[j];
                double payoff = isCall ? std::max(S[i] - K, 0.0) : std::max(K - S[i], 0.0);
                Vnew = std::max(Vnew, payoff);
                error += std::abs(Vnew - V[i]);
                V[i] = Vnew;
            }
            if (error < tol)
                break;
        }
        V[0] = isCall ? 0.0 : K * std::exp(-r * (T - t_curr));
        V[N] = isCall ? (Smax - K * std::exp(-r * (T - t_curr))) : 0.0;
        V_hist[t] = V;
    }
    return V_hist;
}


std::vector<std::vector<double>> fdm_exponential_integral_vector(int N, double Smax, double T, double K,
                                                                  double r, double sigma, bool isCall) {
    double dS = Smax / N;
    double dt = 0.01;
    int M = static_cast<int>(T / dt);

    std::vector<double> S(N + 1);
    for (int i = 0; i <= N; ++i)
        S[i] = i * dS;

    std::vector<double> V(N + 1), V_new(N + 1), rhs(N + 1);
    std::vector<std::vector<double>> V_hist(M + 1, std::vector<double>(N + 1));

    // Final payoff at maturity (t = T)
    for (int i = 0; i <= N; ++i)
        V[i] = isCall ? std::max(S[i] - K, 0.0) : std::max(K - S[i], 0.0);

    V_hist[M] = V;

    for (int t = M - 1; t >= 0; --t) {
        for (int i = 1; i < N; ++i) {
            double d2V = (V[i - 1] - 2 * V[i] + V[i + 1]) / (dS * dS);
            double dV  = (V[i + 1] - V[i - 1]) / (2 * dS);
            rhs[i] = 0.5 * sigma * sigma * S[i] * S[i] * d2V
                   + r * S[i] * dV - r * V[i];
        }

        // Boundary conditions
        double tau = T - t * dt;
        V_new[0]  = isCall ? 0.0 : K * std::exp(-r * tau);
        V_new[N]  = isCall ? (Smax - K * std::exp(-r * tau)) : 0.0;

        for (int i = 1; i < N; ++i) {
            V_new[i] = V[i] + dt * rhs[i];
            V_new[i] *= std::exp(-r * dt);  // exponential integrator step
            if (V_new[i] < 0.0) V_new[i] = 0.0; // safeguard
        }

        V = V_new;
        V_hist[t] = V;
    }

    return V_hist;
}

std::vector<std::vector<double>> binomial_tree_surface(int N, double T, double K,
                                                       double r, double sigma, bool is_call,
                                                       bool is_american, double S0) {
    double dt = T / N;
    double u = std::exp(sigma * std::sqrt(dt));
    double d = 1.0 / u;
    double p = (std::exp(r * dt) - d) / (u - d);

    std::vector<std::vector<double>> surface(N + 1, std::vector<double>(N + 1));

    // Terminal nodes
    for (int i = 0; i <= N; ++i) {
        double S = S0 * std::pow(u, N - i) * std::pow(d, i);
        surface[N][i] = is_call ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
    }

    // Backward iteration
    for (int t = N - 1; t >= 0; --t) {
        for (int i = 0; i <= t; ++i) {
            double S = S0 * std::pow(u, t - i) * std::pow(d, i);
            double cont_val = std::exp(-r * dt) * (p * surface[t + 1][i] + (1 - p) * surface[t + 1][i + 1]);
            double intrinsic = is_call ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
            surface[t][i] = is_american ? std::max(cont_val, intrinsic) : cont_val;
        }
    }

    return surface;
}

std::vector<std::vector<double>> exponential_integral_surface(
    int N, int M, double Smax, double T,
    double K, double r, double sigma,
    bool isCall
) {
    double dS = Smax / N;
    double dt = T / M;

    std::vector<double> S(N + 1);
    for (int i = 0; i <= N; ++i)
        S[i] = i * dS;

    std::vector<std::vector<double>> grid(M + 1, std::vector<double>(N + 1));

    // Final time step: payoff
    for (int i = 0; i <= N; ++i)
        grid[M][i] = isCall ? std::max(S[i] - K, 0.0) : std::max(K - S[i], 0.0);

    // Time stepping (backward in time)
    for (int j = M - 1; j >= 0; --j) {
        double discount = std::exp(-r * dt);  // Exponential integrator

        for (int i = 1; i < N; ++i) {
            double Si = S[i];
            double d2V = (grid[j + 1][i - 1] - 2 * grid[j + 1][i] + grid[j + 1][i + 1]) / (dS * dS);
            double dV = (grid[j + 1][i + 1] - grid[j + 1][i - 1]) / (2 * dS);

            double diffusion = 0.5 * sigma * sigma * Si * Si * d2V;
            double convection = r * Si * dV;
            double discount_term = -r * grid[j + 1][i];

            double rhs = diffusion + convection + discount_term;
            grid[j][i] = discount * (grid[j + 1][i] + dt * rhs);
        }

        // Boundary conditions (at time j)
        double tau = T - j * dt;
        grid[j][0] = isCall ? 0.0 : K * std::exp(-r * tau);
        grid[j][N] = isCall ? (Smax - K * std::exp(-r * tau)) : 0.0;
    }

    return grid;
}

std::vector<std::vector<double>> american_psor_surface(
    int N, int M, double Smax, double T,
    double K, double r, double sigma,
    bool isCall, double omega, int maxIter, double tol
) {
    double dS = Smax / N;
    double dt = T / M;
    std::vector<double> S(N + 1);
    for (int i = 0; i <= N; ++i) S[i] = i * dS;

    std::vector<std::vector<double>> grid(M + 1, std::vector<double>(N + 1));
    for (int i = 0; i <= N; ++i) {
        grid[M][i] = isCall ? std::max(S[i] - K, 0.0) : std::max(K - S[i], 0.0);
    }

    std::vector<double> a(N - 1), b(N - 1), c(N - 1);
    for (int i = 1; i < N; ++i) {
        double sigma_sq = sigma * sigma;
        a[i - 1] = -0.5 * dt * (sigma_sq * i * i - r * i);
        b[i - 1] = 1 + dt * (sigma_sq * i * i + r);
        c[i - 1] = -0.5 * dt * (sigma_sq * i * i + r * i);
    }

    std::vector<double> V_old(N - 1), V_new(N - 1);
    for (int j = M - 1; j >= 0; --j) {
        for (int i = 1; i < N; ++i)
            V_old[i - 1] = grid[j + 1][i];

        for (int iter = 0; iter < maxIter; ++iter) {
            for (int i = 1; i < N; ++i) {
                int k = i - 1;
                double rhs = a[k] * V_new[std::max(0, k - 1)] +
                             b[k] * V_new[k] +
                             c[k] * V_new[std::min(N - 2, k + 1)];
                rhs = V_old[k] + (rhs - b[k] * V_new[k]);
                V_new[k] = V_new[k] + omega * (rhs - V_new[k]);

                double exercise = isCall ? std::max(S[i] - K, 0.0) : std::max(K - S[i], 0.0);
                V_new[k] = std::max(V_new[k], exercise);
            }

            if (std::inner_product(V_old.begin(), V_old.end(), V_new.begin(), 0.0,
                                   std::plus<>(), [](double x, double y) { return std::abs(x - y); }) < tol)
                break;

            V_old = V_new;
        }

        for (int i = 1; i < N; ++i) grid[j][i] = V_new[i - 1];
        grid[j][0] = isCall ? 0.0 : K * std::exp(-r * (T - j * dt));
        grid[j][N] = isCall ? Smax - K * std::exp(-r * (T - j * dt)) : 0.0;
    }

    return grid;
}

// American PSOR Solver for Option Pricing
std::vector<double> american_psor_vector(
    int N, int M, double Smax, double T,
    double K, double r, double sigma, bool is_call,
    double omega, int maxIter, double tol)
{
    double dS = Smax / N;
    double dt = T / M;

    std::vector<double> S(N + 1);
    for (int i = 0; i <= N; ++i)
        S[i] = i * dS;

    std::vector<double> V(N + 1);         // Option value
    std::vector<double> payoff(N + 1);    // Early exercise payoff

    // Set terminal condition
    for (int i = 0; i <= N; ++i)
        payoff[i] = is_call ? std::max(S[i] - K, 0.0) : std::max(K - S[i], 0.0);

    V = payoff;

    std::vector<double> a(N + 1), b(N + 1), c(N + 1);

    // Coefficients for the tridiagonal matrix
    for (int i = 1; i < N; ++i)
    {
        double sigma2 = sigma * sigma;
        a[i] = 0.5 * dt * (sigma2 * i * i - r * i);
        b[i] = 1.0 + dt * (sigma2 * i * i + r);
        c[i] = 0.5 * dt * (-(sigma2 * i * i + r * i));
    }

    // PSOR time-stepping loop
    for (int t = M - 1; t >= 0; --t)
    {
        std::vector<double> V_old = V;
        for (int iter = 0; iter < maxIter; ++iter)
        {
            double error = 0.0;
            for (int i = 1; i < N; ++i)
            {
                double rhs = a[i] * V[i - 1] + (1.0 - b[i]) * V[i] + c[i] * V[i + 1];
                double V_new = V[i] + omega * (rhs - V[i]) / b[i];

                // American constraint: early exercise condition
                V_new = std::max(V_new, payoff[i]);

                error += std::abs(V_new - V[i]);
                V[i] = V_new;
            }

            if (error < tol)
                break;
        }

        // Boundary conditions
        V[0] = is_call ? 0.0 : K * std::exp(-r * (T - t * dt));
        V[N] = is_call ? (Smax - K * std::exp(-r * (T - t * dt))) : 0.0;
    }

    return V;
}


std::vector<std::vector<double>> fdm_time_fractional_surface(int N, int M, double Smax, double T, double K,
                                                             double r, double sigma, bool isCall, double beta) {
    if (beta <= 0.0 || beta >= 1.0) {
        throw std::invalid_argument("beta must be in (0,1)");
    }

    double dS = Smax / N;
    double dt = T / M;

    std::vector<double> S(N + 1);
    std::vector<double> V(N + 1), V_new(N + 1);
    std::vector<std::vector<double>> V_hist(M + 1, std::vector<double>(N + 1));
    std::vector<double> weights(M + 1, 0.0);

    for (int i = 0; i <= N; ++i) {
        S[i] = i * dS;
        V[i] = isCall ? std::max(S[i] - K, 0.0) : std::max(K - S[i], 0.0);
        V_hist[0][i] = V[i];
    }

    weights[0] = 1.0;
    for (int k = 1; k <= M; ++k) {
        weights[k] = weights[k - 1] * (1.0 - (1.0 + beta) / k);
    }

    for (int t = 1; t <= M; ++t) {
        for (int i = 1; i < N; ++i) {
            double d2V = (V[i - 1] - 2 * V[i] + V[i + 1]) / (dS * dS);
            double dV = (V[i + 1] - V[i - 1]) / (2 * dS);
            double frac_sum = 0.0;

            for (int k = 1; k <= t; ++k) {
                frac_sum += weights[k] * V_hist[t - k][i];
            }

            V_new[i] = V[i] + dt * (0.5 * sigma * sigma * S[i] * S[i] * d2V + r * S[i] * dV - r * V[i])
                       + (pow(dt, -beta) / tgamma(2 - beta)) * frac_sum;

            if (V_new[i] < 0.0) V_new[i] = 0.0;
        }

        V_new[0] = isCall ? 0.0 : K * std::exp(-r * (T - t * dt));
        V_new[N] = isCall ? (Smax - K * std::exp(-r * (T - t * dt))) : 0.0;

        V = V_new;
        V_hist[t] = V;
    }

    return V_hist;
}

