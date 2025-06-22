#include "finite_difference.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

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

// ======= Exponential Integral FDM =======
double fdm_exponential_integral(int N, double Smax, double T, double K,
                               double r, double sigma, bool isCall, double S0) {
    double dS = Smax / N;
    double dt = 0.01;
    int M = static_cast<int>(T / dt);

    std::vector<double> V(N + 1), V_new(N + 1), rhs(N + 1), S(N + 1);

    for (int i = 0; i <= N; ++i) {
        S[i] = i * dS;
        V[i] = isCall ? std::max(S[i] - K, 0.0) : std::max(K - S[i], 0.0);
    }

    for (int t = 0; t < M; ++t) {
        for (int i = 1; i < N; ++i) {
            double d2V = (V[i - 1] - 2 * V[i] + V[i + 1]) / (dS * dS);
            double dV = (V[i + 1] - V[i - 1]) / (2 * dS);
            rhs[i] = 0.5 * sigma*sigma*S[i]*S[i]*d2V + r*S[i]*dV - r*V[i];
        }
        for (int i = 1; i < N; ++i) {
            V_new[i] = V[i] + dt * rhs[i];
            if (V_new[i] < 0.0) V_new[i] = 0.0;
        }
        V = V_new;
    }
    return interpolate_result(V, S, S0);
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
// ======= Time-Fractional FDM =======
double fdm_time_fractional(int N, int M, double Smax, double T, double K,
                          double r, double sigma, bool isCall, double beta, double S0) {
    // Throw if beta out of (0,1) to satisfy tests requiring exceptions
    if (beta <= 0.0 || beta >= 1.0) {
        throw std::invalid_argument("beta must be in (0,1)");
    }


    double dS = Smax / N;
    double dt = T / M;

    std::vector<double> V(N + 1), V_new(N + 1), S(N + 1);
    std::vector<std::vector<double>> V_hist(M + 1, std::vector<double>(N + 1));
    std::vector<double> weights(M + 1, 0.0);

    // Initialize asset prices and payoff at maturity
    for (int i = 0; i <= N; ++i) {
        S[i] = i * dS;
        V[i] = isCall ? std::max(S[i] - K, 0.0) : std::max(K - S[i], 0.0);
        V_hist[0][i] = V[i];
    }

    // Compute weights for fractional derivative
    weights[0] = 1.0;
    for (int k = 1; k <= M; ++k) {
        weights[k] = weights[k - 1] * (1.0 - (1.0 + beta) / k);
    }

    // Time-stepping loop for fractional PDE
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

            if (V_new[i] < 0.0) V_new[i] = 0.0;  // clamp negative values
        }

        // Boundary conditions
        V_new[0] = isCall ? 0.0 : K * std::exp(-r * (T - t * dt));
        V_new[N] = isCall ? (Smax - K * std::exp(-r * (T - t * dt))) : 0.0;

        V = V_new;
        V_hist[t] = V;
    }

    return interpolate_result(V, S, S0);
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
        std::vector<double> V = fdm_explicit_vector(N, M, Smax, T, K, r, sigma, isCall);
        double dx = Smax / N;
        std::vector<double> d2V = compact_4th_order_second_derivative(V, dx);
        std::vector<double> S(N + 1);
        for (int i = 0; i <= N; ++i)
            S[i] = i * dx;
        return interpolate_result(d2V, S, S0);
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
