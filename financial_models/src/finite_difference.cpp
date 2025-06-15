#include "finite_difference.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <vector>

using namespace std;

// ======= Tridiagonal Solver =======
vector<double> solve_tridiagonal(const vector<double>& a, const vector<double>& b,
                                 const vector<double>& c, const vector<double>& d) {
    int n = b.size();
    vector<double> x(n), cp(n), dp(n);
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

// ======= Explicit FDM =======
vector<double> fdm_explicit(int N, int M, double Smax, double T, double K,
                            double r, double sigma, bool isCall) {
    double dS = Smax / N;
    double dt = T / M;
    vector<double> V(N + 1), Vnew(N + 1);

    for (int i = 0; i <= N; ++i) {
        double S = i * dS;
        V[i] = isCall ? max(S - K, 0.0) : max(K - S, 0.0);
    }

    for (int t = 0; t < M; ++t) {
        for (int i = 1; i < N; ++i) {
            double j = static_cast<double>(i);
            double a = 0.5 * dt * (sigma * sigma * j * j - r * j);
            double b = 1 - dt * (sigma * sigma * j * j + r);
            double c = 0.5 * dt * (sigma * sigma * j * j + r * j);
            Vnew[i] = a * V[i - 1] + b * V[i] + c * V[i + 1];
        }
        double t_curr = (t + 1) * dt;
        Vnew[0] = isCall ? 0.0 : K * exp(-r * (T - t_curr));
        Vnew[N] = isCall ? (Smax - K * exp(-r * (T - t_curr))) : 0.0;
        V = Vnew;
    }

    return V;
}

// ======= Implicit FDM =======
vector<double> fdm_implicit(int N, int M, double Smax, double T, double K,
                            double r, double sigma, bool isCall) {
    double dS = Smax / N;
    double dt = T / M;
    vector<double> V(N + 1);

    for (int i = 0; i <= N; ++i) {
        double S = i * dS;
        V[i] = isCall ? max(S - K, 0.0) : max(K - S, 0.0);
    }

    vector<double> a(N - 1), b(N - 1), c(N - 1), d(N - 1);

    for (int t = 0; t < M; ++t) {
        for (int i = 1; i < N; ++i) {
            double j = static_cast<double>(i);
            a[i - 1] = -0.5 * dt * (sigma * sigma * j * j - r * j);
            b[i - 1] = 1 + dt * (sigma * sigma * j * j + r);
            c[i - 1] = -0.5 * dt * (sigma * sigma * j * j + r * j);
            d[i - 1] = V[i];
        }
        vector<double> Vnew = solve_tridiagonal(a, b, c, d);
        for (int i = 1; i < N; ++i)
            V[i] = Vnew[i - 1];

        double t_curr = (t + 1) * dt;
        V[0] = isCall ? 0.0 : K * exp(-r * (T - t_curr));
        V[N] = isCall ? (Smax - K * exp(-r * (T - t_curr))) : 0.0;
    }

    return V;
}

// ======= Crank-Nicolson FDM =======
vector<double> fdm_crank_nicolson(int N, int M, double Smax, double T, double K,
                                  double r, double sigma, bool isCall, bool rannacher_smoothing) {
    (void)rannacher_smoothing;

    double dS = Smax / N;
    double dt = T / M;
    vector<double> V(N + 1);

    for (int i = 0; i <= N; ++i) {
        double S = i * dS;
        V[i] = isCall ? max(S - K, 0.0) : max(K - S, 0.0);
    }

    vector<double> a(N - 1), b(N - 1), c(N - 1), d(N - 1);

    for (int t = 0; t < M; ++t) {
        for (int i = 1; i < N; ++i) {
            double j = static_cast<double>(i);
            double alpha = 0.25 * dt * (sigma * sigma * j * j - r * j);
            double beta  = -0.5 * dt * (sigma * sigma * j * j + r);
            double gamma = 0.25 * dt * (sigma * sigma * j * j + r * j);

            a[i - 1] = -alpha;
            b[i - 1] = 1 - beta;
            c[i - 1] = -gamma;

            d[i - 1] = alpha * V[i - 1] + (1 + beta) * V[i] + gamma * V[i + 1];
        }

        vector<double> Vnew = solve_tridiagonal(a, b, c, d);
        for (int i = 1; i < N; ++i)
            V[i] = Vnew[i - 1];

        double t_curr = (t + 1) * dt;
        V[0] = isCall ? 0.0 : K * exp(-r * (T - t_curr));
        V[N] = isCall ? (Smax - K * exp(-r * (T - t_curr))) : 0.0;
    }

    return V;
}

// ======= American PSOR FDM =======
vector<double> fdm_american_psor(int N, int M, double Smax, double T, double K,
                                 double r, double sigma, bool isCall,
                                 double omega, int maxIter, double tol) {
    double dS = Smax / N;
    double dt = T / M;

    vector<double> S(N + 1), V(N + 1), oldV(N + 1);
    for (int i = 0; i <= N; ++i) {
        S[i] = i * dS;
        V[i] = isCall ? max(S[i] - K, 0.0) : max(K - S[i], 0.0);
    }

    vector<double> a(N - 1), b(N - 1), c(N - 1), rhs(N - 1);

    for (int t = M - 1; t >= 0; --t) {
        for (int i = 1; i < N; ++i) {
            double j = static_cast<double>(i);
            a[i - 1] = -0.5 * dt * (sigma * sigma * j * j - r * j);
            b[i - 1] = 1 + dt * (sigma * sigma * j * j + r);
            c[i - 1] = -0.5 * dt * (sigma * sigma * j * j + r * j);
            rhs[i - 1] = V[i];
        }

        oldV = V;
        for (int k = 0; k < maxIter; ++k) {
            double error = 0.0;
            for (int i = 1; i < N; ++i) {
                double j = i - 1;
                double y = (j > 0 ? a[j] * V[i - 1] : 0.0)
                         + ( b[j] * V[i])
                         + (j < N-2 ? c[j] * V[i + 1] : 0.0)
                         - rhs[j];

                double Vnew = V[i] - omega * y / b[j];
                double payoff = isCall ? max(S[i] - K, 0.0) : max(K - S[i], 0.0);
                Vnew = max(Vnew, payoff);

                error += abs(Vnew - V[i]);
                V[i] = Vnew;
            }

            if (error < tol)
                break;
        }

        double t_curr = t * dt;
        V[0] = isCall ? 0.0 : K * exp(-r * (T - t_curr));
        V[N] = isCall ? (Smax - K * exp(-r * (T - t_curr))) : 0.0;
    }

    return V;
}

// ======= Exponential Integral FDM =======
vector<double> fdm_exponential_integral(int N, double Smax, double T, double K,
                                        double r, double sigma, bool isCall) {
    double dS = Smax / N;
    double dt = 0.01;
    int M = static_cast<int>(T / dt);
    vector<double> V(N + 1), V_new(N + 1), rhs(N + 1);

    for (int i = 0; i <= N; ++i) {
        double S = i * dS;
        V[i] = isCall ? max(S - K, 0.0) : max(K - S, 0.0);
    }

    for (int t = 0; t < M; ++t) {
        for (int i = 1; i < N; ++i) {
            double S = i * dS;
            double d2V = (V[i - 1] - 2 * V[i] + V[i + 1]) / (dS * dS);
            double dV = (V[i + 1] - V[i - 1]) / (2 * dS);
            rhs[i] = 0.5 * sigma * sigma * S * S * d2V + r * S * dV - r * V[i];
        }
        for (int i = 1; i < N; ++i)
            V_new[i] = V[i] + dt * rhs[i];
        V = V_new;
    }

    return V;
}

// ======= Time-Fractional FDM =======
vector<double> fdm_time_fractional(int N, int M, double Smax, double T, double K,
                                   double r, double sigma, bool isCall, double beta) {
    if (beta <= 0.0 || beta >= 1.0)
        throw invalid_argument("beta must be in (0, 1)");

    double dS = Smax / N;
    double dt = T / M;
    vector<double> V(N + 1), V_new(N + 1);
    vector<vector<double>> V_hist(M + 1, vector<double>(N + 1));
    vector<double> weights(M + 1, 0.0);

    for (int i = 0; i <= N; ++i) {
        double S = i * dS;
        V[i] = isCall ? max(S - K, 0.0) : max(K - S, 0.0);
        V_hist[0][i] = V[i];
    }

    weights[0] = 1.0;
    for (int k = 1; k <= M; ++k)
        weights[k] = weights[k - 1] * (1.0 - (1.0 + beta) / k);

    for (int t = 1; t <= M; ++t) {
        for (int i = 1; i < N; ++i) {
            double S = i * dS;
            double d2V = (V[i - 1] - 2 * V[i] + V[i + 1]) / (dS * dS);
            double dV = (V[i + 1] - V[i - 1]) / (2 * dS);
            double frac_sum = 0.0;
            for (int k = 1; k <= t; ++k)
                frac_sum += weights[k] * V_hist[t - k][i];
            V_new[i] = V[i] + dt * (0.5 * sigma * sigma * S * S * d2V + r * S * dV - r * V[i])
                       + (pow(dt, -beta) / tgamma(2 - beta)) * frac_sum;
        }
        V = V_new;
        V_hist[t] = V;
    }

    return V;
}

// ======= Compact 4th-order Second Derivative =======
vector<double> compact_4th_order_second_derivative(const vector<double>& V, double dx) {
    int N = V.size();
    vector<double> d2V(N, 0.0);
    for (int i = 2; i < N - 2; ++i) {
        d2V[i] = (-V[i - 2] + 16 * V[i - 1] - 30 * V[i] + 16 * V[i + 1] - V[i + 2]) / (12 * dx * dx);
    }
    return d2V;
}

// ======= Dispatcher =======
vector<double> solve_fdm(const string& method, int N, int M, double Smax, double T, double K,
                         double r, double sigma, bool isCall,
                         double beta, bool rannacher_smoothing) {
    if (method == "explicit")
        return fdm_explicit(N, M, Smax, T, K, r, sigma, isCall);
    else if (method == "implicit")
        return fdm_implicit(N, M, Smax, T, K, r, sigma, isCall);
    else if (method == "crank")
        return fdm_crank_nicolson(N, M, Smax, T, K, r, sigma, isCall, rannacher_smoothing);
    else if (method == "american")
        return fdm_american_psor(N, M, Smax, T, K, r, sigma, isCall, 1.2, 10000, 1e-6);
    else if (method == "compact")
        return compact_4th_order_second_derivative(fdm_explicit(N, M, Smax, T, K, r, sigma, isCall), Smax / N);
    else if (method == "fractional")
        return fdm_time_fractional(N, M, Smax, T, K, r, sigma, isCall, beta);
    else if (method == "exponential")
        return fdm_exponential_integral(N, Smax, T, K, r, sigma, isCall);
    else
        throw invalid_argument("Unknown method: " + method);
}
