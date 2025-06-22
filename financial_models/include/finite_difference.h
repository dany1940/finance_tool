#ifndef FINITE_DIFFERENCE_ALL_H
#define FINITE_DIFFERENCE_ALL_H

#include <vector>
#include <string>

// ======= Cubic spline interpolation helper =======
double interpolate_result(const std::vector<double>& V, const std::vector<double>& S, double S0);

// ======= Tridiagonal Solver =======
std::vector<double> solve_tridiagonal(
    const std::vector<double>& a,
    const std::vector<double>& b,
    const std::vector<double>& c,
    const std::vector<double>& d
);

// ======= Boundary Condition Helper =======
double boundaryCondition(
    double S, double K, double T, double t, double r, bool isCall
);

// ======= Explicit FDM (returns full vector) =======
std::vector<double> fdm_explicit_vector(
    int N, int M, double Smax, double T, double K,
    double r, double sigma, bool isCall
);

// ======= Crank-Nicolson FDM (returns full vector) =======
std::vector<double> fdm_crank_nicolson_vector(
    int N, int M, double Smax, double T, double K,
    double r, double sigma, bool isCall,
    bool rannacher_smoothing
);

// ======= Finite Difference Methods (return interpolated price at S0) =======
double fdm_explicit(
    int N, int M, double Smax, double T, double K,
    double r, double sigma, bool isCall,
    double S0
);

double fdm_implicit(
    int N, int M, double Smax, double T, double K,
    double r, double sigma, bool isCall,
    double S0
);

double fdm_crank_nicolson(
    int N, int M, double Smax, double T, double K,
    double r, double sigma, bool isCall,
    bool rannacher_smoothing,
    double S0
);

double fdm_american_psor(
    int N, int M, double Smax, double T, double K,
    double r, double sigma, bool isCall,
    double omega, int maxIter, double tol,
    double S0
);

double fdm_exponential_integral(
    int N, double Smax, double T, double K,
    double r, double sigma, bool isCall,
    double S0
);

double fdm_time_fractional(
    int N, int M, double Smax, double T, double K,
    double r, double sigma, bool isCall,
    double beta,
    double S0
);

// ======= Compact 4th-order 2nd derivative (returns full vector) =======
std::vector<double> compact_4th_order_second_derivative(
    const std::vector<double>& V, double dx
);

// ======= Dispatcher Function (returns interpolated price at S0) =======
double solve_fdm(
    const std::string& method,
    int N, int M, double Smax, double T, double K,
    double r, double sigma, bool isCall,
    double beta, bool rannacher_smoothing,
    double S0
);

#endif // FINITE_DIFFERENCE_ALL_H
