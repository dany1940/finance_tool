#ifndef FINITE_DIFFERENCE_ALL_H
#define FINITE_DIFFERENCE_ALL_H

#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>
#include <iostream>

using namespace std;

// ======= Tridiagonal Solver =======
vector<double> solve_tridiagonal(
    const vector<double>& a,
    const vector<double>& b,
    const vector<double>& c,
    const vector<double>& d
);

// ======= Boundary Condition Helper =======
double boundaryCondition(
    double S, double K, double T, double t, double r, bool isCall
);

// ======= Finite Difference Methods Declarations =======

// Explicit Method
vector<double> fdm_explicit(
    int N, int M, double Smax, double T, double K,
    double r, double sigma, bool isCall
);

// Implicit Method
vector<double> fdm_implicit(
    int N, int M, double Smax, double T, double K,
    double r, double sigma, bool isCall
);

// Crank-Nicolson Method with optional Rannacher smoothing
vector<double> fdm_crank_nicolson(
    int N, int M, double Smax, double T, double K,
    double r, double sigma, bool isCall,
    bool rannacher_smoothing = false
);

// American Option using PSOR (Projected Successive Over-Relaxation)
vector<double> fdm_american_psor(
    int N, int M, double Smax, double T, double K,
    double r, double sigma, bool isCall,
    double omega = 1.2, int maxIter = 10000, double tol = 1e-6
);

// Compact 4th-order 2nd derivative (used in high-accuracy schemes)
vector<double> compact_4th_order_second_derivative(
    const vector<double>& V, double dx
);

// Experimental Exponential Integral Method
vector<double> fdm_exponential_integral(
    int N, double Smax, double T, double K,
    double r, double sigma, bool isCall
);

// Time-Fractional FDM (for fractional Black-Scholes equations)
vector<double> fdm_time_fractional(
    int N, int M, double Smax, double T, double K,
    double r, double sigma, bool isCall, double beta
);

// ======= Dispatcher Function (selects method by name) =======
vector<double> solve_fdm(
    const string& method,
    int N, int M, double Smax, double T, double K,
    double r, double sigma, bool isCall,
    double beta = 0.5, bool rannacher_smoothing = false
);

#endif // FINITE_DIFFERENCE_ALL_H
