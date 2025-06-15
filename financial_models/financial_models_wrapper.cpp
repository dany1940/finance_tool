#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // For std::vector bindings

#include "finite_difference.h"
#include "utils.h"
#include "black_scholes.h"
#include "monte_carlo.h"

namespace py = pybind11;

PYBIND11_MODULE(financial_models_wrapper, m) {
    m.doc() = "Python bindings for comprehensive financial models";

    // ======= Black-Scholes Model =======
    m.def("black_scholes", &blackScholes,
          "Black-Scholes option pricing formula",
          py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("sigma"), py::arg("is_call"));

    // ======= Monte Carlo Simulation =======
    m.def("monte_carlo", &monteCarloBlackScholes,
          "Monte Carlo simulation for option pricing",
          py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("sigma"), py::arg("is_call"),
          py::arg("num_simulations"));

    // ======= Finite Difference Methods =====

    // Explicit FDM
    m.def("explicit_fdm", &fdm_explicit,
          "Explicit Finite Difference method",
          py::arg("N"), py::arg("M"), py::arg("Smax"), py::arg("T"),
          py::arg("K"), py::arg("r"), py::arg("sigma"), py::arg("is_call"));

    // Implicit FDM
    m.def("implicit_fdm", &fdm_implicit,
          "Implicit Finite Difference method",
          py::arg("N"), py::arg("M"), py::arg("Smax"), py::arg("T"),
          py::arg("K"), py::arg("r"), py::arg("sigma"), py::arg("is_call"));

    // Crank-Nicolson FDM
    m.def("crank_nicolson_fdm", &fdm_crank_nicolson,
          "Crank-Nicolson Finite Difference method",
          py::arg("N"), py::arg("M"), py::arg("Smax"), py::arg("T"),
          py::arg("K"), py::arg("r"), py::arg("sigma"),
          py::arg("is_call"), py::arg("rannacher_smoothing") = false);

    // American Option using PSOR
    m.def("american_psor_fdm", &fdm_american_psor,
          "American Option using PSOR method",
          py::arg("N"), py::arg("M"), py::arg("Smax"), py::arg("T"),
          py::arg("K"), py::arg("r"), py::arg("sigma"),
          py::arg("is_call"),
          py::arg("omega") = 1.2,
          py::arg("maxIter") = 10000,
          py::arg("tol") = 1e-6);

    // 4th Order Compact Method
    m.def("compact_fdm", &compact_4th_order_second_derivative,
          "Compact 4th-order Finite Difference approximation for second derivative",
          py::arg("V"), py::arg("dx"));

    // Time-Fractional FDM
    m.def("fractional_fdm", &fdm_time_fractional,
          "Time-Fractional Finite Difference method",
          py::arg("N"), py::arg("M"), py::arg("Smax"), py::arg("T"),
          py::arg("K"), py::arg("r"), py::arg("sigma"),
          py::arg("is_call"), py::arg("beta"));

    // Exponential Integral Method
    m.def("exponential_integral_fdm", &fdm_exponential_integral,
          "Exponential Integral Semi-Discrete FD method",
          py::arg("N"), py::arg("Smax"), py::arg("T"),
          py::arg("K"), py::arg("r"), py::arg("sigma"), py::arg("is_call"));

    // ======= Utilities =====
    m.def("normal_cdf", &normalCDF,
          "Standard normal cumulative distribution function",
          py::arg("x"));

    m.def("normal_pdf", &normalPDF,
          "Standard normal probability density function",
          py::arg("x"));

    m.def("log_message", &logMessage,
          "Log a message for debugging",
          py::arg("message"));


    // in bindings.cpp or wrapper file
    m.def("solve_fdm", &solve_fdm, py::arg("method"), py::arg("N"), py::arg("M"),
      py::arg("Smax"), py::arg("T"), py::arg("K"),
      py::arg("r"), py::arg("sigma"), py::arg("is_call"),
      py::arg("beta") = 0.5, py::arg("rannacher_smoothing") = false);

}
