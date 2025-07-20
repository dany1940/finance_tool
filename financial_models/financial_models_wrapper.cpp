#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for std::vector bindings

#include "finite_difference.h"
#include "utils.h"
#include "black_scholes.h"
#include "monte_carlo.h"

namespace py = pybind11;

PYBIND11_MODULE(financial_models_wrapper, m) {
    m.doc() = "Python bindings for comprehensive financial models";
    // Map std::invalid_argument to Python ValueError
    py::register_exception<std::invalid_argument>(m, "ValueError");

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

    // ======= Finite Difference Methods - scalar (interpolated) =======
    m.def("explicit_fdm", &fdm_explicit,
          "Explicit Finite Difference method with interpolation",
          py::arg("N"), py::arg("M"), py::arg("Smax"), py::arg("T"),
          py::arg("K"), py::arg("r"), py::arg("sigma"), py::arg("is_call"),
          py::arg("S0"));

    m.def("implicit_fdm", &fdm_implicit,
          "Implicit Finite Difference method with interpolation",
          py::arg("N"), py::arg("M"), py::arg("Smax"), py::arg("T"),
          py::arg("K"), py::arg("r"), py::arg("sigma"), py::arg("is_call"),
          py::arg("S0"));

    m.def("crank_nicolson_fdm", &fdm_crank_nicolson,
          "Crank-Nicolson Finite Difference method with interpolation",
          py::arg("N"), py::arg("M"), py::arg("Smax"), py::arg("T"),
          py::arg("K"), py::arg("r"), py::arg("sigma"),
          py::arg("is_call"), py::arg("rannacher_smoothing") = false,
          py::arg("S0"));

    // ======= Finite Difference Methods - vector (full price grid) =======
    m.def("explicit_fdm_vector", &fdm_explicit_vector,
          "Explicit Finite Difference method returning full price vector",
          py::arg("N"), py::arg("M"), py::arg("Smax"), py::arg("T"),
          py::arg("K"), py::arg("r"), py::arg("sigma"), py::arg("is_call"));

     m.def("implict_fdm_vector", &fdm_implicit_vector,
          "Explicit Finite Difference method returning full price vector",
          py::arg("N"), py::arg("M"), py::arg("Smax"), py::arg("T"),
          py::arg("K"), py::arg("r"), py::arg("sigma"), py::arg("is_call"));




    m.def("crank_nicolson_fdm_vector", &fdm_crank_nicolson_vector,
          "Crank-Nicolson Finite Difference method returning full price vector",
          py::arg("N"), py::arg("M"), py::arg("Smax"), py::arg("T"),
          py::arg("K"), py::arg("r"), py::arg("sigma"),
          py::arg("is_call"), py::arg("rannacher_smoothing") = false);

    // ======= Other FDM methods as before =======
    m.def("american_psor_fdm", &fdm_american_psor,
          "American Option PSOR Finite Difference method with interpolation",
          py::arg("N"), py::arg("M"), py::arg("Smax"), py::arg("T"),
          py::arg("K"), py::arg("r"), py::arg("sigma"),
          py::arg("is_call"),
          py::arg("omega") = 1.2,
          py::arg("maxIter") = 10000,
          py::arg("tol") = 1e-6,
          py::arg("S0"));

    m.def("fractional_fdm", &fdm_time_fractional,
          "Time-Fractional Finite Difference method with interpolation",
          py::arg("N"), py::arg("M"), py::arg("Smax"), py::arg("T"),
          py::arg("K"), py::arg("r"), py::arg("sigma"),
          py::arg("is_call"), py::arg("beta"),
          py::arg("S0"));

    m.def("exponential_integral_fdm", &fdm_exponential_integral,
          "Exponential Integral Semi-Discrete Finite Difference method with interpolation",
          py::arg("N"), py::arg("Smax"), py::arg("T"),
          py::arg("K"), py::arg("r"), py::arg("sigma"),
          py::arg("is_call"), py::arg("S0"));

    m.def("compact_fdm", &compact_4th_order_second_derivative,
          "Compact 4th-order Finite Difference second derivative approximation",
          py::arg("V"), py::arg("dx"));

    // ======= Utilities =======
    m.def("normal_pdf", &normalPDF,
          "Standard normal probability density function",
          py::arg("x"));

    m.def("normal_cdf", &normalCDF,
          "Standard normal cumulative distribution function",
          py::arg("x"));

    m.def("interpolate_result", &interpolate_result,
          "Interpolate option price at spot S0",
          py::arg("V"), py::arg("S"), py::arg("S0"));

    m.def("fdm_explicit_surface", &fdm_explicit_surface,
      "Explicit Finite Difference method returning 2D surface (time vs. space)",
      py::arg("N"), py::arg("M"), py::arg("Smax"), py::arg("T"),
      py::arg("K"), py::arg("r"), py::arg("sigma"), py::arg("is_call"));

    m.def("fdm_implicit_surface", &fdm_implicit_surface,
      "Implicit Finite Difference method returning 2D surface (time vs. space)",
      py::arg("N"), py::arg("M"), py::arg("Smax"), py::arg("T"),
      py::arg("K"), py::arg("r"), py::arg("sigma"), py::arg("is_call"));

   m.def("fdm_crank_nicolson_surface", &fdm_crank_nicolson_surface,
      "Crank-Nicolson Finite Difference method returning 2D surface (time vs. space)",
      py::arg("N"), py::arg("M"), py::arg("Smax"), py::arg("T"),
      py::arg("K"), py::arg("r"), py::arg("sigma"),
      py::arg("is_call"), py::arg("rannacher_smoothing") = false);
    // ======= Dispatcher =======

      // This function dispatches to the appropriate FDM method based on the string input
    m.def("fdm_compact", &fdm_compact,
          "Compact 4th-order Finite Difference method with interpolation",
          py::arg("N"), py::arg("M"), py::arg("Smax"), py::arg("T"),
          py::arg("K"), py::arg("r"), py::arg("sigma"),
          py::arg("is_call"), py::arg("S0"));

     m.def("compact_vector", &fdm_compact_vector,
      py::arg("N"), py::arg("M"), py::arg("Smax"), py::arg("T"),
      py::arg("K"), py::arg("r"), py::arg("sigma"), py::arg("isCall"), py::arg("S0"));

     m.def("compact_surface", &fdm_compact_surface,
      py::arg("N"), py::arg("M"), py::arg("Smax"), py::arg("T"),
      py::arg("K"), py::arg("r"), py::arg("sigma"), py::arg("isCall"));

    m.def("binomial_tree", &binomial_tree,
      "Binomial tree method returning scalar price",
      py::arg("N"),  py::arg("T"), py::arg("K"),
      py::arg("r"), py::arg("sigma"), py::arg("is_call"),
      py::arg("is_american"), py::arg("S0"));

    m.def("binomial_tree_vector", &binomial_tree_vector,
            "Binomial tree method returning full price vector",
            py::arg("N"), py::arg("T"), py::arg("K"),
            py::arg("r"), py::arg("sigma"), py::arg("is_call"),
            py::arg("is_american"), py::arg("S0"), py::arg("Smax") = 0.0);  // Smax dummy

    m.def("fdm_american_psor_vector", &fdm_american_psor_vector,
        "Returns surface (price grid) for American option via PSOR method",
        py::arg("N"), py::arg("M"), py::arg("Smax"), py::arg("T"), py::arg("K"),
        py::arg("r"), py::arg("sigma"), py::arg("isCall"),
        py::arg("omega"), py::arg("maxIter"), py::arg("tol"));

    m.def("fdm_exponential_integral_vector", &fdm_exponential_integral_vector,
        "Returns surface (price grid) using exponential integral FDM scheme",
        py::arg("N"), py::arg("Smax"), py::arg("T"), py::arg("K"),
        py::arg("r"), py::arg("sigma"), py::arg("isCall"));


// === Binomial Tree: Full surface ===
      m.def("binomial_tree_surface", &binomial_tree_surface,
            "Compute full binomial tree surface (time steps and nodes)",
            py::arg("N"), py::arg("T"), py::arg("K"), py::arg("r"),
            py::arg("sigma"), py::arg("is_call"), py::arg("is_american"),
            py::arg("S0"));

      m.def("american_psor_surface", &american_psor_surface,
            "Generate price surface using American PSOR method",
            py::arg("N"), py::arg("M"), py::arg("Smax"), py::arg("T"),
            py::arg("K"), py::arg("r"), py::arg("sigma"),
            py::arg("is_call"), py::arg("omega"),
            py::arg("max_iter"), py::arg("tol"));

       m.def("american_psor_vector", &american_psor_vector,
        py::arg("N"),
        py::arg("M"),
        py::arg("Smax"),
        py::arg("T"),
        py::arg("K"),
        py::arg("r"),
        py::arg("sigma"),
        py::arg("is_call"),
        py::arg("omega"),
        py::arg("maxIter"),
        py::arg("tol"),
        "American option pricing using PSOR (vector output)");

      m.def("exponential_integral_surface", &exponential_integral_surface,
            "Generate price surface using Exponential Integral method",
            py::arg("N"), py::arg("M"), py::arg("Smax"), py::arg("T"),
            py::arg("K"), py::arg("r"), py::arg("sigma"),
            py::arg("is_call"));
        m.def("fractional_vector", &fdm_time_fractional,
          py::arg("N"), py::arg("M"), py::arg("Smax"), py::arg("T"), py::arg("K"),
          py::arg("r"), py::arg("sigma"), py::arg("isCall"), py::arg("beta"), py::arg("S0"));

       m.def("fractional_surface", &fdm_time_fractional_surface,
          py::arg("N"), py::arg("M"), py::arg("Smax"), py::arg("T"), py::arg("K"),
          py::arg("r"), py::arg("sigma"), py::arg("isCall"), py::arg("beta"));
    m.def("solve_fdm", &solve_fdm,
          "Dispatcher for FDM methods returning interpolated price at S0",
          py::arg("method"),
          py::arg("N"),
          py::arg("M"),
          py::arg("Smax"),
          py::arg("T"),
          py::arg("K"),
          py::arg("r"),
          py::arg("sigma"),
          py::arg("is_call"),
          py::arg("beta") = 0.5,
          py::arg("rannacher_smoothing") = false,
          py::arg("S0"));
}
