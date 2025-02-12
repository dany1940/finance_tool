#include <pybind11/pybind11.h>
#include "black_scholes.h"
#include "monte_carlo.h"
#include "finite_difference.h"
#include "utils.h"

namespace py = pybind11;

PYBIND11_MODULE(financial_models_wrapper, m) {
    m.doc() = "Python bindings for financial models";

    // Expose Black-Scholes function
    m.def("black_scholes", &blackScholes,
          "Calculate the Black-Scholes option pricing model",
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"), py::arg("is_call"));

    // Expose Monte Carlo simulation for Black-Scholes option pricing
    m.def("monte_carlo", &monteCarloBlackScholes,
          "Monte Carlo simulation for Black-Scholes option pricing",
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"), py::arg("is_call"), py::arg("num_simulations"));

    // Expose Finite Difference method for Black-Scholes option pricing
    m.def("finite_difference", &finiteDifferenceBlackScholes,
          "Finite Difference method for Black-Scholes option pricing",
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"), py::arg("is_call"), py::arg("grid_size"), py::arg("time_steps"));

    // Expose normal CDF and PDF functions (useful in Black-Scholes, Monte Carlo, etc.)
    m.def("normal_cdf", &normalCDF, "Cumulative distribution function of standard normal distribution", py::arg("x"));
    m.def("normal_pdf", &normalPDF, "Probability density function of standard normal distribution", py::arg("x"));

    // Expose utility functions if needed
    m.def("log_message", &logMessage, "Log a message", py::arg("message"));
}
