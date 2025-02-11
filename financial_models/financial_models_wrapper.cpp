#include <pybind11/pybind11.h>
#include "black_scholes.h"
#include "monte_carlo.h"
#include "finite_difference.h"

namespace py = pybind11;

PYBIND11_MODULE(financial_models, m) {
    m.def("black_scholes", &blackScholes, "Black-Scholes Analytical Solution");
    m.def("monte_carlo", &monteCarloBlackScholes, "Monte Carlo Black-Scholes Simulation");
    m.def("finite_difference", &finiteDifferenceBlackScholes, "Finite Difference Black-Scholes Solver");
}
