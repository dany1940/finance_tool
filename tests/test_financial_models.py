import pytest
import sys
import os

# Ensure Python can find `financial_models.so`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../financial_models")))

import financial_models  # ✅ Imports the shared library correctly

# Test Parameters
S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
num_simulations = 1000000
grid_size = 100
time_steps = 100

@pytest.mark.parametrize("is_call", [True, False])
def test_black_scholes(is_call):
    """Test Black-Scholes function."""
    price = financial_models.black_scholes(S, K, T, r, sigma, is_call)
    assert isinstance(price, float) and price >= 0.0

@pytest.mark.parametrize("is_call", [True, False])
def test_monte_carlo(is_call):
    """Test Monte Carlo function."""
    price = financial_models.monte_carlo(S, K, T, r, sigma, num_simulations, is_call)
    assert isinstance(price, float) and price >= 0.0

@pytest.mark.parametrize("is_call", [True, False])
def test_finite_difference(is_call):
    """Test Finite Difference function."""
    price = financial_models.finite_difference(S, K, T, r, sigma, is_call, grid_size, time_steps)
    assert isinstance(price, float) and price >= 0

if __name__ == "__main__":
    pytest.main()
