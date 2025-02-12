import pytest
import sys
import os

import financial_models


# Ensure Python can find `financial_models.so`
sys.path.append("/Users/FlorinDumitrascu/Desktop/repo/finance/finance-tool/financial_models/financial_models_wrapper.so")

# Test Parameters
S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
num_simulations = 1000000
grid_size = 100
time_steps = 100

@pytest.mark.parametrize("is_call", [True, False])
def test_black_scholes(is_call):
    """Test Black-Scholes function."""
    pass

@pytest.mark.parametrize("is_call", [True, False])
def test_monte_carlo(is_call):
    """Test Monte Carlo function."""
    pass

@pytest.mark.parametrize("is_call", [True, False])
def test_finite_difference(is_call):
    """Test Finite Difference function."""
    pass

