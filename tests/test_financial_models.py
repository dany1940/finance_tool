import pytest
import financial_models # Ensure it correctly imports the wrapped functions

# Test Parameters
S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
num_simulations = 1000000
grid_size = 100
time_steps = 100

# --- BLACK-SCHOLES TESTS ---
@pytest.mark.parametrize("is_call", [True, False])
def test_black_scholes(is_call):
    """Test Black-Scholes function from financial_models."""
    price = financial_models.black_scholes(S, K, T, r, sigma, is_call)
    assert isinstance(price, float) and price >= 0.0, f"Invalid Black-Scholes price: {price}"

# --- MONTE CARLO TESTS ---
@pytest.mark.parametrize("is_call", [True, False])
def test_monte_carlo(is_call):
    """Test Monte Carlo function from financial_models."""
    price = financial_models.monte_carlo(S, K, T, r, sigma, num_simulations, is_call)
    assert isinstance(price, float) and price >= 0.0, f"Invalid Monte Carlo price: {price}"

# --- FINITE DIFFERENCE TESTS ---
@pytest.mark.parametrize("is_call", [True, False])
def test_finite_difference(is_call):
    """Test Finite Difference function from financial_models."""
    price = financial_models.finite_difference(S, K, T, r, sigma, is_call, grid_size, time_steps)
    assert isinstance(price, float) and price >= 0.0, f"Invalid Finite Difference price: {price}"

# --- EXTRA TEST: CHECK MODULE BINDING ---
def test_financial_models_functions():
    """Ensure financial_models exposes all expected functions."""
    expected_functions = {"black_scholes", "monte_carlo", "finite_difference"}
    available_functions = set(dir(financial_models))

    missing = expected_functions - available_functions
    assert not missing, f"Missing expected functions in financial_models: {missing}"
