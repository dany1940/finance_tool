import pytest
import financial_models

S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
num_simulations = 100_000
grid_size = 100
time_steps = 100
beta = 0.9  # for fractional FD (Caputo derivative)

# === Black-Scholes ===
@pytest.mark.parametrize("is_call", [True, False])
def test_black_scholes(is_call):
    price = financial_models.black_scholes(S, K, T, r, sigma, is_call)
    assert isinstance(price, float) and price >= 0.0

# === Monte Carlo ===
@pytest.mark.parametrize("is_call", [True, False])
def test_monte_carlo(is_call):
    price = financial_models.monte_carlo(S, K, T, r, sigma, is_call, num_simulations)
    assert isinstance(price, float) and price >= 0.0

# === Explicit FDM ===
@pytest.mark.parametrize("is_call", [True, False])
def test_explicit_fdm(is_call):
    result = financial_models.explicit_fdm(grid_size, time_steps, S, T, K, r, sigma, is_call)
    assert isinstance(result, list) and all(isinstance(v, float) for v in result)

# === Implicit FDM ===
@pytest.mark.parametrize("is_call", [True, False])
def test_implicit_fdm(is_call):
    result = financial_models.implicit_fdm(grid_size, time_steps, S, T, K, r, sigma, is_call)
    assert isinstance(result, list) and all(isinstance(v, float) for v in result)

# === Crank-Nicolson FDM ===
@pytest.mark.parametrize("is_call", [True, False])
def test_crank_nicolson_fdm(is_call):
    result = financial_models.crank_nicolson_fdm(grid_size, time_steps, S, T, K, r, sigma, is_call)
    assert isinstance(result, list) and all(isinstance(v, float) for v in result)

# === American PSOR FDM ===
@pytest.mark.parametrize("is_call", [True, False])
def test_american_psor_fdm(is_call):
    result = financial_models.american_psor_fdm(grid_size, time_steps, S, T, K, r, sigma, is_call)
    assert isinstance(result, list) and all(isinstance(v, float) for v in result)

# === Compact 4th-order Derivative ===
def test_compact_fdm():
    sample_vec = [float(i) for i in range(grid_size)]
    dx = S / grid_size
    result = financial_models.compact_fdm(sample_vec, dx)
    assert isinstance(result, list) and all(isinstance(v, float) for v in result)

# === Exponential Integral FDM ===
@pytest.mark.parametrize("is_call", [True, False])
def test_exponential_integral_fdm(is_call):
    result = financial_models.exponential_integral_fdm(grid_size, S, T, K, r, sigma, is_call)
    assert isinstance(result, list) and all(isinstance(v, float) for v in result)

# === Time-Fractional FDM ===
@pytest.mark.parametrize("is_call", [True, False])
def test_fractional_fdm(is_call):
    result = financial_models.fractional_fdm(grid_size, time_steps, S, T, K, r, sigma, is_call, beta)
    assert isinstance(result, list) and all(isinstance(v, float) for v in result)

# === Normal Distribution Utilities ===
def test_normal_pdf_and_cdf():
    x = 1.0
    assert isinstance(financial_models.normal_pdf(x), float)
    assert isinstance(financial_models.normal_cdf(x), float)

# === Sanity Check: all functions are bound ===
def test_available_functions():
    expected = {
        "black_scholes", "monte_carlo",
        "explicit_fdm", "implicit_fdm", "crank_nicolson_fdm",
        "american_psor_fdm", "compact_fdm", "exponential_integral_fdm",
        "fractional_fdm", "normal_pdf", "normal_cdf"
    }
    found = set(dir(financial_models))
    missing = expected - found
    assert not missing, f"Missing bindings: {missing}"
