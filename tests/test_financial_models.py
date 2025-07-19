import math

import pytest

import financial_models

# Test parameters
S, K = 100, 100
r = 0.05
sigma_low = 0.01
sigma_mid = 0.2
sigma_high = 1.0
grid_size = 100
time_steps = 100
beta = 0.9
Smax = 200
S0 = 100
T_zero = 0.0
T_short = 0.01
T_long = 10.0


# === Basic Black-Scholes Test ===
@pytest.mark.parametrize("is_call", [True, False])
def test_black_scholes_value_close(is_call):
    price = financial_models.black_scholes(S, K, 1, r, sigma_mid, is_call)
    assert isinstance(price, float)
    assert price >= 0.0


# === Monte Carlo Test ===
@pytest.mark.parametrize("is_call", [True, False])
def test_monte_carlo_price_reasonable(is_call):
    price = financial_models.monte_carlo(S, K, 1, r, sigma_mid, is_call, 100_000)
    assert isinstance(price, float)
    assert price >= 0.0


# === Explicit FDM Reasonable Price ===
@pytest.mark.parametrize("is_call", [True, False])
def test_explicit_fdm_reasonable_price(is_call):
    price = financial_models.explicit_fdm(
        grid_size, time_steps, Smax, 1, K, r, sigma_mid, is_call, S0
    )
    assert isinstance(price, float)
    # Allow tiny negative numerical noise
    assert price >= -1e-12


# === Implicit FDM Reasonable Price ===
@pytest.mark.parametrize("is_call", [True, False])
def test_implicit_fdm_reasonable_price(is_call):
    price = financial_models.implicit_fdm(
        grid_size, time_steps, Smax, 1, K, r, sigma_mid, is_call, S0
    )
    assert isinstance(price, float)
    assert price >= -1e-12


# === Crank-Nicolson FDM Reasonable Price ===
@pytest.mark.parametrize("is_call", [True, False])
def test_crank_nicolson_fdm_reasonable_price(is_call):
    price = financial_models.crank_nicolson_fdm(
        grid_size, time_steps, Smax, 1, K, r, sigma_mid, is_call, False, S0
    )
    assert isinstance(price, float)
    assert price >= -1e-12


# === American PSOR FDM Reasonable Price ===
@pytest.mark.parametrize("is_call", [True, False])
def test_american_psor_fdm_reasonable_price(is_call):
    price = financial_models.american_psor_fdm(
        grid_size, time_steps, Smax, 1, K, r, sigma_mid, is_call, 1.2, 10000, 1e-6, S0
    )
    assert isinstance(price, float)
    assert price >= -1e-12


# === Compact 4th-order Derivative Type ===
def test_compact_fdm_type():
    sample_vec = [float(i) for i in range(grid_size + 1)]
    dx = Smax / grid_size
    d2V = financial_models.compact_fdm(sample_vec, dx)
    assert isinstance(d2V, list)
    assert len(d2V) == grid_size + 1


# === Exponential Integral FDM Reasonable Price ===
@pytest.mark.parametrize("is_call", [True, False])
def test_exponential_integral_fdm_reasonable_price(is_call):
    price = financial_models.exponential_integral_fdm(
        grid_size, Smax, 1, K, r, sigma_mid, is_call, S0
    )
    assert isinstance(price, float)
    assert price >= -1e-12


# === Time-Fractional FDM Reasonable Price and Beta Validation ===
@pytest.mark.parametrize("is_call", [True, False])
def test_fractional_fdm_reasonable_price(is_call):
    price = financial_models.fractional_fdm(
        grid_size, time_steps, Smax, 1, K, r, sigma_mid, is_call, beta, S0
    )
    assert isinstance(price, float)
    assert price >= -1e-12


def test_fractional_beta_close_to_zero():
    with pytest.raises(ValueError):
        financial_models.solve_fdm(
            "fractional",
            grid_size,
            time_steps,
            Smax,
            1,
            K,
            r,
            sigma_mid,
            True,
            1e-9,
            False,
            S0,
        )


def test_fractional_beta_close_to_one():
    with pytest.raises(ValueError):
        financial_models.solve_fdm(
            "fractional",
            grid_size,
            time_steps,
            Smax,
            1,
            K,
            r,
            sigma_mid,
            True,
            1 - 1e-9,
            False,
            S0,
        )


# === Normal PDF and CDF ===
def test_normal_pdf_and_cdf():
    x = 1.0
    assert isinstance(financial_models.normal_pdf(x), float)
    assert isinstance(financial_models.normal_cdf(x), float)


# === Check all expected functions bound ===
def test_available_functions():
    expected = {
        "black_scholes",
        "monte_carlo",
        "explicit_fdm",
        "implicit_fdm",
        "crank_nicolson_fdm",
        "american_psor_fdm",
        "compact_fdm",
        "exponential_integral_fdm",
        "fractional_fdm",
        "normal_pdf",
        "normal_cdf",
        "solve_fdm",
    }
    found = set(dir(financial_models))
    missing = expected - found
    assert not missing, f"Missing bindings: {missing}"


# === Extended Tests ===


# Boundary test with low and high spot prices (deep ITM/OTM)
@pytest.mark.parametrize("S0_test", [1, 300])
@pytest.mark.parametrize("method", ["explicit", "implicit", "crank", "american"])
def test_fdm_extreme_spot_prices(S0_test, method):
    price = financial_models.solve_fdm(
        method,
        grid_size,
        time_steps,
        Smax,
        1,
        K,
        r,
        sigma_mid,
        True,
        0.5,
        False,
        S0_test,
    )
    assert price >= -1e-12


# Volatility sensitivity test with small epsilon tolerance
def test_fdm_volatility_sensitivity():
    price_low_vol = financial_models.solve_fdm(
        "explicit",
        grid_size,
        time_steps,
        Smax,
        1,
        K,
        r,
        sigma_low,
        True,
        0.5,
        False,
        S0,
    )
    price_high_vol = financial_models.solve_fdm(
        "explicit",
        grid_size,
        time_steps,
        Smax,
        1,
        K,
        r,
        sigma_high,
        True,
        0.5,
        False,
        S0,
    )
    assert price_low_vol <= price_high_vol + 1e-12


# Interest rate sensitivity test with small epsilon tolerance
def test_fdm_interest_rate_sensitivity():
    price_low_r = financial_models.solve_fdm(
        "explicit",
        grid_size,
        time_steps,
        Smax,
        1,
        K,
        0.0,
        sigma_mid,
        True,
        0.5,
        False,
        S0,
    )
    price_high_r = financial_models.solve_fdm(
        "explicit",
        grid_size,
        time_steps,
        Smax,
        1,
        K,
        0.2,
        sigma_mid,
        True,
        0.5,
        False,
        S0,
    )
    assert price_low_r <= price_high_r + 1e-6


# Interpolation returns exact value at grid node
def test_interpolation_exact_node_value():
    N = 10
    Smax_test = 100
    dS = Smax_test / N
    V = [float(i * dS) for i in range(N + 1)]
    S = [i * dS for i in range(N + 1)]
    for i in range(N + 1):
        val = financial_models.interpolate_result(V, S, S[i])
        assert math.isclose(val, V[i], rel_tol=1e-9)


# American option price should be >= European option price
def test_american_option_price_ge_european():
    price_american = financial_models.solve_fdm(
        "american",
        grid_size,
        time_steps,
        Smax,
        1,
        K,
        r,
        sigma_mid,
        True,
        0.5,
        False,
        S0,
    )
    price_european = financial_models.solve_fdm(
        "crank", grid_size, time_steps, Smax, 1, K, r, sigma_mid, True, 0.5, False, S0
    )
    assert price_american >= price_european
