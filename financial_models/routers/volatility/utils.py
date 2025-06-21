from typing import List
import yfinance as yf
import numpy as np
import financial_models_wrapper as fm
from crud.stock_analysis_models import CommonParams

# === Historical Volatility ===
def calculate_historical_volatility(ticker: str, start_date: str, end_date: str) -> float:
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError("No data retrieved for the given ticker and date range.")
    returns = np.log(data['Adj Close'] / data['Adj Close'].shift(1)).dropna()
    return np.std(returns) * np.sqrt(252)


# === Interest Rate Resolver ===
def resolve_rate(params: CommonParams) -> float:
    if getattr(params, 'rate_source', "User-defined") == "User-defined":
        return params.r

    # Define yield curve: maturities in years mapped to Yahoo symbols
    yield_curve = {
        0.25: "^IRX",   # 3-month
        10.0: "^TNX",   # 10-year
        30.0: "^TYX",   # 30-year
    }

    rates = {}
    for maturity, symbol in yield_curve.items():
        try:
            df = yf.download(symbol, period="5d", interval="1d", progress=False)
            if df.empty:
                continue
            rate_percent = df["Adj Close"].dropna().iloc[-1]
            rates[maturity] = rate_percent / 100.0  # Convert to decimal
        except Exception:
            continue

    if not rates:
        return 0.04  # Fallback default rate

    maturities = sorted(rates.keys())
    values = [rates[m] for m in maturities]
    return float(np.interp(params.T, maturities, values))

# === Implied Volatility using Brent/Bisection method with C++ Black-Scholes ===
def calculate_implied_volatility(
    S: float, K: float, T: float, r: float,
    market_price: float, is_call: bool,
    tol: float = 1e-6, max_iter: int = 100
) -> float:
    sigma_low = 0.0001
    sigma_high = 5.0

    for _ in range(max_iter):
        sigma_mid = (sigma_low + sigma_high) / 2.0
        bs_price = fm.black_scholes(S, K, T, r, sigma_mid, is_call)
        if abs(bs_price - market_price) < tol:
            return sigma_mid
        elif bs_price > market_price:
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid

    return sigma_mid  # best guess if not converged


# === Value-at-Risk (VaR) via Bootstrapping ===
def bootstrap_var(
    returns: List[float],
    num_samples: int = 1000,
    confidence_level: float = 0.95
) -> float:
    n = len(returns)
    if n == 0:
        raise ValueError("Returns list cannot be empty.")
    simulated_vars = [
        np.percentile(
            np.random.choice(returns, size=n, replace=True),
            (1 - confidence_level) * 100
        )
        for _ in range(num_samples)
    ]
    return float(np.mean(simulated_vars))


# === Dispatcher for Volatility Source ===
def resolve_sigma(params) -> float:
    vol_source = getattr(params, 'vol_source', 'user').lower()

    if vol_source in {"user", "user-defined"}:
        return params.sigma

    elif vol_source == "historical":
        if not hasattr(params, 'ticker') or not hasattr(params, 'start_date') or not hasattr(params, 'end_date'):
            raise ValueError("Missing ticker or date range for historical volatility.")
        return calculate_historical_volatility(
            ticker=params.ticker,
            start_date=params.start_date,
            end_date=params.end_date
        )

    elif vol_source == "implied":
        if not hasattr(params, 'market_price'):
            raise ValueError("Missing market_price for implied volatility.")
        return calculate_implied_volatility(
            S=params.S0,
            K=params.K,
            T=params.T,
            r=params.r,
            market_price=params.market_price,
            is_call=params.is_call
        )

    raise ValueError(f"Unknown volatility source: {params.vol_source}")

def resolve_s0(params: CommonParams) -> float:
    if params.vol_source.lower() == "implied":
        if params.S0 is None:
            raise ValueError("S0 is required for implied volatility calculation.")
        return params.S0

    # For other cases: return S0 if available, else fallback logic
    if params.S0 is not None:
        return params.S0

    if params.ticker:
        try:
            df = yf.download(params.ticker, period="1d", interval="1m", progress=False)
            if not df.empty:
                return float(df["Adj Close"].dropna().iloc[-1])
        except Exception:
            pass

    return params.K  # fallback to strike price
