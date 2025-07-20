import logging
import math

import financial_models_wrapper as fm
from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool
from crud.stock_analysis_models import (
    AmericanParams,
    BinomialSurfaceParams,
    BlackScholesParams,
    CommonParams,
    DispatcherParams,
    FDMResult,
    FractionalParams,
    PSORSurfaceParams,
    ResponseBlackscholes,
    ResultItem,
    SurfaceParams,
    SurfaceResult,
    VectorResult,
)
from routers.volatility.utils import resolve_rate, resolve_sigma

router = APIRouter(prefix="/fdm", tags=["FDM Processing"])
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def wrap_result(result: List[float]) -> FDMResult:
    """
    Wraps the result list into a FDMResult object.
    Converts the result list into ResultItem objects with indices and values.
    If the result is empty, logs a warning and returns an empty FDMResult.
    """
    if not result:
        logger.warning("⚠️ FDM solver returned an empty result list")
        return FDMResult(result=[], final_price=0.0)
    return FDMResult(
        result=[ResultItem(index=i, value=float(v)) for i, v in enumerate(result)],
        final_price=float(result[-1]),
    )


# === Scalar-returning FDM methods: return only final price and empty result list ===


@router.post("/explicit", response_model=FDMResult)
async def run_explicit(params: CommonParams) -> FDMResult:
    """
    Run the explicit finite difference method for option pricing.
    This method computes the option price using the explicit finite difference method
    and returns the final price along with an empty result list.
    """
    logger.info("Running explicit_fdm with parameters: %s", params)
    try:
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        price = await run_in_threadpool(
            fm.explicit_fdm,
            params.N,
            params.M,
            params.Smax,
            params.T,
            params.K,
            rate,
            sigma,
            params.is_call,
            params.S0,
        )
        logger.info(f"✅ Explicit FDM computed price: {price}")
        return FDMResult(result=[], final_price=price)
    except Exception as e:
        logger.exception("❌ Error in explicit_fdm")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/implicit", response_model=FDMResult)
async def run_implicit(params: CommonParams) -> FDMResult:
    """
    Run the implicit finite difference method for option pricing.
    This method computes the option price using the implicit finite difference method
    and returns the final price along with an empty result list.
    """
    logger.info("Running implicit_fdm with parameters: %s", params)
    try:
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        price = await run_in_threadpool(
            fm.implicit_fdm,
            params.N,
            params.M,
            params.Smax,
            params.T,
            params.K,
            rate,
            sigma,
            params.is_call,
            params.S0,
        )
        return FDMResult(result=[], final_price=price)
    except Exception as e:
        logger.exception("❌ Error in implicit_fdm")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/crank", response_model=FDMResult)
async def run_crank(params: CommonParams) -> FDMResult:
    """
    Run the Crank-Nicolson finite difference method for option pricing.
    This method computes the option price using the Crank-Nicolson finite difference method
    and returns the final price along with an empty result list.
    """
    try:
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        price = await run_in_threadpool(
            fm.crank_nicolson_fdm,
            params.N,
            params.M,
            params.Smax,
            params.T,
            params.K,
            rate,
            sigma,
            params.is_call,
            False,  # rannacher smoothing default false
            params.S0,
        )
        return FDMResult(result=[], final_price=price)
    except Exception as e:
        logger.exception("❌ Error in crank_nicolson_fdm")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/psor", response_model=FDMResult)
async def run_american(params: AmericanParams) -> FDMResult:
    """
    Run the PSOR finite difference method for American option pricing.
    This method computes the option price using the PSOR finite difference method
    and returns the final price along with an empty result list.
    """
    try:
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        price = await run_in_threadpool(
            fm.american_psor_fdm,
            params.N,
            params.M,
            params.Smax,
            params.T,
            params.K,
            rate,
            sigma,
            params.is_call,
            params.omega,
            params.maxIter,
            params.tol,
            params.S0,
        )
        return FDMResult(result=[], final_price=price)
    except Exception as e:
        logger.exception("❌ Error in american_psor_fdm")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fractional", response_model=FDMResult)
async def run_fractional(params: FractionalParams) -> FDMResult:
    """
    Run the fractional finite difference method for option pricing.
    This method computes the option price using the fractional finite difference method
    and returns the final price along with an empty result list.
    """
    logger.info("Running fractional_fdm with parameters: %s", params)
    try:
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        price = await run_in_threadpool(
            fm.fractional_fdm,
            params.N,
            params.M,
            params.Smax,
            params.T,
            params.K,
            rate,
            sigma,
            params.is_call,
            params.beta,
            params.S0,

        )
        return FDMResult(result=[], final_price=price)
    except Exception as e:
        logger.exception("❌ Error in fractional_fdm")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/exponential", response_model=FDMResult)
async def run_exponential(params: CommonParams) -> FDMResult:
    """
    Run the exponential integral finite difference method for option pricing.
    This method computes the option price using the exponential integral finite difference method
    and returns the final price along with an empty result list.
    """
    logger.info("Running exponential_integral_fdm with parameters: %s", params)
    try:
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        price = await run_in_threadpool(
            fm.exponential_integral_fdm,
            params.N,
            params.Smax,
            params.T,
            params.K,
            rate,
            sigma,
            params.is_call,
            params.S0,
        )
        return FDMResult(result=[], final_price=price)
    except Exception as e:
        logger.exception("❌ Error in exponential_integral_fdm")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/psor_surface", response_model=SurfaceResult)
async def fdm_psor_surface(params: PSORSurfaceParams) -> SurfaceResult:
    """
    Run the PSOR finite difference method for American option pricing surface.
    This method computes the option price surface using the PSOR finite difference method
    and returns the price surface along with the grid axes.
    """
    try:
        logger.info("Running PSOR FDM surface...")

        # Ensure non-zero positive divisors
        N = max(1, params.N)
        M = max(1, params.M)
        T = max(params.T, 1e-8)
        Smax = max(params.Smax, 1e-8)

        raw = fm.american_psor_surface(
            N,
            M,
            Smax,
            T,
            params.K,
            params.r,
            params.sigma,
            params.is_call,
            params.omega,
            params.maxIter,
            params.tol,
        )

        # === Safety: Convert surface to 2D list ===
        if isinstance(raw, np.ndarray):
            surface = raw.tolist()
        elif isinstance(raw[0], (int, float)):
            try:
                surface = [raw[i * (N + 1) : (i + 1) * (N + 1)] for i in range(M + 1)]
            except Exception as e:
                raise ValueError(f"Failed to reshape flat vector into surface: {e}")
        elif isinstance(raw[0], list):
            surface = raw
        else:
            raise ValueError(
                "Unsupported surface data format returned from fm.american_psor_surface"
            )

        # === Grid Axes (safe from div-zero) ===
        S_grid = [i * (Smax / N) for i in range(N + 1)]
        t_grid = [i * (T / M) for i in range(M + 1)]

        return {"S_grid": S_grid, "t_grid": t_grid, "price_surface": surface}

    except Exception as e:
        logger.error(f"❌ PSOR surface error: {e}")
        raise HTTPException(status_code=500, detail=f"PSOR surface error: {str(e)}")


# === Exponential Integral Surface Endpoint ===
@router.post("/exponential_surface", response_model=SurfaceResult)
async def fdm_exponential_surface(params: SurfaceParams) -> SurfaceResult:
    """
    Run the exponential integral finite difference method for option pricing surface.
    This method computes the option price surface using the exponential integral finite difference method
    and returns the price surface along with the grid axes.
    """
    try:
        logger.info("Running exponential integral FDM surface...")
        surface = fm.fdm_exponential_integral_vector(
            params.N,
            params.Smax,
            params.T,
            params.K,
            params.r,
            params.sigma,
            params.is_call,
        )
        S_grid = [i * (params.Smax / params.N) for i in range(params.N + 1)]
        t_grid = [i * (params.T / params.M) for i in range(params.M + 1)]
        return {"S_grid": S_grid, "t_grid": t_grid, "price_surface": surface}
    except Exception as e:
        logger.error(f"Exponential Integral FDM surface error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compact", response_model=FDMResult)
async def run_compact(params: CommonParams) -> FDMResult:
    """
    Run the compact finite difference method for option pricing.
    This method computes the option price using the compact finite difference method
    and returns the final price along with an empty result list.
    """
    try:
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        price = await run_in_threadpool(
            fm.compact_fdm,  # must match signature in your C++/pybind wrapper
            params.N,
            params.M,
            params.Smax,
            params.T,
            params.K,
            rate,
            sigma,
            params.is_call,
            params.S0,
        )
        return FDMResult(result=[], final_price=price)
    except Exception as e:
        logger.exception("❌ Error in compact_fdm")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dispatcher", response_model=FDMResult)
async def run_dispatcher(params: DispatcherParams) -> FDMResult:
    """
    Dispatcher for finite difference methods.
    This endpoint selects the appropriate finite difference method based on the input parameters
    and returns the final price along with an empty result list.
    """
    try:
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        price = await run_in_threadpool(
            fm.solve_fdm,
            params.method,
            params.N,
            params.M,
            params.Smax,
            params.T,
            params.K,
            rate,
            sigma,
            params.is_call,
        )
        return FDMResult(result=[], final_price=price)
    except Exception as e:
        logger.exception("❌ Error in dispatcher")
        raise HTTPException(status_code=500, detail=str(e))


# === Black-Scholes ===


@router.post("/black_scholes", response_model=ResponseBlackscholes)
async def run_black_scholes(params: BlackScholesParams) -> ResponseBlackscholes:
    """
    Run the Black-Scholes model for option pricing.
    This method computes the option price using the Black-Scholes formula
    and returns the price in a structured response.
    """
    try:
        price = await run_in_threadpool(
            fm.black_scholes,
            params.S,
            params.K,
            params.T,
            params.r,
            params.sigma,
            params.is_call,
        )
        return {"price": price}
    except Exception as e:
        logger.exception("❌ Error in black_scholes")
        raise HTTPException(status_code=500, detail=str(e))


# === Helper to wrap vector results with price grid ===
def wrap_vector_result(vector: List[float], Smax: float, N: int) -> VectorResult:
    """
    Wraps the vector result into a VectorResult object.
    Converts the vector into a price grid based on Smax and N.
    If the vector is empty, returns an empty grid and a final price of 0.0.
    """
    dS = Smax / N
    S_grid = [i * dS for i in range(N + 1)]
    final_price = vector[-1] if vector else 0.0
    return VectorResult(S_grid=S_grid, prices=vector, final_price=final_price)


# === Explicit FDM Vector Endpoint ===
@router.post("/explicit_vector", response_model=VectorResult)
async def explicit_vector(params: CommonParams) -> VectorResult:
    """
    Run the explicit finite difference method for option pricing vector.
    This method computes the option price vector using the explicit finite difference method
    and returns the price grid along with the final price.
    """
    try:
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        vector = await run_in_threadpool(
            fm.explicit_fdm_vector,
            params.N,
            params.M,
            params.Smax,
            params.T,
            params.K,
            rate,
            sigma,
            params.is_call,
        )
        return wrap_vector_result(vector, params.Smax, params.N)
    except Exception as e:
        logger.exception("Error in explicit_fdm_vector")
        raise HTTPException(status_code=500, detail=str(e))


# === Implicit FDM Vector Endpoint === not impelmented
@router.post("/implicit_vector", response_model=VectorResult)
async def implicit_vector(params: CommonParams) -> VectorResult:
    """
    Run the implicit finite difference method for option pricing vector.
    This method computes the option price vector using the implicit finite difference method
    and returns the price grid along with the final price.
    """
    logger.info("Running implicit_fdm_vector with parameters: %s", params)
    try:
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        vector = await run_in_threadpool(
            fm.implict_fdm_vector,
            params.N,
            params.M,
            params.Smax,
            params.T,
            params.K,
            rate,
            sigma,
            params.is_call,
        )
        return wrap_vector_result(vector, params.Smax, params.N)
    except Exception as e:
        logger.exception("Error in implicit_fdm_vector")
        raise HTTPException(status_code=500, detail=str(e))


# === Crank-Nicolson FDM Vector Endpoint ===
@router.post("/crank_vector", response_model=VectorResult)
async def crank_vector(params: CommonParams) -> VectorResult:
    """
    Run the Crank-Nicolson finite difference method for option pricing vector.
    This method computes the option price vector using the Crank-Nicolson finite difference method
    and returns the price grid along with the final price.
    """
    logger.info("Running crank_nicolson_fdm_vector with parameters: %s", params)
    try:
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        vector = await run_in_threadpool(
            fm.crank_nicolson_fdm_vector,
            params.N,
            params.M,
            params.Smax,
            params.T,
            params.K,
            rate,
            sigma,
            params.is_call,
            False,  # rannacher smoothing default false
        )
        return wrap_vector_result(vector, params.Smax, params.N)
    except Exception as e:
        logger.exception("Error in crank_nicolson_fdm_vector")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explicit_surface", response_model=SurfaceResult)
async def fdm_explicit_surface(params: SurfaceParams) -> SurfaceResult:
    """
    Run the explicit finite difference method for option pricing surface.
    This method computes the option price surface using the explicit finite difference method
    and returns the price surface along with the grid axes.
    """
    try:
        if params.N <= 0 or params.M <= 0:
            raise ValueError("N and M must be positive integers.")
        if params.Smax <= 0 or params.T <= 0:
            raise ValueError("Smax and T must be positive.")
        if params.K <= 0 or params.sigma < 0:
            raise ValueError("Strike and volatility must be valid.")

        logger.info("Running explicit FDM surface...")
        surface = fm.fdm_explicit_surface(
            params.N,
            params.M,
            params.Smax,
            params.T,
            params.K,
            params.r,
            params.sigma,
            params.is_call,
        )

        dS = params.Smax / params.N
        dt = params.T / params.M

        S_grid = [i * dS for i in range(params.N + 1)]
        t_grid = [i * dt for i in range(params.M + 1)]

        # Sanitize surface
        sanitized = [[v if math.isfinite(v) else 0.0 for v in row] for row in surface]

        return {"S_grid": S_grid, "t_grid": t_grid, "price_surface": sanitized}
    except ZeroDivisionError:
        logger.error("Division by zero in explicit surface calculation.")
        raise HTTPException(status_code=400, detail="Division by zero.")
    except ValueError as ve:
        logger.error(f"Invalid explicit surface input: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Explicit FDM surface error: {e}")
        raise HTTPException(
            status_code=500, detail="Unexpected explicit surface error."
        )


@router.post("/implicit_surface", response_model=SurfaceResult)
async def fdm_implicit_surface(params: SurfaceParams) -> SurfaceResult:
    """
    Run the implicit finite difference method for option pricing surface.
    This method computes the option price surface using the implicit finite difference method
    and returns the price surface along with the grid axes.
    """
    logger.info("Running implicit_fdm_surface with parameters: %s", params)
    try:
        if params.N <= 0 or params.M <= 0:
            raise ValueError("N and M must be positive integers.")
        if params.Smax <= 0 or params.T <= 0:
            raise ValueError("Smax and T must be positive.")
        if params.K <= 0 or params.sigma < 0:
            raise ValueError("Strike and volatility must be valid.")

        logger.info("Running implicit FDM surface...")
        surface = fm.fdm_implicit_surface(
            params.N,
            params.M,
            params.Smax,
            params.T,
            params.K,
            params.r,
            params.sigma,
            params.is_call,
        )

        dS = params.Smax / params.N
        dt = params.T / params.M

        S_grid = [i * dS for i in range(params.N + 1)]
        t_grid = [i * dt for i in range(params.M + 1)]

        sanitized = [[v if math.isfinite(v) else 0.0 for v in row] for row in surface]

        return {"S_grid": S_grid, "t_grid": t_grid, "price_surface": sanitized}
    except ZeroDivisionError:
        logger.error("Division by zero in implicit surface calculation.")
        raise HTTPException(status_code=400, detail="Division by zero.")
    except ValueError as ve:
        logger.error(f"Invalid implicit surface input: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Implicit FDM surface error: {e}")
        raise HTTPException(
            status_code=500, detail="Unexpected implicit surface error."
        )


@router.post("/crank_surface", response_model=SurfaceResult)
async def fdm_crank_surface(params: SurfaceParams) -> SurfaceResult:
    """
    Run the Crank-Nicolson finite difference method for option pricing surface.
    This method computes the option price surface using the Crank-Nicolson finite difference method
    and returns the price surface along with the grid axes.
    """
    logger.info("Running crank_nicolson_fdm_surface with parameters: %s", params)
    try:
        # Guard against invalid parameters early
        if params.N <= 0 or params.M <= 0:
            raise ValueError("N and M must be positive integers.")
        if params.Smax <= 0 or params.T <= 0:
            raise ValueError("Smax and T must be positive.")
        if params.sigma < 0 or params.K <= 0:
            raise ValueError("Sigma and Strike must be non-negative.")

        logger.info(f"Running Crank-Nicolson FDM surface with: {params.dict()}")

        surface = await run_in_threadpool(
            fm.fdm_crank_nicolson_surface,
            params.N,
            params.M,
            params.Smax,
            params.T,
            params.K,
            params.r,
            params.sigma,
            params.is_call,
        )

        # Replace invalid float values in the surface
        sanitized = [[v if math.isfinite(v) else 0.0 for v in row] for row in surface]

        dS = params.Smax / params.N
        dt = params.T / params.M

        if not math.isfinite(dS) or not math.isfinite(dt) or dS <= 0 or dt <= 0:
            raise ValueError("Computed grid steps (dS or dt) are invalid.")

        S_grid = [i * dS for i in range(params.N + 1)]
        t_grid = [j * dt for j in range(params.M + 1)]

        return {
            "S_grid": S_grid,
            "t_grid": t_grid,
            "price_surface": sanitized,
        }

    except ZeroDivisionError:
        logger.error("Division by zero in FDM surface calculation.")
        raise HTTPException(
            status_code=400, detail="Division by zero encountered in FDM logic."
        )
    except ValueError as ve:
        logger.error(f"Invalid input for FDM surface: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected Crank-Nicolson error: {e}")
        raise HTTPException(
            status_code=500, detail="Unexpected error in FDM surface calculation."
        )


# === Binomial Tree Methods ===
@router.post("/binomial", response_model=FDMResult)
async def binomial_scalar(params: BinomialSurfaceParams) -> FDMResult:
    """
    Run the binomial tree method for option pricing.
    This method computes the option price using the binomial tree method
    and returns the final price along with an empty result list.
    """
    logger.info("Running binomial tree method with parameters: %s", params)
    try:
        # === Input validation ===
        if (
            params.N <= 0
            or params.T <= 0
            or params.K <= 0
            or params.S0 <= 0
            or params.sigma <= 0
        ):
            raise HTTPException(
                status_code=400, detail="All inputs (N, T, K, S0, sigma) must be > 0"
            )

        sigma = resolve_sigma(params)
        rate = resolve_rate(params)

        price = await run_in_threadpool(
            fm.binomial_tree,
            params.N,
            params.T,
            params.K,
            rate,
            sigma,
            params.is_call,
            params.is_american,
            params.S0,
        )

        # === Output validation ===
        if math.isnan(price) or math.isinf(price):
            raise HTTPException(
                status_code=500,
                detail="Model returned invalid numerical result (NaN or Inf)",
            )

        return FDMResult(result=[], final_price=price)

    except HTTPException:
        raise  # re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Binomial price error: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during binomial pricing."
        )


# === Binomial Tree Vector Endpoint ===
@router.post("/binomial_vector", response_model=VectorResult)
async def binomial_vector(params: BinomialSurfaceParams) -> VectorResult:
    """
    Run the binomial tree method for option pricing vector.
    This method computes the option price vector using the binomial tree method
    and returns the price grid along with the final price.
    """
    try:
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        vector = await run_in_threadpool(
            fm.binomial_tree_vector,
            params.N,
            params.T,
            params.K,
            rate,
            sigma,
            params.is_call,
            params.is_american,
            params.S0,
            params.Smax,
        )
        dS = params.Smax / params.N
        S_grid = [i * dS for i in range(params.N + 1)]
        final_price = vector[0] if vector else 0.0
        return VectorResult(S_grid=S_grid, prices=vector, final_price=final_price)
    except Exception as e:
        logger.error(f"Binomial vector error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/binomial_surface", response_model=SurfaceResult)
async def binomial_surface(params: BinomialSurfaceParams) -> SurfaceResult:
    """
    Run the binomial tree method for option pricing surface.
    This method computes the option price surface using the binomial tree method
    and returns the price surface along with the grid axes.
    """
    logger.info("Running binomial tree surface with parameters: %s", params)
    try:
        if params.N <= 0:
            raise HTTPException(status_code=400, detail="N must be > 0")

        surface = fm.binomial_tree_surface(
            params.N,
            params.T,
            params.K,
            params.r,
            params.sigma,
            params.is_call,
            params.is_american,
            params.S0,
        )

        S_grid = [params.S0 * (1 + i / params.N) for i in range(params.N + 1)]
        t_grid = [params.T * (i / params.N) for i in range(params.N + 1)]

        return {"S_grid": S_grid, "t_grid": t_grid, "price_surface": surface}

    except Exception as e:
        logger.error(f"Binomial surface error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/psor_vector", response_model=VectorResult)
async def fdm_psor_vector(params: PSORSurfaceParams) -> VectorResult:
    """
    Run the PSOR finite difference method for American option pricing vector.
    This method computes the option price vector using the PSOR finite difference method
    and returns the price grid along with the final price.
    """
    logger.info("Running PSOR vector method with parameters: %s", params)
    try:
        logger.info("Running PSOR vector method...")

        # Safety clamp for zero or invalid divisions
        N = max(1, params.N)
        Smax = max(1e-8, params.Smax)
        dS = Smax / N

        # Call pybind-wrapped C++ function
        V = fm.american_psor_vector(
            N,
            params.M,
            params.Smax,
            params.T,
            params.K,
            params.r,
            params.sigma,
            params.is_call,
            params.omega,
            params.maxIter,
            params.tol,
        )

        S_grid = [i * dS for i in range(N + 1)]
        final_price = V[-1] if V else 0.0
        return VectorResult(S_grid=S_grid, prices=V, final_price=final_price)

    except Exception as e:
        logger.exception("❌ PSOR vector error")
        raise HTTPException(status_code=500, detail=str(e))
