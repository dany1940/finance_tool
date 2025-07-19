import logging
import math
from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool
from typing import List, Dict

from crud.stock_analysis_models import (
    CommonParams, AmericanParams, FractionalParams, CompactParams,
    DispatcherParams, BootstrapParams, BootstrapResult, FDMResult, ResultItem,
    BlackScholesParams, ResponseBlackscholes, VectorResult, SurfaceResult, SurfaceParams, PSORSurfaceParams, BinomialSurfaceParams
)

import financial_models_wrapper as fm
from routers.volatility.utils import resolve_sigma, resolve_rate

router = APIRouter(prefix="/fdm", tags=["FDM Processing"])
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def wrap_result(result: List[float]) -> FDMResult:
    if not result:
        logger.warning("⚠️ FDM solver returned an empty result list")
        return FDMResult(result=[], final_price=0.0)
    return FDMResult(
        result=[ResultItem(index=i, value=float(v)) for i, v in enumerate(result)],
        final_price=float(result[-1])
    )

# === Scalar-returning FDM methods: return only final price and empty result list ===

@router.post("/explicit", response_model=FDMResult)
async def run_explicit(params: CommonParams):
    try:
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        price = await run_in_threadpool(
            fm.explicit_fdm,
            params.N, params.M, params.Smax,
            params.T, params.K, rate, sigma,
            params.is_call,
            params.S0
        )
        logger.info(f"✅ Explicit FDM computed price: {price}")
        return FDMResult(result=[], final_price=price)
    except Exception as e:
        logger.exception("❌ Error in explicit_fdm")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/implicit", response_model=FDMResult)
async def run_implicit(params: CommonParams):
    try:
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        price = await run_in_threadpool(
            fm.implicit_fdm,
            params.N, params.M, params.Smax,
            params.T, params.K, rate, sigma,
            params.is_call,
            params.S0
        )
        return FDMResult(result=[], final_price=price)
    except Exception as e:
        logger.exception("❌ Error in implicit_fdm")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/crank", response_model=FDMResult)
async def run_crank(params: CommonParams):
    try:
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        price = await run_in_threadpool(
            fm.crank_nicolson_fdm,
            params.N, params.M, params.Smax,
            params.T, params.K, rate, sigma,
            params.is_call,
            False,  # rannacher smoothing default false
            params.S0
        )
        return FDMResult(result=[], final_price=price)
    except Exception as e:
        logger.exception("❌ Error in crank_nicolson_fdm")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/psor", response_model=FDMResult)
async def run_american(params: AmericanParams):
    try:
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        price = await run_in_threadpool(
            fm.american_psor_fdm,
            params.N, params.M, params.Smax,
            params.T, params.K, rate, sigma,
            params.is_call,
            params.omega, params.maxIter, params.tol,
            params.S0
        )
        return FDMResult(result=[], final_price=price)
    except Exception as e:
        logger.exception("❌ Error in american_psor_fdm")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fractional", response_model=FDMResult)
async def run_fractional(params: FractionalParams):
    try:
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        price = await run_in_threadpool(
            fm.fractional_fdm,
            params.N, params.M, params.Smax,
            params.T, params.K, rate, sigma,
            params.is_call,
            params.beta,
            params.S0
        )
        return FDMResult(result=[], final_price=price)
    except Exception as e:
        logger.exception("❌ Error in fractional_fdm")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/exponential", response_model=FDMResult)
async def run_exponential(params: CommonParams):
    try:
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        price = await run_in_threadpool(
            fm.exponential_integral_fdm,
            params.N, params.Smax,
            params.T, params.K, rate, sigma,
            params.is_call,
            params.S0
        )
        return FDMResult(result=[], final_price=price)
    except Exception as e:
        logger.exception("❌ Error in exponential_integral_fdm")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/psor_surface", response_model=SurfaceResult)
async def fdm_psor_surface(params: PSORSurfaceParams):
    try:
        logger.info("Running PSOR FDM surface...")
        surface = fm.american_psor_surface(
            params.N, params.M, params.Smax, params.T,
            params.K, params.r, params.sigma, params.is_call,
            params.omega, params.maxIter, params.tol
        )
        S_grid = [i * (params.Smax / params.N) for i in range(params.N + 1)]
        t_grid = [i * (params.T / params.M) for i in range(params.M + 1)]

        return {
            "S_grid": S_grid,
            "t_grid": t_grid,
            "price_surface": surface
        }

    except Exception as e:
        logger.error(f"❌ PSOR surface error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Exponential Integral Surface Endpoint ===
@router.post("/exponential_surface", response_model=SurfaceResult)
async def fdm_exponential_surface(params: SurfaceParams):
    try:
        logger.info("Running exponential integral FDM surface...")
        surface = fm.fdm_exponential_integral_vector(
            params.N, params.Smax, params.T, params.K,
            params.r, params.sigma, params.is_call
        )
        S_grid = [i * (params.Smax / params.N) for i in range(params.N + 1)]
        t_grid = [i * (params.T / params.M) for i in range(params.M + 1)]
        return {
            "S_grid": S_grid,
            "t_grid": t_grid,
            "price_surface": surface
        }
    except Exception as e:
        logger.error(f"Exponential Integral FDM surface error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compact", response_model=FDMResult)
async def run_compact(params: CommonParams):
    try:
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        price = await run_in_threadpool(
            fm.compact_fdm, # must match signature in your C++/pybind wrapper
            params.N, params.M, params.Smax,
            params.T, params.K, rate, sigma,
            params.is_call,
            params.S0
        )
        return FDMResult(result=[], final_price=price)
    except Exception as e:
        logger.exception("❌ Error in compact_fdm")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dispatcher", response_model=FDMResult)
async def run_dispatcher(params: DispatcherParams):
    try:
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        price = await run_in_threadpool(
            fm.solve_fdm,
            params.method, params.N, params.M,
            params.Smax, params.T, params.K,
            rate, sigma,
            params.is_call
        )
        return FDMResult(result=[], final_price=price)
    except Exception as e:
        logger.exception("❌ Error in dispatcher")
        raise HTTPException(status_code=500, detail=str(e))

# === Black-Scholes ===

@router.post("/black_scholes", response_model=ResponseBlackscholes)
async def run_black_scholes(params: BlackScholesParams):
    try:
        price = await run_in_threadpool(
            fm.black_scholes,
            params.S, params.K, params.T, params.r, params.sigma, params.is_call
        )
        return {"price": price}
    except Exception as e:
        logger.exception("❌ Error in black_scholes")
        raise HTTPException(status_code=500, detail=str(e))




# === Helper to wrap vector results with price grid ===
def wrap_vector_result(vector: List[float], Smax: float, N: int) -> VectorResult:
    dS = Smax / N
    S_grid = [i * dS for i in range(N + 1)]
    final_price = vector[-1] if vector else 0.0
    return VectorResult(S_grid=S_grid, prices=vector, final_price=final_price)


# === Explicit FDM Vector Endpoint ===
@router.post("/explicit_vector", response_model=VectorResult)
async def explicit_vector(params: CommonParams):
    try:
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        vector = await run_in_threadpool(
            fm.explicit_fdm_vector,
            params.N, params.M, params.Smax,
            params.T, params.K, rate, sigma,
            params.is_call
        )
        return wrap_vector_result(vector, params.Smax, params.N)
    except Exception as e:
        logger.exception("Error in explicit_fdm_vector")
        raise HTTPException(status_code=500, detail=str(e))


# === Implicit FDM Vector Endpoint === not impelmented
@router.post("/implicit_vector", response_model=VectorResult)
async def implicit_vector(params: CommonParams):
    try:
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        vector = await run_in_threadpool(
            fm.implict_fdm_vector,
            params.N, params.M, params.Smax,
            params.T, params.K, rate, sigma,
            params.is_call
        )
        return wrap_vector_result(vector, params.Smax, params.N)
    except Exception as e:
        logger.exception("Error in implicit_fdm_vector")
        raise HTTPException(status_code=500, detail=str(e))


# === Crank-Nicolson FDM Vector Endpoint ===
@router.post("/crank_vector", response_model=VectorResult)
async def crank_vector(params: CommonParams):
    try:
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        vector = await run_in_threadpool(
            fm.crank_nicolson_fdm_vector,
            params.N, params.M, params.Smax,
            params.T, params.K, rate, sigma,
            params.is_call,
            False  # rannacher smoothing default false
        )
        return wrap_vector_result(vector, params.Smax, params.N)
    except Exception as e:
        logger.exception("Error in crank_nicolson_fdm_vector")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explicit_surface", response_model=SurfaceResult)
async def fdm_explicit_surface(params: SurfaceParams):
    try:
        if params.N <= 0 or params.M <= 0:
            raise ValueError("N and M must be positive integers.")
        if params.Smax <= 0 or params.T <= 0:
            raise ValueError("Smax and T must be positive.")
        if params.K <= 0 or params.sigma < 0:
            raise ValueError("Strike and volatility must be valid.")

        logger.info("Running explicit FDM surface...")
        surface = fm.fdm_explicit_surface(
            params.N, params.M, params.Smax, params.T,
            params.K, params.r, params.sigma, params.is_call
        )

        dS = params.Smax / params.N
        dt = params.T / params.M

        S_grid = [i * dS for i in range(params.N + 1)]
        t_grid = [i * dt for i in range(params.M + 1)]

        # Sanitize surface
        sanitized = [
            [v if math.isfinite(v) else 0.0 for v in row]
            for row in surface
        ]

        return {
            "S_grid": S_grid,
            "t_grid": t_grid,
            "price_surface": sanitized
        }
    except ZeroDivisionError:
        logger.error("Division by zero in explicit surface calculation.")
        raise HTTPException(status_code=400, detail="Division by zero.")
    except ValueError as ve:
        logger.error(f"Invalid explicit surface input: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Explicit FDM surface error: {e}")
        raise HTTPException(status_code=500, detail="Unexpected explicit surface error.")



@router.post("/implicit_surface", response_model=SurfaceResult)
async def fdm_implicit_surface(params: SurfaceParams):
    try:
        if params.N <= 0 or params.M <= 0:
            raise ValueError("N and M must be positive integers.")
        if params.Smax <= 0 or params.T <= 0:
            raise ValueError("Smax and T must be positive.")
        if params.K <= 0 or params.sigma < 0:
            raise ValueError("Strike and volatility must be valid.")

        logger.info("Running implicit FDM surface...")
        surface = fm.fdm_implicit_surface(
            params.N, params.M, params.Smax, params.T,
            params.K, params.r, params.sigma, params.is_call
        )

        dS = params.Smax / params.N
        dt = params.T / params.M

        S_grid = [i * dS for i in range(params.N + 1)]
        t_grid = [i * dt for i in range(params.M + 1)]

        sanitized = [
            [v if math.isfinite(v) else 0.0 for v in row]
            for row in surface
        ]

        return {
            "S_grid": S_grid,
            "t_grid": t_grid,
            "price_surface": sanitized
        }
    except ZeroDivisionError:
        logger.error("Division by zero in implicit surface calculation.")
        raise HTTPException(status_code=400, detail="Division by zero.")
    except ValueError as ve:
        logger.error(f"Invalid implicit surface input: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Implicit FDM surface error: {e}")
        raise HTTPException(status_code=500, detail="Unexpected implicit surface error.")


@router.post("/crank_surface", response_model=SurfaceResult)
async def fdm_crank_surface(params: SurfaceParams):
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
            params.N, params.M, params.Smax, params.T,
            params.K, params.r, params.sigma, params.is_call
        )

        # Replace invalid float values in the surface
        sanitized = [
            [v if math.isfinite(v) else 0.0 for v in row]
            for row in surface
        ]

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
        raise HTTPException(status_code=400, detail="Division by zero encountered in FDM logic.")
    except ValueError as ve:
        logger.error(f"Invalid input for FDM surface: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected Crank-Nicolson error: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error in FDM surface calculation.")


# === Binomial Tree Methods ===
@router.post("/binomial", response_model=FDMResult)
async def binomial_scalar(params: AmericanParams):
    try:
        # === Input validation ===
        if params.N <= 0 or params.T <= 0 or params.K <= 0 or params.S0 <= 0 or params.sigma <= 0:
            raise HTTPException(status_code=400, detail="All inputs (N, T, K, S0, sigma) must be > 0")

        sigma = resolve_sigma(params)
        rate = resolve_rate(params)

        price = await run_in_threadpool(
            fm.binomial_tree,
            params.N, params.T, params.K, rate, sigma,
            params.is_call, params.is_american, params.S0
        )

        # === Output validation ===
        if math.isnan(price) or math.isinf(price):
            raise HTTPException(status_code=500, detail="Model returned invalid numerical result (NaN or Inf)")

        return FDMResult(result=[], final_price=price)

    except HTTPException:
        raise  # re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Binomial price error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during binomial pricing.")


# === Binomial Tree Vector Endpoint ===
@router.post("/binomial_vector", response_model=VectorResult)
async def binomial_vector(params: AmericanParams):
    try:
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        vector = await run_in_threadpool(
            fm.binomial_tree_vector,
            params.N, params.T, params.K, rate, sigma,
            params.is_call, params.is_american, params.S0, params.Smax
        )
        dS = params.Smax / params.N
        S_grid = [i * dS for i in range(params.N + 1)]
        final_price = vector[0] if vector else 0.0
        return VectorResult(S_grid=S_grid, prices=vector, final_price=final_price)
    except Exception as e:
        logger.error(f"Binomial vector error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/binomial_surface", response_model=SurfaceResult)
async def binomial_surface(params: BinomialSurfaceParams):
    try:
        if params.N <= 0:
            raise HTTPException(status_code=400, detail="N must be > 0")

        surface = fm.binomial_tree_surface(
            params.N, params.T, params.K, params.r,
            params.sigma, params.is_call, params.is_american, params.S0
        )

        S_grid = [params.S0 * (1 + i / params.N) for i in range(params.N + 1)]
        t_grid = [params.T * (i / params.N) for i in range(params.N + 1)]

        return {
            "S_grid": S_grid,
            "t_grid": t_grid,
            "price_surface": surface
        }

    except Exception as e:
        logger.error(f"Binomial surface error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
