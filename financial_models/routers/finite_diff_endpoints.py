from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool
from typing import List, Dict
import logging

from crud.stock_analysis_models import (
    CommonParams, AmericanParams, FractionalParams, CompactParams,
    DispatcherParams, BootstrapParams, BootstrapResult, FDMResult, ResultItem,
    BlackScholesParams, ResponseBlackscholes, VectorResult
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

@router.post("/american", response_model=FDMResult)
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

# === Vector-returning FDM methods: return result vector and final price ===

@router.post("/compact", response_model=FDMResult)
async def run_compact(params: CompactParams):
    try:
        result = await run_in_threadpool(fm.compact_fdm, V=params.V, dx=params.dx)
        return wrap_result(result)
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


# === Implicit FDM Vector Endpoint ===
@router.post("/implicit_vector", response_model=VectorResult)
async def implicit_vector(params: CommonParams):
    try:
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        vector = await run_in_threadpool(
            fm.implicit_fdm_vector,
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
