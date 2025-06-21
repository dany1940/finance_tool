from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool
from typing import List
import logging
import numpy as np
import yfinance as yf

from crud.stock_analysis_models import (
    CommonParams, AmericanParams, FractionalParams, CompactParams,
    DispatcherParams, BootstrapParams, BootstrapResult, FDMResult, ResultItem
)

import financial_models_wrapper as fm
from routers.volatility.utils import resolve_sigma, resolve_rate

# === Router and Logger Configuration ===
router = APIRouter(prefix="/fdm", tags=["FDM Processing"])
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# === Utility: Wrap raw list output to FDMResult model ===
def wrap_result(result: List[float]) -> FDMResult:
    if not result:
        logger.warning("⚠️ FDM solver returned an empty result list")
    return FDMResult(
        result=[ResultItem(index=i, value=float(v)) for i, v in enumerate(result or [])],
        final_price=float(result[-1]) if result else 0.0
    )


# === Endpoint: Explicit FDM ===
@router.post("/explicit", response_model=FDMResult)
async def run_explicit(params: CommonParams):
    try:
        logger.info(f"[explicit] T={params.T}, N={params.N}, M={params.M}")
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        logger.info(f"[explicit] sigma={sigma}, rate={rate}")
        result = await run_in_threadpool(
            fm.explicit_fdm, params.N, params.M, params.Smax,
            params.T, params.K, rate, sigma, params.is_call
        )
        logger.info(f"[explicit] Output size: {len(result)}")
        return wrap_result(result)
    except Exception as e:
        logger.exception("❌ Error in explicit_fdm")
        raise HTTPException(status_code=500, detail=str(e))


# === Endpoint: Implicit FDM ===
@router.post("/implicit", response_model=FDMResult)
async def run_implicit(params: CommonParams):
    try:
        logger.info(f"[implicit] T={params.T}, N={params.N}, M={params.M}")
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        logger.info(f"[implicit] sigma={sigma}, rate={rate}")
        result = await run_in_threadpool(
            fm.implicit_fdm, params.N, params.M, params.Smax,
            params.T, params.K, rate, sigma, params.is_call
        )
        logger.info(f"[implicit] Output size: {len(result)}")
        return wrap_result(result)
    except Exception as e:
        logger.exception("❌ Error in implicit_fdm")
        raise HTTPException(status_code=500, detail=str(e))


# === Endpoint: Crank-Nicolson FDM ===
@router.post("/crank", response_model=FDMResult)
async def run_crank(params: CommonParams):
    try:
        logger.info(f"[crank] T={params.T}, N={params.N}, M={params.M}")
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        logger.info(f"[crank] sigma={sigma}, rate={rate}")
        result = await run_in_threadpool(
            fm.crank_nicolson_fdm, params.N, params.M, params.Smax,
            params.T, params.K, rate, sigma, params.is_call, False
        )
        logger.info(f"[crank] Output size: {len(result)}")
        return wrap_result(result)
    except Exception as e:
        logger.exception("❌ Error in crank_nicolson_fdm")
        raise HTTPException(status_code=500, detail=str(e))


# === Endpoint: American (PSOR) FDM ===
@router.post("/american", response_model=FDMResult)
async def run_american(params: AmericanParams):
    try:
        logger.info(f"[american] T={params.T}, N={params.N}, M={params.M}, ω={params.omega}, tol={params.tol}, maxIter={params.maxIter}")
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        logger.info(f"[american] sigma={sigma}, rate={rate}")
        result = await run_in_threadpool(
            fm.american_psor_fdm, params.N, params.M, params.Smax,
            params.T, params.K, rate, sigma, params.is_call,
            params.omega, params.maxIter, params.tol
        )
        logger.info(f"[american] Output size: {len(result)}")
        return wrap_result(result)
    except Exception as e:
        logger.exception("❌ Error in american_psor_fdm")
        raise HTTPException(status_code=500, detail=str(e))


# === Endpoint: Fractional FDM ===
@router.post("/fractional", response_model=FDMResult)
async def run_fractional(params: FractionalParams):
    try:
        logger.info(f"[fractional] T={params.T}, N={params.N}, M={params.M}, β={params.beta}")
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        logger.info(f"[fractional] sigma={sigma}, rate={rate}")
        result = await run_in_threadpool(
            fm.fractional_fdm, params.N, params.M, params.Smax,
            params.T, params.K, rate, sigma, params.is_call, params.beta
        )
        logger.info(f"[fractional] Output size: {len(result)}")
        return wrap_result(result)
    except Exception as e:
        logger.exception("❌ Error in fractional_fdm")
        raise HTTPException(status_code=500, detail=str(e))


# === Endpoint: Exponential Integral FDM ===
@router.post("/exponential", response_model=FDMResult)
async def run_exponential(params: CommonParams):
    try:
        logger.info(f"[exponential] T={params.T}, N={params.N}")
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        logger.info(f"[exponential] sigma={sigma}, rate={rate}")
        result = await run_in_threadpool(
            fm.exponential_integral_fdm, params.N, params.Smax,
            params.T, params.K, rate, sigma, params.is_call
        )
        logger.info(f"[exponential] Output size: {len(result)}")
        return wrap_result(result)
    except Exception as e:
        logger.exception("❌ Error in exponential_integral_fdm")
        raise HTTPException(status_code=500, detail=str(e))


# === Endpoint: Compact FDM (with V and dx) ===
@router.post("/compact", response_model=FDMResult)
async def run_compact(params: CompactParams):
    try:
        logger.info(f"[compact] V={params.V}, dx={params.dx}")
        result = await run_in_threadpool(fm.compact_fdm, V=params.V, dx=params.dx)
        logger.info(f"[compact] Output size: {len(result)}")
        return wrap_result(result)
    except Exception as e:
        logger.exception("❌ Error in compact_fdm")
        raise HTTPException(status_code=500, detail=str(e))


# === Endpoint: Dispatcher (Dynamic FDM Router) ===
@router.post("/dispatcher", response_model=FDMResult)
async def run_dispatcher(params: DispatcherParams):
    try:
        logger.info(f"[dispatcher] method={params.method}, T={params.T}, N={params.N}, M={params.M}")
        sigma = resolve_sigma(params)
        rate = resolve_rate(params)
        logger.info(f"[dispatcher] sigma={sigma}, rate={rate}")
        result = await run_in_threadpool(
            fm.solve_fdm, params.method, params.N, params.M,
            params.Smax, params.T, params.K, rate, sigma, params.is_call
        )
        logger.info(f"[dispatcher] Output size: {len(result)}")
        return wrap_result(result)
    except Exception as e:
        logger.exception("❌ Error in dispatcher")
        raise HTTPException(status_code=500, detail=str(e))
