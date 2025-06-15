from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool
from typing import List
import logging

from crud.stock_analysis_models import (
    CommonParams,
    AmericanParams,
    FractionalParams,
    CompactParams,
    DispatcherParams,
    FDMResult
)
# Import the financial models wrapper
import financial_models_wrapper as financial_models

router = APIRouter(prefix="/fdm", tags=["FDM Processing"])
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# === Endpoint: Explicit FDM ===
@router.post("/explicit", response_model=FDMResult)
async def run_explicit(params: CommonParams):
    try:
        result = await run_in_threadpool(financial_models.explicit_fdm, **params.dict())
        return {"result": result}
    except Exception as e:
        logger.exception("Error in explicit_fdm")
        raise HTTPException(status_code=500, detail=str(e))


# === Endpoint: Implicit FDM ===
@router.post("/implicit", response_model=FDMResult)
async def run_implicit(params: CommonParams):
    try:
        result = await run_in_threadpool(financial_models.implicit_fdm, **params.dict())
        return {"result": result}
    except Exception as e:
        logger.exception("Error in implicit_fdm")
        raise HTTPException(status_code=500, detail=str(e))


# === Endpoint: Crank-Nicolson FDM ===
@router.post("/crank", response_model=FDMResult)
async def run_crank(params: CommonParams):
    try:
        result = await run_in_threadpool(
            financial_models.crank_nicolson_fdm,
            **params.dict(),
            rannacher_smoothing=False
        )
        return {"result": result}
    except Exception as e:
        logger.exception("Error in crank_nicolson_fdm")
        raise HTTPException(status_code=500, detail=str(e))


# === Endpoint: American PSOR FDM ===
@router.post("/american", response_model=FDMResult)
async def run_american(params: AmericanParams):
    try:
        result = await run_in_threadpool(financial_models.american_psor_fdm, **params.dict())
        return {"result": result}
    except Exception as e:
        logger.exception("Error in american_psor_fdm")
        raise HTTPException(status_code=500, detail=str(e))


# === Endpoint: Fractional FDM ===
@router.post("/fractional", response_model=FDMResult)
async def run_fractional(params: FractionalParams):
    try:
        result = await run_in_threadpool(financial_models.fractional_fdm, **params.dict())
        return {"result": result}
    except Exception as e:
        logger.exception("Error in fractional_fdm")
        raise HTTPException(status_code=500, detail=str(e))


# === Endpoint: Exponential Integral FDM ===
@router.post("/exponential", response_model=FDMResult)
async def run_exponential(params: CommonParams):
    try:
        result = await run_in_threadpool(financial_models.exponential_integral_fdm, **params.dict())
        return {"result": result}
    except Exception as e:
        logger.exception("Error in exponential_integral_fdm")
        raise HTTPException(status_code=500, detail=str(e))


# === Endpoint: Compact Scheme ===
@router.post("/compact", response_model=FDMResult)
async def run_compact(params: CompactParams):
    try:
        result = await run_in_threadpool(financial_models.compact_fdm, V=params.V, dx=params.dx)
        return {"result": result}
    except Exception as e:
        logger.exception("Error in compact_fdm")
        raise HTTPException(status_code=500, detail=str(e))


# === Endpoint: Dispatcher ===
@router.post("/dispatcher", response_model=FDMResult)
async def run_dispatcher(params: DispatcherParams):
    try:
        result = await run_in_threadpool(
            financial_models.solve_fdm,
            method=params.method,
            N=params.N,
            M=params.M,
            Smax=params.Smax,
            T=params.T,
            K=params.K,
            r=params.r,
            sigma=params.sigma,
            is_call=params.is_call,
        )
        return {"result": result}
    except Exception as e:
        logger.exception("Error in dispatcher")
        raise HTTPException(status_code=500, detail=str(e))
