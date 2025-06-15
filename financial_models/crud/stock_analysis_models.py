from typing import Dict, Any, List, Literal
from pydantic import BaseModel, Field


class StockAnalysisItem(BaseModel):
    count: int
    null_count: int
    mean: float
    std: float
    min: float
    percentile_25: float
    median: float
    percentile_75: float
    max: float


class StockAnalysisResponse(BaseModel):
    summary: Dict[str, StockAnalysisItem]
    stock_data: List[Dict[str, Any]]


class CommonParams(BaseModel):
    N: int = Field(..., gt=0)
    M: int = Field(..., gt=0)
    Smax: float = Field(..., gt=0)
    T: float = Field(..., gt=0)
    K: float = Field(..., gt=0)
    r: float
    sigma: float = Field(..., gt=0)
    is_call: bool

class AmericanParams(CommonParams):
    omega: float = Field(default=1.2, gt=0)
    maxIter: int = Field(default=10000, gt=0)
    tol: float = Field(default=1e-6, gt=0)

class FractionalParams(CommonParams):
    beta: float = Field(..., gt=0, le=1)

class CompactParams(BaseModel):
    V: List[float]
    dx: float = Field(..., gt=0)

class DispatcherParams(CommonParams):
    method: Literal["explicit", "implicit", "crank", "exponential"]

# ===== Response Model =====

class FDMResult(BaseModel):
    result: List[float]
