from typing import Dict, Any, List, Literal, Optional
from pydantic import BaseModel, Field
from datetime import date

# === Summary Stats (Optional for Bootstrap) ===
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

# === Core FDM Request Parameters ===
class CommonParams(BaseModel):
    N: int = Field(50, ge=3)
    M: int = Field(50)
    Smax: float = Field(100.0)
    T: float = Field(1.0)
    K: float = Field(100.0)
    r: float = Field(0.05)
    sigma: float = Field(0.2)
    is_call: bool = Field(True)
    cfl: Optional[bool] = False
    S0: Optional[float] = Field(None, description="Spot price")
    option_style: Optional[Literal["European", "American"]] = "European"
    vol_source: Optional[Literal["User-defined",  "Implied"]] = "User-defined"
    grid_scheme: Optional[Literal["uniform", "adaptive"]] = "uniform"



# === American PSOR ===
class AmericanParams(CommonParams):
    omega: float = Field(1.2, gt=0)
    maxIter: int = Field(10000, gt=0)
    tol: float = Field(1e-6, gt=0)

# === Fractional FDM ===
class FractionalParams(CommonParams):
    beta: float = Field(0.5, gt=0, le=1)

# === Compact Derivative ===
class CompactParams(BaseModel):
    V: List[float]
    dx: float = Field(0.1, gt=0)

# === Dispatcher FDM ===
class DispatcherParams(CommonParams):
    method: Literal["explicit", "implicit", "crank", "exponential"] = Field("crank")

# === Bootstrap Parameters ===
class BootstrapParams(BaseModel):
    ticker: str = Field(..., description="e.g. 'AAPL'")
    start_date: date
    end_date: date
    num_samples: int = Field(1000, gt=0)
    confidence_levels: List[float] = Field(default=[0.95, 0.99])

    class Config:
        schema_extra = {
            "example": {
                "ticker": "AAPL",
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "num_samples": 1000,
                "confidence_levels": [0.95, 0.99]
            }
        }

# === Bootstrap Output ===
class BootstrapResult(BaseModel):
    values: List[float]
    mean: float
    variance: float
    var_95: float
    var_99: float

# === FDM Output ===
class ResultItem(BaseModel):
    index: int
    value: float

class FDMResult(BaseModel):
    result: List[ResultItem]
    final_price: float

class BlackScholesParams(BaseModel):
    S: float = Field(..., description="Spot price")
    K: float = Field(..., description="Strike price")
    T: float = Field(..., description="Time to maturity")
    r: float = Field(..., description="Risk-free rate")
    sigma: float = Field(..., description="Volatility")
    is_call: bool = Field(..., description="Call or Put option")


# === Pydantic Response Model for Vector Results ===
class VectorResult(BaseModel):
    S_grid: List[float]
    prices: List[float]
    final_price: float


class ResponseBlackscholes(BaseModel):
    price: float = Field(..., description="Calculated option price using Black-Scholes formula")



# === Request & Response Models ===
class SurfaceParams(BaseModel):
    N: int
    M: int
    Smax: float
    T: float
    K: float
    r: float
    sigma: float
    is_call: bool


class SurfaceResult(BaseModel):
    surface: List[List[float]]
