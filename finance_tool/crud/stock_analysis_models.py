from typing import Dict, Any, List
from pydantic import BaseModel


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


