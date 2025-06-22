from pydantic import BaseModel, Field
from typing import List, Optional

class YahooExchange(BaseModel):
    Date: str
    Ticker: str
    Open: float
    High: float
    Low: float
    Close: float
    Volume: int

    class Config:
        orm_mode = True  # To make it compatible with ORM models (optional)

class YahooExchangeSummary(BaseModel):
    statistic: List[str]
    Date: List[str]
    Ticker: List[str]
    Open: List[float]
    High: List[float]
    Low: List[float]
    Close: List[float]
    Volume: List[int]

    class Config:
        orm_mode = True  # To make it compatible with ORM models (optional)


class YahooExchangeResponse(BaseModel):
    yahoo_data: List[YahooExchange]


# Endpoint response model
class StockDataResponse(BaseModel):
    status: str
    message: Optional[str] = None
    data: YahooExchangeSummary

