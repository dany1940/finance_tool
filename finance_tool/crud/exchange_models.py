from pydantic import BaseModel


class YahooExchange(BaseModel):
    Date: str
    Ticker: str
    Open: float
    High: float
    Low: float
    Close: float
    Volume: int

    class Config:
        orm_mode = True
