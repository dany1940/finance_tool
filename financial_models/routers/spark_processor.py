import datetime
import logging
from typing import List

from fastapi import APIRouter, HTTPException, Query

from crud.exchange_models import YahooExchange, YahooExchangeResponse
from data_transformation.analyze_data import analyze_stock_data
from data_transformation.fetch_data import fetch_polygon_data, fetch_yahoo_data

router = APIRouter(prefix="/stocks", tags=["Stock Processing"])
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dropdown options
AVAILABLE_TICKERS = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"]
AVAILABLE_SOURCES = ["yahoo", "polygon"]


async def fetch_stock_data(tickers: List[str], source: str, start: str, end: str) -> List[YahooExchange]:
    """
    Fetch stock data from the specified source and return it as a list of YahooExchange objects.
    """
    logger.info(f"Fetching stock data for {tickers} from {source.upper()} between {start} and {end}")
    if source == "yahoo":
        return await fetch_yahoo_data(tickers, start, end)
    elif source == "polygon":
        return await fetch_polygon_data(tickers, start, end)
    else:
        raise HTTPException(
            status_code=400, detail="Invalid source. Use 'yahoo' or 'polygon'."
        )


@router.get("/stocks/download", response_model=YahooExchangeResponse)
async def download_stock_data(
    tickers: List[str] = Query(
        ["AAPL"],
        title="Stock Tickers",
        description="Select one or more stock symbols",
        enum=AVAILABLE_TICKERS,
    ),
    source: str = Query(
        "yahoo",
        title="Data Source",
        description="Select data source",
        enum=AVAILABLE_SOURCES,
    ),
    start: datetime.date = Query(
        datetime.date.today() - datetime.timedelta(days=365), title="Start Date"
    ),
    end: datetime.date = Query(datetime.date.today(), title="End Date"),
):
    """
    Downloads stock data based on timeframe, tickers, and source.
    Returns JSON serialized data.
    """

    try:
        data = await fetch_stock_data(tickers, source, start, end)

        yahoo_data = [
            YahooExchange(
                Date=data["Date"],
                Ticker=data["Ticker"],
                Open=data["Open"],
                High=data["High"],
                Low=data["Low"],
                Close=data["Close"],
                Volume=data["Volume"],
            )
            for data in data.to_dicts()
        ]
        return YahooExchangeResponse(yahoo_data=yahoo_data)
    except Exception as e:
        logger.error(f"Error fetching stock data: {e}")
        return {"error": str(e)}


@router.get("/analyze")
async def analyze_stocks(
    tickers: List[str] = Query(
        ["AAPL"],
        title="Stock Tickers",
        description="Select one or more stock symbols",
        enum=AVAILABLE_TICKERS,
    ),
    source: str = Query(
        "yahoo",
        title="Data Source",
        description="Select data source",
        enum=AVAILABLE_SOURCES,
    ),
    start: datetime.date = Query(
        datetime.date.today() - datetime.timedelta(days=365), title="Start Date"
    ),
    end: datetime.date = Query(datetime.date.today(), title="End Date"),
):
    logger.info(f"Analyzing stock data for {tickers} from {source.upper()}")

    df = await fetch_stock_data(tickers, source, start, end)
    analysis = analyze_stock_data(df)
    return {"summary": analysis}
