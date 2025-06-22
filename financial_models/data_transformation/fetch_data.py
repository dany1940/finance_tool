import asyncio
import yfinance as yf
import requests
import logging
import polars as pl
import pyarrow as pa
import pandas as pd
from data_transformation.cache_manager import get_cached_data, cache_data
from fastapi import HTTPException




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = "your_polygon_api_key"  # Replace with your actual Polygon.io API key


async def fetch_yahoo_data(tickers: list, start: str, end: str):
    """
    Fetch historical stock data from Yahoo Finance and return as a Polars DataFrame.
    Caches results for performance.
    """
    cache_key = f"yahoo:{','.join(tickers)}:{start}:{end}"
    cached_data = get_cached_data(cache_key)

    if cached_data:
        logger.info(f"✅ Cache hit for {tickers} ({start} - {end})")
        return pl.DataFrame(cached_data)  # ✅ Return cached data

    try:
        # ✅ Fetch data for multiple tickers
        df = yf.download(tickers, start=start, end=end, group_by="ticker")
        logger.info(f"📥 Downloaded Yahoo Finance data for {tickers}")

        # ✅ Normalize multi-index DataFrame
        if isinstance(df.columns, pd.MultiIndex):
            df = df.stack(level=0).reset_index()
        for col in df.select_dtypes(include=["datetime"]).columns:
            df[col] = df[col].dt.strftime("%Y-%m-%d")
        # ✅ Convert to Polars DataFrame
        arrow_table = pa.Table.from_pandas(df)
        pl_df = pl.from_arrow(arrow_table)

        # ✅ Cache result
        cache_data(cache_key, pl_df.to_dicts())
        logger.info(f"✅ Cached Yahoo Finance data for {tickers} ({start} - {end})")
        return pl_df

    except Exception as e:
        logger.error(f"❌ Yahoo Finance error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Yahoo Finance error: {str(e)}")


async def fetch_polygon_data(tickers: list, start: str, end: str):
    cache_key = f"polygon:{','.join(tickers)}:{start}:{end}"
    cached_data = get_cached_data(cache_key)
    if cached_data:
        return pl.DataFrame(cached_data)

    async def fetch_ticker_data(ticker):
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}?apiKey={API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()["results"]
        return None

    tasks = [fetch_ticker_data(ticker) for ticker in tickers]
    results = await asyncio.gather(*tasks)

    all_data = []
    for ticker, data in zip(tickers, results):
        if data:
            for record in data:
                record["symbol"] = ticker
                all_data.append(record)

    if not all_data:
        raise HTTPException(status_code=404, detail="No data found for requested stocks")

    pl_df = pl.DataFrame(all_data)
    cache_data(cache_key, pl_df.to_dicts())  # Cache result
    return pl_df
