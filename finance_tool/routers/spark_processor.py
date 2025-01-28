from fastapi import APIRouter
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, max, min, stddev
import yfinance as yf
import pandas as pd

# Initialize the router
router = APIRouter()

# Initialize the Spark session
spark = SparkSession.builder \
    .appName("SparkProcessor") \
    .getOrCreate()

def fetch_historical_data(symbol: str, start_date: str, end_date: str):
    """
    Fetch historical stock data from Yahoo Finance.
    :param symbol: Stock ticker symbol (e.g., "AAPL").
    :param start_date: Start date for the historical data (e.g., "2022-01-01").
    :param end_date: End date for the historical data (e.g., "2023-01-01").
    :return: A Pandas DataFrame containing the historical stock data.
    """
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date, interval="1d")
    if not data.empty:
        return data.reset_index() # Reset index to make it Spark-compatible
    return pd.DataFrame() # Return empty DataFrame if no data is available

@router.post("/historical")
def analyze_historical_data(symbol: str, start_date: str, end_date: str):
    """
    Analyze historical stock data using Spark.
    :param symbol: Stock ticker symbol.
    :param start_date: Start date for historical data.
    :param end_date: End date for historical data.
    :return: Analysis results.
    """
    # Fetch historical data from Yahoo Finance
    historical_data = fetch_historical_data(symbol, start_date, end_date)

    if historical_data.empty:
        return {"error": f"No data found for {symbol} between {start_date} and {end_date}"}

    # Convert Pandas DataFrame to Spark DataFrame
    df = spark.createDataFrame(historical_data)

    # Perform analysis on the Spark DataFrame
    analysis = df.selectExpr(
        "avg(Close) as average_close",
        "max(Close) as max_close",
        "min(Close) as min_close",
        "stddev(Close) as volatility"
    ).collect()

    # Convert the results to a Python dictionary
    results = {row.asDict().keys()[0]: row.asDict() for row in analysis}

    return {"symbol": symbol, "analysis": results}

@router.post("/process-realtime")
def process_realtime_data(kafka_messages: list):
    """
    Process real-time data consumed from Kafka using Spark.
    :param kafka_messages: List of messages fetched from Kafka.
    :return: Processed analysis results.
    """
    if not kafka_messages:
        return {"error": "No real-time data to process"}

    # Convert Kafka messages (list of dictionaries) to a Spark DataFrame
    df = spark.createDataFrame(kafka_messages)

    # Perform analysis (e.g., average, max, min prices)
    analysis = df.groupBy("symbol").agg(
        avg("close").alias("average_close"),
        max("close").alias("max_close"),
        min("close").alias("min_close"),
        stddev("close").alias("volatility")
    ).collect()

    # Convert the results to a Python dictionary
    results = [row.asDict() for row in analysis]

    return {"realtime_analysis": results}
