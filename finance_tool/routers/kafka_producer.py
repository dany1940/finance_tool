import json
import time
import logging
from threading import Thread
from confluent_kafka import Producer
import yfinance as yf
from fastapi import APIRouter

router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kafka producer configuration
producer_conf = {
    "bootstrap.servers": "localhost:9092",
    "client.id": "fastapi-producer",
}
producer = Producer(producer_conf)


def fetch_realtime_data(symbol):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="1d", interval="1m")
    if not data.empty:
        return {
            "symbol": symbol,
            "timestamp": data.index[-1].isoformat(),
            "open": data["Open"].iloc[-1],
            "high": data["High"].iloc[-1],
            "low": data["Low"].iloc[-1],
            "close": data["Close"].iloc[-1],
            "volume": int(data["Volume"].iloc[-1]),
        }
    return None


def produce_messages_continuously():
    """
    This function runs continuously in the background to fetch and send stock data to Kafka.
    """
    symbols = ["AAPL", "GOOGL", "MSFT"]  # Example stocks
    while True:
        for symbol in symbols:
            stock_data = fetch_realtime_data(symbol)
            if stock_data:
                producer.produce(
                    topic="stock_data",
                    key=stock_data["symbol"],
                    value=json.dumps(stock_data),
                )
                producer.flush()
                logger.info(f"📊 Produced data: {stock_data}")
        time.sleep(60)  # Fetch data every 60 seconds


def start_producer():
    """
    Starts the Kafka producer in a separate thread when FastAPI launches.
    """
    Thread(target=produce_messages_continuously, daemon=True).start()
