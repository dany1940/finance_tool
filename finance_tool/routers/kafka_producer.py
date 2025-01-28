from confluent_kafka import Producer
import json
import time
from threading import Thread
import yfinance as yf

# Kafka producer configuration
producer_conf = {
    "bootstrap.servers": "localhost:9092",
    "client.id": "fastapi-producer",
    "enable.idempotence": True,
}
producer = Producer(producer_conf)

# Function to fetch stock data from Yahoo Finance
def fetch_realtime_data(symbol: str):
    """
    Fetch the latest stock data for a given symbol using Yahoo Finance.
    """
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="1d", interval="1m")
    if not data.empty:
        latest = data.iloc[-1]
        return {
            "symbol": symbol,
            "timestamp": latest.name.strftime("%Y-%m-%d %H:%M:%S"),
            "open": latest["Open"],
            "high": latest["High"],
            "low": latest["Low"],
            "close": latest["Close"],
            "volume": latest["Volume"],
        }
    return None

# Continuous producer function
def produce_messages_continuously():
    """
    Continuously produce stock data to Kafka.
    """
    symbols = ["AAPL", "GOOGL", "MSFT"]  # Stock symbols to track
    while True:
        for symbol in symbols:
            stock_data = fetch_realtime_data(symbol)
            if stock_data:
                producer.produce(
                    topic="stock_data",
                    key=stock_data["symbol"],
                    value=json.dumps(stock_data),
                    callback=lambda err, msg: print(f"Produced: {msg.topic()}") if not err else print(f"Error: {err}")
                )
        producer.flush()
        time.sleep(60)  # Fetch real-time data every minute

# Start the producer in a background thread
def start_producer():
    """
    Start the Kafka producer in a separate thread.
    """
    producer_thread = Thread(target=produce_messages_continuously, daemon=True)
    producer_thread.start()
