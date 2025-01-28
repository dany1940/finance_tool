from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from confluent_kafka import Consumer
from models import StockData
from db import get_db
import json
from threading import Thread

# Initialize router
router = APIRouter()

# Kafka consumer configuration
consumer_conf = {
    "bootstrap.servers": "localhost:9092",
    "group.id": "fastapi-consumer",
    "auto.offset.reset": "earliest",
}
consumer = Consumer(consumer_conf)

# Save data to PostgreSQL
async def save_to_db(db: AsyncSession, stock_data: dict):
    new_data = StockData(
        symbol=stock_data["symbol"],
        timestamp=stock_data["timestamp"],
        open=stock_data["open"],
        high=stock_data["high"],
        low=stock_data["low"],
        close=stock_data["close"],
        volume=stock_data["volume"]
    )
    db.add(new_data)
    await db.commit()

# Continuous consumer function
def consume_messages_continuously(db: AsyncSession):
    """
    Continuously consume messages from Kafka.
    """
    consumer.subscribe(["stock_data"])
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"Error: {msg.error()}")
                continue

            # Parse the Kafka message
            stock_data = json.loads(msg.value().decode("utf-8"))
            print(f"Consumed: {stock_data}")

            # Save to database
            asyncio.run(save_to_db(db, stock_data))
    finally:
        consumer.close()



# Start the consumer in a background thread
def start_consumer():
    """
    Start the Kafka consumer in a separate thread.
    """
    consumer_thread = Thread(
        target=lambda: consume_messages_continuously(next(get_db())),
        daemon=True
    )
    consumer_thread.start()

