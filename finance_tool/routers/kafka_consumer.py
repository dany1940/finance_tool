import asyncio
import json
import logging
from confluent_kafka import Consumer
from sqlalchemy.ext.asyncio import AsyncSession
from finance_tool.db import get_db
from finance_tool.models import StockData
from fastapi import APIRouter
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
consumer_conf = {
    "bootstrap.servers": "localhost:9092",
    "group.id": "fastapi-consumer",
    "auto.offset.reset": "earliest",
}
router = APIRouter()
consumer = Consumer(consumer_conf)


async def save_to_db(stock_data: dict, db_session: AsyncSession):
    """
    Save stock data to the database with a proper session.
    """
    async for db_session in db_session:
        try:
            # Assuming stock_data['timestamp'] is a timezone-aware datetime object
            timestamp = stock_data["timestamp"]
            timestamp = datetime.fromisoformat(stock_data["timestamp"])
            # Convert timezone-aware datetime to naive (removing tzinfo)
            if timestamp.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=None)

            # Insert the stock data into the database
            new_data = StockData(
                symbol=stock_data["symbol"],
                timestamp=timestamp,
            )
            logger.info(f"Saving data: {new_data}")
            db_session.add(new_data)
            await db_session.flush()
            await db_session.commit()
            logger.info(f"✅ Saved to DB: {stock_data}")
        except Exception as e:
            logger.error(f"❌ Error saving to DB: {e}") # Log the error
            await db_session.rollback()
        finally:
            await db_session.close()
            logger.info("🔒 Database session closed.")


def consume_messages_continuously():
    """
    Consume messages from Kafka and save them to the database.
    """
    consumer.subscribe(["stock_data"])
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error(f"Consumer error: {msg.error()}")
                continue
            stock_data = json.loads(msg.value())
            logger.info(f"Received: {stock_data}")

            # Fix: Ensure each message gets its own session
            asyncio.run(save_to_db(stock_data, get_db()))
    except Exception as e:
        logger.error(f"❌ Error consuming message: {e}")
    finally:
        consumer.close()


def start_consumer():
    """
    Start Kafka consumer in a separate thread.
    """
    from threading import Thread
    consumer_thread = Thread(target=consume_messages_continuously, daemon=True)
    consumer_thread.start()
    consumer_thread.join()  # Wait for the thread to finish
