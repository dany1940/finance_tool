import asyncio
import json
import logging
from confluent_kafka import Consumer
from finance_tool.db import get_db
from finance_tool.models import BinanceModel, CoinbaseModel
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kafka Consumer Configuration
consumer_conf = {
    "bootstrap.servers": "localhost:9092",
    "group.id": "fastapi-consumer",
    "auto.offset.reset": "earliest",
}
consumer = Consumer(consumer_conf)

shutdown_event = asyncio.Event()  # Shutdown event for graceful shutdown

async def save_to_db(data: dict, db_session):
    """
    Save stock data to the database, choosing between Binance and Coinbase based on the 'exchange' field
    """
    try:
        exchange_name = data.get("exchange")

        # Convert timestamp to a naive datetime object
        timestamp = datetime.fromisoformat(data["timestamp"])
        if timestamp.tzinfo is not None:
            timestamp = timestamp.replace(tzinfo=None)

        if exchange_name == "binance":
            binance_data = BinanceModel(
                symbol=data["symbol"],
                price=data["price"],
                best_ask=data["best_ask"],
                best_ask_size=data["best_ask_size"],
            )
            db_session.add(binance_data)
            await db_session.commit()
            logger.info(f"📥 Data saved to Binance table: {binance_data}")

        elif exchange_name == "coinbase":
            coinbase_data = CoinbaseModel(
                symbol=data["symbol"],
                price=data["price"],
                best_ask=data["best_ask"],
                best_ask_size=data["best_ask_size"],
            )
            db_session.add(data)
            await db_session.flush() # Use await for async flush
            await db_session.commit() # Use await for async commit
            logger.info(f"📥 Data saved to Coinbase table: {coinbase_data}")

        else:
            logger.warning(f"⚠️ Unsupported exchange: {exchange_name}")

    except Exception as e:
        logger.error(f"❌ Error saving to DB: {e}", exc_info=True)
        await db_session.rollback()


async def consume_kafka_messages():
    """
    Asynchronously consume messages from Kafka and save them to PostgreSQL.
    Gracefully handle shutdown.
    """
    consumer.subscribe(["binance_ticker", "coinbase_ticker"])  # Subscribe to both topics

    try:
        while True:
            if shutdown_event.is_set():  # Check if shutdown event is triggered
                logger.info("🔴 Shutdown event triggered. Stopping Kafka consumer.")
                break

            msg = consumer.poll(1.0)  # Poll for messages with 1-second timeout
            if msg is None:
                continue
            if msg.error():
                logger.error(f"Consumer error: {msg.error()}")
                continue

            data = json.loads(msg.value())
            logger.info(f"📥 Received Kafka message: {data}")
            # Ensure each message gets its own session
            async for db_session in get_db():
                await save_to_db(data, db_session)
    except Exception as e:
        logger.error(f"❌ Kafka Consumer Error: {e}", exc_info=True)
    finally:
        consumer.close()


def start_consumer():
    """
    Start Kafka consumer in a background task.
    """
    loop = asyncio.get_event_loop()
    loop.create_task(consume_kafka_messages())
    logger.info("🚀 Kafka Consumer Started!")

def stop_consumer():
    """
    Stop Kafka consumer by setting the shutdown event.
    """
    shutdown_event.set()  # Trigger the shutdown event
    logger.info("⚠️ Kafka Consumer is stopping...")
