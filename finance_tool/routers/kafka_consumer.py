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
consumer.subscribe(["binance_ticker", "coinbase_ticker"])  # Subscribe to both topics

shutdown_event = asyncio.Event()  # Shutdown event for graceful shutdown

async def save_to_db(data: dict, db_session):
    """
    Save stock data to the database, choosing between Binance and Coinbase based on the 'exchange' field
    """
    try:
        exchange_name = data.get("exchange", "").lower()
        logger.info(f"📥 Saving data to DB for exchange: {exchange_name}")
        # Convert timestamp to a naive datetime object
        if exchange_name.lower() == "coinbase":
            data = CoinbaseModel(
                unique_id=data["uniqueId"],
                symbol=data["product_id"],
                trade_id=data["trade_id"]

            )
            logger.info(f"📥 Data saved to Coinbase table: {data}")

        elif exchange_name.lower() == "binance":
            data = BinanceModel(
                unique_id=data["uniqueId"],
                event_time=data["E"],
                is_maker=data["M"],
                timestamp=data["T"],
                price=float(data["p"]),
                quantity=float(data["q"]),
                symbol=data["s"],
                trade_id=data["t"],
            )

            logger.info(f"📥 Data saved to Binance table: {data}")

        else:
            logger.warning(f"⚠️ Unsupported exchange: {exchange_name}")
        db_session.add(data)

        await db_session.flush()
        await db_session.commit() # Use await for async flush

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
