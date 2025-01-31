import asyncio
import json
import logging
from fastapi import FastAPI, WebSocket
from confluent_kafka import Consumer
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
from finance_tool.db import get_db
from finance_tool.models import StockData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Kafka Consumer Configuration
consumer_conf = {
    "bootstrap.servers": "localhost:9092",
    "group.id": "fastapi-consumer",
    "auto.offset.reset": "earliest",
}
consumer = Consumer(consumer_conf)


async def save_to_db(stock_data: dict, db_session: AsyncSession):
    """
    Save stock data to the database.
    """
    try:
        # Convert timestamp to a naive datetime object
        timestamp = datetime.fromisoformat(stock_data["timestamp"])
        if timestamp.tzinfo is not None:
            timestamp = timestamp.replace(tzinfo=None)

        new_data = StockData(
            symbol=stock_data["symbol"],
            price=stock_data["price"],
            timestamp=timestamp,
        )

        logger.info(f"📥 Saving to DB: {new_data}")
        db_session.add(new_data)
        await db_session.flush()
        await db_session.commit()
        logger.info(f"✅ Data saved successfully: {stock_data}")

    except Exception as e:
        logger.error(f"❌ Error saving to DB: {e}", exc_info=True)
        await db_session.rollback()
    finally:
        await db_session.close()
        logger.info("🔒 Database session closed.")


async def consume_kafka_messages():
    """
    Asynchronously consume messages from Kafka and save them to PostgreSQL.
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
            logger.info(f"📥 Received Kafka message: {stock_data}")

            # Use an async database session
            async with get_db() as db_session:
                await save_to_db(stock_data, db_session)

    except Exception as e:
        logger.error(f"❌ Kafka Consumer Error: {e}", exc_info=True)
    finally:
        consumer.close()


@app.on_event("startup")
async def startup_event():
    """
    Start Kafka consumer as a background task when FastAPI starts.
    """
    loop = asyncio.get_event_loop()
    loop.create_task(consume_kafka_messages())
    logger.info("🚀 Kafka Consumer Started!")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time stock updates.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"📥 WebSocket received: {data}")

            # Broadcast data to all connected clients
            await websocket.send_text(f"ACK: {data}")
    except Exception as e:
        logger.error(f"❌ WebSocket Error: {e}", exc_info=True)
