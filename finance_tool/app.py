import asyncio
import logging
from fastapi import FastAPI, WebSocket
from finance_tool.routers.kafka_consumer import start_consumer, stop_consumer  # Import the consumer functions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """
    Start Kafka consumer as a background task when FastAPI starts.
    """
    start_consumer()
    logger.info("🚀 FastAPI application started, Kafka consumer is running!")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Stop Kafka consumer when FastAPI shuts down.
    """
    stop_consumer()
    logger.info("⚠️ FastAPI is shutting down, Kafka consumer is stopping...")

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
