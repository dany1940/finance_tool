import logging
from fastapi import FastAPI, WebSocket
from finance_tool.routers.kafka_consumer import router as kafka_router # Import router
from finance_tool.routers.spark_processor import router as spark_router # Import router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Real-Time Stock Data API",
    description="This API provides real-time stock data from Binance and Coinbase.",
    version="1.1",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ✅ Register Kafka Consumer Routes (Now Optional)
app.include_router(kafka_router)
app.include_router(spark_router)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time stock updates."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"📥 WebSocket received: {data}")
            await websocket.send_text(f"ACK: {data}")
    except Exception as e:
        logger.error(f"❌ WebSocket Error: {e}", exc_info=True)
