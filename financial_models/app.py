# app.py

import logging

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
import fdm_gui  # This will automatically call fdm_gui layout
from nicegui import ui
from routers.finite_diff_endpoints import router as fdm_router
from routers.spark_processor import router as spark_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Real-Time Stock Data API",
    description="This API provides real-time stock data from Binance and Coinbase.",
    version="1.1",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Mount FastAPI routers
app.include_router(spark_router)
app.include_router(fdm_router)

# Mount NiceGUI UI to FastAPI (this will automatically trigger layout rendering)
app.mount("/downloads", StaticFiles(directory="downloads"), name="downloads")


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time communication.
    Accepts incoming messages and sends back an acknowledgment.
    """
    logger.info("🔗 WebSocket connection established")
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"📥 WebSocket received: {data}")
            await websocket.send_text(f"ACK: {data}")
    except Exception as e:
        logger.error(f"❌ WebSocket Error: {e}", exc_info=True)


ui.run_with(app, title="Real-Time Stock Data API")
