from finance_tool.routers import kafka_producer, kafka_consumer, spark_processor
from finance_tool.db import engine
from finance_tool.models import Base
import logging
from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from finance_tool.db import get_db
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(docs_url="/docs", redoc_url=None)

# Include routers
app.include_router(kafka_producer.router, prefix="/producer", tags=["Producer"])
app.include_router(kafka_consumer.router, prefix="/consumer", tags=["Consumer"])
app.include_router(spark_processor.router, prefix="/spark", tags=["Spark"])


@app.on_event("startup")
async def startup_event():
    """
    On app startup: Initialize the database and start Kafka producer/consumer.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    kafka_producer.start_producer()
    kafka_consumer.start_consumer()
    logger.info("🚀 App is ready.")



@app.get("/db-test")
async def db_test(db: AsyncSession = Depends(get_db)):
    await db.execute(text("INSERT INTO stock_data (symbol, timestamp) VALUES ('AAPL', '2025-01-28T15:59:00-05:00')"))
    await db.commit()


@app.get("/")
def root():
    return {"message": "FastAPI with Kafka, Spark, and PostgreSQL is running!"}
