from fastapi import FastAPI
import subprocess
import time
from routers.kafka_producer import start_producer
from routers.kafka_consumer import start_consumer
from routers import spark_processor

app = FastAPI()

# Include routers for Spark processing
app.include_router(spark_processor.router, prefix="/spark", tags=["Spark Processor"])

def start_zookeeper():
    """
    Start Zookeeper server.
    """
    print("Starting Zookeeper...")
    subprocess.Popen(
        ["zookeeper-server-start", "/usr/local/etc/kafka/zookeeper.properties"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )
    time.sleep(5)  # Allow Zookeeper to initialize

def start_kafka():
    """
    Start Kafka server.
    """
    print("Starting Kafka...")
    subprocess.Popen(
        ["kafka-server-start", "/usr/local/etc/kafka/server.properties"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )
    time.sleep(5)  # Allow Kafka to initialize

@app.on_event("startup")
def startup_event():
    """
    Automatically start Zookeeper, Kafka, Kafka producer, and Kafka consumer.
    """
    print("Starting services...")
    start_zookeeper()  # Start Zookeeper
    start_kafka()      # Start Kafka
    start_producer()   # Start Kafka producer
    start_consumer()   # Start Kafka consumer
    print("All services started successfully!")

@app.get("/")
def root():
    """
    Root endpoint to check if the application is running.
    """
    return {"message": "FastAPI with Kafka and Spark is running!"}
