import json
import logging
from datetime import timedelta

import redis

# Initialize Redis
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cache_data(key: str, data: dict, ttl: int = 600):
    """
    Store stock data in Redis cache with a TTL (default: 10 minutes).
    """
    redis_client.setex(key, timedelta(seconds=ttl), json.dumps(data))
    logger.info(f"✅ Data cached: {key}")


def get_cached_data(key: str):
    """
    Retrieve stock data from Redis cache if available.
    """
    cached_data = redis_client.get(key)
    if cached_data:
        logger.info(f"♻️ Serving from cache: {key}")
        return json.loads(cached_data)
    return None
