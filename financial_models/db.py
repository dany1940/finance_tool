import os

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in the environment variables.")

# Create Async Database Engine
engine = create_async_engine(DATABASE_URL, echo=True, future=True, poolclass=NullPool)

# Create Session Factory
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()


# Dependency for FastAPI Routes
async def get_db():
    async with async_session() as session:
        yield session
