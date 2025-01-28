import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Load environment variables from .env file
load_dotenv()

# Get database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL")

# Create the database engine
engine = create_async_engine(DATABASE_URL, echo=True)

# Create a session factory
async_session = sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)

# Dependency to get the database session
async def get_db():
    async with async_session() as session:
        yield session
