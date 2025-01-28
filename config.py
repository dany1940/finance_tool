import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db") # Default to SQLite
    SECRET_KEY = os.getenv("SECRET_KEY", "defaultsecret")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

settings = Settings()
