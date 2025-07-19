from sqlalchemy import BigInteger, Boolean, Column, Float, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


# Coinbase Model
class CoinbaseModel(Base):
    __tablename__ = "coinbase_data"

    # Mapping the JSON keys to model columns
    unique_id = Column(String, primary_key=True)  # unique_id - Unique ID for the trade
    symbol = Column("symbol", String)  # product_id - Symbol (BTC-USD, etc.)
    trade_id = Column("trade_id", Integer)  # trade_id - Unique trade ID


class BinanceModel(Base):
    __tablename__ = "binance_data"

    # Mapping the JSON keys to model columns
    unique_id = Column(String, primary_key=True)  # Unique ID for the trade
    event_time = Column(BigInteger)  # E - Event time
    is_maker = Column(Boolean)  # M - Is the trade maker
    timestamp = Column(BigInteger)  # T - Timestamp of the trade
    price = Column(Float)  # p - Price of the asset
    quantity = Column(Float)  # q - Quantity of the asset traded
    symbol = Column(String)  # s - Symbol (BTCUSDT, etc.)
    trade_id = (Column(BigInteger),)  # t - Trade ID
