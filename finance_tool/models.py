from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


# Coinbase Model
class CoinbaseModel(Base):
    __tablename__ = 'coinbase_data'

    # Mapping the JSON keys to model columns
    timestamp = Column("time", DateTime, primary_key=True) # time - Timestamp of the trade
    price = Column("price", Float) # price - Price of the asset in the trade
    volume = Column("last_size", Float) # last_size - Volume of the asset traded
    symbol = Column("product_id", String) # product_id - Symbol (BTC-USD, etc.)
    trade_id = Column("trade_id", Integer) # trade_id - Unique trade ID

    def __init__(self, timestamp, price, volume, symbol, trade_id):
        self.timestamp = datetime.utcfromtimestamp(timestamp / 1000) if isinstance(timestamp, int) else timestamp # Convert milliseconds to datetime
        self.price = price
        self.volume = volume
        self.symbol = symbol
        self.trade_id = trade_id

    def __repr__(self):
        return f"<CoinbaseModel(trade_id={self.trade_id}, symbol={self.symbol}, price={self.price}, volume={self.volume}, timestamp={self.timestamp})>"


# Binance Model
class BinanceModel(Base):
    __tablename__ = 'binance_data'

    # Mapping the JSON keys to model columns
    event_time = Column("E", Integer, primary_key=True) # E - Event time
    is_maker = Column("M", Boolean) # M - Is the trade maker
    timestamp = Column("T", Integer) # T - Timestamp of the trade
    price = Column("p", Float) # p - Price of the asset
    quantity = Column("q", Float) # q - Quantity of the asset traded
    symbol = Column("s", String) # s - Symbol (BTCUSDT, etc.)
    trade_id = Column("t", Integer) # t - Trade ID

    def __init__(self, event_time, is_maker, timestamp, price, quantity, symbol, trade_id):
        self.event_time = event_time
        self.is_maker = is_maker
        self.timestamp = datetime.utcfromtimestamp(timestamp / 1000) if isinstance(timestamp, int) else timestamp # Convert milliseconds to datetime
        self.price = price
        self.quantity = quantity
        self.symbol = symbol
        self.trade_id = trade_id

    def __repr__(self):
        return f"<BinanceModel(trade_id={self.trade_id}, symbol={self.symbol}, price={self.price}, quantity={self.quantity}, timestamp={self.timestamp}, is_maker={self.is_maker})>"
