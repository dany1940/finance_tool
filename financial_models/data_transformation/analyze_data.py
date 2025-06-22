from pyspark.sql import SparkSession
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_spark_session():
    return SparkSession.builder.appName("StockAnalysis").getOrCreate()


def estimate_data_size(df):
    return df.shape[0] * df.shape[1] * 8 / 1e6


def analyze_stock_data(df):
    data_size = estimate_data_size(df)

    if data_size >= 50000:
        logger.info("Using Apache Spark for large dataset processing")
        spark = get_spark_session()
        spark_df = spark.createDataFrame(df.to_pandas())
        return spark_df.describe().toPandas().to_dict()

    logger.info("Using Polars for lightweight data processing")
    return df.describe().to_dict(as_series=False)
