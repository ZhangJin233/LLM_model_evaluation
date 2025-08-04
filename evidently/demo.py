import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from pyspark.sql.types import *
from pyspark.sql import SparkSession
import pyspark.sql.functions as fun
from pyspark.conf import SparkConf
from pyspark.sql.functions import *
from pyspark.sql import types as T
from pyspark.sql import SparkSession
from pyspark.sql import functions as F, DataFrame
from functools import reduce

# 1. 直接读取 GCS 上的 Parquet 文件为 DataFrame
# gcs_path = "gcs://<your-bucket>/<path>/training_data.parquet"
# df = pd.read_parquet(gcs_path, storage_options={"token": "cloud"})  
# print(f"Loaded dataset with shape: {df.shape}")

# 1. 直接读取 GCS 上的 Parquet 文件为 DataFrame
spark = (
    SparkSession.builder.master("local[*]")
    .appName("Apple")
    .config("spark.driver.memory", "8g")
    .config("spark.sql.source.partitionOverwriteMode", "dynamic")
    .config("spark.sql.ansi.enabled", "false")
    .getOrCreate()
)

reference_data = "/Users/jinzhang/Documents/date=2025-07-13/partitionId=0/part-00007-8ce856a1-3bbe-4fc2-9bea-3d5a6d6eb47e.c000.snappy.parquet" 
current_data = "/Users/jinzhang/Documents/date=2025-07-13/partitionId=1/part-00027-8ce856a1-3bbe-4fc2-9bea-3d5a6d6eb47e.c000.snappy.parquet"
df_ref = spark.read.parquet(reference_data)
df_cur = spark.read.parquet(current_data)
# df.show(100, False)

# 将PySpark DataFrame转换为pandas DataFrame
pandas_df_ref = df_ref.toPandas()
pandas_df_cur = df_cur.toPandas()

# 2. 使用 Evidently 生成数据质量报告
data = Report(metrics=[DataDriftPreset()])
report = data.run(current_data=pandas_df_cur, reference_data=pandas_df_ref)

# 3. 将报告保存为 HTML 文件
report.save_html("data_quality_report.html")
print("Evidently Data Quality report saved to data_quality_report.html")
