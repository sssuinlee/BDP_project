from pyspark.sql import SparkSession
from pyspark.sql.functions import *

if __name__ == "__main__":
	spark = SparkSession.builder.appName("test").getOrCreate()
	
	# train data 로드
	train_data = spark.read.json("hdfs:///user/maria_dev/otto-recommender-system/train.jsonl")
	
	# column 분리 
	train_data = train_data.select(col("session").alias("customer_id"), explode("events").alias("events")) \
			.select("customer_id", col("events.aid").alias("product_id"), date_format(to_timestamp("events.ts"),'dd-MM-yy HH:mm').alias("time_stamp"), \
			col("events.type").alias("event_type"))
	
	# timestamp column 제거
	train_data = train_data.drop(col("time_stamp"))

	# parquet 형태로 hdfs에 저장
	train_data.write.parquet("hdfs:///user/maria_dev/otto-recommender-system/train_data.parquet")
