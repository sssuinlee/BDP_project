from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import *

if __name__ == "__main__":
	spark = SparkSession.builder.appName("test").getOrCreate()
	
	#Preprocessing
	train_data = spark.read.json("hdfs:///user/maria_dev/otto-recommender-system/train.jsonl")
	train_data = train_data.select(col("session").alias("customer_id"), explode("events").alias("events")) \
			.select("customer_id", col("events.aid").alias("product_id"), date_format(to_timestamp("events.ts"),'dd-MM-yy HH:mm').alias("time_stamp"), \
			col("events.type").alias("event_type"))

	train_data.schema
	train_data.show()
