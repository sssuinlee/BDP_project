from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import IndexToString
from pyspark.ml.linalg import DenseVector
from pyspark.ml.feature import OneHotEncoder
from sparktorch import serialize_torch_obj, SparkTorch
import torch
import torch.nn as nn

if __name__ == "__main__":
	# parquet data load 후 DataFrame 변환
	spark = SparkSession.builder.appName("rnn").getOrCreate()
	train_data = spark.read.parquet("hdfs:///user/maria_dev/otto-recommender-system/train.parquet")
	train_data.show(10)
	print("filtering 전 row 개수", train_data.count())
	print("\nfiltering 전 schema", train_data.schema)
	print("\nfiltering 전 datatype", train_data.dtypes)
	print("\nfiltering 전 pid 개수", train_data.select("product_id").distinct().count())
	print("\nfiltering 전 session 개수", train_data.select("customer_id").distinct().count())
	
	# 데이터 분포 확인
	count_attention = train_data.groupBy("product_id").agg(count("customer_id").alias("attention_to_pid")).orderBy(col("attention_to_pid"))
	attention_aggr = count_attention.groupBy("attention_to_pid").agg(count("product_id").alias("count")).orderBy(col("attention_to_pid"))

	# 관심 횟수 100번 이하인 상품을 포함하는 tuple  잘라내기
	pid_filter = count_attention.where("attention_to_pid > 1000").select("attention_to_pid", col("product_id").alias("filtered_product_id"))
	train_filter = train_data.join(pid_filter, train_data["product_id"]==pid_filter["filtered_product_id"])
	drop_cols = ["filtered_product_id", "time_stamp", "attention_to_pid"]
	train_filter = train_filter.drop(*drop_cols)
	train_filter.show(10)
	print("filtering 후 row 개수", train_filter.count())
	print("\nfiltering 후 schema", train_filter.schema)
	print("\nfiltering 후 datatype", train_filter.dtypes)
	print("\nfiltering 후 pid 개수", train_filter.select("product_id").distinct().count())
#	print("\nfiltering 후 session 개수", train_filter.select("customer_id").distinct().count())
	
	# One-Hot-Encoding
	# string -> numeric type으로 인코딩
	encoder_pid = StringIndexer(inputCol='product_id', outputCol='pid_enc')
	encoder_event = StringIndexer(inputCol='event_type', outputCol='event_enc')
	train_enc = encoder_pid.fit(train_filter).transform(train_filter)
	train_enc = encoder_event.fit(train_enc).transform(train_enc)
	print("\nEncoding 후")
	train_enc.show(10)

	# 인코딩 전 컬럼 삭제
	drop_cols = ["product_id", "event_type"]
	test = train_enc.drop(*drop_cols)
	test = test.where("customer_id < 50")

	# 인코딩된 컬럼으로 ohe
	ohe_pid = OneHotEncoder(inputCol='pid_enc', outputCol='pid_ohe', dropLast=False)
	ohe_event = OneHotEncoder(inputCol='event_enc', outputCol='event_ohe', dropLast=False)
	train_ohe = ohe_pid.transform(test)
	train_ohe = ohe_event.transform(train_ohe)
	train_ohe.show(10)

	train_ohe.write.parquet("hdfs:///user/maria_dev/otto-recommender-system/train_ohe.parquet")
