from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import IndexToString
from pyspark.ml.linalg import DenseVector, SparseVector
from pyspark.ml.feature import OneHotEncoder
from sparktorch import serialize_torch_obj, SparkTorch
import torch
import torch.nn as nn

from pyspark import SparkConf, SparkContext

def sparseVector_join(x):
	key = x[0]
	v1 = x[1][0]
	v2 = x[1][1]
	size = v1.size + v2.size
	maxIndex = v1.size
	combined_indices = v1.indices.tolist() + list(map(lambda i : i + maxIndex, v2.indices))
	combined_values = v1.values.tolist() + v2.values.tolist()
	return (key, SparseVector(size, combined_indices, combined_values))

def row_parse(row):
	return (row['customer_id'], (row['pid_ohe'], row['event_ohe']))

def groupBy_session(x1, x2):
	if(isinstance(x1, list) and isinstance(x2, list)):
		x1.extend(x2)
		return x1
	if(isinstance(x1, list)):
		x1.append(x2)
		return x1
	elif(isinstance(x2, list)):
		x2.append(x1)
		return x2
	else:
		l = []
		l.append(x1)
		l.append(x2)
		return l

def slice_action(x):
	action = x[1]
	if(isinstance(action, SparseVector)):
		return None
	length = len(action)
	size = 5
	input_batch = []
	target_batch = []
	if(length < 6):
		return None
	for i in range(0, length-size-1):
		input_chunk = []
		for j in range(i, i+size):
			input_chunk.append(action[j])
		input_batch.append(input_chunk.copy())
		target_batch.append(action[i+size])
	return (x[0], (input_batch, target_batch))

if __name__=="__main__":
	spark = SparkSession.builder.appName("rnn2").getOrCreate()
	train_ohe = spark.read.parquet("hdfs:///user/maria_dev/otto-recommender-system/train_ohe.parquet")
	drop_cols = ["pid_enc", "event_enc"]
	train_ohe = train_ohe.drop(*drop_cols)
	train_ohe.show()
	print(train_ohe.rdd.take(10))

	# 2개의 sparse vector 합치기
	train_rdd = train_ohe.rdd.map(row_parse)
	print("\n", train_rdd.take(10))
	train_rdd = train_rdd.map(sparseVector_join)
	print("\n", train_rdd.take(10))	

	# session별로 활동 모으기
	train_rdd = train_rdd.reduceByKey(groupBy_session)
	print("\n", train_rdd.take(10))

	# session 활동 5개씩 잘라서 넣기
	train_rdd = train_rdd.map(slice_action)
	print("\n", train_rdd.take(10))
	
	# RNN 모델 객체 생성
	model = nn.RNN(input_size=34671, hidden_size=2, dropout=0)
	torch_obj = serialize_torch_obj(
		model = model,
		criterion = nn.CrossEntropyLoss(),
		optimizer=torch.optim.Adam,
		lr=0.01)
