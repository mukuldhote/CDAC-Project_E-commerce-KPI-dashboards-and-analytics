from kafka.admin import KafkaAdminClient, NewTopic
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col,expr,to_json,struct
from pyspark.sql.types import StructType, StructField, StringType, IntegerType,DatetimeConverter,DateType,DoubleType,TimestampType,FloatType
import json
from time import sleep
import concurrent.futures


spark = SparkSession.builder \
    .appName('KafkaConsumerVisualization') \
    .enableHiveSupport() \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
kafka_bootstrap_servers = '127.0.0.1:9092'
dash_topic='dash_topic'
kafka_topics = ['orders','customers','order_items', 'order_payments','order_reviews']


# - olist_orders
#  0   order_id                       object        
#  1   customer_id                    object        
#  2   order_status                   object        
#  3   order_purchase_timestamp       datetime64[ns]
#  4   order_approved_at              datetime64[ns]
#  5   order_delivered_carrier_date   datetime64[ns]
#  6   order_delivered_customer_date  datetime64[ns]
#  7   order_estimated_delivery_date  datetime64[ns]
csv_schema1 = StructType([
    StructField("order_id1", StringType(), True),
    StructField("customer_id1", StringType(), True),
    StructField("order_status", StringType(), True),
    StructField("order_purchase_timestamp", TimestampType(), True),
    StructField("order_approved_at", TimestampType(), True),
    StructField("order_delivered_carrier_date", TimestampType(), True),
    StructField("order_delivered_customer_date", TimestampType(), True),
    StructField("order_estimated_delivery_date", TimestampType(), True)
])

# - olist_customer
#  0   customer_id               object
#  1   customer_unique_id        object
#  2   customer_zip_code_prefix  int64 
#  3   customer_city             object
#  4   customer_state            object
csv_schema2 = StructType([
    StructField("customer_id", StringType(), True),
    StructField("customer_unique_id", StringType(), True),
    StructField("customer_zip_code_prefix", IntegerType(), True),
    StructField("customer_city", StringType(), True),
    StructField("customer_state", StringType(), True)
    
])

# - olist_order_items
# 0   order_id             object        
# 1   order_item_id        int64         
# 2   product_id           object        
# 3   seller_id            object        
# 4   shipping_limit_date  datetime64[ns]
# 5   price                float64       
# 6   freight_value        float64  
csv_schema3 = StructType([
    StructField("order_id", StringType(), True),
    StructField("order_item_id", IntegerType(), True),
    StructField("product_id", StringType(), True),
    StructField("seller_id", StringType(), True),
    StructField("shipping_limit_date", TimestampType(), True),
    StructField("price", FloatType(), True),
    StructField("freight_value", FloatType(), True)
])

# - olist_order_payments
# 0   order_id              103886 non-null  object 
# 1   payment_sequential    103886 non-null  int64  
# 2   payment_type          103886 non-null  object 
# 3   payment_installments  103886 non-null  int64  
# 4   payment_value         103886 non-null  float64
csv_schema4 = StructType([
    StructField("order_id", StringType(), True),
    StructField("payment_sequential", IntegerType(), True),
    StructField("payment_type", StringType(), True),
    StructField("payment_installments", IntegerType(), True),
    StructField("payment_value", FloatType(), True)
])

# - olist_order_reviews
# 0   review_id                99224 non-null  object
# 1   order_id                 99224 non-null  object
# 2   review_score             99224 non-null  int64 
# 3   review_comment_title     11568 non-null  object
# 4   review_comment_message   40977 non-null  object
# 5   review_creation_date     99224 non-null  object
# 6   review_answer_timestamp  99224 non-null  object
csv_schema5 = StructType([
    StructField("review_id", StringType(), True),
    StructField("order_id", StringType(), True),
    StructField("review_score", IntegerType(), True),
    StructField("review_creation_date", StringType(), True),
    StructField("review_answer_timestamp", StringType(), True)
])


# Read data from Kafka using Structured Streaming
orders = spark.readStream \
    .format('kafka') \
    .option('kafka.bootstrap.servers', '127.0.0.1:9092') \
    .option("subscribe", kafka_topics[0]) \
    .load() \
    .selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), csv_schema1).alias("data")) \
    .select("data.*")

customers = spark.readStream \
    .format('kafka') \
    .option('kafka.bootstrap.servers', '127.0.0.1:9092') \
    .option("subscribe", kafka_topics[1]) \
    .load() \
    .selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), csv_schema2).alias("data")) \
    .select("data.*")

order_items = spark.readStream \
    .format('kafka') \
    .option('kafka.bootstrap.servers', '127.0.0.1:9092') \
    .option("subscribe", kafka_topics[2]) \
    .load() \
    .selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), csv_schema3).alias("data")) \
    .select("data.*")

order_payments = spark.readStream \
    .format('kafka') \
    .option('kafka.bootstrap.servers', '127.0.0.1:9092') \
    .option("subscribe", kafka_topics[3]) \
    .load() \
    .selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), csv_schema4).alias("data")) \
    .select("data.*")


order_reviews = spark.readStream \
    .format('kafka') \
    .option('kafka.bootstrap.servers', '127.0.0.1:9092') \
    .option("subscribe", kafka_topics[4]) \
    .load() \
    .selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), csv_schema5).alias("data")) \
    .select("data.*")


# sellers
# 0   seller_id               3095 non-null   object
# 1   seller_zip_code_prefix  3095 non-null   int64
# 2   seller_city             3095 non-null   object
# 3   seller_state            3095 non-null   object
sellers = StructType([
    StructField("seller_id", StringType(), True),
    StructField("seller_zip_code_prefix", IntegerType(), True),
    StructField("seller_city", StringType(), True),
    StructField("seller_state", StringType(), True)
])


# products
# 0   product_id                  32951 non-null  object
# 1   product_category_name       32341 non-null  object
# 2   product_name_lenght         32341 non-null  float64
# 3   product_description_lenght  32341 non-null  float64
# 4   product_photos_qty          32341 non-null  float64
# 5   product_weight_g            32949 non-null  float64
# 6   product_length_cm           32949 non-null  float64
# 7   product_height_cm           32949 non-null  float64
# 8   product_width_cm            32949 non-null  float64

products = StructType([
    StructField("product_id", StringType(), True),
    StructField("product_category_name", StringType(), True),
    StructField("product_name_lenght", FloatType(), True),
    StructField("product_description_lenght", FloatType(), True),
    StructField("product_photos_qty", FloatType(), True),
    StructField("product_weight_g", FloatType(), True),
    StructField("product_length_cm", FloatType(), True),
    StructField("product_height_cm", FloatType(), True),
    StructField("product_width_cm", FloatType(), True)
])



olist_products = spark.read.schema(products).parquet("hdfs:///user/talentum/products").fillna({'product_category_name': 'unknown', 'product_name_lenght': 48.0,'product_description_lenght':771.0,'product_photos_qty':2.0}) #<-- to create throught spark to parquet to hdfs
olist_sellers = spark.read.schema(sellers).parquet("hdfs:///user/talentum/sellers")#<-- to create

#else try how = "left"
join_1c=[
orders['customer_id1']==customers['customer_id']
]

join_1=orders.join(customers,join_1c).drop('customer_id')

join_2c=[
join_1['order_id1']==order_items['order_id']
]

join_2=join_1.join(order_items,join_2c).drop('order_id')

join_3c=[
join_2['order_id1']==order_payments['order_id']
]

join_3=join_2.join(order_payments,join_3c).drop('order_id')

join_4c=[
join_3['order_id1']==order_reviews['order_id']
]

join_4=join_3.join(order_reviews,join_4c).drop('order_id')


join_5=join_4.join(olist_products,'product_id').join(olist_sellers,'seller_id')


# create check point in hdfs and construct the table in hive before hand
def writeToDash(writeDF, _):
    writeDF.select(to_json(struct("*")).alias("value"))\
	.writeStream \
        .format('kafka') \
        .option('kafka.bootstrap.servers', '127.0.0.1:9092') \
        .option('topic', 'dash_topic') \
	.option('checkpointLocation',"file:///home/talentum/Videos")\
        .start()
    #.option('checkpointLocation', '/hdfs') \
    

def writeToHive1(writeDF, _):
    writeDF.write \
        .mode('append') \
        .saveAsTable('ml_table')

#qury=join_5.writeStream \
#    .foreachBatch(writeToDash) \
#    .foreachBatch(writeToHive1) \
#    .outputMode("append") \
#    .start()

query = join_5 \
    .writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()
