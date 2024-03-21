from pyspark.sql import SparkSession
import os
import sys
import data_ingestion
import data_transformation
import processed_data_put
from pyspark.sql import SparkSession
class Pipeline:

    def run_pipeline(self):

        print("Running Pipeline")
        ingest_process = data_ingestion.Ingest(self.spark)
        customer_df,order_items_df,order_payments_df,order_reviews,orders_df,products_df,sellers_df,product_category_name_translation_df = ingest_process.ingest_data()
        
        tranform_process = data_transformation.Transform(self.spark)
        transformed_df = tranform_process.transform_data(customer_df,order_items_df,order_payments_df,order_reviews,orders_df,products_df,sellers_df,product_category_name_translation_df)
        

        persist_process = processed_data_put.Persist(self.spark)
        persist_process.persist_data(transformed_df)
        return

    def create_spark_session(self):
        # A class level variable
      
        self.spark = SparkSession.builder \
            .appName("my Ecommerce spark app") \
            .enableHiveSupport().getOrCreate()

if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.create_spark_session()
    pipeline.run_pipeline()

