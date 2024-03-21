from pyspark.sql import functions as F
class Transform:

    # TODO 3 - Add a constructor here
    def __init__(self, spark):
        self.spark = spark

    def transform_data(self,customer_df,order_items_df,order_payments_df,order_reviews,orders_df,products_df,sellers_df,product_category_name_translation_df):
        print("Transforming Data Frames")
        orders_df=orders_df[orders_df['order_status']=='delivered']

	#filling null values of order_approved_at column with order_purchase_timestamp
        orders_df = orders_df.withColumn('order_delivered_customer_date',F.coalesce(F.col('order_delivered_customer_date'),F.col('order_estimated_delivery_date')))

	#droping order_delivered_carrier_date column as it is not relevant also its missing values couldn't be replaced 
        orders_df = orders_df.drop('order_delivered_carrier_date')
        orders_df = orders_df.drop('order_approved_at')
        products_df = products_df.fillna("unknown",subset='product_category_name')
        products_df = products_df.drop('product_name_lenght')
        products_df = products_df.drop('product_description_lenght')
        products_df = products_df.drop('product_photos_qty')
        products_df = products_df.drop('product_weight_g')
        products_df = products_df.drop('product_length_cm')
        products_df = products_df.drop('product_height_cm')
        products_df = products_df.drop('product_width_cm')
        order_reviews = order_reviews.dropna(how='any',subset=['order_id','review_id','review_score','review_creation_date',
                                                       'review_answer_timestamp'])
        order_reviews = order_reviews.drop('review_comment_title')
        order_reviews = order_reviews.drop('review_comment_message')
        sellers_df = sellers_df.drop('seller_zip_code_prefix')
        joined_df = orders_df.join(order_items_df,on='order_id',how='inner').join(order_payments_df,on='order_id',how='inner').join(customers_df,on='customer_id',how='inner').join(order_reviews,on='order_id',how='inner').join(products_df,on='product_id',how='inner').join(sellers_df,on='seller_id',how='inner').join(product_category_name_translation_df,on='product_category_name',how='inner')


        
        return joined_df
