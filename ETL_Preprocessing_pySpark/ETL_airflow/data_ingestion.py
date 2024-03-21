class Ingest:

    def __init__(self, spark):
        # A class level variable
        self.spark = spark

    def ingest_data(self):
        print("Ingesting from HDFS")

        dir_path = 'hdfs:///user/talentum/olist_dataset/'
        customers_df = spark.read.csv(dir_path+'olist_customers_dataset.csv',inferSchema=True,header=True)
        order_items_df = spark.read.csv(dir_path+'olist_order_items_dataset.csv',inferSchema=True,header=True)
        order_payments_df = spark.read.csv(dir_path+'olist_order_payments_dataset.csv',inferSchema=True,header=True)
        order_reviews = spark.read.csv(dir_path+'olist_order_reviews_dataset.csv',inferSchema=True,header=True)
        orders_df = spark.read.csv(dir_path+'olist_orders_dataset.csv',inferSchema=True,header=True)
        products_df = spark.read.csv(dir_path+'olist_products_dataset.csv',inferSchema=True,header=True)
        sellers_df = spark.read.csv(dir_path+'olist_sellers_dataset.csv',inferSchema=True,header=True)
        product_category_name_translation_df = spark.read.csv(dir_path+'product_category_name_translation.csv',inferSchema=True,header=True)
        return customer_df,order_items_df,order_payments_df,order_reviews,orders_df,products_df,sellers_df,product_category_name_translation_df
