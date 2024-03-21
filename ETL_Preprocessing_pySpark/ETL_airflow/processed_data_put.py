class Persist:
    def __init__(self, spark):
        self.spark = spark

    def persist_data(self, joined_df):
        print("Putting the processed data back on hdfs ")
        
        joined_df.write.csv("hdfs:///user/talentum/combined_dataset.csv")
