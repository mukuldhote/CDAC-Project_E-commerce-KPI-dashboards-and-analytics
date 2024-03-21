from kafka import KafkaProducer
import pandas as pd
import json
from time import sleep
import concurrent.futures

def run_producer(file, topic, producer):
    counter = 0
    for chunk in pd.read_csv(file, chunksize=35):
        for _, row in chunk.iterrows():
            row_dict = row.to_dict()
            data = json.dumps(row_dict, default=str).encode('utf-8')
            producer.send(topic=topic, key=str(counter).encode(), value=data)
            counter += 1
        sleep(19)  # Optional: Introduce a small delay for better sequencing single file so 10 to sync
        # fi we using 5 tables then use 20 second to 20 second dash board

# Kafka configuration
kafka_bootstrap_servers = '127.0.0.1:9092'

# Create a Kafka producer
producer = KafkaProducer(bootstrap_servers=kafka_bootstrap_servers)

# List of files to process
files = ['joined_data_set.csv']

# List of topics
kafka_topics = ['newer']

with concurrent.futures.ThreadPoolExecutor() as executor:
    # Submit tasks to create topics concurrently
    # Submit tasks to run producers concurrently
    futures = [executor.submit(run_producer, file, topic, producer) for file, topic in zip(files, kafka_topics)]
    concurrent.futures.wait(futures)

producer.close()
