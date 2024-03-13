import csv
import json
from time import sleep
from json import dumps
from kafka import KafkaProducer
import numpy as np
from datetime import datetime
import time

#Define producer
producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x: 
                         dumps(x).encode('utf-8'))



#Send random data to topic - sine with noise in range 2-4

tab_data = [3, 4, 4, 4, 4, 5, 5, 5]
tab_data_csv = []

timestamp = time.time()

for e in range(5000):
    timestamp += 5*60
    ran = float(np.random.normal(15, 5))
    
    data = {"ftr_vector" : [ran],
			"timestamp": timestamp}
    
    producer.send('features_braila_pressure5770_anomaly', value=data)
    producer.send('features_braila_pressure5771_anomaly', value=data)
    sleep(0.1) #one data point each second

