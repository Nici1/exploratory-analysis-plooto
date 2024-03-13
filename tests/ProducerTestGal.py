import csv
import json
from time import sleep
from json import dumps
from kafka import KafkaProducer
import numpy as np
from datetime import datetime

#Define producer
producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x: 
                         dumps(x).encode('utf-8'))


# "load real data from Continental, send it to kafka topic"

stored_data = np.loadtxt("../data/Continental/1", skiprows=1, delimiter = ",", usecols=(1,))

for i in range(len(stored_data)):

    # "Artificially add some anomalies"
    if(i%20 == 0):
        ran = np.random.choice([-1, 1])*5
    else:
        ran = 0
    value = stored_data[i] + ran
    data = {"test_value" : [value],
			"timestamp": str(datetime.now())}

    producer.send('anomaly_detection', value=data)
    sleep(1) #one data point each second
