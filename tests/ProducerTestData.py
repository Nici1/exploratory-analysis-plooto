import csv
import json
from time import sleep
from json import dumps
from kafka import KafkaProducer
import numpy as np
import datetime
import pandas as pd
from ast import literal_eval

#Define producer
producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x: 
                         dumps(x).encode('utf-8'))


"load real data from Continental, send it to kafka topic"

df = pd.read_csv("../data/consumer/braila_pressure5770.csv", delimiter = ",")
#df = pd.read_csv("../data/consumer/conductivity_data.csv", delimiter = ",")

#df['time'] = pd.to_datetime(df['time'],unit='s')
#values = df['value']
#times = df['time']

values = df['ftr_vector'].values
times = df['timestamp'].values

#df['time'] = pd.to_datetime(df['time'],unit='s')

start = datetime.datetime(2021, 1, 4)

end = datetime.datetime(2021, 3, 15)

index = pd.date_range(start, end)

#averages = []
#times = []
#for i in index:
#    start_date = i
#    end_date = i + datetime.timedelta(hours=24)
#    mask = (df['time'] > start_date) & (df['time'] <= end_date) & (df['flow_rate_value']<1000)
#    sub_df = df.loc[mask]
#    if(len(sub_df) > 0 and sum(sub_df['flow_rate_value']) > 0):
#        averages.append(np.average(sub_df['flow_rate_value'][sub_df['flow_rate_value'] !=0]))
#        times.append(i)

for i in range(1000):

    "Artificially add some anomalies"
    #if(i%20 == 0):
    #    ran = np.random.choice([-1, 1])*5
    #else:
    #    ran = 0
    value = literal_eval(values[i])
    print(value)
    anomaly = 0
    if (i%20 == 0):
        anomaly = -0.03
    data = {"ftr_vector" : value,
			"timestamp": times[i]}

	
    producer.send("pressure_topic_5770", value=data)
    sleep(1) #one data point each second
