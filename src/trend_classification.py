from typing import Any, Dict, List
import numpy as np
import sys
import json
from pandas.core.frame import DataFrame
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from ast import literal_eval
#sys.path.insert(0,'./src')
#sys.path.insert(1, 'C:/Users/Matic/SIHT/anomaly_det/anomalyDetection/')
from src.anomalyDetection import AnomalyDetectionAbstract
from isolationForest import IsolationForest
from output import OutputAbstract, TerminalOutput, FileOutput, KafkaOutput
from visualization import VisualizationAbstract, GraphVisualization,\
    HistogramVisualization, StatusPointsVisualization
from normalization import NormalizationAbstract, LastNAverage,\
    PeriodicLastNAverage

class Trend_Classification(AnomalyDetectionAbstract):
    name: str = "Trend_Classification"

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None,
                  algorithm_indx: int = None) -> None:
        super().configure(conf, configuration_location=configuration_location,
                          algorithm_indx=algorithm_indx)

        # Train configuration
        self.num_samples = conf["num_samples"]
        self.N = conf["N"]
        self.averaging = conf["averaging"]
        self.prediction_conv = conf["prediction_conv"]
        self.train_noise = conf["train_noise"]
        self.warning_stages = conf["warning_stages"]
        self.amp_scale = conf["amp_scale"]

        self.memory = [] #sample memory
        self.prediction_memory = [] #NN result memory

        self.FV = [None]*self.N #Feature vector

        if(self.num_samples is not None):
            self.train_model()
        else:
            raise Exception("Model train configuration not given.")

    def message_insert(self, message_value: Dict[Any, Any]) -> Any:
        if(not self.check_ftr_vector(message_value=message_value)):
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE

            # if the names in the msg are wrong, there would be an error here
            #self.normalization_output_visualization(status=status,
            #                                    status_code=status_code,
            #                                    value=message_value["ftr_vector"],
            #                                    timestamp=message_value["timestamp"])
            
            # Remenber status for unittests
            self.status = status
            self.status_code = status_code
            return status, status_code

        super().message_insert(message_value)

        # Check feature vector
        

        value = message_value["ftr_vector"]
        value = value[0]

        if(self.use_cols is not None):
            value = []
            for el in range(len(message_value["ftr_vector"])):
                if(el in self.use_cols):
                    value.append(message_value["ftr_vector"][el])
        else:
            value = message_value["ftr_vector"]

        timestamp = message_value["timestamp"]

        status = self.UNDEFINED
        status_code = self.UNDEFIEND_CODE


        if(len(self.memory)<self.averaging):
            self.memory.append(value)
            self.FV.append(np.average(self.memory))
            self.FV = self.FV[-self.N:]
        else:
            self.memory.append(value)
            self.memory = self.memory[-self.averaging:]
            self.FV = np.append(self.FV, np.average(self.memory))
            self.FV = self.FV[-self.N:]

        if(None in self.FV):
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE
        else:
            amplitude = (max(self.FV) - min(self.FV))/self.amp_scale
            if(max(self.FV)==min(self.FV)):
                prediction = 1
            else:
                self.FV = (self.FV - np.average(self.FV))/max(self.FV) - min(self.FV)
                prediction = np.argmax(self.model.predict(np.atleast_2d(self.FV)), axis = 1)
            self.prediction_memory.append(prediction[0])

        if(len(self.prediction_memory) > self.prediction_conv):
            self.prediction_memory = self.prediction_memory[-self.prediction_conv:]
        
        if(len(self.prediction_memory) >= self.prediction_conv):
            averaged_prediction = np.average(self.prediction_memory) #on the interval [0, 2]

            value_normalized = 1 + (averaged_prediction-1)*amplitude 
            if(value_normalized > 2):
                status = "Error: measurement above upper limit"
                status_code = -1
            elif(value_normalized < 0):
                status = "Error: measurement below lower limit"
                status_code = self.ERROR_CODE
            else:
                for stage in range(len(self.warning_stages)):
                    if(value_normalized > 1+self.warning_stages[stage]):
                        status = "Warning" + str(stage) + \
                            ": trend close to upper limit."
                        status_code = self.WARNING_CODE
                    elif(value_normalized < 1-self.warning_stages[stage]):
                        status = "Warning" + str(stage) + \
                            ": trend close to lower limit."
                        status_code = self.WARNING_CODE
                    else:
                        status = self.OK
                        status_code = self.OK_CODE
                        break
        
        self.status = status
        self.status_code = status_code

        self.normalization_output_visualization(status=status,
                                                status_code=status_code,
                                                value=message_value["ftr_vector"],
                                                timestamp=timestamp)

        
        return status, status_code


    def train_model(self, train_file: str = None, train_dataframe: DataFrame = None) -> None:
        x_train = []
        labels = []
        #create random trending samples with noise
        for i in range(self.num_samples):
            a = np.random.randint(0, 3)
            noise = np.random.normal(loc = 0, scale = self.train_noise, size = self.N)
            
            if(a == 1):
                #no trend
                sample = np.ones(self.N)*0.5 + noise
            elif(a == 2):
                #uptrend
                sample = np.linspace(0, 1, self.N) + noise
            else:
                #downtrend
                sample = np.linspace(1, 0, self.N) + noise
                
            label = np.array([0, 0, 0])
            label[a] = 1

            x_train.append((sample-np.average(sample))/(max(sample) - min(sample)))
            labels.append(label)
        x_train = np.array(x_train)
        labels = np.array(labels)

        # Fit classification model to random samples (if there was enoug samples to
        # construct at leat one feature)
        if(len(x_train) > 0):
            self.model = tf.keras.Sequential()
            self.model.add(tf.keras.layers.Dense(self.N,activation='relu'))
            self.model.add(tf.keras.layers.Dense(int(self.N/2)))
            self.model.add(tf.keras.layers.Dense(3, activation='softmax'))

            self.model.compile(optimizer =tf.keras.optimizers.Adam(lr = 0.0001, beta_1 = 0.95), loss = 'categorical_crossentropy')

            batch_size = 10

            self.model.fit(x_train,labels, epochs =50, batch_size = batch_size, validation_data = None, verbose = 1)

