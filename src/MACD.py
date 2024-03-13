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
from anomalyDetection import AnomalyDetectionAbstract
from isolationForest import IsolationForest
from output import OutputAbstract, TerminalOutput, FileOutput, KafkaOutput
from visualization import VisualizationAbstract, GraphVisualization,\
    HistogramVisualization, StatusPointsVisualization
from normalization import NormalizationAbstract, LastNAverage,\
    PeriodicLastNAverage

class MACD(AnomalyDetectionAbstract):
    name: str = "MACD"

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
        self.period1 = conf["period1"]
        self.period2 = conf["period2"]
        self.warning_stages = conf["warning_stages"]
        self.UL = conf["UL"]
        self.LL = conf["LL"]
        self.filtering = conf["filtering"]

        self.EMA1 = 0
        self.EMA2 = 0
        self.counter = 0


    def message_insert(self, message_value: Dict[Any, Any]) -> Any:
        if(self.filtering is not None and eval(self.filtering) is not None):
            #extract target time and tolerance
            target_time, tolerance = eval(self.filtering)
            message_value = super().filter_by_time(message_value, target_time, tolerance)

        # Check feature vector
        if(not self.check_ftr_vector(message_value=message_value)):
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE

            # Remenber status for unittests
            self.status = status
            self.status_code = status_code
            return status, status_code

        super().message_insert(message_value)

        
        value = message_value["ftr_vector"]
        value = value[0]

        feature_vector = super().feature_construction(value=message_value['ftr_vector'],
                                                      timestamp=message_value['timestamp'])

        timestamp = message_value["timestamp"]

        if (feature_vector == False):
            # If this happens the memory does not contain enough samples to
            # create all additional features.
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE
        else:
            value = feature_vector[0]


            if(self.counter == 0):
                self.EMA1 = value
                self.EMA2 = value
            else:
                self.EMA1 = value*2/(self.period1 + 1) + self.EMA1*(1 - 2/(self.period1 + 1))
                self.EMA2 = value*2/(self.period2 + 1) + self.EMA2*(1 - 2/(self.period2 + 1))

            value_normalized = 2*((self.EMA1 - self.EMA2) - (self.UL + self.LL)/2)/(self.UL - self.LL)
            
            if(value_normalized > 1):
                status = "Error: MACD above upper limit"
                status_code = -1
            elif(value_normalized < -1):
                status = "Error: MACD below lower limit"
                status_code = -1
            else:
                for stage in range(len(self.warning_stages)):
                    if(value_normalized > self.warning_stages[stage]):
                        status = "Warning" + str(stage) + \
                            ": MACD close to upper limit."
                        status_code = 0
                    elif(value_normalized < -self.warning_stages[stage]):
                        status = "Warning" + str(stage) + \
                            ": MACD close to lower limit."
                        status_code = 0
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

            self.counter +=1
        return status, status_code

