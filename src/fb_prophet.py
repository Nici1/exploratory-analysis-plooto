from typing import Any, Dict, List
import sys
from pandas._libs.tslibs.timestamps import Timestamp
from pandas.core.frame import DataFrame
import pandas as pd
from prophet import Prophet
import numpy as np

from anomalyDetection import AnomalyDetectionAbstract
from output import OutputAbstract, TerminalOutput, FileOutput, KafkaOutput
from visualization import VisualizationAbstract, GraphVisualization,\
    HistogramVisualization, StatusPointsVisualization
from normalization import NormalizationAbstract, LastNAverage,\
    PeriodicLastNAverage

class fb_Prophet(AnomalyDetectionAbstract):
    name: str = "Prophet"

    uncertainty_interval: float

    memory_dataframe: pd.DataFrame()
    history_file: str
    min_samples: int
    max_samples: int
    retrain_interval: int
    samples_since_retrain: int

    model: Any
    intervals: List[List[float]]

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None,
                  algorithm_indx: int = None) -> None:
        super().configure(conf, configuration_location=configuration_location,
                          algorithm_indx=algorithm_indx)

        # Controlling the anomaly detection recall
        self.uncertainty_interval = conf["uncertainty_interval"]

        # Saved data information
        if("retrain_interval" in conf):
            self.retrain_interval = conf["retrain_interval"]
        else:
            self.retrain_interval = 1

        self.forecast_horizons = conf["forecast_horizons"]
        self.min_samples = conf["min_samples"]
        self.max_samples = conf["max_samples"]
        self.samples_since_retrain = 0

        # Read from history file, trim it and store as memory_dataframe (and
        # save to memory location)
        if("history_file" in conf):
            self.history_file = conf['history_file']
            self.memory_dataframe = pd.read_csv(conf["history_file"])
            self.memory_dataframe = self.memory_dataframe.iloc[-self.max_samples:]
        else:
            self.memory_dataframe = pd.DataFrame({
                "ds": [],
                "y": []}
            )

        self.memory_dataframe.to_csv(self.history_file, index=False)

        # First train
        self.train_model()


    def message_insert(self, message_value: Dict[Any, Any]) -> Any:
        super().message_insert(message_value)

        #print(f'message inserted: {message_value}', flush = True)
        # Check feature vector
        if(not self.check_ftr_vector(message_value=message_value)):
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE
            #print('check NOK', flush = True)
           
            # Remenber status for unittests
            self.status = status
            self.status_code = status_code
            return status, status_code

        feature_vector = super().feature_construction(value=message_value['ftr_vector'],
                                                      timestamp=message_value['timestamp'])
        ##print(feature_vector)
        if (feature_vector == False):
            # If this happens the memory does not contain enough samples to
            # create all additional features.
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE
        else:

            # Extract the monitored value
            timestamp = message_value["timestamp"]
            string_timestamp = pd.to_datetime(timestamp, unit="s")
            #print(feature_vector)
            value = feature_vector[0]

            #print(self.memory_dataframe)
            # Check if there is enough samples
            if(self.memory_dataframe.shape[0] < self.min_samples):

                # Add sample to dataframe for retrain
                self.memory_dataframe = self.memory_dataframe.append({"ds": string_timestamp, "y": value},
                                            ignore_index=True)

                #print(f'Memory dataframe now has {self.memory_dataframe.shape[0]} samples')
                
                status = self.UNDEFINED
                status_code = self.UNDEFIEND_CODE

                # Normalization output and visualization
                self.normalization_output_visualization(status=status,
                                                        status_code=status_code,
                                                        value=value,
                                                        timestamp=timestamp)
            
                # Remenber status for unittests
                self.status = status
                self.status_code = status_code
                # Try to retrain
                self.train_model()

                return status, status_code

            #find boundary with closest timestamp to inserted message
            b_idx = np.argmin([abs(string_timestamp - self.intervals[:,0][i]) for i in range(len(self.intervals))])
            
            # Check if the value is in boundries
            boundries = self.intervals[b_idx]
            if (value < boundries[2]):
                status = "Error: Value below the lower limit"
                status_code = self.ERROR_CODE
            elif(value > boundries[3]):
                status = "Error: Value over the upper limit"
                status_code = self.ERROR_CODE
            else:
                status = self.OK
                status_code = self.OK_CODE

            # Add sample to dataframe for retrain
            self.memory_dataframe = self.memory_dataframe.append({"ds": string_timestamp, "y": value},
                                        ignore_index=True)
            self.memory_dataframe = self.memory_dataframe.iloc[-self.max_samples:]
            self.memory_dataframe = self.memory_dataframe.reset_index(drop = True)

            self.status = status
            self.status_code = status_code

            # Normalization outptu and visualization
            self.normalization_output_visualization(status=status,
                                                    status_code=status_code,
                                                    value=value,
                                                    suggested_value = boundries[1],
                                                    timestamp=timestamp)

            # Increase the samples since retrain and check if retrain is needed
            self.samples_since_retrain += 1
            if (self.samples_since_retrain%self.retrain_interval == 0):
                self.samples_since_retrain = 0
                self.train_model()

        return status, status_code

    def train_model(self):
        # Check if enough samples in memory
        self.memory_dataframe = self.memory_dataframe[-self.max_samples:]
        print(f'{len(self.memory_dataframe)}')
        if(self.memory_dataframe.shape[0] < self.min_samples):
            print(f'Not enough samples({self.memory_dataframe.shape[0]}) - exiting training', flush = True)
            return

        # Initialize new model
        self.model = Prophet(interval_width=self.uncertainty_interval)

        # Fit the model
        self.model.fit(self.memory_dataframe)
        # Make future dataframe and make predictions
        future = self.model.make_future_dataframe(periods=self.forecast_horizons[0], freq = self.forecast_horizons[1])
        forecast = self.model.predict(future)

        # Save bounds to 2D array

        #TODO: manage forecast horizons, connect message insert timestamp to correct forecast
        self.intervals = []
        for i, row in forecast.iterrows():
            self.intervals.append([row.loc["ds"], row.loc["yhat"], row.loc["yhat_lower"], row.loc["yhat_upper"]])

        self.intervals = np.array(self.intervals)

        self.memory_dataframe.to_csv(self.history_file, index=False)

        print('Training successful', flush = True)
