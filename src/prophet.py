from typing import Any, Dict, List
import sys
from pandas._libs.tslibs.timestamps import Timestamp
from pandas.core.frame import DataFrame
import pandas as pd
import fbprophet as Prophet

from anomalyDetection import AnomalyDetectionAbstract
from output import OutputAbstract, TerminalOutput, FileOutput, KafkaOutput
from visualization import VisualizationAbstract, GraphVisualization,\
    HistogramVisualization, StatusPointsVisualization
from normalization import NormalizationAbstract, LastNAverage,\
    PeriodicLastNAverage

class Prophet(AnomalyDetectionAbstract):
    name: str = "Prophet"

    uncertainty_interval: float

    memory_dataframe: Dataframe
    memory_location: str
    history_file: str
    samples_in_store: int
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
        if("restrain_interval"in conf):
            self.retrain_interval = conf["retrain_interval"]
        else:
            self.retrain_interval = 1
        self.samples_in_store = conf["samples_in_store"]
        self.memory_location = conf["memory_location"]
        self.samples_since_retrain = 0

        # Read from history file, trim it and store as memory_dataframe (and
        # save to memory location)
        self.memory_dataframe = pd.read_csv(conf["history_file"])
        self.memory_dataframe = self.memory_dataframe.iloc[-self.samples_in_store:]
        self.memory_dataframe.to_csv(self.memory_location, Index=False)

        # First train
        self.train_model()


    def message_insert(self, message_value: Dict[Any, Any]) -> Any:
        super().message_insert(message_value)

        # Check feature vector
        if(not self.check_ftr_vector(message_value=message_value)):
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE
           
            # Remenber status for unittests
            self.status = status
            self.status_code = status_code
            return status, status_code

        # Extract the monitored value
        timestamp = message_value["tiemstamp"]
        string_timestamp = pd.to_datetime(timestamp, unit="s")
        value = message_value["ftr_vector"]
        value = value[0]

        # Check if there is enough samples
        if(self.memory_dataframe.shape[0] != self.samples_in_store):
            # Add sample to dataframe for retrain
            self.memory_dataframe.append({"ds": string_timestamp, "y": value},
                                         ignore_index=True)
            
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE

            # Normalization outptu and visualization
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

        # Check if the value is in boundries
        boundries = self.intervals[self.samples_since_retrain]
        if (value < boundries[0]):
            status = "Error: Value below the lower limit"
            status_code = self.ERROR_CODE
        elif(value > boundries[1]):
            status = "Error: Value over the upper limit"
            status_code = self.ERROR_CODE
        else:
            status = self.OK
            status_code = self.OK_CODE

        # Add sample to dataframe for retrain
        self.memory_dataframe.append({"ds": string_timestamp, "y": value},
                                     ignore_index=True)
        self.memory_dataframe = self.memory_dataframe.iloc[-self.samples_in_store:]

        self.status = status
        self.status_code = status_code

        # Normalization outptu and visualization
        self.normalization_output_visualization(status=status,
                                                status_code=status_code,
                                                value=value,
                                                timestamp=timestamp)

        # Increase the samples since retrain and check if retrain is needed
        self.samples_since_retrain += 1

        if (self.samples_since_retrain%self.retrain_interval == 0):
            self.train_model()

        return status, status_code

    def train_model(self):
        # Check if enough samples in memory
        if(self.memory_dataframe.shape[0] != self.samples_in_store):
            return

        # May be changed or parametrized
        interval_width = 0.9

        # Initialize new model
        self.model = Prophet.Prophet(seasonality_mode='multiplicative', 
                                     interval_width=interval_width,
                                     changepoint_range = self.changepoint_range)

        # Fit the model
        self.model.fit(self.memory_dataframe)
        # Make future dataframe and make predictions
        future = self.model.make_future_dataframe(periods=self.retrain_interval)
        forecast = self.model.predict(future)

        # Save bounds to 2D array
        self.intervals = []
        for i, row in forecast.iterrows():
            self.intervals.append([row.loc["yhat_lower"], row.loc["yhat_upper"]])
