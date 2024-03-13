from typing import Any, Dict, List
import numpy as np
import sys

sys.path.insert(0,'./src')
sys.path.insert(1, 'C:/Users/Matic/SIHT/anomaly_det/anomalyDetection/')
from anomalyDetection import AnomalyDetectionAbstract
from output import OutputAbstract, TerminalOutput, FileOutput, KafkaOutput
from visualization import VisualizationAbstract, GraphVisualization,\
    HistogramVisualization, StatusPointsVisualization
from normalization import NormalizationAbstract, LastNAverage,\
    PeriodicLastNAverage

class EMA_Percentile(AnomalyDetectionAbstract):
    name: str = "EMA_percentile"

    UL: float
    LL: float
    N: int
    smoothing: float
    EMA: List[float]
    numbers: List[float]
    timestamp: List[Any]

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None,
                  algorithm_indx: int = None) -> None:
        super().configure(conf, configuration_location=configuration_location,
                          algorithm_indx=algorithm_indx)

        self.percentile = conf["percentile"]
        self.percentiles = None
        self.window = conf["window"]
        self.start_on = conf["start_on"]
        self.period = conf["period"]
        self.smoothing = 2/(self.period+1)
        self.EMA = []
        self.sigma = []
        self.numbers = []
        self.timestamps = []

    def message_insert(self, message_value: Dict[Any, Any]) -> Any:
        super().message_insert(message_value)

        # Check feature vector
        if(not self.check_ftr_vector(message_value=message_value)):
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE

            self.status = status
            self.status_code = status_code
            return status, status_code

        value = message_value["ftr_vector"]
        value = value[0]

        self.numbers.append(value)
        self.timestamps.append(message_value['timestamp'])
        if (len(self.numbers) > self.period):
            self.numbers = self.numbers[-self.period:]
            self.timestamps = self.timestamps[-self.period:]

        #Calculate exponential moving average
        if(len(self.EMA) == 0):
            self.EMA.append(self.numbers[-1])
        else:
            new = self.numbers[-1] * self.smoothing + self.EMA[-1] *\
                (1-self.smoothing)
            self.EMA.append(new)
        
        if(len(self.EMA) > self.window):
            self.EMA = self.EMA[-self.window:]


        #untill the window is full, leave undefined
        if(len(self.EMA) >= self.start_on):
            self.percentiles = np.percentile(self.EMA, [100-self.percentile, self.percentile])


        #Perform error checks
        if(self.percentiles is not None):
            if(self.EMA[-1] > self.percentiles[1]):
                status = "Error: EMA in upper percentile"
                status_code = self.ERROR_CODE
            elif(self.EMA[-1] < self.percentiles[0]):
                status = "Error: EMA in lower percentile"
                status_code = self.ERROR_CODE
            else:
                status = self.OK
                status_code = self.OK_CODE
        else:
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE

        self.status = status
        self.status_code = status_code

        # Does not use general normalization_output_visualization method
        # because of custom visualization
        for output in self.outputs:
            output.send_out(timestamp=message_value["timestamp"],
                            algorithm=self.name, status=status,
                            value=message_value['ftr_vector'],
                            status_code=status_code)

        lines = [self.numbers[-1]]
        if(self.visualization is not None):
            self.visualization.update(value=lines, timestamp=message_value["timestamp"],
            status_code = status_code)

        return status, status_code
