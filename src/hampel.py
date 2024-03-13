from typing import Any, Dict
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

class Hampel(AnomalyDetectionAbstract):
    name: str = "Hampel"

    W: int
    new_value: float
    K: float


    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None,
                  algorithm_indx: int = None) -> None:
        super().configure(conf, configuration_location=configuration_location,
                          algorithm_indx=algorithm_indx)
        if ('W' in conf):
            self.W = conf['W']
            self.memory = [None] * (2*self.W + 1)
        else:
            self.W = None

        self.K = conf["K"]
        self.n_sigmas = conf["n_sigmas"]
        self.count = 0


    def message_insert(self, message_value: Dict[Any, Any]) -> Any:
        super().message_insert(message_value)
        
        # Check feature vector
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

        value = message_value["ftr_vector"]
        value = value[0]

        # extract from message
        value = message_value["ftr_vector"]
        value = value[0]
        timestamp = message_value["timestamp"]

        status = self.OK
        status_code = self.OK_CODE

        suggested_value = None

        if(self.W is not None):
            self.memory.append(value)
            self.memory = self.memory[-(2*self.W + 1):]

        #Nothing is done until there is enough values to apply the window
        if (self.count < (2*self.W + 1)):
            if(self.memory[self.W + 1] is not None):
                suggested_value = self.memory[self.W + 1]
                status = self.UNDEFINED
                status_code = self.UNDEFIEND_CODE
            else:
                suggested_value = None
                status = self.UNDEFINED
                status_code = self.UNDEFIEND_CODE
        
        else:
            median = np.median(self.memory)
            S0 = self.K * np.median(np.abs(self.memory - median))
            if(np.abs(self.memory[self.W+1] - median) > self.n_sigmas * S0):
                suggested_value = median
                status = "Anomaly detected"
                status_code = self.ERROR_CODE
            else:
                suggested_value = self.memory[self.W+1]
                status = self.OK
                status_code = self.OK_CODE
  


        # custom output because the normalization is in algorithm
        # Outputs and visualizations
        output_value = self.memory[self.W+1]
        if(output_value is not None):
            for output in self.outputs:
                output.send_out(timestamp=timestamp, status=status,
                                suggested_value=suggested_value,
                                value=value,
                                status_code=status_code, algorithm=self.name)

        if(suggested_value is not None):
            if(self.visualization is not None):
                # print(status_code)
                lines = [suggested_value]
                self.visualization.update(value=lines, timestamp=timestamp,
                                        status_code=status_code)
        
        self.count += 1

        return status, status_code
