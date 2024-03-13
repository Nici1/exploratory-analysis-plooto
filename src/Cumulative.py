from typing import Any, Dict, List
import sys
import numpy as np

sys.path.insert(0,'./src')
sys.path.insert(1, 'C:/Users/Matic/SIHT/anomaly_det/anomalyDetection/')
from anomalyDetection import AnomalyDetectionAbstract
from output import OutputAbstract, TerminalOutput, FileOutput, KafkaOutput
from visualization import VisualizationAbstract, GraphVisualization,\
    HistogramVisualization, StatusPointsVisualization
from normalization import NormalizationAbstract, LastNAverage,\
    PeriodicLastNAverage

class Cumulative(AnomalyDetectionAbstract):
    """ works with 1D data to recognize trends by cumulative difference of values.
    """
    name: str = "Cumulative"

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)


    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None,
                  algorithm_indx: int = None) -> None:
        super().configure(conf, configuration_location=configuration_location,
                          algorithm_indx=algorithm_indx)

        self.cumulative_sum = 0
        self.memory = []

        self.decay = conf["decay"]
        self.averaging = conf["averaging"]
        self.running_mean = 0

        self.warning_stages = conf["warning_stages"]
        self.warning_stages.sort()

    def message_insert(self, message_value: Dict[Any, Any]) -> Any:
        super().message_insert(message_value)

        # Check feature vector
        if(not self.check_ftr_vector(message_value=message_value)):
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE
            #self.normalization_output_visualization(status=status,
            #                                    status_code=status_code,
            #                                    value=message_value["ftr_vector"],
            #                                    timestamp=message_value["timestamp"])
            
            # Remenber status for unittests
            self.status = status
            self.status_code = status_code
            return status, status_code

        # Extract value and timestamp
        value = message_value["ftr_vector"]
        value = value[0]
        timestamp = message_value["timestamp"]

        previous = self.running_mean

        if(len(self.memory) < self.averaging):
            self.memory.append(value)
            self.running_mean = np.average(self.memory)
        else:
            self.memory.append(value)
            self.memory = self.memory[-self.averaging:]
            self.running_mean = np.average(self.memory)
        
        delta = (self.running_mean - previous)/np.abs(self.running_mean)
        self.cumulative_sum += delta 
        self.cumulative_sum *= (1-self.decay)

        status = self.OK
        status_code = self.OK_CODE

        # Check limits

        value_normalized = self.cumulative_sum
        if(value_normalized > 1):
            status = "Error: measurement above upper limit"
            status_code = -1
        elif(value_normalized < -1):
            status = "Error: measurement below lower limit"
            status_code = self.ERROR_CODE
        else:
            for stage in range(len(self.warning_stages)):
                if(value_normalized > self.warning_stages[stage]):
                    status = "Warning" + str(stage) + \
                        ": measurement close to upper limit."
                    status_code = self.WARNING_CODE
                elif(value_normalized < -self.warning_stages[stage]):
                    status = "Warning" + str(stage) + \
                        ": measurement close to lower limit."
                    status_code = self.WARNING_CODE
                else:
                    break
        
        # Remenber status for unittests
        self.status = status
        self.status_code = status_code

        self.normalization_output_visualization(status=status,
                                                status_code=status_code,
                                                value=message_value["ftr_vector"],
                                                timestamp=timestamp)

        return status, status_code
