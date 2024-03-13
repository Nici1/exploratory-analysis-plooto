from typing import Any, Dict, List
import numpy as np
import statistics
import sys
import math

sys.path.insert(0,'./src')
sys.path.insert(1, 'C:/Users/Matic/SIHT/anomaly_det/anomalyDetection/')
from anomalyDetection import AnomalyDetectionAbstract
from output import OutputAbstract, TerminalOutput, FileOutput, KafkaOutput
from visualization import VisualizationAbstract, GraphVisualization,\
    HistogramVisualization, StatusPointsVisualization
from normalization import NormalizationAbstract, LastNAverage,\
    PeriodicLastNAverage

class Welford(AnomalyDetectionAbstract):
    name: str = "Welford"

    UL: float
    LL: float
    N: int
    count: int
    Memory: List[float]
    X: float
    mean: float
    s: float
    warning_stages: List[float]

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None,
                  algorithm_indx: int = None) -> None:
        super().configure(conf, configuration_location=configuration_location,
                          algorithm_indx=algorithm_indx)

        if ('N' in conf):
            self.N = conf['N']
            self.Memory = [None] * self.N
        else:
            self.N = None

        self.UL = self.LL = None
        self.count = 0
        self.filtering = conf["filtering"]

        self.X = conf["X"]
        self.warning_stages = conf["warning_stages"]
        self.warning_stages.sort()

    def message_insert(self, message_value: Dict[Any, Any]) -> Any:
        super().message_insert(message_value)
        
        if(self.filtering is not None and eval(self.filtering) is not None):
            #extract target time and tolerance
            target_time, tolerance = eval(self.filtering)
            message_value = super().filter_by_time(message_value, target_time, tolerance)

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

        #extract from message
        value = message_value["ftr_vector"]
        value = value[0]
        timestamp = message_value["timestamp"]

        feature_vector = super().feature_construction(value=message_value['ftr_vector'],
                                                      timestamp=message_value['timestamp'])
        

        if (feature_vector == False):
            # If this happens the memory does not contain enough samples to
            # create all additional features.
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE
        else:
            value = feature_vector[0]

            # First value is undefined
            if (self.count == 0):
                self.mean = value
                self.s = 0
                status = self.UNDEFINED
                status_code = self.UNDEFIEND_CODE
            # Infinite stream and has at least 2 elements so far, or finite stream
            # with full memory
            elif ((self.N is None and self.count > 1) or
                (self.N is not None and self.N <= self.count)):

                # If LL != UL then calculate normalized value
                if(self.UL != self.LL):
                    value_normalized = 2*(value - (self.UL + self.LL)/2) / \
                        (self.UL - self.LL)
                # If LL == UL and value == LL then there is no error
                elif(value == self.LL):
                    value_normalized = 0
                # Else ther is always error
                else:
                    value_normalized = float("inf")

                status = self.OK
                status_code = self.OK_CODE

                if(value_normalized > 1):
                    status = "Error: measurement above upper limit"
                    status_code = -1
                elif(value_normalized < -1):
                    status = "Error: measurement below lower limit"
                    status_code = -1
                else:
                    for stage in range(len(self.warning_stages)):
                        if(value_normalized > self.warning_stages[stage]):
                            status = "Warning" + str(stage) + \
                                ": measurement close to upper limit."
                            status_code = 0
                        elif(value_normalized < -self.warning_stages[stage]):
                            status = "Warning" + str(stage) + \
                                ": measurement close to lower limit."
                            status_code = 0
                        else:
                            break
            else:
                status = self.UNDEFINED
                status_code = self.UNDEFIEND_CODE

            self.status = status
            self.status_code = status_code

            # Outputs and visualizations
            for output in self.outputs:
                output.send_out(timestamp=timestamp, status=status,
                                value=message_value["ftr_vector"],
                                status_code=status_code, algorithm=self.name)

            if(self.visualization is not None):
                lines = [value]
                self.visualization.update(value=lines, timestamp=timestamp,
                                        status_code=status_code)
            
            self.count += 1
            # adjust mean and stdev for the current window
            if(self.N is not None):
                self.Memory.append(value)
                self.Memory = self.Memory[-self.N:]
                # If the memory is not full calculate stdev, mean LL and UL
                if(self.count >= self.N):
                    #print(f'{self.memory = }')
                    self.mean = statistics.mean(self.Memory)
                    self.s = statistics.stdev(self.Memory)

                    # if standard deviation is 0 it causes error in the next
                    # iteration (UL and LL are the same) so a small number is
                    # choosen instead 
                    if(self.s == 0):
                        self.s = np.nextafter(0, 1)

                    self.LL = self.mean - self.X*self.s
                    self.UL = self.mean + self.X*self.s
            # Adjust mean and stdev for all data till this point
            elif(self.count > 1):
                # Actual welfords algorithm formulas
                new_mean = self.mean + (value - self.mean)*1./self.count
                new_s = self.s + (value - self.mean)* \
                            (value - new_mean)
                self.mean = new_mean
                self.s = new_s

                # if standard deviation is 0 it causes error in the next
                # iteration (UL and LL are the same) so a small number is
                # choosen instead 
                if(self.s == 0):
                    self.s = np.nextafter(0, 1)

                self.LL = self.mean - self.X*(math.sqrt(self.s/self.count))
                self.UL = self.mean + self.X*(math.sqrt(self.s/self.count))
    
        return status, status_code
