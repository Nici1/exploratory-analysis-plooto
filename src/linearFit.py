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

class LinearFit(AnomalyDetectionAbstract):
    name: str = "LinearFit"
    slope: float
    average: float
    memory: list
    UL: float
    LL: float
    N: int
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

        self.LL = conf["LL"]
        self.UL = conf["UL"]
        self.confidence_norm = conf["confidence_norm"]
        self.warning_stages = conf["warning_stages"]
        self.warning_stages.sort()
        self.timestamps = []
        self.count = 0


        if ('N' in conf):
            self.N = conf['N']
            self.memory = []
        else:
            self.N = None

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

        value = message_value["ftr_vector"]
        value = value[0]
        timestamp = message_value["timestamp"]

        self.memory.append(value)
        self.timestamps.append(message_value['timestamp'])
        if (len(self.memory) > self.N):
            self.memory = self.memory[-self.N:]
            self.timestamps = self.timestamps[-self.N:]

        slope = None
        #Calculate the fit coefficients
        if(self.count < self.N):
            pass
        else:
            x = np.array(range(len(self.memory)))
            y = self.memory
            a, residuals, rank, singular_values, rcond = np.polyfit(x, y, deg = 1, full = True)
            slope, average = a
        

        #Normalize the slope to the range LL - UL
        status = self.UNDEFINED
        status_code = self.UNDEFIEND_CODE
        value_normalized = 0
        if(slope is not None):
            value_normalized = 2*(slope - (self.UL + self.LL)/2) / \
                (self.UL - self.LL)
            # print(value_normalized)

            status = self.OK
            status_code = self.OK_CODE


            print(residuals/value)

            #only post errors and warnings if the quality of the fit is good (mse)
            if(residuals/value > self.confidence_norm):
                status = self.UNDEFINED
                status_code = self.UNDEFIEND_CODE
            
            #Perform error and warning checks
            elif(value_normalized > 1):
                status = "Error: slope above upper limit"
                status_code = -1
            elif(value_normalized < -1):
                status = "Error: slope below lower limit"
                status_code = -1
            else:
                for stage in range(len(self.warning_stages)):
                    if(value_normalized > self.warning_stages[stage]):
                        status = "Warning" + str(stage) + \
                            ": slope close to upper limit."
                        status_code = 0
                    elif(value_normalized < -self.warning_stages[stage]):
                        status = "Warning" + str(stage) + \
                            ": slope close to lower limit."
                        status_code = 0
                    else:
                        break


        self.status = status
        self.status_code = status_code

        # Customend to outputs
        for output in self.outputs:
            output.send_out(timestamp=message_value["timestamp"],
                            algorithm=self.name, status=status,
                            value=message_value['ftr_vector'],
                            status_code=status_code)

        # Send to visualization
        lines = [message_value['ftr_vector']]
        timestamp = self.timestamps[-1]
        if(self.visualization is not None):
            self.visualization.update(value=lines, timestamp=message_value["timestamp"],
            status_code = status_code)
        self.count += 1

        return status, status_code
