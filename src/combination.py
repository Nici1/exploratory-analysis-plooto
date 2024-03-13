from abc import ABC, abstractclassmethod
from typing import Any, Dict, List, Tuple
import sys

sys.path.insert(0,'./src')
sys.path.insert(1, 'C:/Users/Matic/SIHT/anomaly_det/anomalyDetection/')

# Algorithm imports
from anomalyDetection import AnomalyDetectionAbstract
from borderCheck import BorderCheck
from welford import Welford
from EMA import EMA
from filtering import Filtering
from isolationForest import IsolationForest
from GAN import GAN
from PCA import PCA
from hampel import Hampel
from linearFit import LinearFit
from MACD import MACD
import time
import numpy as np

from output import OutputAbstract, TerminalOutput, FileOutput, KafkaOutput
from visualization import VisualizationAbstract, GraphVisualization,\
    HistogramVisualization, StatusPointsVisualization
from normalization import NormalizationAbstract, LastNAverage,\
    PeriodicLastNAverage

class Combination(AnomalyDetectionAbstract):
    """ Usses different anomaly detection algorithms and then determines
    output based on all of them
    """

    name: str = "Combination"
    anomaly_algorithms: List["AnomalyDetectionAbstract"]
    status_determiner: "StatusDeterminer"

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)


    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None,
                  algorithm_indx: int = None) -> None:
        super().configure(conf, configuration_location=configuration_location,
                          algorithm_indx=algorithm_indx)

        # ALGORITHMS
        anomaly_algorithm_names = conf["anomaly_algorithms"]
        anomaly_algorithms_configurations = conf["anomaly_algorithms_configurations"]

        # Check number of algorithms and configurations
        assert len(anomaly_algorithm_names) == len(anomaly_algorithms_configurations), "Number of anomaly detection algorithms does not match the number of configurations."

        # Initialize and configure algorithms
        self.anomaly_algorithms = []
        for alg_indx in range(len(anomaly_algorithm_names)):
            anomaly_algorithm = eval(anomaly_algorithm_names[alg_indx])
            anomaly_algorithm.configure(anomaly_algorithms_configurations[alg_indx],
                                        configuration_location=self.configuration_location,
                                        algorithm_indx=self.algorithm_indx)
            anomaly_algorithm.index_in_combination = alg_indx
            self.anomaly_algorithms.append(anomaly_algorithm)

        # Status determiner initalization
        self.status_determiner = eval(conf["status_determiner"])
        self.status_determiner.configure(conf["status_determiner_conf"])

    def message_insert(self, message_value: Dict[Any, Any]) -> Any:
        super().message_insert(message_value)

        # No need to check feature vector since every algorithm is going to do
        # that
        

        # Get statuses from all algorithms
        statuses = []
        status_messages = []
        for algorithm in self.anomaly_algorithms:
            to_insert = message_value.copy()
            status_message, status_code = algorithm.message_insert(message_value=to_insert)
            statuses.append(status_code)
            status_messages.append(status_message)
        
        # Get fina status
        final_status_code, status = self.status_determiner.determine_status(statuses=statuses, status_messages = status_messages, timestamp = message_value["timestamp"])

        self.status_code = final_status_code
        self.status = status

        # Output
        self.normalization_output_visualization(status_code=final_status_code,
                                                status=status,
                                                value=message_value["ftr_vector"],
                                                timestamp=message_value["timestamp"])

        return status, final_status_code


class StatusDeterminer(ABC):
    # Statuses
    UNDEFINED = "Undefined"
    ERROR = "Error"
    WARNING = "Warning"
    OK = "OK"

    # Status codes
    UNDEFIEND_CODE = 2
    ERROR_CODE = -1
    WARNING_CODE = 0
    OK_CODE = 1

    @abstractclassmethod
    def determine_status(self, statuses: List[int] ) -> Tuple[int, str]:
        pass
    @abstractclassmethod
    def configure(self, conf: Dict[Any, Any] = None):
        pass


class AND(StatusDeterminer):
    # Returns error if all statuses are error, warning if all statuses are
    # warrning (or error) and OK otherwise. Undefined statuses are ignored.
    def configure(self, conf: Dict[Any, Any] = None):
        pass

    def determine_status(self, statuses,status_messages, timestamp):
        # 1-OK, 0-warrning, -1-error. We search for the highest status
        # (except 2) and return it
        max_status = -2
        for status in statuses:
            if(status > max_status and status != 2):
                max_status = status
        
        status_message = ""
        if(max_status == -1):
            status_message = self.ERROR
        elif(max_status == 0):
            status_message = self.WARNING
        elif(max_status == 1):
            status_message = self.OK
        elif(max_status == -2):
            # If status remains -2 it means that all statuses were undefined
            max_status = self.UNDEFIEND_CODE
            status_message = self.UNDEFINED
        
        return max_status, status_message


class OR(StatusDeterminer):
    # Returns error if at least one status is error, warning if at least one
    # status is warrning (or error) and OK otherwise. Undefined statuses are
    # ignored.
    def configure(self, conf: Dict[Any, Any] = None):
        pass

    def determine_status(self, statuses, status_messages, timestamp):
        # 1-OK, 0-warrning, -1-error. We search for the lowest status and
        # return it
        min_status = 2
        for status in statuses:
            if(status < min_status):
                min_status = status
        
        status_message = ""
        if(min_status == -1):
            status_message = self.ERROR
        elif(min_status == 0):
            status_message = self.WARNING
        elif(min_status == 1):
            status_message = self.OK
        elif(min_status == 2):
            status_message = self.UNDEFINED
        
        return min_status, status_message

class PercentScore(StatusDeterminer):
    #Returns a score between 0 and 1; 1-all algorithms signaled an error, 0-all algorithms OK

    def configure(self, conf: Dict[Any, Any] = None):
        self.memory = []
        self.interval = conf["interval"]
        self.data_interval = conf["data_interval"]
        self.num_in_interval = int(self.interval/self.data_interval)

    def determine_status(self, statuses, status_messages, timestamp):
        #sum up the statuses, divide by max score
        max_score = len(statuses)*2
        score = 0
        for status in statuses:
            if(status == 0):
                score +=1
            elif(status == -1):
                score +=2
            else:
                score +=0
        
        final_score = score/max_score
        status_message = "PercentScore"

        if(final_score is not None):
            if(timestamp<1e10):
                self.memory.append([final_score, timestamp])
            else:
                self.memory.append([final_score, timestamp/1000])
        now = self.memory[-1][1]

        tmstps = np.array(self.memory)[:,1]
        idx =[i for i, j in enumerate(tmstps) if j>(now-self.interval)]
        self.memory = list(np.array(self.memory)[idx])

        try:
            convoluted_score = np.sum(np.array(self.memory)[:,0])/max(self.num_in_interval, len(self.memory))
        except:
            convoluted_score = 0
        
        return convoluted_score, status_message

    
class PercentScore_Alicante(StatusDeterminer):
    #Returns a score between 0 and 1; 1-all algorithms signaled an error, 0-all algorithms OK
    #Specific for alicante salinity - only includes sudden jumps in the upwards direction, not sudden drops

    def configure(self, conf: Dict[Any, Any] = None):
        self.memory = []
        self.interval = conf["interval"]
        self.data_interval = conf["data_interval"]
        self.num_in_interval = int(self.interval/self.data_interval)

    def determine_status(self, statuses, status_messages, timestamp):
        #sum up the statuses, divide by max score
        max_score = len(statuses)*2
        score = 0
        for s, s_m in zip(statuses, status_messages):
            if('upper' in s_m):    # only include anomalies of type .... close to upper limit
                if(s == 0):
                    score +=1
                elif(s == -1):
                    score +=2
                else:
                    score +=0
            else:
                score += 0
        
        final_score = score/max_score
        status_message = "PercentScore"

        if(final_score is not None):
            if(timestamp<1e10):
                self.memory.append([final_score, timestamp])
            else:
                self.memory.append([final_score, timestamp/1000])
        now = self.memory[-1][1]

        tmstps = np.array(self.memory)[:,1]
        idx =[i for i, j in enumerate(tmstps) if j>(now-self.interval)]
        self.memory = list(np.array(self.memory)[idx])

        try:
            convoluted_score = np.sum(np.array(self.memory)[:,0])/max(self.num_in_interval, len(self.memory))
        except:
            convoluted_score = 0
        
        return convoluted_score, status_message
