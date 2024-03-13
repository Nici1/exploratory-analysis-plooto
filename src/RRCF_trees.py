from typing import Any, Dict, List
import numpy as np
import sys
import json
from pandas.core.frame import DataFrame
import pandas as pd
import rrcf
from ast import literal_eval

sys.path.insert(0,'./src')
#sys.path.insert(1, 'C:/Users/Matic/SIHT/anomaly_det/anomalyDetection/')
from anomalyDetection import AnomalyDetectionAbstract
from output import OutputAbstract, TerminalOutput, FileOutput, KafkaOutput
from visualization import VisualizationAbstract, GraphVisualization,\
    HistogramVisualization, StatusPointsVisualization
from normalization import NormalizationAbstract, LastNAverage,\
    PeriodicLastNAverage

class RRCF_trees(AnomalyDetectionAbstract):
    name: str = "RRCF"

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None,
                  algorithm_indx: int = None) -> None:
        super().configure(conf, configuration_location=configuration_location,
                          algorithm_indx=algorithm_indx)

        #configuration
        self.num_trees = conf["num_trees"]
        self.tree_size = conf["tree_size"]

        self.filtering = conf["filtering"]

        #initialise index
        self.index = 0

        #initialise a forest of empty trees
        self.forest = []
        for _ in range(self.num_trees):
            tree = rrcf.RCTree()
            self.forest.append(tree)


    def message_insert(self, message_value: Dict[Any, Any]) -> Any:
        if(self.filtering is not None and eval(self.filtering) is not None):
            #extract target time and tolerance
            target_time, tolerance = eval(self.filtering)
            message_value = self.filter_by_time(message_value, target_time, tolerance)

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

        #do feature construction only if single values are fed into the algorithm
        if(len(message_value['ftr_vector']) == 1):
            feature_vector = super().feature_construction(value=message_value['ftr_vector'],
                                                      timestamp=message_value['timestamp'])
            
        else:
            feature_vector = list(message_value['ftr_vector'])

        value = feature_vector

        timestamp = message_value["timestamp"]

        if (feature_vector == False):
            # If this happens the memory does not contain enough samples to
            # create all additional features.
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE
        else:
            feature_vector = np.array(feature_vector)

            anomaly_score = 0

            #Insert point into the trees.
            for tree in self.forest:
                #forget earliest point, if the maximal size is reached
                if(len(tree.leaves) >= self.tree_size):
                    tree.forget_point(self.index - self.tree_size)

                tree.insert_point(feature_vector, index = self.index)

                #compute anomaly score
                anomaly_score += tree.codisp(self.index)/self.num_trees
            self.index +=1
            status_code = anomaly_score
            status = "RRCF anomaly score"

            self.normalization_output_visualization(status=status,
                                                    status_code=status_code,
                                                    value=value,
                                                    timestamp=timestamp)
        
        self.status = status
        self.status_code = status_code
        
        return status, status_code