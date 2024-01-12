from typing import Any, Dict, List
import sys
import numpy as np

sys.path.insert(0, "./src")
from anomalyDetection import AnomalyDetectionAbstract
from output import OutputAbstract, TerminalOutput, FileOutput, KafkaOutput
from visualization import (
    VisualizationAbstract,
    GraphVisualization,
    HistogramVisualization,
    StatusPointsVisualization,
)
from normalization import NormalizationAbstract, LastNAverage, PeriodicLastNAverage


class Percentile(AnomalyDetectionAbstract):
    # method specific
    percentile_range: List[float]
    buff: List[float]
    buff_size: int

    name: str = "Percentile"

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if conf is not None:
            self.configure(conf)

    def configure(
        self,
        conf: Dict[Any, Any] = None,
        configuration_location: str = None,
        algorithm_indx: int = None,
    ) -> None:
        super().configure(
            conf,
            configuration_location=configuration_location,
            algorithm_indx=algorithm_indx,
        )

        self.filtering = conf["filtering"]
        self.percentile_range = conf["percentile_range"]
        self.buff_size = conf["buff_size"]
        self.buff = []

    def message_insert(self, message_value: Dict[Any, Any]) -> Any:
        super().message_insert(message_value)
        if self.filtering is not None and eval(self.filtering) is not None:
            # extract target time and tolerance
            target_time, tolerance = eval(self.filtering)
            message_value = super().filter_by_time(
                message_value, target_time, tolerance
            )

        # If the feature vector is multidimensional then we would need to convert the string of the feature array into an actual array, otherwise inputs of
        # dimension are written and sent as non string values
        if self.input_vector_size > 1:
            message_value["ftr_vector"] = eval(message_value["ftr_vector"][0])

        # Check feature vector
        # if feature vector is not OK, then return undefined codes
        if not self.check_ftr_vector(message_value=message_value):
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE

            self.status = status
            self.status_code = status_code
            return status, status_code

        # Extract value and timestamp
        value = message_value["ftr_vector"]
        # We only consider the first component of the feature vector
        value = value[0]
        timestamp = message_value["timestamp"]

        # create feature vector based on the last value
        feature_vector = super().feature_construction(
            value=message_value["ftr_vector"], timestamp=message_value["timestamp"]
        )

        if feature_vector == False:
            # If this happens the memory does not contain enough samples to
            # create all additional features.
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE
        else:
            value = feature_vector[0]

            # add value to buff and shift if buff is full
            self.buff.append(value)
            if len(self.buff) > self.buff_size:
                self.buff.pop(0)

                percentiles = np.percentile(self.buff, self.percentile_range)

                # Identify anomalies
                if self.buff[-1] < percentiles[0]:
                    status = self.ERROR  # "Error: measurement above upper limit"
                    status_code = self.ERROR_CODE
                elif self.buff[-1] > percentiles[1]:
                    status = self.ERROR
                    status_code = self.ERROR_CODE
                elif self.buff[-1] == percentiles[0] or self.buff[-1] == percentiles[1]:
                    status = "The value is on the percentile border"
                    status_code = self.UNDEFIEND_CODE
                else:
                    status = self.OK
                    status_code = self.OK_CODE
                # Remenber status for unittests
                self.status = status
                self.status_code = status_code

                # check why this is here
                self.normalization_output_visualization(
                    status=status,
                    status_code=status_code,
                    value=message_value["ftr_vector"],
                    timestamp=timestamp,
                )

                return status, status_code
