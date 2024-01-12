from typing import Any, Dict, List
import sys

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


class BorderCheck(AnomalyDetectionAbstract):
    """works with 1D data and checks if the value is above, below or close
    to guven upper and lower limits
    """

    UL: float
    LL: float
    warning_stages: List[float]
    name: str = "Border check"

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

        self.LL = conf["LL"]
        self.UL = conf["UL"]
        self.filtering = conf["filtering"]

        self.warning_stages = conf["warning_stages"]
        self.warning_stages.sort()

    def message_insert(self, message_value: Dict[Any, Any]) -> Any:
        super().message_insert(message_value)
        if self.filtering is not None and eval(self.filtering) is not None:
            # extract target time and tolerance
            target_time, tolerance = eval(self.filtering)
            message_value = super().filter_by_time(
                message_value, target_time, tolerance
            )

        # Check feature vector
        if not self.check_ftr_vector(message_value=message_value):
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE
            # self.normalization_output_visualization(status=status,
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

            # Normalize value
            value_normalized = (
                2 * (value - (self.UL + self.LL) / 2) / (self.UL - self.LL)
            )
            status = self.OK
            status_code = self.OK_CODE

            # Check limits
            if value_normalized > 1:
                status = "Error: measurement above upper limit"
                status_code = -1
            elif value_normalized < -1:
                status = "Error: measurement below lower limit"
                status_code = self.ERROR_CODE
            else:
                for stage in range(len(self.warning_stages)):
                    if value_normalized > self.warning_stages[stage]:
                        status = (
                            "Warning"
                            + str(stage)
                            + ": measurement close to upper limit."
                        )
                        status_code = self.WARNING_CODE
                    elif value_normalized < -self.warning_stages[stage]:
                        status = (
                            "Warning"
                            + str(stage)
                            + ": measurement close to lower limit."
                        )
                        status_code = self.WARNING_CODE
                    else:
                        break

            # Remenber status for unittests
            self.status = status
            self.status_code = status_code

            self.normalization_output_visualization(
                status=status,
                status_code=status_code,
                value=message_value["ftr_vector"],
                timestamp=timestamp,
            )

        return status, status_code
