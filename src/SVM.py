from typing import Any, Dict
import sys
from pandas.core.frame import DataFrame
import pandas as pd
from ast import literal_eval
from sklearn.svm import OneClassSVM
import json
import numpy as np
from scipy.spatial import distance
from ast import literal_eval

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


class SVM(AnomalyDetectionAbstract):
    name: str = "SVM"

    # retrain information
    samples_from_retrain: int  # A variable that tracks when to trigger the model retrainning since i.e. when samples_from_retrain >= retrain_interval
    retrain_interval: int  # An integer representing the number of samples recieved by the anomaly detection component that trigger model retraining.
    samples_for_retrain: int  #  An integer representing the number of most recent samples that are used to retrain the model.
    retrain_file: (
        str  # Path and file name of the file in which retrain data will be stored
    )
    trained: bool
    memory_dataframe: DataFrame

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

        # Retrain configuration
        if "retrain_interval" in conf:
            self.retrain_counter = 0
            self.retrain_interval = conf["retrain_interval"]
            self.retrain_file = conf["retrain_file"]
            self.samples_from_retrain = 0
            if "samples_for_retrain" in conf:
                self.samples_for_retrain = conf["samples_for_retrain"]
            else:
                self.samples_for_retrain = None

            # Retrain memory initialization
            # Retrain memory is of shape [timestamp, ftr_vector]
            if "train_data" in conf:
                self.memory_dataframe = pd.read_csv(
                    conf["train_data"],
                    skiprows=0,
                    delimiter=",",
                    converters={"ftr_vector": literal_eval},
                )
                if self.samples_for_retrain is not None:
                    self.memory_dataframe = self.memory_dataframe.iloc[
                        -self.samples_for_retrain :
                    ]
            else:
                columns = ["timestamp", "ftr_vector"]
                self.memory_dataframe = pd.DataFrame(columns=columns)
        else:
            self.retrain_interval = None
            self.samples_for_retrain = None
            self.memory_dataframe = None

        # Initialize model
        self.trained = False
        if "train_data" in conf:
            self.train_model(conf["train_data"])
        elif self.retrain_interval is not None:
            self.model = OneClassSVM(gamma="auto")
        else:
            raise Exception(
                "The configuration must specify either \
                            load_model_from, train_data or train_interval"
            )

    def message_insert(self, message_value: Dict[Any, Any]) -> Any:
        # message_value = literal_eval(message_value)[0]
        # print(literal_eval(message_value["ftr_vector"][0]))

        # If the feature vector is multidimensional then we would need to convert the string of the feature array into an actual array, otherwise inputs of
        # dimension are written and sent as non string values
        if self.input_vector_size > 1:
            message_value["ftr_vector"] = literal_eval(message_value["ftr_vector"][0])
        super().message_insert(message_value)

        # Check feature vector
        if not self.check_ftr_vector(message_value=message_value):
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE

            # Remenber status for unittests
            self.status = status
            self.status_code = status_code
            return status, status_code

        # Filter ftr_vector by use_cols if neccessery
        if self.use_cols is not None:
            value = []
            for el in range(len(message_value["ftr_vector"])):
                if el in self.use_cols:
                    value.append(message_value["ftr_vector"][el])
        else:
            value = message_value["ftr_vector"]
        timestamp = message_value["timestamp"]

        # Feature construction
        feature_vector = super().feature_construction(value=value, timestamp=timestamp)

        if not feature_vector or not self.trained:
            # If this happens the memory does not contain enough samples to
            # create all additional features.
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE
        else:
            # Check if sample is at least for treshold away from all core
            # samples

            if self.model.predict([feature_vector]) == -1:
                anomalous = True
            else:
                anomalous = False

            if anomalous:
                status = "Error: outlier detected"
                status_code = -1
            else:
                status = self.OK
                status_code = self.OK_CODE

        self.normalization_output_visualization(
            status=status, status_code=status_code, value=value, timestamp=timestamp
        )

        # Remember for unittests
        self.status = status
        self.status_code = status_code

        # Add to memory for retrain and execute retrain if needed
        if self.retrain_interval is not None:
            # Add to memory (timestamp and ftr_vector seperate so it does not
            # ceuse error)
            new_row = {"timestamp": timestamp, "ftr_vector": value}
            self.memory_dataframe = self.memory_dataframe.append(
                new_row, ignore_index=True
            )

            # Cut if needed
            if self.samples_for_retrain is not None:
                self.memory_dataframe = self.memory_dataframe.iloc[
                    -self.samples_for_retrain :
                ]
            self.samples_from_retrain += 1

            # Retrain if needed (and possible)
            if self.samples_from_retrain >= self.retrain_interval and (
                self.samples_for_retrain == self.memory_dataframe.shape[0]
                or self.samples_for_retrain is None
            ):
                self.samples_from_retrain = 0
                self.train_model(train_dataframe=self.memory_dataframe)
                self.retrain_counter += 1

    def train_model(
        self, train_file: str = None, train_dataframe: DataFrame = None
    ) -> None:
        if train_dataframe is not None:
            # This is in case of retrain
            df = train_dataframe

            # Save train_dataframe to file and change the config file so the
            # next time the model will train from that file
            path = self.retrain_file
            df.to_csv(path, index=False)

            with open("configuration/" + self.configuration_location) as conf:
                whole_conf = json.load(conf)
                if (
                    whole_conf["anomaly_detection_alg"][self.algorithm_indx]
                    == "Combination()"
                ):
                    whole_conf["anomaly_detection_conf"][self.algorithm_indx][
                        "anomaly_algorithms_configurations"
                    ][self.index_in_combination]["train_data"] = path
                else:
                    whole_conf["anomaly_detection_conf"][self.algorithm_indx][
                        "train_data"
                    ] = path

            with open("configuration/" + self.configuration_location, "w") as conf:
                json.dump(whole_conf, conf)

        elif train_file is not None:
            # Read csv and eval ftr_vector strings
            df = pd.read_csv(
                train_file,
                skiprows=0,
                delimiter=",",
                usecols=(
                    0,
                    1,
                ),
                converters={"ftr_vector": literal_eval},
            )
        else:
            raise Exception("train_file or train_dataframe must be specified.")

        # Extract list of ftr_vectors and list of timestamps
        ftr_vector_list = df["ftr_vector"].tolist()
        timestamp_list = df["timestamp"].tolist()
        # print(ftr_vector_list)
        # Create a new  dataframe with features as columns
        # print("Feature vector list: ", ftr_vector_list)

        # If the feature vector is single dimensional, for some reason pd.DataFrame.from_records does not work hence the following:
        if self.input_vector_size > 1:
            df = pd.DataFrame.from_records(ftr_vector_list)
        else:
            df = pd.DataFrame(ftr_vector_list)

        df.insert(loc=0, column="timestamp", value=timestamp_list)
        # print(df)

        # Transfer to numpy and extract data and timestamps
        df = df.to_numpy()

        timestamps = np.array(df[:, 0])

        data = np.array(df[:, 1 : (1 + self.input_vector_size)])

        # Requires special feature construction so it does not mess with the
        # feature-construction memory

        features = self.training_feature_construction(data=data, timestamps=timestamps)
        # print(features)
        # Fit DBscan model to data (if there was enoug samples to
        # construct at leat one feature)

        if len(features) > 0:
            # train model
            self.model = OneClassSVM(gamma="auto").fit(features)

            self.trained = True
