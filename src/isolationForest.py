from typing import Any, Dict, List
import numpy as np
import sys
import json
import pickle
from pandas.core.frame import DataFrame
import sklearn.ensemble
import pandas as pd
from ast import literal_eval

sys.path.insert(0, "./src")
sys.path.insert(1, "C:/Users/Matic/SIHT/anomaly_det/anomalyDetection/")
from anomalyDetection import AnomalyDetectionAbstract
from output import OutputAbstract, TerminalOutput, FileOutput, KafkaOutput
from visualization import (
    VisualizationAbstract,
    GraphVisualization,
    HistogramVisualization,
    StatusPointsVisualization,
)
from normalization import NormalizationAbstract, LastNAverage, PeriodicLastNAverage


class IsolationForest(AnomalyDetectionAbstract):
    name: str = "Isolation forest"

    # method specific
    N: int
    max_samples: int
    model_name: str
    memory: List[float]
    isolation_score: float
    warning_stages: List[float]

    # retrain information
    samples_from_retrain: int
    retrain_interval: int
    samples_for_retrain: int
    retrain_file: str
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

        # Train configuration
        self.max_features = conf["train_conf"]["max_features"]
        self.model_name = conf["train_conf"]["model_name"]
        self.max_samples = conf["train_conf"]["max_samples"]
        self.input_vector_size = conf["input_vector_size"]

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
        if "load_model_from" in conf:
            self.model = self.load_model(conf["load_model_from"])
        elif "train_data" in conf:
            self.train_model(conf["train_data"])
        elif self.retrain_interval is not None:
            self.model = sklearn.ensemble.IsolationForest(
                max_samples=self.max_samples, max_features=self.max_features
            )
        else:
            raise Exception(
                "The configuration must specify either \
                            load_model_from, train_data or train_interval"
            )

    def message_insert(self, message_value: Dict[Any, Any]) -> Any:
        super().message_insert(message_value)

        # Check feature vector
        if not self.check_ftr_vector(message_value=message_value):
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE

            # if the names in the msg are wrong, there would be an error here
            # self.normalization_output_visualization(status=status,
            #                                    status_code=status_code,
            #                                    value=message_value["ftr_vector"],
            #                                    timestamp=message_value["timestamp"])

            # Remenber status for unittests
            self.status = status
            self.status_code = status_code
            return status, status_code

        # value = message_value["ftr_vector"]
        # value = value[0]

        if self.use_cols is not None:
            value = []
            for el in range(len(message_value["ftr_vector"])):
                if el in self.use_cols:
                    value.append(message_value["ftr_vector"][el])
        else:
            value = message_value["ftr_vector"]

        timestamp = message_value["timestamp"]
        feature_vector = super().feature_construction(value=value, timestamp=timestamp)
        # print("feature vector:", feature_vector)

        if not feature_vector or not self.trained:
            # If this happens the memory does not contain enough samples to
            # create all additional features.
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE
        else:
            feature_vector = np.array(feature_vector)
            # Model prediction
            isolation_score = self.model.predict(feature_vector.reshape(1, -1))
            if isolation_score == 1:
                status = self.OK
                status_code = self.OK_CODE
            elif isolation_score == -1:
                status = "Error: outlier detected"
                status_code = -1
            else:
                status = self.UNDEFINED
                status_code = self.UNDEFIEND_CODE

        self.normalization_output_visualization(
            status=status, status_code=status_code, value=value, timestamp=timestamp
        )
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
        return status, status_code

    def save_model(self, filename: str) -> None:
        with open("models/" + filename, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, filename: str) -> None:
        with open(filename, "rb") as f:
            clf = pickle.load(f)
        return clf

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
            # Load data from location stored in "filename" (ussually for
            # initial model training)

            # Changed 25.3. by Gal
            # df = pd.read_csv(train_file, skiprows=1, delimiter = ",")

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
        ftr_vector_list = df["ftr_vector"].values.tolist()
        timestamp_list = df["timestamp"].values.tolist()
        # Create a new  dataframe with features as columns
        df = pd.DataFrame(ftr_vector_list)
        df.insert(loc=0, column="timestamp", value=timestamp_list)
        # Transfer to numpy and extract data and timestamps
        df = df.to_numpy()
        timestamps = np.array(df[:, 0])
        data = np.array(df[:, 1 : (1 + self.input_vector_size)])

        # Requires special feature construction so it does not mess with the
        # feature-construction memory
        features = self.training_feature_construction(data=data, timestamps=timestamps)

        # Fit IsolationForest model to data (if there was enoug samples to
        # construct at leat one feature)
        if len(features) > 0:
            self.model = sklearn.ensemble.IsolationForest(
                max_samples=self.max_samples, max_features=self.max_features
            ).fit(features)

            self.save_model(self.model_name)
            self.trained = True
