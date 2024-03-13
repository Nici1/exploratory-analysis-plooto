from abc import ABC, abstractmethod
import csv
import json
import sys

from typing import Any, Dict, List

sys.path.insert(0, "./src")
# Algorithm imports
from src.anomalyDetection import AnomalyDetectionAbstract
from borderCheck import BorderCheck
from welford import Welford
from EMA import EMA
from filtering import Filtering
from isolationForest import IsolationForest
from GAN import GAN
from PCA import PCA
from hampel import Hampel
from linearFit import LinearFit
from combination import Combination
from trend_classification import Trend_Classification
from Cumulative import Cumulative
from MACD import MACD
from DBscan import DBscan
from percentile import Percentile
from Kmeans import Kmeans
from SVM import SVM
from LSTM import LSTM


# TODO: imports
# from fb_prophet import fb_Prophet
# from RRCF_trees import RRCF_trees

from kafka import KafkaConsumer, TopicPartition
from json import loads
import matplotlib.pyplot as plt
from time import sleep
import numpy as np
import pandas as pd
import datetime


class ConsumerAbstract(ABC):
    configuration_location: str

    def __init__(self, configuration_location: str = None) -> None:
        self.configuration_location = configuration_location

    @abstractmethod
    def configure(self, con: Dict[Any, Any], configuration_location: str) -> None:
        self.configuration_location = configuration_location

    @abstractmethod
    def read(self) -> None:
        pass

    # rewrites anomaly detection configuration
    def rewrite_configuration(self, anomaly_detection_conf: Dict[str, Any]) -> None:
        with open(self.configuration_location) as c:
            conf = json.load(c)
            conf["anomaly_detection_conf"] = anomaly_detection_conf

        with open(self.configuration_location, "w") as c:
            json.dump(conf, c)


class ConsumerKafka(ConsumerAbstract):
    anomalies: List["AnomalyDetectionAbstract"]
    anomaly_names: List[str]
    anomaly_configurations: List[Any]
    consumer: KafkaConsumer

    def __init__(
        self, conf: Dict[Any, Any] = None, configuration_location: str = None
    ) -> None:
        super().__init__(configuration_location=configuration_location)
        if conf is not None:
            self.configure(con=conf)
        elif configuration_location is not None:
            # Read config file
            with open("configuration/" + configuration_location) as data_file:
                conf = json.load(data_file)
            self.configure(con=conf)
        else:
            print("No configuration was given")

    def configure(self, con: Dict[Any, Any] = None) -> None:
        if con is None:
            print("No configuration was given")
            return

        if "filtering" in con:
            self.filtering = con["filtering"]
        else:
            self.filtering = None

        self.topics = con["topics"]
        self.consumer = KafkaConsumer(
            bootstrap_servers=con["bootstrap_servers"],
            auto_offset_reset=con["auto_offset_reset"],
            enable_auto_commit=con["enable_auto_commit"],
            group_id=con["group_id"],
            value_deserializer=eval(con["value_deserializer"]),
        )
        self.consumer.subscribe(self.topics)

        # Initialize a list of anomaly detection algorithms, each for a
        # seperate topic
        self.anomaly_names = con["anomaly_detection_alg"]
        self.anomaly_configurations = con["anomaly_detection_conf"]
        # check if the lengths of configurations, algorithms and topics are
        # the same
        assert len(self.anomaly_names) == len(self.topics) and len(self.topics) == len(
            self.anomaly_configurations
        ), "Number of algorithms, configurations and topics does not match"
        self.anomalies = []
        algorithm_indx = 0
        for anomaly_name in self.anomaly_names:
            anomaly = eval(anomaly_name)
            anomaly.configure(
                self.anomaly_configurations[algorithm_indx],
                configuration_location=self.configuration_location,
                algorithm_indx=algorithm_indx,
            )
            self.anomalies.append(anomaly)
            algorithm_indx += 1

    def read(self) -> None:
        for message in self.consumer:
            # Get topic and insert into correct algorithm
            # print(message)
            topic = message.topic
            # print('topic: ' + str(topic), flush=True)

            algorithm_indxs = []

            for i, j in enumerate(self.topics):
                if j == topic:
                    algorithm_indxs.append(i)
            # print(f'{algorithm_indxs = }')

            # this line was replaced with above loop (to insert the message into several algorithms at the same time)
            # algorithm_indx = self.topics.index(topic)

            for algorithm_indx in algorithm_indxs:
                # check if this topic needs filtering
                if (
                    self.filtering is not None
                    and eval(self.filtering[algorithm_indx]) is not None
                ):
                    # extract target time and tolerance
                    target_time, tolerance = eval(self.filtering[algorithm_indx])
                    message = self.filter_by_time(message, target_time, tolerance)

                if message is not None:
                    value = message.value

                    self.anomalies[algorithm_indx].message_insert(value)

    def filter_by_time(self, message, target_time, tolerance):
        # convert to timedelta objects

        # Convert unix timestamp to datetime format (with seconds unit if
        # possible alse miliseconds)

        # print('filering; timestamp: ' + str(message.value['timestamp']), flush=True)
        try:
            timestamp = pd.to_datetime(message.value["timestamp"], unit="s")
        except pd._libs.tslibs.np_datetime.OutOfBoundsDatetime:
            timestamp = pd.to_datetime(message.value["timestamp"], unit="ms")

        # timestamp = pd.to_datetime(message.value['timestamp'], unit='s')
        time = timestamp.time()
        target_time = datetime.time(target_time[0], target_time[1], target_time[2])
        tol = datetime.timedelta(
            hours=tolerance[0], minutes=tolerance[1], seconds=tolerance[2]
        )
        date = datetime.date(1, 1, 1)
        datetime1 = datetime.datetime.combine(date, time)
        datetime2 = datetime.datetime.combine(date, target_time)

        # Return message only if timestamp is within tolerance
        # print((max(datetime2, datetime1) - min(datetime2, datetime1)))
        # print(tol)
        if (max(datetime2, datetime1) - min(datetime2, datetime1)) < tol:
            return message
        else:
            return None


class ConsumerFile(ConsumerAbstract):
    anomaly: "AnomalyDetectionAbstract"
    file_name: str
    file_path: str

    def __init__(
        self, conf: Dict[Any, Any] = None, configuration_location: str = None
    ) -> None:
        super().__init__(configuration_location=configuration_location)
        if conf is not None:
            self.configure(con=conf)
        elif configuration_location is not None:
            # Read config file
            with open("configuration/" + configuration_location) as data_file:
                conf = json.load(data_file)
            self.configure(con=conf)
        else:
            print("No configuration was given")

    def configure(self, con: Dict[Any, Any] = None) -> None:
        self.file_name = con["file_name"]
        self.file_path = self.file_name

        # Variables that are needed for the operation of Kafka
        self.anomaly_names = con["anomaly_detection_alg"]
        self.anomaly_configurations = con["anomaly_detection_conf"]

        if "filtering" in con:
            self.filtering = con["filtering"]
        else:
            self.filtering = None

        assert len(self.anomaly_names) == len(
            self.anomaly_configurations
        ), "Number of algorithms and configurations does not match"

        # Expects a list but only requires the first element
        self.anomaly = eval(con["anomaly_detection_alg"][0])
        anomaly_configuration = con["anomaly_detection_conf"][0]
        self.anomaly.configure(
            anomaly_configuration,
            configuration_location=self.configuration_location,
            algorithm_indx=0,
        )

        self.anomalies = []
        algorithm_indx = 0
        for anomaly_name in self.anomaly_names:
            anomaly = eval(anomaly_name)
            anomaly.configure(
                self.anomaly_configurations[algorithm_indx],
                configuration_location=self.configuration_location,
                algorithm_indx=algorithm_indx,
            )
            self.anomalies.append(anomaly)
            algorithm_indx += 1

    def read(self) -> None:
        if self.file_name[-4:] == "json":
            self.read_JSON()
        elif self.file_name[-3:] == "csv":
            self.read_csv()
        else:
            print("Consumer file type not supported.")
            sys.exit(1)

    def read_JSON(self):
        with open(self.file_path) as json_file:
            data = json.load(json_file)
            tab = data["data"]
        for d in tab:
            for i, a in enumerate(self.anomalies):
                self.anomalies[i].message_insert(d)

    def read_csv(self):
        with open(self.file_path, "r") as read_obj:
            csv_reader = csv.reader(read_obj)

            header = next(csv_reader)

            try:
                timestamp_index = header.index("timestamp")
            except ValueError:
                timestamp_index = None
            other_indicies = [i for i, x in enumerate(header) if (x != "timestamp")]

            # Iterate over each row in the csv using reader object
            for row in csv_reader:
                d = {}
                if timestamp_index is not None:
                    timestamp = row[timestamp_index]
                    try:
                        timestamp = float(timestamp)
                    except ValueError:
                        pass
                    d["timestamp"] = timestamp

                try:
                    ftr_vector = [float(row[i]) for i in other_indicies]
                except:
                    ftr_vector = [row[i] for i in other_indicies]

                d["ftr_vector"] = ftr_vector
                message = d

                for i, a in enumerate(self.anomalies):
                    if (
                        self.filtering is not None
                        and eval(self.filtering[i]) is not None
                    ):
                        # extract target time and tolerance
                        target_time, tolerance = eval(self.filtering[i])
                        message = self.filter_by_time(d, target_time, tolerance)

                    if message is not None:
                        self.anomalies[i].message_insert(d)

    def filter_by_time(self, message, target_time, tolerance):
        # convert to timedelta objects

        # Convert unix timestamp to datetime format (with seconds unit if
        # possible alse miliseconds)

        # print('filering; timestamp: ' + str(message['timestamp']), flush=True)
        try:
            timestamp = pd.to_datetime(message["timestamp"], unit="s")
        except pd._libs.tslibs.np_datetime.OutOfBoundsDatetime:
            timestamp = pd.to_datetime(message["timestamp"], unit="ms")

        # timestamp = pd.to_datetime(message.value['timestamp'], unit='s')
        time = timestamp.time()
        target_time = datetime.time(target_time[0], target_time[1], target_time[2])
        tol = datetime.timedelta(
            hours=tolerance[0], minutes=tolerance[1], seconds=tolerance[2]
        )
        date = datetime.date(1, 1, 1)
        datetime1 = datetime.datetime.combine(date, time)
        datetime2 = datetime.datetime.combine(date, target_time)

        # Return message only if timestamp is within tolerance
        # print((max(datetime2, datetime1) - min(datetime2, datetime1)))
        # print(tol)
        # print('razlika: ' + str((max(datetime2, datetime1) - min(datetime2, datetime1))), flush=True)
        if (max(datetime2, datetime1) - min(datetime2, datetime1)) < tol:
            # print('filtriral!', flush=True)
            return message
        else:
            # print('Nisem :(', flush=True)
            return None


class ConsumerFileKafka(ConsumerKafka, ConsumerFile):
    anomaly: "AnomalyDetectionAbstract"
    file_name: str
    file_path: str

    def __init__(
        self, conf: Dict[Any, Any] = None, configuration_location: str = None
    ) -> None:
        super().__init__(configuration_location=configuration_location)
        if conf is not None:
            self.configure(con=conf)
        elif configuration_location is not None:
            # Read config file
            with open("configuration/" + configuration_location) as data_file:
                conf = json.load(data_file)
            self.configure(con=conf)
        else:
            print("No configuration was given")

    def configure(self, con: Dict[Any, Any] = None) -> None:
        # File configuration
        self.file_name = con["file_name"]
        self.file_path = "./data/consumer/" + self.file_name

        # Kafka configuration
        self.topics = con["topics"]
        self.consumer = KafkaConsumer(
            bootstrap_servers=con["bootstrap_servers"],
            auto_offset_reset=con["auto_offset_reset"],
            enable_auto_commit=con["enable_auto_commit"],
            group_id=con["group_id"],
            value_deserializer=eval(con["value_deserializer"]),
        )
        self.consumer.subscribe(self.topics)

        # Expects a list but only requires the first element
        self.anomaly = eval(con["anomaly_detection_alg"][0])
        anomaly_configuration = con["anomaly_detection_conf"][0]
        self.anomaly.configure(
            anomaly_configuration,
            configuration_location=self.configuration_location,
            algorithm_indx=0,
        )

    def read(self) -> None:
        ConsumerFile.read(self)

        # expects only one topic
        for message in self.consumer:
            value = message.value
            self.anomaly.message_insert(value)
