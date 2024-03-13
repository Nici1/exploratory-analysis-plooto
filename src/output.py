from abc import abstractmethod
from abc import ABC
import json
import csv
from json import dumps
import os
import logging
from typing import Any, Dict
from kafka import KafkaProducer
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS

# from kafka.admin import KafkaAdminClient, NewTopic


class OutputAbstract(ABC):
    send_ok: bool

    def __init__(self) -> None:
        pass

    @abstractmethod
    def configure(self, conf: Dict[Any, Any]) -> None:
        if "send_ok" in conf:
            self.send_ok = conf["send_ok"]
        else:
            self.send_ok = True

    @abstractmethod
    def send_out(
        self,
        value: Any,
        suggested_value: Any,
        status: str,
        timestamp: Any,
        status_code: int = None,
        algorithm: str = "Unknown",
    ) -> None:
        pass


class TerminalOutput(OutputAbstract):
    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if conf is not None:
            self.configure(conf=conf)

    def configure(self, conf: Dict[Any, Any] = None) -> None:
        super().configure(conf=conf)
        # Nothing to configure
        pass

    def send_out(
        self,
        value: Any,
        suggested_value: Any = None,
        status: str = "",
        timestamp: Any = 0,
        status_code: int = None,
        algorithm: str = "Unknown",
    ) -> None:
        # Send to kafka only if an anomaly is detected (or if it is specified
        # that ok values are to be sent)
        if status_code != 1 or self.send_ok:
            o = (
                str(timestamp)
                + ": "
                + status
                + "(value: "
                + str(value)
                + ")"
                + ", Algorithm: "
                + algorithm
            )
            if suggested_value is not None:
                o = o + ", Suggested value: " + str(suggested_value)
            print(o)


class FileOutput(OutputAbstract):
    file_name: str
    file_path: str
    mode: str

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if conf is not None:
            self.configure(conf=conf)

    def configure(self, conf: Dict[Any, Any] = None) -> None:
        super().configure(conf=conf)

        self.file_name = conf["file_name"]
        self.mode = conf["mode"]
        self.file_path = "log/" + self.file_name

        # make log folder if one does not exist
        dir = "./log"
        if not os.path.isdir(dir):
            os.makedirs(dir)

        # If mode is write clear the file
        if self.mode == "w":
            if self.file_name[-4:] == "json":
                with open(self.file_path, "w") as f:
                    d = {"data": []}
                    json.dump(d, f)
            elif self.file_name[-3:] == "txt":
                open(self.file_path, "w").close()
            elif self.file_name[-3:] == "csv":
                with open(self.file_path, "w", newline="") as csvfile:
                    fieldnames = [
                        "timestamp",
                        "status",
                        "status_code",
                        "value",
                        "suggested_value",
                        "algorithm",
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()

    def send_out(
        self,
        value: Any = None,
        suggested_value: Any = None,
        status: str = "",
        timestamp: Any = None,
        status_code: int = None,
        algorithm: str = "Unknown",
    ) -> None:
        # Send to kafka only if an anomaly is detected (or if it is specified
        # that ok values are to be sent)
        # print("here")
        if status_code != 1 or self.send_ok:
            if self.file_name[-4:] == "json":
                self.write_JSON(
                    value=value,
                    status=status,
                    timestamp=timestamp,
                    status_code=status_code,
                    algorithm=algorithm,
                    suggested_value=suggested_value,
                )
            elif self.file_name[-3:] == "txt":
                self.write_txt(
                    value=value,
                    status=status,
                    timestamp=timestamp,
                    status_code=status_code,
                    algorithm=algorithm,
                    suggested_value=suggested_value,
                )
            elif self.file_name[-3:] == "csv":
                self.write_csv(
                    value=value,
                    status=status,
                    timestamp=timestamp,
                    status_code=status_code,
                    algorithm=algorithm,
                    suggested_value=suggested_value,
                )
            else:
                print("Output file type not supported.")

    def write_JSON(
        self,
        value: Any,
        timestamp: Any,
        status: str = "",
        status_code: int = None,
        algorithm: str = "Unknown",
        suggested_value: Any = None,
    ) -> None:
        # Construct the object to write
        to_write = {"algorithm": algorithm}
        if value is not None:
            to_write["value"] = value
        if status != "":
            to_write["status"] = status
        if timestamp is not None:
            to_write["timestamp"] = timestamp
        if status_code is not None:
            to_write["status_code"] = status_code
        if suggested_value is not None:
            to_write["suggested_value"] = suggested_value

        with open(self.file_path) as json_file:
            data = json.load(json_file)
            temp = data["data"]
            temp.append(to_write)
        with open(self.file_path, "w") as f:
            json.dump(data, f)

    def write_txt(
        self,
        value: Any,
        status: str = "",
        timestamp: Any = 0,
        status_code: int = None,
        algorithm: str = "Unknown",
        suggested_value: Any = None,
    ) -> None:
        # print(self.file_path)
        with open(self.file_path, "a") as txt_file:
            o = (
                str(timestamp)
                + ": "
                + status
                + "(value: "
                + str(value)
                + ")"
                + ", Algorithm: "
                + algorithm
            )
            if suggested_value is not None:
                o = o + ", Suggested value: " + suggested_value
            o = o + "\n"
            txt_file.write(o)

    def write_csv(
        self,
        value: Any,
        status: str = "",
        timestamp: Any = 0,
        status_code: int = None,
        algorithm: str = "Unknown",
        suggested_value: Any = None,
    ) -> None:
        # Construct the object to write
        to_write = {"algorithm": algorithm}
        to_write["value"] = value
        to_write["status"] = status
        to_write["timestamp"] = timestamp
        to_write["status_code"] = status_code
        to_write["suggested_value"] = suggested_value

        with open(self.file_path, "a", newline="") as csv_file:
            fieldnames = [
                "timestamp",
                "status",
                "status_code",
                "value",
                "suggested_value",
                "algorithm",
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow(to_write)


class KafkaOutput(OutputAbstract):
    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        # print(conf)
        if conf is not None:
            self.configure(conf=conf)

    def configure(self, conf: Dict[Any, Any]) -> None:
        super().configure(conf=conf)
        self.node_id = conf["node_id"]
        if "has_suggested_value" in conf:
            self.has_suggested_value = conf["has_suggested_value"]
        self.producer = KafkaProducer(
            bootstrap_servers=["localhost:9092"],
            value_serializer=lambda x: dumps(x).encode("utf-8"),
        )

    def send_out(
        self,
        suggested_value: Any = None,
        status: str = "",
        timestamp: Any = None,
        status_code: int = None,
        value: Any = None,
        algorithm: str = "Unknown",
    ) -> None:
        # Send to kafka only if an anomaly is detected (or if it is specified
        # that ok values are to be sent)
        if status_code != 1 or self.send_ok:
            # Construct the object to write
            to_write = {"algorithm": algorithm}
            if value is not None:
                to_write["value"] = value
            if status != "":
                to_write["status"] = status
            if timestamp is not None:
                to_write["timestamp"] = timestamp
            if status_code is not None:
                to_write["status_code"] = status_code
            if (suggested_value is not None) and self.has_suggested_value:
                to_write["suggested_value"] = suggested_value

            kafka_topic = "anomalies_" + str(self.node_id)

            self.producer.send(kafka_topic, value=to_write)


class InfluxOutput(OutputAbstract):
    ip: str
    port: str
    token: str
    org: str
    bucket: str
    measurement: str
    tags: Dict
    # s, ms, us, ns
    unix_time_format: str
    has_suggested_value: bool

    influx_writer: Any

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        logging.basicConfig(
            format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
        )
        if conf is not None:
            self.configure(conf=conf)

    def configure(self, conf: Dict[Any, Any] = None) -> None:
        super().configure(conf=conf)

        # Configura writer
        self.ip = conf["ip"]
        self.port = conf["port"]
        self.token = conf["token"]
        self.org = conf["org"]
        url = "http://" + self.ip + ":" + self.port

        self.influx_writer = InfluxDBClient(
            url=url, token=self.token, org=self.org
        ).write_api(write_options=ASYNCHRONOUS)

        self.bucket = conf["bucket"]
        self.measurement = conf["measurement"]
        self.tags = eval(conf["tags"])
        self.has_suggested_value = conf["has_suggested_value"]

        self.unix_time_format = conf["unix_time_format"]
        # Check if format is is acceptable
        if (
            self.unix_time_format != "s"
            and self.unix_time_format != "ms"
            and self.unix_time_format != "ns"
            and self.unix_time_format != "us"
        ):
            logging.error("Invalid unix_time_format at %s.", self.measurement)
            exit(1)

    def send_out(
        self,
        suggested_value: Any = None,
        status: str = "",
        timestamp: Any = None,
        status_code: int = None,
        value: Any = "",
        algorithm: str = "Unknown",
    ) -> None:
        print(
            "measurement "
            + self.measurement
            + "; influx seind out call "
            + str(value)
            + ", "
            + str(status),
            flush=True,
        )

        # Send to kafka only if an anomaly is detected (or if it is specified
        # that ok values are to be sent)
        if status_code != 1 or self.send_ok:
            # Construct the object to write

            # Check and set value
            to_write = {"algorithm": algorithm}
            if value is not None:
                # Writes to influx only the first element in feature vector
                to_write["value"] = value[0]
            else:
                logging.warning(
                    "Missing value in influxdb output %s.", self.measurement
                )
                return
            # Set status
            to_write["status"] = status

            # Check and set status_code
            if status_code is not None:
                to_write["status_code"] = status_code
            else:
                logging.warning(
                    "Missing status_code in influxdb output %s.", self.measurement
                )

            # Suggested value is optional so it may not be recorded
            if self.has_suggested_value:
                # Check and set status_code
                if suggested_value is not None:
                    to_write["suggested_value"] = suggested_value
                else:
                    logging.warning(
                        "Missing suggested_value in influxdb output %s. Try setting has_suggested_value to False",
                        self.measurement,
                    )

            # Change timestamp to ns
            if self.unix_time_format == "s":
                timestamp = int(timestamp * 1000000000)
            elif self.unix_time_format == "ms":
                timestamp = int(timestamp * 1000000)
            elif self.unix_time_format == "us":
                timestamp = int(timestamp * 1000)

            # Write to database
            # print("measurement " + self.measurement + " sending", flush=True)
            # print(timestamp, flush=True)
            self.influx_writer.write(
                self.bucket,
                self.org,
                [
                    {
                        "measurement": self.measurement,
                        "tags": self.tags,
                        "fields": to_write,
                        "time": timestamp,
                    }
                ],
            )
            # print("measurement " + self.measurement + " done", flush=True)
