from abc import abstractmethod, ABC
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import sys
import logging
from statistics import mean
import datetime
from ast import literal_eval


sys.path.insert(0, "./src")
sys.path.insert(1, "C:/Users/Matic/SIHT/anomaly_det/anomalyDetection/")
from output import OutputAbstract, TerminalOutput, FileOutput, KafkaOutput, InfluxOutput
from visualization import (
    VisualizationAbstract,
    GraphVisualization,
    HistogramVisualization,
    StatusPointsVisualization,
)
from normalization import NormalizationAbstract, LastNAverage, PeriodicLastNAverage


class AnomalyDetectionAbstract(ABC):
    configuration_location: str

    # needed if there are more anomaly detection algorithms
    algorithm_indx: int

    memory_size: int
    memory: List[List[Any]]
    averages: List[List[int]]
    periodic_averages: List[List[Tuple[int, List[int]]]]
    shifts: List[List[int]]
    time_features: List[str]
    name: str

    use_cols: List[int]
    input_vector_size: int
    outputs: List["OutputAbstract"]
    visualization: "VisualizationAbstract"
    normalization: "NormalizationAbstract"

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

    def __init__(self) -> None:
        # Logging configuration
        logging.basicConfig(
            format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
        )

        self.memory = []

    @abstractmethod
    def message_insert(self, message_value: Dict[Any, Any]) -> Any:
        # logging when message recieved (this line can be commented)
        # logging.info("%s recieved message.", self.name)

        pass

    def filter_by_time(self, message, target_time, tolerance) -> Any:
        try:
            timestamp = pd.to_datetime(message["timestamp"], unit="s")
        except pd._libs.tslibs.np_datetime.OutOfBoundsDatetime:
            timestamp = pd.to_datetime(message["timestamp"], unit="ms")

        time = timestamp.time()

        target_time = datetime.time(target_time[0], target_time[1], target_time[2])
        tol = datetime.timedelta(
            hours=tolerance[0], minutes=tolerance[1], seconds=tolerance[2]
        )
        date = datetime.date(1, 1, 1)
        datetime1 = datetime.datetime.combine(date, time)
        datetime2 = datetime.datetime.combine(date, target_time)

        if (max(datetime2, datetime1) - min(datetime2, datetime1)) < tol:
            return message
        else:
            return None

    @abstractmethod
    def configure(
        self,
        conf: Dict[Any, Any],
        configuration_location: str = None,
        algorithm_indx: int = None,
    ) -> None:
        self.configuration_location = configuration_location

        self.conf = conf

        # If algorithm is initialized from consumer kafka it has this
        # specified
        self.algorithm_indx = algorithm_indx

        # FEATURE CONSTRUCTION CONFIGURATION
        self.input_vector_size = conf["input_vector_size"]

        if "braila_fall_feature" in conf:
            self.braila_fall = conf["braila_fall_feature"]
        else:
            self.braila_fall = []

        if "averages" in conf:
            self.averages = conf["averages"]
        else:
            self.averages = []

        if "periodic_averages" in conf:
            self.periodic_averages = conf["periodic_averages"]
        else:
            self.periodic_averages = []

        if "shifts" in conf:
            self.shifts = conf["shifts"]
        else:
            self.shifts = []

        if "time_features" in conf:
            self.time_features = conf["time_features"]
        else:
            self.time_features = []

        if "max_memory" in conf:
            self.max_memory = conf["max_memory"]
        else:
            self.max_memory = 0

        if "time_average_shifts" in conf:
            self.time_average_shifts = conf["time_average_shifts"]
            self.last_sample = 0
        else:
            self.time_average_shifts = []
            self.last_sample = 0

        # Finds the largest element among averages and shifts
        if len(self.shifts) == 0:
            max_shift = 0
        else:
            max_shifts = []
            for shifts in self.shifts:
                if len(shifts) == 0:
                    max_shifts.append(0)
                else:
                    max_shifts.append(max(shifts))
            max_shift = max(max_shifts) + 1

        if len(self.averages) == 0:
            max_average = 0
        else:
            max_averages = []
            for averages in self.averages:
                if len(averages) == 0:
                    max_averages.append(0)
                else:
                    max_averages.append(max(averages))
            max_average = max(max_averages)

        if len(self.periodic_averages) == 0:
            max_periodic_average = 0
        else:
            max_periodic_average = 0
            for feature_avgs in self.periodic_averages:
                for period_tuple in feature_avgs:
                    period = period_tuple[0]
                    # assumes period specifies at least one average
                    max_avg = max(period_tuple[1])
                    # Memory required to calculate the average
                    required_memory = 1 + (period * (max_avg - 1))
                    if required_memory > max_periodic_average:
                        max_periodic_average = required_memory

        # one because of feature construction memory management
        self.memory_size = max(
            max_shift, max_average, max_periodic_average, self.max_memory, 1
        )

        # OUTPUT/VISUALIZATION INITIALIZATION & CONFIGURATION
        self.outputs = [eval(o) for o in conf["output"]]
        output_configurations = conf["output_conf"]
        for o in range(len(self.outputs)):
            # print(output_configurations[o])
            # print(self.outputs[o])
            self.outputs[o].configure(conf=output_configurations[o])
        if "visualization" in conf:
            self.visualization = eval(conf["visualization"])
            visualization_configurations = conf["visualization_conf"]
            self.visualization.configure(visualization_configurations)
        else:
            self.visualization = None

        # NORMALIZATION INITIALIZATION & CONFIGURATION
        if "normalization" in conf:
            self.normalization = eval(conf["normalization"])
            normalization_configuration = conf["normalization_conf"]
            self.normalization.configure(normalization_configuration)
        else:
            self.normalization = None

        # If specified save which columns to use
        if "use_cols" in conf:
            self.use_cols = conf["use_cols"]
        else:
            self.use_cols = None

    def check_ftr_vector(self, message_value: Dict[Any, Any]) -> bool:
        # Check for ftr_vector field
        if message_value == None:
            return False
        if not "ftr_vector" in message_value:
            print(
                f"{self.name}: ftr_vector field was not contained in message.",
                flush=True,
            )
            return False

        # Check for timestamp field
        if not "timestamp" in message_value:
            print(
                f"{self.name}: timestamp field was not contained in message.",
                flush=True,
            )
            return False

        # Check vector length
        # print(len(message_value["ftr_vector"]))
        if len(message_value["ftr_vector"]) != self.input_vector_size:
            print(
                "%s: Given test value does not satisfy input vector size. Feature vector: %s",
                self.name,
                "["
                + ",".join([str(elem) for elem in message_value["ftr_vector"]])
                + "]",
                flush=True,
            )
            return False

        # Check if feature vector contains a string
        if any(type(x) == str for x in message_value["ftr_vector"]):
            print(f"{self.name}: Feature vector contains a string.", flush=True)
            return False

        # Check if feature vector contains None
        if any(x == None for x in message_value["ftr_vector"]):
            print(f"{self.name}: Feature vector contains a None.", flush=True)
            return False

        if any(np.isnan(x) for x in message_value["ftr_vector"]):
            print(f"{self.name}: Feature vector contains a None.", flush=True)
            return False

        # Test if timestamp is of type int
        if not (
            isinstance(message_value["timestamp"], int)
            or isinstance(message_value["timestamp"], float)
        ):
            logging.warning(
                "%s: Timestamp not in correct format: %s",
                self.name,
                message_value["timestamp"],
            )
            return False

        # Test if timestamp is valid
        timestamp_ok = False

        try:
            pd.to_datetime(message_value["timestamp"], unit="s")
            timestamp_ok = True
        except pd._libs.tslibs.np_datetime.OutOfBoundsDatetime:
            try:
                pd.to_datetime(message_value["timestamp"], unit="ms")
                timestamp_ok = True
            except pd._libs.tslibs.np_datetime.OutOfBoundsDatetime:
                pass
        if not timestamp_ok:
            logging.warning(
                "%s: Invalid timestamp: %s", self.name, message_value["timestamp"]
            )
            return False

        return True

    def change_last_record(self, value: List[Any]) -> None:
        self.memory[-1] = value

    def training_feature_construction(self, data, timestamps) -> List[Any]:
        # Saves memory so it can be restored
        memory_backup = self.memory
        self.memory = []

        # Aditional feature construction
        features = []
        for i in range(len(data)):
            value = data[i].tolist()
            timestamp = timestamps[i]

            feature_vector = self.feature_construction(value=value, timestamp=timestamp)

            if feature_vector is not False:
                if not np.isnan(np.array(feature_vector)).any():
                    features.append(np.array(feature_vector))

        self.memory = memory_backup
        return features

    def feature_construction(
        self, value: List[Any], timestamp: str
    ) -> Union[Any, bool]:
        # print("The value is ", value)
        # Add new value to memory and slice it
        if timestamp < 1e10:
            self.memory.append([value, timestamp])
        else:
            # Convert to seconds
            self.memory.append([value, timestamp / 1000])

        self.memory = self.memory[-self.memory_size :]

        # print("Memory length ", len(self.memory))
        # print("Memory size ", self.memory_size)
        if len(self.memory) < self.memory_size:
            # print("tuka?")
            # The memory does not contain enough records for all shifts and
            # averages to be created
            return False

        # create new value to be returned
        new_value = value.copy()

        # Create average features
        new_value.extend(self.average_construction())

        # Create periodic averages
        new_value.extend(self.periodic_average_construction())

        # Create shifted features
        new_value.extend(self.shift_construction())

        # Create time features
        new_value.extend(self.time_features_construction(timestamp))

        # print(f'{new_value = }')
        if len(self.time_average_shifts) != 0:
            new_value.extend(self.time_averages())
        # print("feature construction")
        # print(f'{new_value = }')
        if self.use_cols is not None:
            try:
                return list(np.array(new_value)[self.use_cols])
            except Exception as e:
                print(f"Exception occurred here: {e}")
                return False
        else:
            # print(new_value)
            return new_value

    def average_construction(self) -> None:
        averages = []

        # Loop through all features
        for feature_index in range(len(self.averages)):
            # Loop through all horizons we want the average of
            for interval in self.averages[feature_index]:
                # Add last interval numbers to values array
                values = []
                for sample_indx in range(len(self.memory)):
                    if sample_indx == interval:
                        break
                    # print(self.memory)
                    values.append(
                        np.array(self.memory)[:, 0][-(sample_indx + 1)][feature_index]
                    )

                averages.append(mean(values))

        return averages

    def periodic_average_construction(self) -> None:
        # construct periodic averages

        periodic_averages = []
        # Loop through features
        for feature_indx in range(len(self.periodic_averages)):
            # Loop through indexes for different features
            for period_indx in range(len(self.periodic_averages[feature_indx])):
                # Extract period and a list of how N-s (number of samples to
                # take from this sequence)
                period = self.periodic_averages[feature_indx][period_indx][0]
                averages = self.periodic_averages[feature_indx][period_indx][1]

                # Loop through N-s (number of samples to take from this
                # sequence)
                for average in averages:
                    # Construct a list of semples with this period from memory
                    periodic_list = []
                    # Loop through samples (in opposite direction) and if they
                    # are of right period add them to the list
                    for i in range(self.memory_size):
                        if len(periodic_list) == average:
                            # Enough samples
                            break
                        if i % period == 0:
                            periodic_list.append(
                                np.array(self.memory)[:, 0][self.memory_size - (i + 1)][
                                    feature_indx
                                ]
                            )

                    # print("periodic list:")
                    # print(periodic_list)

                    # Append average of the list to features
                    avg = mean(periodic_list)
                    periodic_averages.append(avg)

        return periodic_averages

    def shift_construction(self) -> None:
        shifts = []

        # Loop through all features
        for feature_index in range(len(self.shifts)):
            # Loop through all shift values
            for look_back in self.shifts[feature_index]:
                shifts.append(
                    np.array(self.memory)[:, 0][self.memory_size - (look_back + 1)][
                        feature_index
                    ]
                )

        return shifts

    def time_features_construction(self, tmstp: Any) -> None:
        time_features = []

        try:
            dt = pd.to_datetime(tmstp, unit="s")
        except pd._libs.tslibs.np_datetime.OutOfBoundsDatetime:
            dt = pd.to_datetime(tmstp, unit="ms")

        # Requires datetime format
        # Check for keywords specified in time_features
        if "month" in self.time_features:
            time_features.append(int(dt.month))
        if "day" in self.time_features:
            time_features.append(int(dt.day))
        if "weekday" in self.time_features:
            time_features.append(int(dt.weekday()))
        if "hour" in self.time_features:
            time_features.append(int(dt.hour))
        if "minute" in self.time_features:
            time_features.append(int(dt.minute))

        return time_features

    def time_averages(self) -> None:
        shifts = []

        num, period = self.time_average_shifts
        current_time = self.memory[-1][1]

        self.memory = [i for i in self.memory if i]
        self.memory = [i for i in self.memory if type(i) == list]
        self.memory = [i for i in self.memory if len(i) == 2]

        # print(f'{len(self.memory) = }')
        # print(f'{self.memory[-1][1] - self.last_sample = }')

        # print(self.memory[0][1])
        # print(current_time - num * period)

        if abs(self.memory[-1][1] - self.last_sample) < period:
            return []
        # if the length of the memory is long enough
        elif self.memory[0][1] < (current_time - num * period):
            buff = []
            shift = 0
            for sample in self.memory[::-1]:
                if sample[1] > (current_time - (shift + 1) * period):
                    buff.append(sample[0])
                    # print(sample[1], (current_time - (shift + 1) * period))
                else:
                    shift += 1
                    try:
                        print(shifts)
                        shifts.append(np.mean(buff))
                    except:
                        return []
                    buff = []

        if len(shifts) < num or np.isnan(np.array(shifts)).any():
            # print("len of shifts", len(shifts))
            # print("num", num)
            # print("Memory: ", self.memory[0][1])
            # print((current_time - num * period))

            return []
        else:
            self.last_sample = self.memory[-1][1]

            # print("Memory: ", self.memory)
            # print("Shifts: ", shifts[:num][::-1])
            return shifts[:num][::-1]
        # returns only the shifts, no timestamp; frequency of data is decreased

    def normalization_output_visualization(
        self,
        status_code: int,
        status: str,
        value: List[Any],
        timestamp: Any,
        suggested_value=None,
    ) -> None:
        # Normalize if needed or just add the value
        normalized = None

        if suggested_value is not None:
            normalized = suggested_value

        elif self.normalization is not None:
            if status_code == -1:
                normalized = self.normalization.get_normalized(value=value)
                if normalized != False:
                    self.change_last_record(value=normalized)
                else:
                    normalized = None
            else:
                self.normalization.add_value(value=value)

        for output in self.outputs:
            output.send_out(
                timestamp=timestamp,
                status=status,
                suggested_value=normalized,
                value=value,
                status_code=status_code,
                algorithm=self.name,
            )

        if self.visualization is not None:
            lines = [value[0]]
            self.visualization.update(
                value=lines, timestamp=timestamp, status_code=status_code
            )
