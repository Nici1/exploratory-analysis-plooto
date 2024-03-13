from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import numpy as np


class NormalizationAbstract(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def configure(self, conf: Dict[Any, Any]) -> None:
        pass

    @abstractmethod
    def add_value(self, value: Any) -> None:
        pass

    @abstractmethod
    def get_normalized(self, value: Any) -> Union[List[Any], bool]:
        pass


class LastNAverage(NormalizationAbstract):
    memory: List[List[Any]]
    
    # Configuration
    N: int

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf=conf)

    def configure(self, conf: Dict[Any, Any] = None) -> None:
        self.N = conf["N"]

        self.memory = []

    def add_value(self, value: Any) -> None:
        self.memory.append(value)
        self.memory = self.memory[-self.N:]

    def get_normalized(self ,value: Any) -> Union[List[Any], bool]:
        super().get_normalized(value=value)

        if(self.N == len(self.memory)):
            # Return vector of averaged values
            np_memory = np.array(self.memory)
            normalized = np.mean(np_memory, axis=0).tolist()
            self.add_value(value=normalized)
            return normalized
        else:
            self.add_value(value=value)
            return False


class PeriodicLastNAverage(NormalizationAbstract):
    memory: List[List[Any]]
    memory_len: int
    
    # Configuration
    period: int
    N: int

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf=conf)

    def configure(self, conf: Dict[Any, Any] = None) -> None:
        self.N = conf["N"]
        self.period = conf["period"]

        self.memory_len = ((self.N-1) * self.period)+1
        self.memory = []

    def add_value(self, value: Any) -> None:
        super().add_value(value)
        self.memory.append(value)
        self.memory = self.memory[-self.memory_len:]

    def get_normalized(self, value: Any) -> Union[List[Any], bool]:
        super().get_normalized(value)

        if(self.memory_len == len(self.memory)):
            np_memory = np.array(self.memory[::self.period])
            normalized = np.mean(np_memory, axis=0).tolist()
            self.add_value(value=normalized)
            return normalized
        else:
            self.add_value(value=value)
            return False

