from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np
import matplotlib.pyplot as plt
import time
plt.style.use('dark_background')


class VisualizationAbstract(ABC):
    @abstractmethod
    def configure(self, conf: Dict[Any, Any] = None) -> None:
        pass

    @abstractmethod
    def update(self, timestamp: Any, value: List[Any]) -> None:
        pass

class GraphVisualization(VisualizationAbstract):
    num_of_points: int
    num_of_lines: int
    linestyles: List[str]
    lines: List[List[float]]

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf=conf)
        else:
            default = {"num_of_points": 50,
                       "num_of_lines": 1,
                       "linestyles": ['ro']}
            self.configure(conf=default)

    def configure(self, conf: Dict[Any, Any] = None) -> None:
        self.num_of_points = conf["num_of_points"]
        self.num_of_lines = conf["num_of_lines"]
        self.linestyles = conf["linestyles"]
        if("demo_pause" in conf):
            self.pause = conf["demo_pause"]
        else:
            self.pause = 0.1
        self.lines = [[] for _ in range(self.num_of_lines)]
        self.count = 0
        pass

    def update(self, value: List[Any], timestamp: Any = 0,
               status_code: int = None) -> None:
        assert self.num_of_lines <= len(value), "Configuration specifies more lines that were given."
        # value is an array
        # [lastvalue, current_moving_average, current_moving_average+sigma,
        # current_moving_average-sigma] how many of those you actually need
        # depends on num of lines
        y_data = [None]*self.num_of_lines

        # define or update lines
        if(self.lines[0] == []):
            plt.ion()
            fig_graph = plt.figure(figsize=(13, 6))
            ax_graph = [None] * self.num_of_lines

            x_data = [self.count]

            for i in range(self.num_of_lines):
                ax_graph[i] = fig_graph.add_subplot(111)
            y_data = value.copy()
            for i in range(self.num_of_lines):
                self.lines[i], = ax_graph[i].plot(x_data, y_data[i],
                                                  self.linestyles[i],
                                                  alpha=0.8)
                plt.show()

        elif(len(self.lines[0].get_data()[0]) < self.num_of_points):

            x_data = [None] * self.num_of_points
            x_data = np.append(x_data, self.lines[0].get_data()[0])
            x_data = np.append(x_data, self.count)
            x_data = x_data[-self.num_of_points:]
            for i in range(self.num_of_lines):
                y_data[i] = [None]*self.num_of_points
                y_data[i] = np.append(y_data[i], self.lines[i].get_data()[1])
                y_data[i] = np.append(y_data[i], value[i])
                y_data[i] = y_data[i][-self.num_of_points:]
        else:
            x_data = self.lines[0].get_data()[0]
            x_data = np.append(x_data, self.count)
            x_data = x_data[-self.num_of_points:]

            for i in range(self.num_of_lines):
                y_data[i] = self.lines[i].get_data()[1]
                y_data[i] = np.append(y_data[i], value[i])
                y_data[i] = y_data[i][-self.num_of_points:]

        for i in range(self.num_of_lines):
            self.lines[i].set_ydata(y_data[i])
            self.lines[i].set_xdata(x_data)

        # plot limits correction
        if(value is not None):
            if(isinstance(self.lines[0].get_data()[1], np.ndarray)):

                Min = min(filter(lambda x: x is not None,self.lines[0].get_data()[1]))
                Max = max(filter(lambda x: x is not None,self.lines[0].get_data()[1]))
                for i in range(len(self.lines)):
                    if (min(filter(lambda x: x is not None,self.lines[i].get_data()[1])) <= Min):
                        Min = min(filter(lambda x: x is not None,self.lines[i].get_data()[1]))
                    if (max(filter(lambda x: x is not None,self.lines[i].get_data()[1])) >= Max):
                        Max = max(filter(lambda x: x is not None,self.lines[i].get_data()[1])) 
            else:
                Min = min(value)-0.1
                Max = max(value)+0.1
        plt.subplot(111).set_ylim([Min - 0.1, Max + 0.1])
        plt.subplot(111).set_xlim([min(filter(lambda x: x is not None, x_data)) - 0.1, max(filter(lambda x: x is not None, x_data))+0.1])
        plt.pause(self.pause)

        self.count += 1

class StatusPointsVisualization(VisualizationAbstract):
    num_of_points: int
    num_of_lines: int
    linestyles: List[str]
    lines: List[List[float]]
    colors: List[str]
    ax_graph: Any
    count: int

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf=conf)

    def configure(self, conf: Dict[Any, Any] = None) -> None:
        self.num_of_points = conf["num_of_points"]
        self.num_of_lines = conf["num_of_lines"]
        self.linestyles = conf["linestyles"]
        if("demo_pause" in conf):
            self.pause = conf["demo_pause"]
        else:
            self.pause = 0.1

        self.lines = [[] for _ in range(self.num_of_lines)]
        self.colors = []
        self.count = 0

    def update(self, value: List[Any], timestamp: Any = 0,
               status_code: int = None) -> None:
        assert self.num_of_lines <= len(value), "Configuration specifies more lines that were given."
        # value is an array
        # [lastvalue, current_moving_average, current_moving_average+sigma,
        # current_moving_average-sigma] how many of those you actually need
        # depends on num of lines

        start = time.time()

        self.colors.append(self.get_color(status_code=status_code))
        self.colors = self.colors[-self.num_of_points:]

        # define lines
        if self.lines[0] == []:
            plt.ion()
            fig_graph = plt.figure(figsize=(13, 6))
            self.ax_graph = [None] * self.num_of_lines
            for i in range(self.num_of_lines):
                self.ax_graph[i] = fig_graph.add_subplot(111)
            
            self.x_data_lables = []
            self.x_data_lables = np.append(self.x_data_lables, timestamp)
            self.x_data_lables = self.x_data_lables[-self.num_of_points:]
            
            self.x_data = []
            self.x_data = np.append(self.x_data, self.count)
            self.x_data = self.x_data[-self.num_of_points:]
            
            y_data = value.copy()

            

            self.lines[0] = self.ax_graph[0].scatter(self.x_data, y_data[0], c=self.colors[0])
            for i in range(1, self.num_of_lines):
                self.lines[i], = self.ax_graph[i].plot(self.x_data, y_data[i],
                                            self.linestyles[i], alpha=0.8)
                plt.xticks(self.x_data, self.x_data_lables)
                plt.show()
            return

        # less points than there could be
        elif (len(self.lines[0].get_offsets()[:, 0]) < self.num_of_points):
            self.x_data_lables = np.append(self.x_data_lables, timestamp)
            self.x_data_lables = self.x_data_lables[-self.num_of_points:]

            self.x_data = np.append(self.x_data, self.count)
            self.x_data = self.x_data[-self.num_of_points:]

            y_data = [None]*self.num_of_lines

            y_data[0] = np.append(y_data[0], self.lines[0].get_offsets()[:, 1])
            y_data[0] = np.append(y_data[0], value[0])
            y_data[0] = y_data[0][-self.num_of_points:]

            for i in range(1, self.num_of_lines):
                y_data[i] = [None]*self.num_of_points
                y_data[i] = np.append(y_data[i], self.lines[i].get_data()[1])
                y_data[i] = np.append(y_data[i], value[i])
                y_data[i] = y_data[i][-self.num_of_points:]
        else:
            self.x_data = np.append(self.x_data, self.count)
            self.x_data = self.x_data[-self.num_of_points:]

            self.x_data_lables = np.append(self.x_data_lables, timestamp)
            self.x_data_lables = self.x_data_lables[-self.num_of_points:]

            y_data = [None]*self.num_of_lines

            y_data[0] = self.lines[0].get_offsets()[:, 1]
            y_data[0] = np.append(y_data[0], value[0])
            y_data[0] = y_data[0][-self.num_of_points:]

            for i in range(1, self.num_of_lines):
                y_data[i] = self.lines[i].get_data()[1]
                y_data[i] = np.append(y_data[i], value[i])
                y_data[i] = y_data[i][-self.num_of_points:]


        for i in range(1, self.num_of_lines):
            self.lines[i].set_ydata(y_data[i])
            self.lines[i].set_xdata(self.x_data)

        y_data = self.lines[0].get_offsets()[:, 1]
        y_data = np.append(y_data, value[0])
        y_data = y_data[-self.num_of_points:]

        self.lines[0] = self.ax_graph[0].scatter(self.x_data, y_data, c=self.colors)

        # plot limits correction
        #if(value is not None):
            #if (min(value) <= self.lines[0].axes.get_ylim()[0]) or (max(value) >= self.lines[0].axes.get_ylim()[1]):

        #a new object is added to the figure each cycle, remove the previous line
        plt.gca().get_children()[0].remove()
        
        self.ax_graph[0].set_ylim([min(filter(lambda x: x is not None, self.lines[0].get_offsets()[:,1])) - 1,
                                        max(filter(lambda x: x is not None, self.lines[0].get_offsets()[:,1])) + 1])
        
        plt.subplot(111).set_xlim([float(min(filter(lambda x: x is not None, self.x_data))) - 1, float(max(filter(lambda x: x is not None, self.x_data)))+1])
        plt.pause(self.pause)

        self.count += 1

    def get_color(self, status_code: int) -> str:
        if(status_code == 1):
            # OK
            return "w"
        elif (status_code == 0):
            # Warning
            return "y"
        elif(status_code == -1):
            # Error
            return "r"
        elif(status_code == 2):
            # Undefined
            return "b"
        else:
            print("Visualization: Invalid status code")
            exit(1)


class HistogramVisualization(VisualizationAbstract):
    num_of_bins: int
    range: List[int]
    bins: Any
    bin_vals: List[Any]
    line: List[Any]

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf=conf)

    def configure(self, conf: Dict[Any, Any] = None) -> None:
        self.num_of_bins = conf["num_of_bins"]
        self.range = conf["range"]
        if("demo_pause" in conf):
            self.pause = conf["demo_pause"]
        else:
            self.pause = 0.1
        self.bins = np.linspace(self.range[0], self.range[1], self.num_of_bins)
        self.bin_vals = np.zeros(len(self.bins))
        self.line = []

    def update(self, value: List[Any], timestamp: Any = 0,
               status_code: int = None) -> None:
        if (self.line == []):
            fig_hist = plt.figure(figsize=(13, 6))
            ax_hist = fig_hist.add_subplot(111)

            self.bin_vals[np.digitize([value[0]], self.bins)] = 1
            self.line, = ax_hist.step(self.bins, self.bin_vals)
        else:
            self.bin_vals[np.digitize(value[0], self.bins)] += 1
            self.line.set_ydata(self.bin_vals)

        if(value[0] is not None):
            if (max(self.bin_vals) >= self.line.axes.get_ylim()[1]):
                plt.subplot(111).set_ylim([0, max(self.bin_vals)+1])

        plt.pause(self.pause)