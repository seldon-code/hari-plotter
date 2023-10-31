from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Iterator, List, Optional, Union

import numpy as np

from .hari_dynamics import HariDynamics
from .hari_graph import HariGraph
from .simulation import Simulation


class Interface(ABC):
    REQUIRED_TYPE = None
    available_classes = {}

    def __init__(self, data):
        if not isinstance(data, self.REQUIRED_TYPE):
            raise ValueError(
                f"data must be an instance of {self.REQUIRED_TYPE}")
        self.data = data

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.REQUIRED_TYPE:
            Interface.available_classes[cls.REQUIRED_TYPE] = cls

    @classmethod
    def create_interface(cls, data):
        interface_class = cls.available_classes.get(type(data))
        if interface_class:
            return interface_class(data)
        else:
            raise ValueError("Invalid data type, no matching interface found.")

    @classmethod
    def info(cls):
        mappings = ', '.join(
            [f"{key.__name__} -> {value.__name__}" for key, value in cls.available_classes.items()])
        return f"Available Classes: {mappings}"

    @abstractmethod
    def images(self):
        raise NotImplementedError(
            "This method must be implemented in subclasses")

    @abstractmethod
    def groups(self):
        raise NotImplementedError(
            "This method must be implemented in subclasses")

    def _calculate_mean_node_values(self, group: List, params: List[str]) -> dict:
        # Dictionary to hold cumulative values for each node and each parameter
        node_values_accumulator = defaultdict(lambda: defaultdict(list))
        results = defaultdict(list)

        # Process each image in the group
        for image in group:
            node_values = self.get_node_values(image, params)

            # Accumulate the parameter values for each node
            for node, parameters in node_values.items():
                for param, value in parameters.items():
                    node_values_accumulator[node][param].append(value)

        # Calculate results for each node and parameter
        for node, parameters in node_values_accumulator.items():
            results['node'].append(node)
            for param, values in parameters.items():
                if param.startswith("max_"):
                    # Compute max value for parameters starting with 'max_'
                    results[param].append(max(values))
                elif param.startswith("min_"):
                    # Compute min value for parameters starting with 'min_'
                    results[param].append(min(values))
                else:
                    # Compute mean value for other parameters
                    results[param].append(np.mean(values))

        return results

    def _calculate_node_values(self, group: List, params: List[str]) -> dict:
        # Dictionary to hold values for each node and each parameter
        node_values_accumulator = defaultdict(lambda: defaultdict(list))
        results = defaultdict(lambda: defaultdict(list))

        # Process each image in the group
        for image in group:
            node_values = self.get_node_values(image, params)

            # print(f'{len(node_values.keys()) = }')
            # print(f'{node_values[0].keys() = }')
            # print(f'{node_values[0]["neighbor_mean_opinion"] = }')

            # Accumulate the parameter values for each node
            for node, parameters in node_values.items():
                for param, value in parameters.items():
                    node_values_accumulator[node][param].append(value)

        # Transfer values from the accumulator to the results
        for node, parameters in node_values_accumulator.items():
            for param, values in parameters.items():
                results[node][param] = values

        return results

    @abstractmethod
    def mean_node_values(self, params: List[str]) -> Iterator[dict]:
        raise NotImplementedError(
            "This method must be implemented in subclasses")

    @abstractmethod
    def node_values(self, params: List[str]) -> Iterator[dict]:
        raise NotImplementedError(
            "This method must be implemented in subclasses")


class HariGraphInterface(Interface):
    REQUIRED_TYPE = HariGraph

    def images(self) -> Iterator[dict]:
        image = self.data.copy()
        image.time = 0
        yield image

    def groups(self):
        raise NotImplementedError(
            "This method must be implemented in subclasses")

    def get_node_values(self, image, params: List[str]) -> dict:
        return self.data.gatherer.gather(params)

    def mean_node_values(self, params: List[str]) -> Iterator[dict]:
        data = {'data': self._calculate_mean_node_values(
            [None])}  # No group for single image
        data['time'] = 0
        yield data

    def node_values(self, params: List[str]) -> Iterator[dict]:
        data = {'data': self._calculate_node_values(
            [None])}  # No group for single image
        data['time'] = 0
        yield data


class HariDynamicsInterface(Interface):
    REQUIRED_TYPE = HariDynamics

    def images(self) -> Iterator[dict]:
        for group in self.data.groups:
            image = self.data[group[-1]].copy()
            time = group[-1]
            yield {'image': image, 'time': time}

    def groups(self):
        raise NotImplementedError(
            "This method must be implemented in subclasses")

    def get_node_values(self, image, params: List[str]) -> dict:
        return self.data[image].gatherer.gather(params)

    def mean_node_values(self, params: List[str]) -> Iterator[dict]:
        for group in self.data.groups:
            data = {'data': self._calculate_mean_node_values(group, params)}
            data['time'] = group[-1]
            yield data

    def node_values(self, params: List[str]) -> Iterator[dict]:
        for group in self.data.groups:
            data = {'data': self._calculate_node_values(group, params)}
            data['time'] = group[-1]
            yield data


class SimulationInterface(Interface):
    REQUIRED_TYPE = Simulation

    def images(self) -> Iterator[dict]:
        for group in self.data.dynamics.groups:
            image = self.data.dynamics[group[-1]].copy()
            time = group[-1] * self.data.model.params.get("dt", 1)
            yield {'image': image, 'time': time}

    def groups(self):
        raise NotImplementedError(
            "This method must be implemented in subclasses")

    def get_node_values(self, image, params: List[str]) -> dict:
        return self.data.dynamics[image].gatherer.gather(params)

    def mean_node_values(self, params: List[str]) -> Iterator[dict]:
        for group in self.data.dynamics.groups:
            data = {'data': self._calculate_mean_node_values(group, params)}
            data['time'] = group[-1] * self.data.model.params.get("dt", 1)
            yield data

    def node_values(self, params: List[str]) -> Iterator[dict]:
        for group in self.data.dynamics.groups:
            data = {'data': self._calculate_node_values(group, params)}
            data['time'] = group[-1] * self.data.model.params.get("dt", 1)
            yield data
