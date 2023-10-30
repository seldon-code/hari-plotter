from abc import ABC, abstractmethod, abstractmethod
from typing import Iterator

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

    @abstractmethod
    def mean_node_values(self):
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

    def opinions(self):
        raise NotImplementedError(
            "This method must be implemented in subclasses")

    def neighbors_opinions(self):
        raise NotImplementedError(
            "This method must be implemented in subclasses")


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

    def mean_node_values(self) -> Iterator[dict]:
        for group in self.data.groups:
            nodes = []
            opinions = []
            sizes = []
            max_opinions = []  # To store the maximum value of max_opinion for each node
            min_opinions = []  # To store the minimum value of min_opinion for each node

            # Dictionary to hold cumulative values for each node
            node_opinions = {}
            node_sizes = {}
            node_max_opinions = {}
            node_min_opinions = {}

            for image in group:
                node_values = self.data[image].node_values

                for node, value in node_values['opinion'].items():
                    if node not in node_opinions:
                        node_opinions[node] = []
                    node_opinions[node].append(value)

                    if node not in node_sizes:
                        node_sizes[node] = []
                    node_sizes[node].append(node_values['size'][node])

                    # Check and store max_opinion values
                    if 'max_opinion' in node_values:
                        if node not in node_max_opinions:
                            node_max_opinions[node] = []
                        node_max_opinions[node].append(
                            node_values['max_opinion'][node])

                    # Check and store min_opinion values
                    if 'min_opinion' in node_values:
                        if node not in node_min_opinions:
                            node_min_opinions[node] = []
                        node_min_opinions[node].append(
                            node_values['min_opinion'][node])

            # Calculate mean opinion, size and max/min opinion values for each
            # node
            for node, opinion_values in node_opinions.items():
                nodes.append(node)
                opinions.append(np.mean(opinion_values))
                sizes.append(np.mean(node_sizes[node]))

                # Add max_opinion or None
                if node_max_opinions:
                    max_opinions.append(max(node_max_opinions[node]))
                else:
                    max_opinions.append(None)

                # Add min_opinion or None
                if node_min_opinions:
                    min_opinions.append(min(node_min_opinions[node]))
                else:
                    min_opinions.append(None)

            time = group[-1]
            yield {
                'nodes': nodes,
                'opinions': opinions,
                'sizes': sizes,
                'max_opinions': max_opinions,
                'min_opinions': min_opinions,
                'time': time
            }

    def node_values(self) -> Iterator[dict]:
        for group in self.data.groups:
            nodes = []
            opinions = []
            sizes = self.data[image].cluster_size.values()
            for image in group:
                for node, opinion in self.data[image].opinions.items():
                    nodes.append({'image_number': image, 'node': node})
                    opinions.append(opinion)
            time = group[-1]
            yield {'nodes': nodes, 'opinions': np.array(opinions), 'sizes': sizes, 'time': time}

    def opinion_neighbor_mean_opinion_pairs(self) -> Iterator[dict]:
        for group in self.data.groups:
            nodes = []
            opinions = []
            neighbor_mean_opinions = []
            for image in group:
                for node, (opinion, neighbor_mean_opinion) in self.data[image].get_opinion_neighbor_mean_opinion_pairs_dict(
                ).items():
                    nodes.append({'image_number': image, 'node': node})
                    opinions.append(opinion)
                    neighbor_mean_opinions.append(neighbor_mean_opinion)
            time = group[-1]
            yield {'nodes': nodes, 'opinions': np.array(opinions), 'neighbor_mean_opinions': np.array(opinions), 'time': time}


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

    def opinions(self):
        raise NotImplementedError(
            "This method must be implemented in subclasses")

    def neighbors_opinions(self):
        raise NotImplementedError(
            "This method must be implemented in subclasses")
