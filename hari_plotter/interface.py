from abc import ABC, abstractmethod, abstractmethod
from typing import Iterator

import numpy as np

from .hari_dynamics import HariDynamics
from .hari_graph import HariGraph
from .simulation import Simulation

from typing import List, Optional, Union


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

    def _calculate_mean_node_values(self, group: List) -> dict:
        nodes = []
        opinions = []
        sizes = []
        max_opinions = []  # Store the maximum value of max_opinion for each node
        min_opinions = []  # Store the minimum value of min_opinion for each node

        # Dictionary to hold cumulative values for each node
        node_opinions = {}
        node_sizes = {}
        node_max_opinions = {}
        node_min_opinions = {}

        for image in group:
            node_values = self.get_node_values(image)

            for node, value in node_values['opinion'].items():
                node_opinions.setdefault(node, []).append(value)
                node_sizes.setdefault(node, []).append(
                    node_values['size'][node])

                # Check and store max_opinion values
                if 'max_opinion' in node_values:
                    node_max_opinions.setdefault(node, []).append(
                        node_values['max_opinion'][node])
                elif 'inner_opinions' in node_values and node in node_values['inner_opinions']:
                    max_inner_opinion = max(
                        node_values['inner_opinions'][node].values(), default=None)
                    if max_inner_opinion is not None:
                        node_max_opinions.setdefault(
                            node, []).append(max_inner_opinion)

                # Check and store min_opinion values
                if 'min_opinion' in node_values:
                    node_min_opinions.setdefault(node, []).append(
                        node_values['min_opinion'][node])
                elif 'inner_opinions' in node_values and node in node_values['inner_opinions']:
                    min_inner_opinion = min(
                        node_values['inner_opinions'][node].values(), default=None)
                    if min_inner_opinion is not None:
                        node_min_opinions.setdefault(
                            node, []).append(min_inner_opinion)

        # Calculate mean opinion, size, and max/min opinion values for each node
        for node, opinion_values in node_opinions.items():
            nodes.append(node)
            opinions.append(np.mean(opinion_values))
            sizes.append(np.mean(node_sizes[node]))
            max_opinions.append(max(node_max_opinions.get(node, [None])))
            min_opinions.append(min(node_min_opinions.get(node, [None])))

        return {
            'nodes': nodes,
            'opinions': opinions,
            'sizes': sizes,
            'max_opinions': max_opinions,
            'min_opinions': min_opinions
        }

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

    def get_node_values(self, image) -> dict:
        return self.data.node_values

    def mean_node_values(self) -> Iterator[dict]:
        data = self._calculate_mean_node_values(
            [None])  # No group for single image
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

    def get_node_values(self, image) -> dict:
        return self.data[image].node_values

    def mean_node_values(self) -> Iterator[dict]:
        for group in self.data.groups:
            data = self._calculate_mean_node_values(group)
            data['time'] = group[-1]
            yield data

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

    def get_node_values(self, image) -> dict:
        return self.data.dynamics[image].node_values

    def mean_node_values(self) -> Iterator[dict]:
        for group in self.data.dynamics.groups:
            data = self._calculate_mean_node_values(group)
            data['time'] = group[-1] * self.data.model.params.get("dt", 1)
            yield data
