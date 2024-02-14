from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np


class NodeGatherer(ABC):
    """
    Abstract base class representing a node gatherer. It provides a framework for extracting specific attributes
    or parameters of nodes within a graph.

    Attributes:
        _methods (dict): Dictionary mapping property names to the associated method.
        G (nx.Graph): The graph containing the nodes.
    """
    _methods = {}

    def __init__(self, G):
        super().__init__()
        self.G = G

    @classmethod
    def parameter(cls, property_name):
        """
        Class method decorator to register methods that provide various properties of nodes.

        Parameters:
        -----------
        property_name : str
            The name of the property that the method provides.

        Raises:
        -------
        ValueError: 
            If the property name is already defined.
        """
        def decorator(method):
            if property_name in cls._methods:
                raise ValueError(
                    f"Property {property_name} is already defined.")
            cls._methods[property_name] = method
            return method
        return decorator

    @property
    def methods(self) -> list:
        """Returns a list of property names that have been registered."""
        return list(self._methods.keys())

    def gather_unprocessed(self, param: Union[str, List[str]]) -> dict:
        """
        Extracts the desired parameter(s) for all nodes in the graph, and includes node names/IDs.

        Parameters:
            param (Union[str, List[str]]): The parameter name or a list of parameter names to extract.

        Returns:
            dict: A dictionary containing the extracted parameters.
        """
        if isinstance(param, str):
            param = [param]

        result = dict()

        for p in param:
            if p not in self._methods:
                raise ValueError(
                    f'{p} is not a valid method. Valid methods are: {self.methods}')

            method = self._methods[p]
            method_result = method(self)
            # Use None or a default value for missing data
            result[p] = method_result

        return result

    def gather(self, param: Union[str, List[str]]) -> dict:
        """
        Extracts the desired parameter(s) for all nodes in the graph, and includes node names/IDs.

        Parameters:
            param (Union[str, List[str]]): The parameter name or a list of parameter names to extract.

        Returns:
            dict: A dictionary containing the extracted parameters, plus a 'nodes' key with a list of node names/IDs.
        """
        if isinstance(param, str):
            param = [param]

        result = self.gather_unprocessed(param)

        transformed_result = {}

        # Assuming all parameters have the same nodes, sort them once
        nodes = sorted(next(iter(result.values())).keys())
        transformed_result['nodes'] = nodes

        # Iterate over each parameter in the result
        for parameter, values in result.items():

            # Extract the values for the sorted nodes
            sorted_values = [values.get(node, None) for node in nodes]

            # Store in the transformed result
            transformed_result[parameter] = sorted_values

        return transformed_result

    def gather_everything(self) -> dict:
        return self.gather_unprocessed(self._methods.keys())

    @abstractmethod
    def merge(self, nodes: (List[dict])) -> dict:
        """
        Abstract method to gather properties or parameters for a merged node based on input nodes.

        Parameters:
            nodes (List[dict]): A list of nodes to merge.

        Returns:
            dict: Dictionary with merged node attributes.
        """
        raise NotImplementedError(
            "This method must be implemented in subclasses")


class DefaultNodeGatherer(NodeGatherer):
    """
    Default implementation of the NodeGatherer. This class provides methods to extract default parameters from nodes
    within a graph.
    """

    def merge(self, node_ids: List[dict]) -> dict:
        """
        Merges given nodes and computes combined attributes such as 'inner_opinions', 'cluster_size', and 'label'.

        Parameters:
            node_ids (List[dict]): A list of node dictionaries, each containing node attributes.

        Returns:
            dict: A dictionary with merged node attributes.
        """
        if not node_ids:
            raise ValueError("The input list of nodes must not be empty.")

        nodes = [self.G.nodes[node_id] for node_id in node_ids]

        size = sum(node.get('cluster_size', len(node)) for node in nodes)

        # Gather all opinions of the nodes being merged using node labels/identifiers as keys
        inner_opinions = {}

        for node, node_id in zip(nodes, node_ids):
            # Check if node has 'inner_opinions', if not, create one
            if 'inner_opinions' in node:
                inner_opinions.update(node['inner_opinions'])
            else:
                if len(node_id) != 1:
                    warnings.warn(
                        f"The size of the node {node_id} is {size}, higher than one. Assuming that all opinions in this cluster were equal. This is not typical behavior, check that that it corresponds to your intention.")
                inner_opinions[node_id[0]] = node['opinion']

        return {
            'cluster_size': size,
            'opinion': sum(node.get('cluster_size', len(node)) * node['opinion'] for node in nodes) / size,
            'inner_opinions': inner_opinions
        }

    @NodeGatherer.parameter("opinion")
    def opinion(self) -> dict:
        """Returns a dictionary mapping node IDs to their respective opinions."""
        return self.G.opinions

    @NodeGatherer.parameter("cluster_size")
    def cluster_size(self) -> dict:
        """Returns a dictionary mapping node IDs to their cluster sizes."""
        return {node: self.G.nodes[node].get('cluster_size', len(node)) for node in self.G.nodes}

    @NodeGatherer.parameter("importance")
    def importance(self) -> dict:
        """Returns a dictionary mapping node IDs to their importance based on influence and size."""
        importance_dict = {}
        size_dict = self.cluster_size()

        for node in self.G.nodes:
            influences_sum = sum(data['influence']
                                 for _, _, data in self.G.edges(node, data=True))
            importance_dict[node] = influences_sum / \
                size_dict[node] if size_dict[node] != 0 else 0

        return importance_dict

    @NodeGatherer.parameter("neighbor_mean_opinion")
    def neighbor_mean_opinion(self) -> dict:
        """Returns a dictionary mapping node IDs to the mean opinion of their neighbors."""
        data = {}
        opinions = self.G.opinions
        for node in self.G.nodes():
            node_opinion = opinions[node]
            neighbors = list(self.G.neighbors(node))
            if neighbors:
                mean_neighbor_opinion = sum(
                    opinions[neighbor] for neighbor in neighbors) / len(neighbors)
                data[node] = mean_neighbor_opinion
            else:
                data[node] = np.nan

        return data

    @NodeGatherer.parameter('inner_opinions')
    def inner_opinions(self) -> dict:
        """Returns a dictionary mapping node IDs to their inner opinions or the main opinion if missing."""
        return {node: self.G.nodes[node].get('inner_opinions', {node: self.G.nodes[node]['opinion']}) for node in self.G.nodes}

    @NodeGatherer.parameter('max_opinion')
    def max_opinion(self) -> dict:
        """Returns a dictionary mapping node IDs to the maximum opinion value among their inner opinions."""
        return {node: max((self.G.nodes[node].get('inner_opinions', {None: self.G.nodes[node]['opinion']})).values()) for node in self.G.nodes}

    @NodeGatherer.parameter('min_opinion')
    def min_opinion(self) -> dict:
        """Returns a dictionary mapping node IDs to the minimum opinion value among their inner opinions."""
        return {node: min((self.G.nodes[node].get('inner_opinions', {None: self.G.nodes[node]['opinion']})).values()) for node in self.G.nodes}

    @NodeGatherer.parameter('label')
    def min_opinion(self) -> dict:
        return {node: self.G.nodes[node].get("label", None) for node in self.G.nodes}


class ActivityDefaultNodeGatherer(DefaultNodeGatherer):
    def merge(self, node_ids: List[dict]) -> dict:
        """
        Merges given nodes and computes combined attributes such as 'inner_opinions', 'cluster_size', and 'label'.

        Parameters:
            node_ids (List[dict]): A list of node dictionaries, each containing node attributes.

        Returns:
            dict: A dictionary with merged node attributes.
        """
        if not node_ids:
            raise ValueError("The input list of nodes must not be empty.")

        nodes = [self.G.nodes[node_id] for node_id in node_ids]

        size = sum(node.get('cluster_size', len(node)) for node in nodes)

        activity = sum(node.get('activity', np.nan) for node in nodes)

        # Gather all opinions of the nodes being merged using node labels/identifiers as keys
        inner_opinions = {}

        for node, node_id in zip(nodes, node_ids):
            # Check if node has 'inner_opinions', if not, create one
            if 'inner_opinions' in node:
                inner_opinions.update(node['inner_opinions'])
            else:
                if len(node_id) != 1:
                    warnings.warn(
                        f"The size of the node {node_id} is {size}, higher than one. Assuming that all opinions in this cluster were equal. This is not typical behavior, check that that it corresponds to your intention.")
                inner_opinions[node_id[0]] = node['opinion']

        return {
            'cluster_size': size,
            'opinion': sum(node.get('cluster_size', len(node)) * node['opinion'] for node in nodes) / size,
            'inner_opinions': inner_opinions,
            'activity': activity
        }

    @NodeGatherer.parameter('activity')
    def activity(self) -> dict:
        return {node: self.G.nodes[node].get("activity", np.nan) for node in self.G.nodes}
