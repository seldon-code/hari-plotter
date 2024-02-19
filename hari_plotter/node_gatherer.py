from __future__ import annotations

import copy
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np


class NodeEdgeGatherer(ABC):
    """
    Abstract base class representing a node gatherer. It provides a framework for extracting specific attributes
    or parameters of nodes within a graph.

    Attributes:
        node_parameter_logger : Class to handle names and associated method.
        G (nx.Graph): The graph containing the nodes.
    """
    class ParameterLogger:
        def __init__(self) -> None:
            self.parameters = {}

        def __call__(self, property_name):
            """
            Makes the ParameterLogger instance callable. It can be used as a decorator to register parameters.

            Parameters:
            -----------
            property_name : str
                The name of the property that the parameter provides.
            """
            def decorator(class_parameter):
                if property_name in self.parameters:
                    raise ValueError(
                        f"Property {property_name} is already defined.")
                self.parameters[property_name] = class_parameter
                return class_parameter
            return decorator

        def copy(self):
            # Create a new instance of ParameterLogger
            new_instance = NodeEdgeGatherer.ParameterLogger()
            new_instance.parameters = copy.deepcopy(
                self.parameters)  # Deep copy the dictionary
            return new_instance

        def keys(self):
            return self.parameters.keys()

        def __getitem__(self, key):
            # Returns None if key is not found
            return self.parameters.get(key, None)

        def __setitem__(self, key, value):
            self.parameters[key] = value

        def __contains__(self, key):
            return key in self.parameters

        def __str__(self):
            return str(self.parameters)

        def __len__(self):
            return len(self.parameters)

        def __iter__(self):
            return iter(self.parameters)

    node_parameter_logger = ParameterLogger()
    edge_parameter_logger = ParameterLogger()

    def __init__(self, G):
        super().__init__()
        self.G = G

    @property
    def node_parameters(self) -> list:
        """Returns a list of property names that have been registered."""
        return list(self.node_parameter_logger.keys())

    @property
    def edge_parameters(self) -> list:
        """Returns a list of property names that have been registered."""
        return list(self.edge_parameter_logger.keys())

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
            if p not in self.node_parameter_logger:
                raise ValueError(
                    f'{p} is not a valid parameter. Valid parameters are: {self.parameters}')

            parameter = self.node_parameter_logger[p]
            parameter_result = parameter(self)
            # Use None or a default value for missing data
            result[p] = parameter_result

        return result

    def gather(self, param: Union[str, List[str]]) -> dict:
        """
        Extracts the desired parameter(s) for all nodes in the graph, and includes node names/IDs.

        Parameters:
            param (Union[str, List[str]]): The parameter name or a list of parameter names to extract.

        Returns:
            dict: A dictionary containing the extracted parameters, plus a 'Nodes' key with a list of node names/IDs.
        """
        if isinstance(param, str):
            param = [param]

        result = self.gather_unprocessed(param)

        transformed_result = {}

        # Assuming all parameters have the same nodes, sort them once
        nodes = sorted(next(iter(result.values())).keys())
        transformed_result['Nodes'] = nodes

        # Iterate over each parameter in the result
        for parameter, values in result.items():

            # Extract the values for the sorted nodes
            sorted_values = [values.get(node, None) for node in nodes]

            # Store in the transformed result
            transformed_result[parameter] = sorted_values

        return transformed_result

    def gather_everything(self) -> dict:
        return self.gather_unprocessed(self.node_parameter_logger.keys())

    # @abstractmethod
    # def merge(self, nodes: (List[dict])) -> dict:
    #     """
    #     Abstract method to gather properties or parameters for a merged node based on input nodes.

    #     Parameters:
    #         nodes (List[dict]): A list of nodes to merge.

    #     Returns:
    #         dict: Dictionary with merged node attributes.
    #     """
    #     raise NotImplementedError(
    #         "This method must be implemented in subclasses")


class DefaultNodeEdgeGatherer(NodeEdgeGatherer):
    """
    Default implementation of the NodeEdgeGatherer. This class provides methods to extract default parameters from nodes
    within a graph.
    """
    node_parameter_logger = NodeEdgeGatherer.ParameterLogger()
    edge_parameter_logger = NodeEdgeGatherer.ParameterLogger()

    def merge_nodes(self, i: Tuple[int], j: Tuple[int]):
        """
        Merges two nodes in the graph into a new node.

        The new node's opinion is a weighted average of the opinions of
        the merged nodes, and its name is the concatenation of the names
        of the merged nodes. The edges are reconnected to the new node,
        and the old nodes are removed.

        Parameters:
            i Tuple[int]: The identifier for the first node to merge.
            j Tuple[int]: The identifier for the second node to merge.
        """
        if not (isinstance(i, tuple) and isinstance(j, tuple)):
            raise TypeError('Incorrect node format')

        # Merge nodes using the gatherer's merge method
        merged_data = self.merge([i, j])

        # Generate a new node ID
        new_node_id = tuple(sorted(i+j))

        # Add the new merged node to the graph
        self.G.add_node(new_node_id, **merged_data)

        # Reconnect edges
        for u, v, data in list(self.G.edges(data=True)):
            if u in [i, j]:
                if v not in [i, j]:
                    influence = data['Influence']
                    self.G.add_edge(new_node_id, v)
                    self.G[new_node_id][v]['Influence'] = influence
                self.G.remove_edge(u, v)
            elif v in [i, j]:
                if u not in [i, j]:
                    influence = data['Influence']
                    self.G.add_edge(u, new_node_id)
                    self.G[u][new_node_id]['Influence'] = influence
                self.G.remove_edge(u, v)

        # Remove the original nodes
        self.G.remove_node(i)
        self.G.remove_node(j)

    def merge_clusters(self, clusters: List[List[Tuple[int]]], labels: Union[List[str], None] = None, merge_remaining=False):
        """
        Merges clusters of nodes in the graph into new nodes. Optionally merges the remaining nodes into an additional cluster.

        Parameters:
            clusters (Union[List[Set[int]], Dict[int, int]]): A list where each element is a set containing
                                    the IDs of the nodes in a cluster to be merged or a dictionary mapping old node IDs
                                    to new node IDs.
            merge_remaining (bool): If True, merge the nodes not included in clusters into an additional cluster. Default is False.
        """
        labels = labels if labels is not None else [None]*len(clusters)

        # Remaining nodes not in any cluster
        if merge_remaining:
            # All nodes in the graph
            all_nodes = set(self.G.nodes)

            # Record all nodes that are part of the specified clusters
            clustered_nodes = set(
                node for cluster in clusters for node in cluster)
            remaining_nodes = all_nodes - clustered_nodes
            if remaining_nodes:
                clusters.append(list(remaining_nodes))
                labels.append(None)

        for cluster, label in zip(clusters, labels):
            new_node_name = tuple(sorted(sum(cluster, ())))
            merged_attributes = self.merge(cluster)
            self.G.add_node(new_node_name, **merged_attributes)
            if label is not None:
                self.G.nodes[new_node_name]['Label'] = label

            # Reconnect edges
            for old_node_id in cluster:
                for successor in list(self.G.successors(old_node_id)):
                    if successor not in cluster:
                        influence = self.G[old_node_id][successor]['Influence']
                        self.G.add_edge(new_node_name, successor)
                        self.G[new_node_name][successor]['Influence'] = influence

                for predecessor in list(self.G.predecessors(old_node_id)):
                    if predecessor not in cluster:
                        influence = self.G[predecessor][old_node_id]['Influence']
                        self.G.add_edge(predecessor, new_node_name)
                        self.G[predecessor][new_node_name]['Influence'] = influence

                # Remove old node
                self.G.remove_node(old_node_id)

    def merge(self, node_ids: List[dict]) -> dict:
        """
        Merges given nodes and computes combined attributes such as 'Inner opinions', 'Cluster size', and 'Label'.

        Parameters:
            node_ids (List[dict]): A list of node dictionaries, each containing node attributes.

        Returns:
            dict: A dictionary with merged node attributes.
        """
        if not node_ids:
            raise ValueError("The input list of nodes must not be empty.")

        nodes = [self.G.nodes[node_id] for node_id in node_ids]

        size = sum(node.get('Cluster size', len(node)) for node in nodes)

        # Gather all opinions of the nodes being merged using node labels/identifiers as keys
        inner_opinions = {}

        for node, node_id in zip(nodes, node_ids):
            # Check if node has 'Inner opinions', if not, create one
            if 'Inner opinions' in node:
                inner_opinions.update(node['Inner opinions'])
            else:
                if len(node_id) != 1:
                    warnings.warn(
                        f"The size of the node {node_id} is {size}, higher than one. Assuming that all opinions in this cluster were equal. This is not typical behavior, check that that it corresponds to your intention.")
                inner_opinions[node_id[0]] = node['Opinion']

        return {
            'Cluster size': size,
            'Opinion': sum(node.get('Cluster size', len(node)) * node['Opinion'] for node in nodes) / size,
            'Inner opinions': inner_opinions
        }

    @node_parameter_logger("Opinion")
    def opinion(self) -> dict:
        """Returns a dictionary mapping node IDs to their respective opinions."""
        return self.G.opinions

    @node_parameter_logger("Cluster size")
    def cluster_size(self) -> dict:
        """Returns a dictionary mapping node IDs to their cluster sizes."""
        return {node: self.G.nodes[node].get('Cluster size', len(node)) for node in self.G.nodes}

    @node_parameter_logger("Importance")
    def importance(self) -> dict:
        """Returns a dictionary mapping node IDs to their importance based on Influence and Size."""
        importance_dict = {}
        size_dict = self.cluster_size()

        for node in self.G.nodes:
            influences_sum = sum(data['Influence']
                                 for _, _, data in self.G.edges(node, data=True))
            importance_dict[node] = influences_sum / \
                size_dict[node] if size_dict[node] != 0 else 0

        return importance_dict

    @node_parameter_logger("Neighbor mean opinion")
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

    @node_parameter_logger('Inner opinions')
    def inner_opinions(self) -> dict:
        """Returns a dictionary mapping node IDs to their inner opinions or the main opinion if missing."""
        return {node: self.G.nodes[node].get('Inner opinions', {node: self.G.nodes[node]['Opinion']}) for node in self.G.nodes}

    @node_parameter_logger('Max opinion')
    def max_opinion(self) -> dict:
        """Returns a dictionary mapping node IDs to the maximum opinion value among their inner opinions."""
        return {node: max((self.G.nodes[node].get('Inner opinions', {None: self.G.nodes[node]['Opinion']})).values()) for node in self.G.nodes}

    @node_parameter_logger('Min opinion')
    def min_opinion(self) -> dict:
        """Returns a dictionary mapping node IDs to the minimum opinion value among their inner opinions."""
        return {node: min((self.G.nodes[node].get('Inner opinions', {None: self.G.nodes[node]['Opinion']})).values()) for node in self.G.nodes}

    @node_parameter_logger('Label')
    def min_opinion(self) -> dict:
        return {node: self.G.nodes[node].get('Label', None) for node in self.G.nodes}


class ActivityDefaultNodeEdgeGatherer(DefaultNodeEdgeGatherer):
    node_parameter_logger = DefaultNodeEdgeGatherer.node_parameter_logger.copy()
    edge_parameter_logger = DefaultNodeEdgeGatherer.edge_parameter_logger.copy()

    def merge(self, node_ids: List[dict]) -> dict:
        """
        Merges given nodes and computes combined attributes such as 'Inner opinions', 'Cluster size', and 'Label'.

        Parameters:
            node_ids (List[dict]): A list of node dictionaries, each containing node attributes.

        Returns:
            dict: A dictionary with merged node attributes.
        """
        if not node_ids:
            raise ValueError("The input list of nodes must not be empty.")

        nodes = [self.G.nodes[node_id] for node_id in node_ids]

        size = sum(node.get('Cluster size', len(node)) for node in nodes)

        activity = sum(node.get('Activity', np.nan) for node in nodes)

        # Gather all opinions of the nodes being merged using node labels/identifiers as keys
        inner_opinions = {}

        for node, node_id in zip(nodes, node_ids):
            # Check if node has 'Inner opinions', if not, create one
            if 'Inner opinions' in node:
                inner_opinions.update(node['Inner opinions'])
            else:
                if len(node_id) != 1:
                    warnings.warn(
                        f"The size of the node {node_id} is {size}, higher than one. Assuming that all opinions in this cluster were equal. This is not typical behavior, check that that it corresponds to your intention.")
                inner_opinions[node_id[0]] = node['Opinion']

        return {
            'Cluster size': size,
            'Opinion': sum(node.get('Cluster size', len(node)) * node['Opinion'] for node in nodes) / size,
            'Inner opinions': inner_opinions,
            'Activity': activity
        }

    @node_parameter_logger('Activity')
    def activity(self) -> dict:
        return {node: self.G.nodes[node].get('Activity', np.nan) for node in self.G.nodes}
