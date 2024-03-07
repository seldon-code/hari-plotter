from __future__ import annotations

import copy
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union
from warnings import warn

if TYPE_CHECKING:
    from .hari_graph import HariGraph

import networkx as nx
import numpy as np
from scipy.stats import gaussian_kde


class NodeEdgeGatherer(ABC):
    """
    Abstract base class for gathering node and edge attributes within a graph. It defines a framework for extracting
    specific attributes or parameters of nodes and edges, facilitating customized data collection and analysis on
    graph-based structures.

    Attributes:
        node_parameter_logger (ParameterLogger): A logger for registering and tracking node parameters.
        edge_parameter_logger (ParameterLogger): A logger for registering and tracking edge parameters.
        G (HariGraph): The graph instance from NetworkX containing the nodes and edges for analysis.
    """

    class ParameterLogger:
        """
        A utility class used within NodeEdgeGatherer to log and manage parameters associated with nodes or edges.
        It allows for the dynamic registration of parameters to be gathered from the graph.

        Attributes:
            parameters (Dict[str, Any]): A dictionary mapping parameter names to their corresponding functions
                or values within the graph structure.
        """

        def __init__(self) -> None:
            self.parameters: Dict[str, Any] = {}

        def __call__(self, property_name: str):
            """
            Allows the ParameterLogger instance to be used as a decorator for registering parameters.

            Parameters:
                property_name (str): The name of the property that the parameter provides.
            """

            def decorator(class_parameter: Any) -> Any:
                if property_name in self.parameters:
                    raise ValueError(
                        f"Property {property_name} is already defined.")
                self.parameters[property_name] = class_parameter
                return class_parameter

            return decorator

        def copy(self) -> "NodeEdgeGatherer.ParameterLogger":
            """
            Creates a deep copy of the ParameterLogger instance, including its parameters.

            Returns:
                NodeEdgeGatherer.ParameterLogger: A new instance of ParameterLogger with copied parameters.
            """
            new_instance = NodeEdgeGatherer.ParameterLogger()
            new_instance.parameters = copy.deepcopy(self.parameters)
            return new_instance

        def keys(self) -> list:
            """Returns a list of all registered parameter names."""
            return list(self.parameters.keys())

        def __getitem__(self, key: str) -> Any:
            """
            Allows direct access to a parameter's value via its name.

            Parameters:
                key (str): The name of the parameter to access.

            Returns:
                Any: The value or function associated with the given parameter name.
            """
            return self.parameters.get(key, None)

        def __setitem__(self, key: str, value: Any) -> None:
            """
            Allows direct setting of a parameter's value via its name.

            Parameters:
                key (str): The name of the parameter.
                value (Any): The value or function to associate with the parameter name.
            """
            self.parameters[key] = value

        def __contains__(self, key: str) -> bool:
            """
            Checks if a given parameter name is already registered.

            Parameters:
                key (str): The name of the parameter to check.

            Returns:
                bool: True if the parameter is registered, False otherwise.
            """
            return key in self.parameters

        def __str__(self) -> str:
            """Returns a string representation of the parameters dictionary."""
            return str(self.parameters)

        def __len__(self) -> int:
            """Returns the number of registered parameters."""
            return len(self.parameters)

        def __iter__(self):
            """Allows iteration over the registered parameter names."""
            return iter(self.parameters)

    node_parameter_logger = ParameterLogger()
    edge_parameter_logger = ParameterLogger()

    def __init__(self, G: HariGraph) -> None:
        """
        Initializes a NodeEdgeGatherer instance with a specific graph.

        Parameters:
            G (HariGraph): The graph containing the nodes and edges for data gathering.
        """
        self.G = G

    @property
    def node_parameters(self) -> List[str]:
        """Returns a list of all registered node parameter names."""
        return list(self.node_parameter_logger.keys())

    @property
    def edge_parameters(self) -> List[str]:
        """Returns a list of all registered edge parameter names."""
        return list(self.edge_parameter_logger.keys())

    def gather_unprocessed(self, param: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Gathers unprocessed parameter data for all nodes or edges in the graph based on the specified parameter(s).

        Parameters:
            param (Union[str, List[str]]): The parameter name(s) to extract data for.

        Returns:
            Dict[str, Any]: A dictionary containing the raw data for the requested parameter(s).
        """
        if isinstance(param, str):
            param = [param]

        result: Dict[str, Any] = {}

        for p in param:
            if p not in self.node_parameter_logger:
                raise ValueError(
                    f"{p} is not a valid parameter. Valid parameters are: {list(self.node_parameter_logger.keys())}")

            parameter = self.node_parameter_logger[p]
            parameter_result = parameter(self)
            result[p] = parameter_result

        return result

    def gather(self, param: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Gathers and organizes parameter data for all nodes or edges in the graph based on the specified parameter(s).

        Parameters:
            param (Union[str, List[str]]): The parameter name(s) to extract data for.

        Returns:
            Dict[str, Any]: A dictionary containing organized data for the requested parameter(s), including
                            a 'Nodes' key with a list of node names/IDs.
        """
        if isinstance(param, str):
            param = [param]

        result = self.gather_unprocessed(param)

        transformed_result: Dict[str, Any] = {}
        nodes = sorted(next(iter(result.values())).keys())
        transformed_result['Nodes'] = nodes

        for parameter, values in result.items():
            sorted_values = [values.get(node, None) for node in nodes]
            transformed_result[parameter] = sorted_values

        return transformed_result

    def gather_everything(self) -> Dict[str, Any]:
        """
        Gathers data for all registered parameters for nodes or edges in the graph.

        Returns:
            Dict[str, Any]: A dictionary containing data for all registered parameters.
        """
        return self.gather_unprocessed(list(self.node_parameter_logger.keys()))


class DefaultNodeEdgeGatherer(NodeEdgeGatherer):
    """
    Provides a default implementation for the NodeEdgeGatherer abstract base class. This class implements methods
    to extract and manipulate default parameters from nodes within a graph, such as merging nodes and clusters
    based on specific criteria.
    """

    # Inherit parameter loggers for node and edge parameters from the base class
    node_parameter_logger = NodeEdgeGatherer.ParameterLogger()
    edge_parameter_logger = NodeEdgeGatherer.ParameterLogger()

    def merge_nodes(self, i: Tuple[int], j: Tuple[int]) -> None:
        """
        Merges two specified nodes into a single new node within the graph. The new node's attributes are determined
        by the attributes of the merged nodes, and the edges are updated accordingly.

        Parameters:
            i (Tuple[int]): Identifier for the first node to be merged.
            j (Tuple[int]): Identifier for the second node to be merged.
        """
        if not (isinstance(i, tuple) and isinstance(j, tuple)):
            raise TypeError('Node identifiers must be tuples.')

        merged_data = self.merge([i, j])
        new_node_id = tuple(sorted(i + j))

        self.G.add_node(new_node_id, **merged_data)

        for u, v, data in list(self.G.edges(data=True)):
            if u in [i, j] and v not in [i, j]:
                self.G.add_edge(new_node_id, v, **data)
            elif v in [i, j] and u not in [i, j]:
                self.G.add_edge(u, new_node_id, **data)
            self.G.remove_edge(u, v)

        self.G.remove_node(i)
        self.G.remove_node(j)

    def merge_clusters(self, clusters: List[List[Tuple[int]]], labels: Union[List[str], None] = None, merge_remaining: bool = False) -> None:
        """
        Merges groups of nodes (clusters) into single nodes, optionally merging unclustered nodes into an additional
        node. Each new node represents its constituent cluster.

        Parameters:
            clusters (List[List[Tuple[int]]]): A list of clusters, each represented as a list of node identifiers.
            labels (Union[List[str], None], optional): Labels for the new nodes created from each cluster. Defaults to None.
            merge_remaining (bool, optional): If True, merges nodes not included in any cluster into a separate new node. Defaults to False.
        """
        labels = labels or [None] * len(clusters)

        if merge_remaining:
            all_nodes = set(self.G.nodes)
            clustered_nodes = {
                node for cluster in clusters for node in cluster}
            remaining_nodes = list(all_nodes - clustered_nodes)

            if remaining_nodes:
                clusters.append(remaining_nodes)
                labels.append(None)

        for cluster, label in zip(clusters, labels):
            new_node_id = tuple(sorted(sum(cluster, ())))
            merged_attributes = self.merge(cluster)

            self.G.add_node(new_node_id, **merged_attributes)
            if label is not None:
                self.G.nodes[new_node_id]['Label'] = label

            for old_node_id in cluster:
                for successor in list(self.G.successors(old_node_id)):
                    if successor not in cluster:
                        self.G.add_edge(new_node_id, successor,
                                        **self.G[old_node_id][successor])
                for predecessor in list(self.G.predecessors(old_node_id)):
                    if predecessor not in cluster:
                        self.G.add_edge(predecessor, new_node_id,
                                        **self.G[predecessor][old_node_id])
                self.G.remove_node(old_node_id)

    def mean_graph(self, images: List[HariGraph]) -> HariGraph:
        """
        Calculates the mean graph from a list of HariGraph instances. The mean graph's nodes and edges have attributes
        that are the average of the corresponding attributes in the input graphs.

        Parameters:
            images (List['HariGraph']): A list of HariGraph instances from which to calculate the mean graph.

        Returns:
            'HariGraph': A new HariGraph instance representing the mean of the input graphs.
        """
        if not images:
            raise ValueError("Input list of graphs is empty.")
        if len(images) == 1:
            return images[0].copy()

        mean_graph: HariGraph = type(self.G)()
        mean_graph.set_gatherer(type(self))

        # Assure all graphs have the same nodes
        nodes = set(images[0].nodes)
        if not all(set(g.nodes) == nodes for g in images):
            raise ValueError(
                "Not all input graphs have the same set of nodes.")

        # Calculate mean attributes for nodes
        for node in nodes:
            mean_opinion = np.mean([g.nodes[node]['Opinion'] for g in images])
            mean_graph.add_node(node, Opinion=mean_opinion)

        # Calculate mean attributes for edges
        edges_set = set()
        for graph in images:
            edges_set.update(graph.edges)

        for u, v in edges_set:
            influences = []
            for graph in images:
                if graph.has_edge(u, v):
                    influences.append(graph.edges[u, v]['Influence'])
            if influences:  # Only add the edge if at least one graph has it
                mean_influence = np.mean(influences)
                mean_graph.add_edge(u, v, Influence=mean_influence)

        return mean_graph

    def merge(self, node_ids: List[dict]) -> Dict[str, Union[int, float, Dict]]:
        """
        Merges a list of nodes based on their identifiers, calculating combined attributes such as inner opinions,
        cluster size, and aggregate opinions.

        Parameters:
            node_ids (List[dict]): A list of node attribute dictionaries to be merged.

        Returns:
            Dict[str, Union[int, float, Dict]]: A dictionary containing the merged attributes of the nodes.
        """
        if not node_ids:
            raise ValueError("No nodes provided for merging.")

        nodes = [self.G.nodes[node_id] for node_id in node_ids]
        size = sum(node.get('Cluster size', 1) for node in nodes)
        inner_opinions = {}

        for node, node_id in zip(nodes, node_ids):
            if 'Inner opinions' in node:
                inner_opinions.update(node['Inner opinions'])
            else:
                if len(node_id) != 1:
                    warn(f"Node {node_id} has size {size}, which is unusual.")
                inner_opinions[node_id[0]] = node['Opinion']

        return {
            'Cluster size': size,
            'Opinion': sum(node.get('Cluster size', 1) * node['Opinion'] for node in nodes) / size,
            'Inner opinions': inner_opinions
        }

    # Decorators to register node attributes for gathering
    @node_parameter_logger("Opinion")
    def opinion(self) -> Dict[Tuple[int], float]:
        """Returns a mapping of node IDs to their opinions."""
        return {node: data['Opinion'] for node, data in self.G.nodes(data=True)}

    @staticmethod
    def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
        """Kernel Density Estimation with Scipy"""
        # x are the data points, x_grid are the points to estimate the density at
        kde = gaussian_kde(x, bw_method=bandwidth, **kwargs)
        return kde.evaluate(x_grid)

    @node_parameter_logger("Opinion density")
    def opinion_density(self) -> Dict[Tuple[int], float]:
        """Returns a mapping of node IDs to their opinion distribution density."""
        opinions = np.array(list(self.opinion().values()))
        densities = self.kde_scipy(opinions, opinions)
        return {node: density for node, density in zip(self.G.nodes, densities)}

    @node_parameter_logger("Cluster size")
    def cluster_size(self) -> Dict[Tuple[int], int]:
        """Returns a mapping of node IDs to their cluster sizes."""
        return {node: data.get('Cluster size', 1) for node, data in self.G.nodes(data=True)}

    @node_parameter_logger("Importance")
    def importance(self) -> Dict[Tuple[int], float]:
        """Calculates and returns a mapping of node IDs to their importance, based on influence and size."""
        importance_dict = {}
        for node in self.G.nodes:
            influences_sum = sum(data['Influence']
                                 for _, _, data in self.G.edges(node, data=True))
            cluster_size = self.G.nodes[node].get('Cluster size', 1)
            importance_dict[node] = influences_sum / \
                cluster_size if cluster_size else 0
        return importance_dict

    @node_parameter_logger("Neighbor mean opinion")
    def neighbor_mean_opinion(self) -> Dict[Tuple[int], float]:
        """Returns a mapping of node IDs to the average opinion of their neighbors."""
        data = {}
        for node in self.G.nodes:
            neighbors = list(self.G.neighbors(node))
            if neighbors:
                data[node] = np.mean(
                    [self.G.nodes[neighbor]['Opinion'] for neighbor in neighbors])
            else:
                data[node] = np.nan
        return data

    @node_parameter_logger('Inner opinions')
    def inner_opinions(self) -> Dict[Tuple[int], Dict]:
        """Returns a mapping of node IDs to their inner opinions."""
        return {node: data.get('Inner opinions', {node: data['Opinion']}) for node, data in self.G.nodes(data=True)}

    @node_parameter_logger('Max opinion')
    def max_opinion(self) -> Dict[Tuple[int], float]:
        """Returns a mapping of node IDs to the maximum opinion value among their inner opinions."""
        return {node: max(data.get('Inner opinions', {None: data['Opinion']}).values()) for node, data in self.G.nodes(data=True)}

    @node_parameter_logger('Min opinion')
    def min_opinion(self) -> Dict[Tuple[int], float]:
        """Returns a mapping of node IDs to the minimum opinion value among their inner opinions."""
        return {node: min(data.get('Inner opinions', {None: data['Opinion']}).values()) for node, data in self.G.nodes(data=True)}

    @node_parameter_logger('Opinion Standard Deviation')
    def std_opinion(self) -> Dict[Tuple[int], float]:
        """Returns a mapping of node IDs to the standard deviation of opinion values among their inner opinions."""
        return {node: np.std(list(data.get('Inner opinions', {None: data['Opinion']}).values())) for node, data in self.G.nodes(data=True)}

    @node_parameter_logger('Label')
    def label(self) -> Dict[Tuple[int], str]:
        """Returns a mapping of node IDs to their labels."""
        return {node: data.get('Label', None) for node, data in self.G.nodes(data=True)}


class ActivityDefaultNodeEdgeGatherer(DefaultNodeEdgeGatherer):
    """
    Extends the DefaultNodeEdgeGatherer with functionality to handle node activity. It adds the ability to merge
    nodes based on activity levels in addition to the default parameters.
    """

    # Copy parameter loggers to extend with activity-related parameters
    node_parameter_logger = DefaultNodeEdgeGatherer.node_parameter_logger.copy()
    edge_parameter_logger = DefaultNodeEdgeGatherer.edge_parameter_logger.copy()

    def merge(self, node_ids: List[dict]) -> Dict[str, Union[int, float, Dict]]:
        """
        Extends the merge function to include activity in the merged node attributes.

        Parameters:
            node_ids (List[dict]): A list of node dictionaries to be merged.

        Returns:
            Dict[str, Union[int, float, Dict]]: A dictionary containing the merged attributes, including activity.
        """
        base_merged_data = super().merge(node_ids)
        activity = sum(self.G.nodes[node_id].get(
            'Activity', 0) for node_id in node_ids)

        base_merged_data['Activity'] = activity
        return base_merged_data

    def mean_graph(self, images: List[HariGraph]) -> HariGraph:
        """
        Calculates the mean graph from a list of HariGraph instances. The mean graph's nodes and edges have attributes
        that are the average of the corresponding attributes in the input graphs.

        Parameters:
            images (List['HariGraph']): A list of HariGraph instances from which to calculate the mean graph.

        Returns:
            'HariGraph': A new HariGraph instance representing the mean of the input graphs.
        """
        if not images:
            raise ValueError("Input list of graphs is empty.")
        if len(images) == 1:
            return images[0].copy()

        mean_graph: HariGraph = type(self.G)()
        mean_graph.set_gatherer(type(self))

        # Assure all graphs have the same nodes
        nodes = set(images[0].nodes)
        if not all(set(g.nodes) == nodes for g in images):
            raise ValueError(
                "Not all input graphs have the same set of nodes.")

        # Calculate mean attributes for nodes
        for node in nodes:
            mean_opinion = np.mean([g.nodes[node]['Opinion'] for g in images])
            mean_activity = np.mean(
                [g.nodes[node]['Activity'] for g in images])
            mean_graph.add_node(node, Opinion=mean_opinion,
                                Activity=mean_activity)

        # Calculate mean attributes for edges
        edges_set = set()
        for graph in images:
            edges_set.update(graph.edges)

        for u, v in edges_set:
            influences = []
            for graph in images:
                if graph.has_edge(u, v):
                    influences.append(graph.edges[u, v]['Influence'])
            if influences:  # Only add the edge if at least one graph has it
                mean_influence = np.mean(influences)
                mean_graph.add_edge(u, v, Influence=mean_influence)

        return mean_graph

    @node_parameter_logger('Activity')
    def activity(self) -> Dict[Tuple[int], float]:
        """Returns a mapping of node IDs to their activity levels."""
        return {node: data.get('Activity', np.nan) for node, data in self.G.nodes(data=True)}
