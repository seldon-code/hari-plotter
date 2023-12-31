from __future__ import annotations

import json
import os
import random
import re
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import combinations, permutations
from typing import Dict, List, Optional, Set, Union

import networkx as nx
import numpy as np

from .distributions import generate_mixture_of_gaussians


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

    def gather(self, param: Union[str, List[str]]) -> dict:
        """
        Extracts the desired parameter(s) for all nodes in the graph.

        Parameters:
            param (Union[str, List[str]]): The parameter name or a list of parameter names to extract.

        Returns:
            dict: A dictionary containing the extracted parameters. The structure depends on the type of `param`.
        Note:
            Note that output is different if the param is str and List!
            output of 'param' and ['param'] is different!

            If str - dict with nodes as keys and parameter values as values
            {node: parameter value}
            If List - dict with nodes as keys and dict with parameters as keys and parameter values as values as values
            {node: {parameter: parameter value}}
        """
        if isinstance(param, str):
            method = self._methods[param]
            return method(self)

        result = defaultdict(dict)

        for p in param:
            method = self._methods[p]
            # Assuming the method returns a dict with node values
            method_result = method(self)
            for node, value in method_result.items():
                result[node][p] = value

        return dict(result)

    def gather_everything(self) -> dict:
        return self.gather(self._methods.keys())

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

    def merge(self, nodes: List[dict]) -> dict:
        """
        Merges given nodes and computes combined attributes such as 'inner_opinions', 'cluster_size', and 'label'.

        Parameters:
            nodes (List[dict]): A list of node dictionaries, each containing node attributes.

        Returns:
            dict: A dictionary with merged node attributes.
        """
        if not nodes:
            raise ValueError("The input list of nodes must not be empty.")

        size = sum(node.get('cluster_size', len(
            node.get('label', [0, ]))) for node in nodes)

        # Gather all opinions of the nodes being merged using node labels/identifiers as keys
        inner_opinions = {}

        for node in nodes:
            node_label = node.get('label', None)
            if node_label is not None:
                # Check if node has 'inner_opinions', if not, create one
                if 'inner_opinions' in node:
                    inner_opinions.update(node['inner_opinions'])
                else:
                    if len(node_label) != 1:
                        warnings.warn(
                            f"The length of the label in the node is higher than one. Assuming that all opinions in this cluster were equal. This is not typical behavior, check that that it corresponds to your intention. Found in node: {node_label}")
                    for i in node_label:
                        inner_opinions[i] = node['opinion']

        return {
            'cluster_size': size,
            'opinion': sum(node.get('cluster_size', len(node.get('label', [0, ]))) * node['opinion'] for node in nodes) / size,
            'label': [id for node in nodes for id in node['label']],
            'inner_opinions': inner_opinions
        }

    @NodeGatherer.parameter("opinion")
    def opinion(self) -> dict:
        """Returns a dictionary mapping node IDs to their respective opinions."""
        return self.G.opinions

    @NodeGatherer.parameter("cluster_size")
    def cluster_size(self) -> dict:
        """Returns a dictionary mapping node IDs to their cluster sizes."""
        return {node: self.G.nodes[node].get('cluster_size', len(
            self.G.nodes[node].get('label', [0, ]))) for node in self.G.nodes}

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

    @NodeGatherer.parameter("label")
    def label(self) -> dict:
        """Returns a dictionary mapping node IDs to their labels or node IDs if label is missing."""
        return {node: self.G.nodes[node]['label'] for node in self.G.nodes}

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

    @NodeGatherer.parameter('activity')
    def activity(self) -> dict:
        """Returns a dictionary mapping node IDs to their activity levels."""
        return {node: self.G.nodes[node].get('activity', np.nan) for node in self.G.nodes}

    @NodeGatherer.parameter('inner_opinions')
    def inner_opinions(self) -> dict:
        """Returns a dictionary mapping node IDs to their inner opinions or the main opinion if missing."""
        return {node: self.G.nodes[node].get('inner_opinions', {idx: self.G.nodes[node]['opinion'] for idx in self.G.nodes[node]['label']}) for node in self.G.nodes}

    @NodeGatherer.parameter('max_opinion')
    def max_opinion(self) -> dict:
        """Returns a dictionary mapping node IDs to the maximum opinion value among their inner opinions."""
        return {node: max((self.G.nodes[node].get('inner_opinions', {None: self.G.nodes[node]['opinion']})).values()) for node in self.G.nodes}

    @NodeGatherer.parameter('min_opinion')
    def min_opinion(self) -> dict:
        """Returns a dictionary mapping node IDs to the minimum opinion value among their inner opinions."""
        return {node: min((self.G.nodes[node].get('inner_opinions', {None: self.G.nodes[node]['opinion']})).values()) for node in self.G.nodes}


class HariGraph(nx.DiGraph):
    """
    HariGraph extends the DiGraph class of Networkx to offer additional functionality.
    It ensures that each node has a label and provides methods to create, save, and load graphs.
    """

    # ---- Initialization Methods ----

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.generate_labels()
        self.similarity_function = self.default_similarity_function
        self.node_parameter_gatherer = self.default_node_parameter_gatherer
        self.gatherer = DefaultNodeGatherer(self)

    def generate_labels(self):
        """
        Generates labels for the nodes if they don't exist.
        Ensures that each node in the graph has a unique label. If a node lacks a label, its ID will be used.
        """
        if not self.nodes:
            return
        if 'label' not in self.nodes[next(iter(self.nodes))]:
            for i in self.nodes:
                self.nodes[i]['label'] = [i]

    def add_parameters_to_nodes(self, node_ids=None):
        """
        Appends or updates node attributes based on predefined criteria.

        Parameters:
            node_ids (Optional[List[int]]): List of node IDs to which parameters will be added. 
                If None, all nodes will be considered.
        """
        if node_ids is None:
            node_ids = list(self.nodes)

        for node_id in node_ids:
            # Gather parameters for the node
            parameters = self.gatherer.merge([self.nodes[node_id]])
            # Update the node's attributes with the gathered parameters
            self.nodes[node_id].update(parameters)

    def remove_self_loops(self):
        """
        Removes any self-loops present in the graph.

        A self-loop is an edge that connects a node to itself.
        """
        # Iterate over all nodes in the graph
        for node in self.nodes:
            # Check if there is an edge from the node to itself and remove it
            if self.has_edge(node, node):
                self.remove_edge(node, node)

    def assign_random_influences(
            self, mean_influence, influence_range, seed=None):
        """
        Assigns random influence values to all edges of the graph within the given range centered around the mean influence.

        :param mean_influence: float, mean value of the influence.
        :param influence_range: float, the range within which the influence values should fall.
        :param seed: int, random seed (default None).
        """
        if seed is not None:
            random.seed(seed)

        lower_bound = mean_influence - (influence_range / 2)
        upper_bound = mean_influence + (influence_range / 2)

        for u, v in self.edges():
            random_influence = random.uniform(lower_bound, upper_bound)
            self[u][v]['influence'] = random_influence

    def is_degroot_converging(self, tolerance=1e-2) -> bool:
        for node in self.nodes():
            incoming_edges = [(neighbor, node)
                              for neighbor in self.predecessors(node)]
            total_influence = sum(self[u][v]['influence']
                                  for u, v in incoming_edges)
            if not (1 - tolerance <= total_influence <= 1 + tolerance):
                return False
        return True

    def make_degroot_converging(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        for node in self.nodes():
            incoming_edges = [(neighbor, node)
                              for neighbor in self.predecessors(node)]

            # Calculate the total influence of incoming edges for the current
            # node
            total_influence = sum(self[u][v]['influence']
                                  for u, v in incoming_edges)

            # If the total influence is zero, assign random influences to
            # incoming edges
            if total_influence == 0:
                for u, v in incoming_edges:
                    self[u][v]['influence'] = random.random()
                total_influence = sum(self[u][v]['influence']
                                      for u, v in incoming_edges)

            # Adjust the influences proportionally
            for u, v in incoming_edges:
                self[u][v]['influence'] /= total_influence

    @classmethod
    def read_network(cls, network_file: str, opinion_file: str) -> HariGraph:
        """
        Class method to create an instance of HariGraph from the provided files.

        Parameters:
            network_file (str): The path to the network file.
            opinion_file (str): The path to the opinion file.

        Returns:
            HariGraph: An instance of HariGraph.
        """
        # Create an instance of HariGraph
        G = cls()

        # Read network file and add nodes and edges to the graph
        with open(network_file, 'r') as f:
            next(f)  # Skip header line
            for line in f:
                # Split using either space or comma as delimiter
                parts = re.split(r'[ ,]+', line.strip())
                idx_agent = int(parts[0])
                n_neighbors = int(parts[1])
                indices_neighbors = map(int, parts[2:2 + n_neighbors])
                weights = map(float, parts[2 + n_neighbors:])

                # Add nodes with initial opinion 0, opinion will be updated
                # from opinion_file
                G.add_node(idx_agent, opinion=0)

                # Add edges with weights
                for neighbor, weight in zip(indices_neighbors, weights):
                    G.add_edge(neighbor, idx_agent, influence=weight)

        # Read opinion file and update node opinions in the G
        with open(opinion_file, 'r') as f:
            next(f)  # Skip header line
            for line in f:
                # Split using either space or comma as delimiter
                parts = re.split(r'[ ,]+', line.strip())
                idx_agent = int(parts[0])
                opinion = float(parts[1])
                # Check if node exists, if not, add it
                if not G.has_node(idx_agent):
                    G.add_node(idx_agent)
                # Update node opinion
                G.nodes[idx_agent]['opinion'] = opinion
                if len(parts) == 3:
                    G.nodes[idx_agent]['activity'] = float(parts[2])

        # G.remove_self_loops()
        G.generate_labels()
        G.add_parameters_to_nodes()

        return G

    def write_network(self, network_file: str, opinion_file: str, delimiter=','):
        '''
        Save the network structure and node opinions to separate files.
        Attention! This save loses the information about the labels.

        :param network_file: The name of the file to write the network structure to.
        :param opinion_file: The name of the file to write the node opinions to.
        :param delimiter: Delimiter used to separate the opinions in the file (default is comma).
        '''
        # Save network structure
        with open(network_file, 'w') as f:
            # Write header
            f.write(
                f"# idx_agent{delimiter}n_neighbors_in{delimiter}indices_neighbors_in[...]{delimiter}weights_in[...]\n")
            for node in self.nodes:
                # Get incoming neighbors
                neighbors = list(self.predecessors(node))
                weights = [self[neighbor][node]['influence']
                           for neighbor in neighbors]
                # Write each node's information in a separate line
                f.write(
                    f"{node}{delimiter}{len(neighbors)}{delimiter}{delimiter.join(map(str, neighbors))}{delimiter}{delimiter.join(map(str, weights))}\n")

        # Save node opinions
        with open(opinion_file, 'w') as f:
            # Write header
            f.write(f"# idx_agent{delimiter}opinion[...]\n")
            for node, data in self.nodes(data=True):
                # Write each node's opinion opinion in a separate line
                f.write(f"{node}{delimiter}{data['opinion']}\n")

    @classmethod
    def read_json(cls, filename: str) -> HariGraph:
        """
        Reads a HariGraph from a JSON file.

        :param filename (str): The name of the file to read from.
        :return: A new HariGraph instance.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} does not exist.")

        with open(filename, 'r') as file:
            graph_dict = json.load(file)

        G = cls()
        for node in graph_dict["nodes"]:
            G.add_node(node["id"], opinion=node["opinion"],
                       # defaulting to opinion if not present
                       min_opinion=node.get('min_opinion', node["opinion"]),
                       # defaulting to opinion if not present
                       max_opinion=node.get('max_opinion', node["opinion"]),
                       label=node.get('label', [node["id"]]))

        for edge in graph_dict["edges"]:
            G.add_edge(edge["source"], edge["target"],
                       influence=edge["influence"])

        return G

    def write_json(self, filename: str):
        """
        Saves the HariGraph to a JSON file.

        :param filename (str): The name of the file to write to.
        """
        params_to_save = ['opinion', 'cluster_size',
                          'activity', 'inner_opinions']
        graph_dict = {
            "nodes": [
                {"id": n, **{prop: self.gatherer.gather(prop)[n] for prop in params_to_save}} for n in self.nodes()
            ],
            "edges": [{"source": u, "target": v, "influence": self[u][v]["influence"]} for u, v in self.edges()]
        }

        with open(filename, 'w') as file:
            json.dump(graph_dict, file)

    @classmethod
    def guaranteed_connected(cls, n: int) -> HariGraph:
        """
        Creates a guaranteed connected HariGraph instance with n nodes.

        :param n (int): Number of nodes.
        :return: A new HariGraph instance.
        """
        if n < 2:
            raise ValueError("Number of nodes should be at least 2")

        G = cls()
        for i in range(n):
            G.add_node(i, opinion=random.random())

        nodes = list(G.nodes)
        random.shuffle(nodes)
        for i in range(n - 1):
            G.add_edge(nodes[i], nodes[i + 1], influence=random.random())

        additional_edges = random.randint(1, n)
        for _ in range(additional_edges):
            u, v = random.sample(G.nodes, 2)
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v, influence=random.random())
                if random.choice([True, False]) and not G.has_edge(v, u):
                    G.add_edge(v, u, influence=random.random())

        G.generate_labels()
        G.add_parameters_to_nodes()

        return G

    @classmethod
    def by_deletion(cls, n: int, factor: float) -> HariGraph:
        """
        Creates a HariGraph instance by deleting some of the edges of a fully connected graph.

        :param n (int): Number of nodes.
        :param factor (float): Factor representing how many edges to keep.
        :return: A new HariGraph instance.
        """
        if not 0 <= 1 - factor <= 1:
            raise ValueError("Deletion factor must be between 0 and 1")
        if n < 2:
            raise ValueError("Number of nodes should be at least 2")

        G = cls()
        for i in range(n):
            G.add_node(i, opinion=random.random())
        for i in range(n):
            for j in range(n):
                if i != j:
                    G.add_edge(i, j, influence=random.random())

        edges_to_remove = random.sample(
            G.edges, int(len(G.edges) * (1 - factor)))
        G.remove_edges_from(edges_to_remove)

        G.generate_labels()
        G.add_parameters_to_nodes()

        return G

    @classmethod
    def strongly_connected_components(
            cls, cluster_sizes: List[int], inter_cluster_edges: int, mean_opinion: float = 0.5, seed: int = None) -> HariGraph:
        """
        Creates a HariGraph instance with multiple strongly connected components.

        :param cluster_sizes: List[int], sizes of the clusters.
        :param inter_cluster_edges: int, number of edges between the components.
        :param mean_opinion: float, mean opinion of the graph.
        :param seed: int, random seed.
        :return: A new HariGraph instance.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if inter_cluster_edges < len(cluster_sizes):
            raise ValueError(
                "Number of inter-cluster edges should be at least the number of clusters.")

        # Convert cluster_sizes to a list (if it isn't already) and shuffle it
        cluster_sizes = list(cluster_sizes)
        random.shuffle(cluster_sizes)

        # Generate opinions based on the mixture of Gaussians
        total_nodes = sum(cluster_sizes)
        opinions = generate_mixture_of_gaussians(n_samples=total_nodes,
                                                 number_of_peaks=len(
                                                     cluster_sizes),
                                                 opinion_limits=(-1, 1),
                                                 mean_opinion=mean_opinion,
                                                 size_of_each_peak=cluster_sizes,
                                                 seed=seed)
        opinions = sorted(opinions)

        # Step 1: Create the "meta-graph"
        meta_graph = nx.Graph()
        meta_graph.add_nodes_from(range(len(cluster_sizes)))
        edge_counters = {}

        # Ensure the meta-graph is connected by connecting nodes sequentially
        for i in range(len(cluster_sizes) - 1):
            meta_graph.add_edge(i, i + 1)
            edge_counters[(i, i + 1)] = 1
            inter_cluster_edges -= 1
        meta_graph.add_edge(len(cluster_sizes) - 1, 0)
        edge_counters[(len(cluster_sizes) - 1, 0)] = 1
        inter_cluster_edges -= 1

        # Spread the remaining inter-cluster edges across the meta-graph
        while inter_cluster_edges > 0:
            u, v = random.sample(list(meta_graph.nodes), 2)
            if u != v:
                edge_key = tuple(sorted((u, v)))
                if edge_key not in edge_counters:
                    edge_counters[edge_key] = 0
                edge_counters[edge_key] += 1
                inter_cluster_edges -= 1

        # Step 2: Create the actual graph with strongly connected clusters
        G = cls()
        start = 0
        opinion_idx = 0
        for size in cluster_sizes:
            for i in range(start, start + size):
                G.add_node(i, opinion=opinions[opinion_idx])
                opinion_idx += 1
            for i in range(start, start + size):
                for j in range(i + 1, start + size):
                    G.add_edge(i, j)
                    G.add_edge(j, i)
            start += size

        # Step 3: Add inter-cluster edges based on the meta-graph connections
        cluster_starts = [sum(cluster_sizes[:i])
                          for i in range(len(cluster_sizes))]
        for (u, v), count in edge_counters.items():
            for _ in range(count):
                # Pick random nodes from clusters u and v to connect
                source_node = random.choice(
                    range(cluster_starts[u], cluster_starts[u] + cluster_sizes[u]))
                target_node = random.choice(
                    range(cluster_starts[v], cluster_starts[v] + cluster_sizes[v]))
                G.add_edge(source_node, target_node)

        # Assign random influences to the edges of the graph
        G.assign_random_influences(
            mean_influence=0.1, influence_range=0.1, seed=seed)

        G.generate_labels()
        G.add_parameters_to_nodes()

        return G

    def copy(self) -> HariGraph:
        G_copy = super().copy(as_view=False)
        G_copy = HariGraph(G_copy) if not isinstance(
            G_copy, HariGraph) else G_copy
        G_copy.similarity_function = self.similarity_function
        G_copy.node_parameter_gatherer = self.node_parameter_gatherer
        G_copy.gatherer = self.gatherer
        return G_copy

    # ---- Clusterization Methods ----

    def dynamics_step(self, t: float):
        """
        Updates the opinion of each node in the HariGraph instance based on the opinions of its predecessors.

        :param t: The time step factor influencing the dynamics.
        """
        updated_opinions = {}  # Temporary dictionary to store updated opinions

        for i in self.nodes:
            vi = self.nodes[i]['opinion']

            # Predecessors of a node are the start nodes of its incoming edges.
            for j in self.predecessors(i):
                pij = self[j][i]['influence']
                vj = self.nodes[j]['opinion']
                vi += pij * vj * t  # Calculate updated opinion based on each incoming edge

            # Clip the updated opinion to [0, 1]
            # vi = max(0, min(vi, 1))

            updated_opinions[i] = vi

        # Update the opinions in the graph with the calculated updated opinions
        for i, vi in updated_opinions.items():
            self.nodes[i]['opinion'] = vi

    def merge_nodes(self, i: int, j: int):
        """
        Merges two nodes in the graph into a new node.

        The new node's opinion is a weighted average of the opinions of
        the merged nodes, and its label is the concatenation of the labels
        of the merged nodes. The edges are reconnected to the new node,
        and the old nodes are removed.

        Parameters:
            i (int): The identifier for the first node to merge.
            j (int): The identifier for the second node to merge.
        """

        # Get opinions and others parameters using the
        # 'node_parameter_gatherer' method
        parameters = self.gatherer.merge(
            [self.nodes[i], self.nodes[j]])

        # Add a new node to the graph with the calculated opinion, label, and
        # other unpacked parameters
        new_node = max(max(self.nodes), max(
            item for sublist in self.gatherer.label().values() for item in sublist)) + 1
        self.add_node(new_node, **parameters)

        # Reconnect edges
        for u, v, data in list(self.edges(data=True)):
            if u == i or u == j:
                if v != i and v != j:  # Avoid connecting the new node to itself
                    influence = self[u][v]['influence']
                    if self.has_edge(new_node, v):
                        # Sum the influences if both original nodes were
                        # connected to the same node
                        self[new_node][v]['influence'] += influence
                    else:
                        self.add_edge(new_node, v, influence=influence)
                if self.has_edge(
                        u, v):  # Check if the edge exists before removing it
                    self.remove_edge(u, v)
            if v == i or v == j:
                if u != i and u != j:  # Avoid connecting the new node to itself
                    influence = self[u][v]['influence']
                    if self.has_edge(u, new_node):
                        # Sum the influences if both original nodes were
                        # connected to the same node
                        self[u][new_node]['influence'] += influence
                    else:
                        self.add_edge(u, new_node, influence=influence)
                if self.has_edge(
                        u, v):  # Check if the edge exists before removing it
                    self.remove_edge(u, v)

        # Remove the old nodes
        self.remove_node(i)
        self.remove_node(j)

    def find_clusters(self, max_opinion_difference: float = 0.1, min_influence: float = 0.1):
        """
        Finds clusters of nodes in the graph where the difference in the nodes' opinions
        is less than max_opinion_difference, and the influence of i on j is higher than
        min_influence * size(i).

        Parameters:
            max_opinion_difference (float): Maximum allowed difference in the opinions of nodes to form a cluster.
            min_influence (float): Minimum required influence to form a cluster, adjusted by the size of the node.

        Returns:
            List[List[int]]: A list of lists, where each inner list represents a cluster of node identifiers.
        """
        clusters = []
        visited_nodes = set()

        for i in self.nodes:
            if i in visited_nodes:
                continue

            cluster = [i]
            visited_nodes.add(i)

            # Use a list as a simple queue for Breadth-First Search (BFS)
            queue = [i]

            while queue:
                node = queue.pop(0)  # Dequeue a node

                for neighbor in set(self.successors(node)).union(
                        self.predecessors(node)):
                    if neighbor in visited_nodes:
                        continue  # Skip already visited nodes

                    vi = self.nodes[node]['opinion']
                    vj = self.nodes[neighbor]['opinion']
                    size_i = len(self.nodes[node].get('label', [node]))

                    if self.has_edge(node, neighbor):
                        influence_ij = self[node][neighbor]['influence']
                    else:
                        influence_ij = 0

                    if self.has_edge(neighbor, node):
                        influence_ji = self[neighbor][node]['influence']
                    else:
                        influence_ji = 0

                    # Check conditions for being in the same cluster
                    if (abs(vi - vj) <= max_opinion_difference and
                            (influence_ij >= min_influence * size_i or
                             influence_ji >= min_influence * size_i)):
                        cluster.append(neighbor)  # Add to the current cluster
                        visited_nodes.add(neighbor)
                        queue.append(neighbor)  # Enqueue for BFS

            # Add found cluster to the list of clusters
            clusters.append(cluster)

        return clusters

    def merge_by_intervals(self, intervals: List[float]):
        """
        Merges nodes into clusters based on the intervals defined by the input list of opinions.

        Parameters:
            intervals (List[float]): A sorted list of opinions representing the boundaries of the intervals.
        """
        if not intervals:
            raise ValueError("Intervals list cannot be empty")

        # Sort the intervals to ensure they are in ascending order
        intervals = sorted(intervals)
        clusters = []
        # Define the intervals
        intervals = [-np.inf] + intervals + [np.inf]
        for i in range(len(intervals) - 1):
            cluster = []
            lower_bound = intervals[i]
            upper_bound = intervals[i + 1]

            # Iterate over all nodes and assign them to the appropriate cluster
            for node, data in self.nodes(data=True):
                if lower_bound <= data['opinion'] < upper_bound:
                    cluster.append(node)
            if len(cluster) > 0:
                clusters.append(set(cluster))

        # Merge the clusters
        if clusters:
            self.merge_clusters(clusters)

    def get_cluster_mapping(self) -> Dict[int, Set[int]]:
        """
        Generates a mapping of current clusters in the graph.

        The method returns a dictionary where the key is the ID of a current node
        and the opinion is a set containing the IDs of the original nodes that were merged
        to form that node.

        :return: A dictionary representing the current clusters in the graph.
        """
        cluster_mapping = {}
        for node in self.nodes:
            label = self.nodes[node].get('label', [node])
            cluster_mapping[node] = set(label)
        return cluster_mapping

    def merge_clusters(self, clusters: Union[List[Set[int]], Dict[int, int]]):
        """
        Merges clusters of nodes in the graph into new nodes.

        For each cluster, a new node is created whose opinion is the weighted
        average of the opinions of the nodes in the cluster, with the weights
        being the lengths of the labels of the nodes. The new node's label
        is the concatenation of the labels of the nodes in the cluster.
        The edges are reconnected to the new nodes, and the old nodes are removed.

        Parameters:
            clusters (Union[List[Set[int]], Dict[int, int]]): A list where each element is a set containing
                                    the IDs of the nodes in a cluster to be merged or a dictionary mapping old node IDs
                                    to new node IDs.
        """
        # Determine whether clusters are provided as a list or a dictionary
        if isinstance(clusters, dict):
            # Filter out trivial mappings (like {4: {4}})
            clusters = {k: v for k, v in clusters.items()
                        if k not in v or len(v) > 1}
            id_mapping = {old_id: new_id for new_id,
                          old_ids_set in clusters.items() for old_id in old_ids_set}
            new_ids = set(id_mapping.values())
            clusters_list = [set(old_id for old_id, mapped_id in id_mapping.items(
            ) if mapped_id == new_id) for new_id in new_ids]
        elif isinstance(clusters, list):
            id_mapping = {}
            clusters_list = clusters
            new_id_start = max(max(self.nodes), max(
                item for sublist in self.gatherer.label().values() for item in sublist)) + 1
            for i, cluster in enumerate(clusters):
                new_id = new_id_start + i
                for node_id in cluster:
                    assert node_id in self.nodes and node_id not in id_mapping, f"Node {node_id}"\
                        "already exists in the graph or is being merged multiple times."
                    id_mapping[node_id] = new_id
        else:
            raise ValueError(
                "clusters must be a list of sets or a dictionary.")

        for i, cluster in enumerate(clusters_list):
            new_id = id_mapping[next(iter(cluster))]
            parameters = self.gatherer.merge(
                [self.nodes[node_id] for node_id in cluster])
            self.add_node(new_id, **parameters)

        for old_node_id, new_id in id_mapping.items():
            for successor in list(self.successors(old_node_id)):
                mapped_successor = id_mapping.get(successor, successor)
                if mapped_successor != new_id:
                    self.add_edge(new_id, mapped_successor,
                                  influence=self[old_node_id][successor]['influence'])

            for predecessor in list(self.predecessors(old_node_id)):
                mapped_predecessor = id_mapping.get(predecessor, predecessor)
                if mapped_predecessor != new_id:
                    self.add_edge(mapped_predecessor, new_id,
                                  influence=self[predecessor][old_node_id]['influence'])

            self.remove_node(old_node_id)

    def simplify_graph_one_iteration(self):
        """
        Simplifies the graph by one iteration.

        In each iteration, it finds the pair of nodes with the maximum similarity
        and merges them. If labels are not initialized, it initializes them.
        If there is only one node left, no further simplification is possible.

        Returns:
            HariGraph: The simplified graph.
        """
        self.generate_labels()

        if self.number_of_nodes() <= 1:
            warnings.warn(
                "Only one node left, no further simplification possible.")
            return self

        # Find the pair of nodes with the maximum similarity
        max_similarity = -1
        pair = None
        for i, j in combinations(self.nodes, 2):
            similarity = self.compute_similarity(i, j)
            if similarity > max_similarity:
                max_similarity = similarity
                pair = (i, j)

        # Merge the nodes with the maximum similarity
        if pair:
            i, j = pair
            self = self.merge_nodes(i, j)

    # ---- Data Processing Methods ----

    def check_all_paths_exist(self) -> bool:
        """
        Checks if there exists a path between every pair of nodes in the HariGraph instance.

        :return: True if a path exists between every pair of nodes, False otherwise.
        """
        for source, target in permutations(self.nodes, 2):
            if not nx.has_path(self, source, target):
                # print(f"No path exists from {source} to {target}")
                return False
        return True

    @property
    def mean_opinion(self) -> float:
        """
        Calculates the weighted mean opinion of the nodes in the graph.

        For each node, its opinion is multiplied by its weight.
        The weight of a node is the length of its label if defined,
        otherwise, it is assumed to be 1. The method returns the
        sum of the weighted opinions divided by the sum of the weights.

        Returns:
            float: The weighted mean opinion of the nodes in the graph.
                Returns 0 if the total weight is 0 to avoid division by zero.
        """
        total_opinion = 0
        total_weight = 0

        for node in self.nodes:
            opinion = self.nodes[node]['opinion']

            # If label is defined, the weight is the length of the label.
            # If not defined, the weight is assumed to be 1.
            weight = len(self.nodes[node].get('label', [node]))

            total_opinion += opinion * weight
            total_weight += weight

        if total_weight == 0:  # Prevent division by zero
            return 0

        return total_opinion / total_weight

    @staticmethod
    def default_similarity_function(vi: float, vj: float, size_i: int, size_j: int, edge_influence: float, reverse_edge_influence: float,
                                    opinion_coef: float = 1., extreme_coef: float = 0.1, influence_coef: float = 1., size_coef: float = 10.) -> float:
        """
        The default function used to calculate the similarity between two nodes.

        Parameters:
            vi (float): The opinion attribute of node i.
            vj (float): The opinion attribute of node j.
            size_i (int): The length of the label of node i.
            size_j (int): The length of the label of node j.
            edge_influence (float): The influence attribute of the edge from node i to node j, if it exists, else None.
            reverse_edge_influence (float): The influence attribute of the edge from node j to node i, if it exists, else None.
            opinion_coef (float): Coefficient for opinion proximity impact.
            extreme_coef (float): Coefficient for extreme proximity impact.
            influence_coef (float): Coefficient for influence impact.
            size_coef (float): Coefficient for size impact.

        Returns:
            float: The computed similarity between node i and node j.
        """
        # Calculate Value Proximity Impact (high if opinions are close)
        opinion_proximity = opinion_coef * (1 - abs(vi - vj))

        # Calculate Proximity to 0 or 1 Impact (high if opinions are close to 0
        # or 1)
        extreme_proximity = extreme_coef * \
            min(min(vi, vj), min(1 - vi, 1 - vj))

        # Calculate Influence Impact (high if influence is high)
        influence = 0
        label_sum = size_i + size_j

        if edge_influence is not None:  # if an edge exists between the nodes
            influence += (edge_influence * size_j) / label_sum

        if reverse_edge_influence is not None:  # if a reverse edge exists between the nodes
            influence += (reverse_edge_influence * size_i) / label_sum

        influence *= influence_coef  # Apply Influence Coefficient

        # Calculate Size Impact (high if size is low)
        size_impact = size_coef * \
            (1 / (1 + size_i) + 1 / (1 + size_j))

        # Combine the impacts
        return opinion_proximity + extreme_proximity + influence + size_impact

    @staticmethod
    def default_node_parameter_gatherer(nodes):
        """
        Gathers default parameters for a node that is a result of merging given nodes.

        :param nodes: List[Dict], a list of node dictionaries, each containing node attributes.
        :return: Dict, a dictionary with calculated 'inner_opinions', 'cluster_size', and 'label'.
        """
        if not nodes:
            raise ValueError("The input list of nodes must not be empty.")

        size = sum(node.get('cluster_size', len(
            node.get('label', [0, ]))) for node in nodes)

        # Gather all opinions of the nodes being merged using node labels/identifiers as keys
        inner_opinions = {}

        for node in nodes:
            node_label = node.get('label', None)
            if node_label is not None:
                # Check if node has 'inner_opinions', if not, create one
                if 'inner_opinions' in node:
                    inner_opinions.update(node['inner_opinions'])
                else:
                    if len(node_label) != 1:
                        warnings.warn(
                            f"The length of the label in the node is higher than one. Assuming that all opinions in this cluster were equal. This is not typical behavior, check that that it corresponds to your intention. Found in node: {node_label}")
                    for i in node_label:
                        inner_opinions[i] = node['opinion']

        return {
            'cluster_size': size,
            'opinion': sum(node.get('cluster_size', len(node.get('label', [0, ]))) * node['opinion'] for node in nodes) / size,
            'label': [id for node in nodes for id in node['label']],
            'inner_opinions': inner_opinions
        }

    @staticmethod
    def min_max_node_parameter_gatherer(nodes):
        """
        Gathers default parameters for a node that is a result of merging given nodes.

        :param nodes: List[Dict], a list of node dictionaries, each containing node attributes.
        :return: Dict, a dictionary with calculated 'max_opinion' and 'min_opinion'.
        """
        if not nodes:
            raise ValueError("The input list of nodes must not be empty.")

        size = sum(node.get('cluster_size', len(
            node.get('label', [0, ]))) for node in nodes)

        return {
            'cluster_size': size,
            'opinion': sum(node.get('cluster_size', len(node.get('label', [0, ]))) * node['opinion'] for node in nodes) / size,
            'label': [id for node in nodes for id in node['label']],
            'max_opinion': max(node.get('max_opinion', node['opinion']) for node in nodes),
            'min_opinion': min(node.get('min_opinion', node['opinion']) for node in nodes)
        }

    def compute_similarity(self, i, j, similarity_function=None):
        """
        Computes the similarity between two nodes in the graph.

        Parameters:
            i (int): The identifier for the first node.
            j (int): The identifier for the second node.
            similarity_function (callable, optional):
                A custom similarity function to be used for this computation.
                If None, the instance's similarity_function is used.
                Default is None.

        Returns:
            float: The computed similarity opinion between nodes i and j.
        """
        # Check if there is an edge between nodes i and j
        if not self.has_edge(i, j) and not self.has_edge(j, i):
            return -2

        # Extract parameters from node i, node j, and the edge (if exists)
        vi = self.nodes[i]['opinion']
        vj = self.nodes[j]['opinion']

        size_i = len(self.nodes[i].get('label', [i]))
        size_j = len(self.nodes[j].get('label', [j]))

        edge_influence = self[i][j]['influence'] if self.has_edge(
            i, j) else None
        reverse_edge_influence = self[j][i]['influence'] if self.has_edge(
            j, i) else None

        # Choose the correct similarity function and calculate the similarity
        func = similarity_function or self.similarity_function
        return func(vi, vj, size_i, size_j,
                    edge_influence, reverse_edge_influence)

    def position_nodes(self, seed=None):
        """
        Determines the positions of the nodes in the graph using the spring layout algorithm.

        :param seed: int, optional
            Seed for the spring layout algorithm, affecting the randomness in the positioning of the nodes.
            If None, the positioning of the nodes will be determined by the underlying algorithm's default behavior.
            Default is None.

        :return: dict
            A dictionary representing the positions of the nodes in a 2D space, where the keys are node IDs
            and the opinions are the corresponding (x, y) coordinates.
        """
        return nx.spring_layout(self, seed=seed)

    # def get_opinion_neighbor_mean_opinion_pairs(self):
    #     # Extract opinion values for all nodes
    #     opinions = nx.get_node_attributes(self, 'opinion')

    #     x_values = []  # Node's opinion
    #     y_values = []  # Mean opinion of neighbors

    #     for node in self.nodes():
    #         node_opinion = opinions[node]
    #         neighbors = list(self.neighbors(node))
    #         if neighbors:  # Ensure the node has neighbors
    #             mean_neighbor_opinion = sum(
    #                 opinions[neighbor] for neighbor in neighbors) / len(neighbors)
    #             x_values.append(node_opinion)
    #             y_values.append(mean_neighbor_opinion)
    #     return x_values, y_values

    # def get_opinion_neighbor_mean_opinion_pairs_dict(self):
    #     # Extract opinion values for all nodes
    #     opinions = nx.get_node_attributes(self, 'opinion')

    #     mean_neighbor_opinion_dict = {}

    #     for node in self.nodes():
    #         node_opinion = opinions[node]
    #         neighbors = list(self.neighbors(node))
    #         if neighbors:  # Ensure the node has neighbors
    #             mean_neighbor_opinion = sum(
    #                 opinions[neighbor] for neighbor in neighbors) / len(neighbors)
    #             mean_neighbor_opinion_dict[node] = (
    #                 node_opinion, mean_neighbor_opinion)
    #     return mean_neighbor_opinion_dict

    @property
    def opinions(self):
        """
        Returns a dictionary with the opinions of the nodes.
        Key is the node ID, and opinion is the opinion of the node.
        """
        return {node: self.nodes[node]["opinion"] for node in self.nodes}

    @opinions.setter
    def opinions(self, values):
        """
        Sets the opinions of the nodes based on the provided value(s).
        Value(s) can be a single float, a list of floats, or a dictionary with node IDs as keys.
        """
        if isinstance(values, (int, float)):  # Single value provided
            for node in self.nodes:
                self.nodes[node]["opinion"] = values

        elif isinstance(values, list):  # List of values provided
            if len(values) != len(self.nodes):
                raise ValueError(
                    "Length of provided list does not match the number of nodes in the graph.")
            for node, opinion in zip(self.nodes, values):
                self.nodes[node]["opinion"] = opinion

        elif isinstance(values, dict):  # Dictionary provided
            for node, opinion in values.items():
                if node in self.nodes:
                    self.nodes[node]["opinion"] = opinion
                else:
                    raise ValueError(
                        f"Node {node} does not exist in the graph.")

        else:
            raise TypeError(
                "Invalid type for opinions. Expected int, float, list, or dict.")

    def __str__(self):
        return f"<HariGraph with {self.number_of_nodes()} nodes and {self.number_of_edges()} edges>"

    def __repr__(self):
        return f"<HariGraph object at {id(self)}: {self.number_of_nodes()} nodes, {self.number_of_edges()} edges>"
