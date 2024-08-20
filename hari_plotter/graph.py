from __future__ import annotations

import json
import os
import random
from itertools import combinations, permutations
from typing import Optional, Type, Union

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

from .distributions import generate_mixture_of_gaussians
from .node_gatherer import (ActivityDefaultNodeEdgeGatherer,
                            DefaultNodeEdgeGatherer, NodeEdgeGatherer)


class Graph(nx.DiGraph):
    """
    HariGraph extends the NetworkX DiGraph class to provide additional functionalities specific to complex network
    analysis and manipulation. It includes methods for graph parameterization, node merging, influence assignment,
    and serialization/deserialization to/from JSON format.

    Attributes:
        gatherer (NodeEdgeGatherer): An instance of a NodeEdgeGatherer subclass responsible for collecting and
                                    applying node and edge parameters based on defined criteria.
    """

    def __init__(self, incoming_graph_data=None, **attr):
        """
        Initializes the HariGraph instance by extending the NetworkX DiGraph constructor.

        Parameters:
            incoming_graph_data: Input graph data to initialize the graph (default is None).
            attr: Additional attributes to add to the graph.
        """
        super().__init__(incoming_graph_data, **attr)
        # Default gatherer for node and edge parameters
        self.gatherer = DefaultNodeEdgeGatherer(self)

    @property
    def node_parameters(self):
        return self.gatherer.node_parameters

    def set_gatherer(self, new_gatherer: Type[NodeEdgeGatherer]) -> None:
        """
        Sets a new gatherer for collecting and applying node and edge parameters.

        Parameters:
            new_gatherer (Type[NodeEdgeGatherer]): The new gatherer class to be used.
        """
        self.gatherer = new_gatherer(self)

    def add_parameters_to_nodes(self, nodes: Optional[list[tuple[int]]] = None) -> None:
        """
        Adds or updates parameters for the specified nodes based on the current gatherer's criteria. If no nodes
        are specified, parameters are added or updated for all nodes in the graph.

        Parameters:
            nodes (Optional[list[tuple[int]]]): list of node identifiers to update. If None, updates all nodes.
        """
        nodes = nodes or list(self.nodes)
        parameters = self.gatherer.gather_everything()

        for key, value in parameters.items():
            for node in nodes:
                self.nodes[node][key] = value[node]

    @classmethod
    def merge(cls, graphs: list['Graph']) -> 'Graph':
        """
        Merges multiple Graph instances into a single Graph instance.
        Nodes with the same ID across different graphs will be renamed 
        to keep them distinct.

        Parameters:
            graphs (list['Graph']): A list of Graph instances to merge.

        Returns:
            'Graph': A new Graph instance representing the merged graph.
        """
        merged_graph = cls()

        for i, graph in enumerate(graphs):
            # Create a mapping to rename nodes by adding a unique prefix based on graph index
            mapping = {node: tuple([(n, i) for n in node])
                       for node in graph.nodes()}
            renamed_graph = nx.relabel_nodes(graph if isinstance(graph, Graph) else graph.get_graph(
            ), mapping)  # different behavior for graphs and LazyGraphs
            merged_graph = nx.compose(merged_graph, renamed_graph)

        merged_graph.set_gatherer(type(graphs[0].gatherer))

        return merged_graph

    def has_self_loops(self) -> bool:
        """
        Checks if the graph contains any self-loops (edges that connect a node to itself).

        Returns:
            bool: True if there is at least one self-loop in the graph, otherwise False.
        """
        return any(self.has_edge(node, node) for node in self.nodes)

    def remove_self_loops(self) -> None:
        """
        Removes all self-loops from the graph.
        """
        for node in list(self.nodes):
            if self.has_edge(node, node):
                self.remove_edge(node, node)

    def assign_parameter(self, parameter: str, method: None | dict = None) -> None:
        """
        Assigns random opinions to all nodes in the graph.

        Parameters:

            parameter (str): The parameter to be assigned random values.
        """

        if isinstance(method, dict):
            if 'distribution' in method:
                distribution = method['distribution']
                if not distribution:
                    values = np.random.rand(len(self.nodes))
                elif isinstance(distribution, dict):
                    if distribution['type'] == 'uniform':
                        values = np.random.uniform(
                            distribution.get('low', 0.0), distribution.get('high', 1.0), len(self.nodes))
                    elif distribution['type'] == 'power_law':
                        # Calculate power law distribution values
                        epsilon = distribution.get('epsilon', 0.01)
                        gamma = distribution.get('gamma', 2.1)
                        values = np.random.uniform(epsilon, 1, len(self.nodes))
                        values = (values ** (-1 / (gamma - 1)))
                    elif distribution['type'] == 'constant':
                        values = [method['constant']] * len(self.nodes)
                    else:
                        raise ValueError(
                            f'Unknown distribution type: {type(distribution)}')
                else:
                    raise ValueError(
                        f'Invalid distribution type: {type(distribution)}')

            else:
                raise ValueError('Invalid method type')
        else:
            raise ValueError('Invalid method type')

        for node, value in zip(self.nodes, values):
            self.nodes[node][parameter] = value

    def assign_random_influences(self, mean_influence: float, influence_range: float, seed: Optional[int] = None) -> None:
        """
        Assigns random influence values to all edges within a specified range centered around a mean influence value.

        Parameters:
            mean_influence (float): The mean value around which the influence values are centered.
            influence_range (float): The range within which the random influence values will vary.
            seed (Optional[int]): An optional seed for the random number generator for reproducibility (default is None).
        """
        if seed is not None:
            random.seed(seed)

        lower_bound, upper_bound = mean_influence - \
            influence_range / 2, mean_influence + influence_range / 2

        for u, v in self.edges:
            self.edges[u, v]['Influence'] = random.uniform(
                lower_bound, upper_bound)

    def is_degroot_converging(self, tolerance: float = 1e-2) -> bool:
        """
        Checks if the graph's influence structure adheres to the Degroot convergence criteria, i.e., the total
        incoming influence for each node is within a tolerance of 1.

        Parameters:
            tolerance (float): The tolerance within which the total influence must fall to be considered converging.

        Returns:
            bool: True if the graph meets the Degroot convergence criteria, otherwise False.
        """
        return all(1 - tolerance <= sum(self.edges[predecessor, node]['Influence'] for predecessor in self.predecessors(node)) <= 1 + tolerance for node in self.nodes)

    def make_degroot_converging(self, seed: Optional[int] = None) -> None:
        """
        Adjusts the influence values on incoming edges for each node to ensure the graph meets the Degroot convergence
        criteria, i.e., the total incoming influence for each node equals 1.

        Parameters:
            seed (Optional[int]): An optional seed for the random number generator for reproducibility (default is None).
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        for node in self.nodes:
            incoming_edges = [(neighbor, node)
                              for neighbor in self.predecessors(node)]
            total_influence = sum(
                self.edges[edge]['Influence'] for edge in incoming_edges)

            if total_influence == 0:
                for u, v in incoming_edges:
                    self.edges[u, v]['Influence'] = random.random()
                total_influence = sum(
                    self.edges[edge]['Influence'] for edge in incoming_edges)

            for u, v in incoming_edges:
                self.edges[u, v]['Influence'] /= total_influence

    def mean_graph(self, images: list['Graph']) -> 'Graph':
        """
        Calculates the mean graph from a list of HariGraph instances. The mean graph's nodes and edges have attributes
        that are the average of the corresponding attributes in the input graphs.

        Parameters:
            images (list['HariGraph']): A list of HariGraph instances from which to calculate the mean graph.

        Returns:
            'HariGraph': A new HariGraph instance representing the mean of the input graphs.
        """
        return self.gatherer.mean_graph(images)

    @classmethod
    def from_harigraph(cls, harigraph: 'Graph') -> 'Graph':
        """
        Returns the given graph. Created for the sake of consistency with the other methods.
        """
        return harigraph

    @classmethod
    def read_network(cls, network_file: str, opinion_file: str, gatherer: NodeEdgeGatherer | None = None, number_of_bots: int = 0) -> 'Graph':
        """
        Reads a graph structure and node attributes from separate files and initializes a HariGraph instance with this data.

        Parameters:
            network_file (str): Path to the file containing the network's topology, specifying nodes and their connections.
            opinion_file (str): Path to the file containing node attributes, specifically their opinions and optionally activities.
            gatherer (NodeEdgeGatherer | None): type of gatherer to be used
            number_of_bots (int): number of bots. First number_of_bots lines will be interpreted as bots

        Returns:
            HariGraph: An instance of HariGraph populated with nodes, edges, and node attributes based on the provided files.
        """

        G = cls()  # Instantiate a new HariGraph object

        # Open and read the network file. Check if node IDs are represented as tuples (indicated by '&').
        with open(network_file, 'r') as f:
            content = f.read()
            # Determine if node IDs include '&', implying tuple representation.
            has_tuples = '&' in content

        # Based on the presence of '&', define a function to parse node IDs appropriately.
        if has_tuples:
            def parse_node_id(node_id_str):
                # Parse node ID string into a tuple if '&' is present, otherwise keep it as a single integer.
                return tuple(map(int, node_id_str.split('&'))) if '&' in node_id_str else (int(node_id_str),)
        else:
            def parse_node_id(node_id_str):
                # If no '&', node IDs are single integers wrapped in a tuple for consistency.
                return (int(node_id_str),)

        # Process each line in the network file to construct the graph's structure.
        with open(network_file, 'r') as f:
            next(f)  # Skip the header line
            for line in f:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                parts = line.split(',')
                # Extract and parse the node ID
                idx_agent = parse_node_id(parts[0].strip())
                # Number of neighbors for this node
                n_neighbors = int(parts[1])

                # For each neighbor, parse the neighbor's ID and the associated weight (influence).
                for i in range(n_neighbors):
                    neighbor_id = parse_node_id(parts[2 + i].strip())
                    weight = float(parts[2 + n_neighbors + i])
                    # Add an edge with the influence attribute
                    G.add_edge(idx_agent, neighbor_id, Influence=weight)

        # Initialize a flag to track the presence of 'Activity' data in the opinion file.
        has_activity = False

        # Open and process the opinion file to set node attributes.
        with open(opinion_file, 'r') as f:
            next(f)  # Skip the header line
            for line in f:
                parts = line.split(',')
                # Extract and parse the node ID
                idx_agent = parse_node_id(parts[0].strip())
                opinion = float(parts[1])  # Extract the node's opinion

                if not G.has_node(idx_agent):
                    # Add the node if it doesn't already exist in the graph
                    G.add_node(idx_agent)

                # Set the 'Opinion' attribute for the node
                G.nodes[idx_agent]['Opinion'] = opinion
                G.nodes[idx_agent]['Type'] = 'Bot' if idx_agent[0] < number_of_bots else ''

                # If an additional column is present, it represents the node's 'Activity' level.
                if len(parts) > 2:
                    has_activity = True
                    activity = float(parts[2])
                    # Set the 'Activity' attribute for the node
                    G.nodes[idx_agent]['Activity'] = activity

        if gatherer:
            G.set_gatherer(gatherer)
        elif has_activity:
            # If any node's 'Activity' level was provided, use the ActivityDefaultNodeEdgeGatherer for this graph.
            G.set_gatherer(ActivityDefaultNodeEdgeGatherer)

        return G  # Return the constructed HariGraph instance

    def write_network(self, network_file: str, opinion_file: str, delimiter=','):
        """
        Writes the current graph's network structure and node attributes to specified files, using integers to represent node IDs based on a sorted mapping. The output is saved in the sorted order of these integer IDs.

        Parameters:
            network_file (str): Path to the file where the network structure will be saved.
            opinion_file (str): Path to the file where the node attributes, particularly opinions, will be saved.
            delimiter (str): The delimiter used to separate values in the output files.
        """

        # Gather opinions for all nodes using the current gatherer
        opinions = self.gatherer.gather('Opinion')
        activities = self.gatherer.gather(
            'Activity') if 'Activity' in self.gatherer.node_parameters else None

        # Create a mapping from node IDs to integers
        node_mapping = {node: idx for idx,
                        node in enumerate(sorted(self.nodes))}

        # Write the network structure to the specified network file
        with open(network_file, 'w') as f:
            # Write the header line
            f.write(
                f"# idx_agent{delimiter}n_neighbors_in{delimiter}indices_neighbors_in[...]{delimiter}weights_in[...]\n")

            # Iterate over the node mapping in sorted order of integer IDs
            for node, node_idx in sorted(node_mapping.items(), key=lambda x: x[1]):
                # list of nodes influencing the current node
                neighbors = list(self.predecessors(node))
                # list of influence weights from each neighbor
                weights = [self[neighbor][node]['Influence']
                           for neighbor in neighbors]
                # Use the mapping to get integer IDs for the neighbors
                neighbors_idxs = [node_mapping[neighbor]
                                  for neighbor in neighbors]

                # Write the node data line with all necessary information
                if len(neighbors_idxs) == 0:
                    f.write(f"{node_idx}{delimiter}0\n")
                else:
                    f.write(
                        f"{node_idx}{delimiter}{len(neighbors_idxs)}{delimiter}{delimiter.join(map(str, neighbors_idxs))}{delimiter}{delimiter.join(map(str, weights))}\n")

        # Write the node opinions to the specified opinion file
        with open(opinion_file, 'w') as f:
            # Write the header line
            f.write(f"# idx_agent{delimiter}opinion[...]\n")

            # Iterate over the node mapping in sorted order of integer IDs for opinions
            for node, node_idx in sorted(node_mapping.items(), key=lambda x: x[1]):
                if node in opinions['Nodes']:
                    # Retrieve the opinion for the current node ID using the mapping
                    opinion = opinions['Opinion'][opinions['Nodes'].index(
                        node)]
                    if activities is not None and node in activities['Nodes']:
                        activity = activities['Activity'][activities['Nodes'].index(
                            node)]
                        # Write the node opinion line with activity if available
                        f.write(
                            f"{node_idx}{delimiter}{opinion}{delimiter}{activity}\n")
                    else:
                        # Write the node opinion line without activity
                        f.write(f"{node_idx}{delimiter}{opinion}\n")

    @classmethod
    def read_json(cls, filename: str, gatherer: NodeEdgeGatherer | None = None) -> Graph:
        """
        Reads a HariGraph instance from a JSON file that contains both the graph's structure and node attributes.

        Parameters:
            filename (str): Path to the JSON file from which the graph is to be loaded.
            gatherer (NodeEdgeGatherer | None): type of gatherer to be used

        Returns:
            HariGraph: A new HariGraph instance constructed based on the data contained in the JSON file. This method reconstructs the graph's nodes, edges, and associated attributes like opinions and influences.
        """

        with open(filename, 'r') as f:
            data = json.load(f)

        # Convert the JSON data back into a NetworkX graph
        G = json_graph.node_link_graph(data)

        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} does not exist.")

        with open(filename, 'r') as file:
            graph_dict = json.load(file)

        G = json_graph.node_link_graph(data)
        G = cls(G)
        if gatherer:
            G.set_gatherer(gatherer)

        return G

    def write_json(self, filename: str):
        """
        Serializes the graph to a JSON file, including its structure (nodes and edges) and attributes (e.g., opinions).

        Parameters:
            filename (str): The file path where the graph will be saved in JSON format.

        This method creates a JSON representation of the graph, which includes detailed information about nodes and edges along with their attributes, making it suitable for storage, sharing, or further analysis.
        """
        data = json_graph.node_link_data(self)
        # Write the JSON data to a file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def guaranteed_connected(cls, n: int) -> Graph:
        """
        Creates a guaranteed connected HariGraph instance with n nodes.

        :param n (int): Number of nodes.
        :return: A new HariGraph instance.
        """
        if n < 2:
            raise ValueError("Number of nodes should be at least 2")

        G = cls()
        for i in range(n):
            G.add_node((i,))
            G.nodes[(i,)]['Opinion'] = random.random()

        nodes = list(G.nodes)
        random.shuffle(nodes)
        for i in range(n - 1):
            G.add_edge(nodes[i], nodes[i + 1])
            G.edges[nodes[i], nodes[i + 1]]['Influence'] = random.random()

        additional_edges = random.randint(1, n)
        for _ in range(additional_edges):
            u, v = random.sample(G.nodes, 2)
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)
                G.edges[u, v]['Influence'] = random.random()
                if random.choice([True, False]) and not G.has_edge(v, u):
                    G.add_edge(v, u)
                    G.edges[v, u]['Influence'] = random.random()

        G.add_parameters_to_nodes()

        return G

    @classmethod
    def unconnected(cls, n: int) -> Graph:
        """
        Creates a HariGraph instance with n nodes and no edges.

        :param n (int): Number of nodes.
        :return: A new HariGraph instance.
        """
        if n < 2:
            raise ValueError("Number of nodes should be at least 2")

        G = cls()
        for i in range(n):
            G.add_node((i,))
            G.nodes[(i,)]['Opinion'] = random.random()

        G.add_parameters_to_nodes()

        return G

    @classmethod
    def by_deletion(cls, n: int, factor: float) -> Graph:
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
            G.add_node((i,))
            G.nodes[(i,)]['Opinion'] = random.random()
        for i in range(n):
            for j in range(n):
                if i != j:
                    G.add_edge((i,), (j,))
                    G.edges[(i,), (j,)]['Influence'] = random.random()

        edges_to_remove = random.sample(
            G.edges, int(len(G.edges) * (1 - factor)))
        G.remove_edges_from(edges_to_remove)

        G.add_parameters_to_nodes()

        return G

    @classmethod
    def strongly_connected_components(
            cls, cluster_sizes: list[int], inter_cluster_edges: int, mean_opinion: float = 0.5, seed: int = None) -> Graph:
        """
        Creates a HariGraph instance with multiple strongly connected components.

        :param cluster_sizes: list[int], sizes of the clusters.
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
                G.add_node((i,))
                G.nodes[(i,)]['Opinion'] = opinions[opinion_idx]
                opinion_idx += 1
            for i in range(start, start + size):
                for j in range(i + 1, start + size):
                    G.add_edge((i,), (j,))
                    G.add_edge((j,), (i,))
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
                G.add_edge((source_node,), (target_node,))

        # Assign random influences to the edges of the graph
        G.assign_random_influences(
            mean_influence=0.1, influence_range=0.1, seed=seed)

        G.add_parameters_to_nodes()

        return G

    def copy(self) -> Graph:
        G_copy = super().copy(as_view=False)
        G_copy = Graph(G_copy) if not isinstance(
            G_copy, Graph) else G_copy
        G_copy.set_gatherer(type(self.gatherer))
        return G_copy

    # ---- Dynamics example ----

    def dynamics_example_step(self, t: float):
        """
        Updates the opinion of each node in the HariGraph instance based on the opinions of its predecessors.

        :param t: The time step factor influencing the dynamics.
        """
        updated_opinions = {}  # Temporary dictionary to store updated opinions

        for i in self.nodes:
            vi = self.nodes[i]['Opinion']

            # Predecessors of a node are the start nodes of its incoming edges.
            for j in self.predecessors(i):
                pij = self[j][i]['Influence']
                vj = self.nodes[j]['Opinion']
                vi += pij * vj * t  # Calculate updated opinion based on each incoming edge

            # Clip the updated opinion to [0, 1]
            # vi = max(0, min(vi, 1))

            updated_opinions[i] = vi

        # Update the opinions in the graph with the calculated updated opinions
        for i, vi in updated_opinions.items():
            self.nodes[i]['Opinion'] = vi

    # ---- Merge Methods ----

    def get_cluster_mapping(self) -> list[list[tuple[int]]]:
        """
        Generates a list of nodes in the unclustered graph to be clustered to get the current graph
        :return: A list representing the current clusters in the graph.
        """
        return sorted([sorted([(element,) for element in node]) for node in self.nodes])

    def merge_nodes(self, i: tuple[int], j: tuple[int]):
        """
        Merges two nodes in the graph into a new node.

        The new node's opinion is a weighted average of the opinions of
        the merged nodes, and its name is the concatenation of the names
        of the merged nodes. The edges are reconnected to the new node,
        and the old nodes are removed.

        Parameters:
            i tuple[int]: The identifier for the first node to merge.
            j tuple[int]: The identifier for the second node to merge.
        """
        self.gatherer.merge_nodes(i, j)

    def merge_clusters(self, clusters: list[list[tuple[int]]], labels: Union[list[str], None] = None, merge_remaining=False):
        """
        Merges clusters of nodes in the graph into new nodes. Optionally merges the remaining nodes into an additional cluster.

        Parameters:
            clusters (Union[list[Set[int]], dict[int, int]]): A list where each element is a set containing
                                    the IDs of the nodes in a cluster to be merged or a dictionary mapping old node IDs
                                    to new node IDs.
            merge_remaining (bool): If True, merge the nodes not included in clusters into an additional cluster. Default is False.
        """
        self.gatherer.merge_clusters(clusters, labels, merge_remaining)

    # ---- Clusterization Methods ----

    def find_clusters(self, max_opinion_difference: float = 0.1, min_influence: float = 0.1):
        """
        Finds clusters of nodes in the graph where the difference in the nodes' opinions
        is less than max_opinion_difference, and the influence of i on j is higher than
        min_influence * size(i).

        Parameters:
            max_opinion_difference (float): Maximum allowed difference in the opinions of nodes to form a cluster.
            min_influence (float): Minimum required influence to form a cluster, adjusted by the size of the node.

        Returns:
            list[list[int]]: A list of lists, where each inner list represents a cluster of node identifiers.
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

                    vi = self.nodes[node]['Opinion']
                    vj = self.nodes[neighbor]['Opinion']
                    size_i = self.nodes[node].get('cluster_size', len(node))

                    if self.has_edge(node, neighbor):
                        influence_ij = self[node][neighbor]['Influence']
                    else:
                        influence_ij = 0

                    if self.has_edge(neighbor, node):
                        influence_ji = self[neighbor][node]['Influence']
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

    def merge_by_intervals(self, intervals: list[float]):
        """
        Merges nodes into clusters based on the intervals defined by the input list of opinions.

        Parameters:
            intervals (list[float]): A sorted list of opinions representing the boundaries of the intervals.
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
                if lower_bound <= data['Opinion'] < upper_bound:
                    cluster.append(node)
            if len(cluster) > 0:
                clusters.append(cluster)

        # Merge the clusters
        if clusters:
            self.merge_clusters(clusters)

    # ---- Data Processing Methods ----

    def check_all_paths_exist(self) -> bool:
        """
        Checks if there exists a path between every pair of nodes in the HariGraph instance.

        :return: True if a path exists between every pair of nodes, False otherwise.
        """
        for source, target in permutations(self.nodes, 2):
            if not nx.has_path(self, source, target):
                # f"No path exists from {source} to {target}"
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
            opinion = self.nodes[node]['Opinion']

            # If label is defined, the weight is the length of the label.
            # If not defined, the weight is assumed to be 1.
            weight = len(node)

            total_opinion += opinion * weight
            total_weight += weight

        if total_weight == 0:  # Prevent division by zero
            return 0

        return total_opinion / total_weight

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

    def get_graph(self) -> Graph:
        '''Self call for union formatting with LazyHariGraph'''
        return self

    @property
    def opinions(self):
        """
        Returns a dictionary with the opinions of the nodes.
        Key is the node ID, and opinion is the opinion of the node.
        """
        return {node: self.nodes[node]['Opinion'] for node in self.nodes}

    @opinions.setter
    def opinions(self, values: Union[int, float, dict[tuple[int]:float]]):
        if isinstance(values, (int, float)):
            for node in self.nodes:
                self.nodes[node]['Opinion'] = values
        elif isinstance(values, dict):
            for node, opinion in values.items():
                if node in self.nodes:
                    self.nodes[node]['Opinion'] = opinion
                else:
                    raise ValueError(
                        f"Node {node} does not exist in the graph.")
        else:
            raise TypeError(
                f'Values input type for the opinions {type(values)} is not supported')

    def __str__(self):
        return f"<HariGraph with {self.number_of_nodes()} nodes and {self.number_of_edges()} edges>"

    def __repr__(self):
        return f"<HariGraph object at {id(self)}: {self.number_of_nodes()} nodes, {self.number_of_edges()} edges>"
