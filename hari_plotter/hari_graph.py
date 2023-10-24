import json
import math
import os
import random
import re
import warnings
from collections import Counter
from itertools import combinations, permutations

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .distibutions import generate_mixture_of_gaussians


class HariGraph(nx.DiGraph):
    """
    HariGraph extends the DiGraph class of Networkx to offer additional functionality.
    It ensures that each node has a label and provides methods to create, save, and load graphs.
    """

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.generate_labels()
        self.similarity_function = self.default_similarity_function
        self.node_parameter_gatherer = self.default_node_parameter_gatherer

    def generate_labels(self):
        """Generates labels for the nodes if they don't exist."""
        if not self.nodes:
            return
        if 'label' not in self.nodes[next(iter(self.nodes))]:
            for i in self.nodes:
                self.nodes[i]['label'] = [i]

    def add_parameters_to_nodes(self, node_ids=None):
        """
        Adds parameters to specified nodes using the node_parameter_gatherer method.
        If node_ids is None, parameters are added to all nodes in the graph.

        :param node_ids: List[int], a list of identifiers for the nodes to which parameters will be added.
        """
        if node_ids is None:
            node_ids = list(self.nodes)

        for node_id in node_ids:
            # Gather parameters for the node
            parameters = self.node_parameter_gatherer([self.nodes[node_id]])
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

    def assign_random_influences(self, mean_influence, influence_range, seed=None):
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

    def is_degroot_converging(self, tolerance=1e-2):
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

            # Calculate the total influence of incoming edges for the current node
            total_influence = sum(self[u][v]['influence']
                                  for u, v in incoming_edges)

            # If the total influence is zero, assign random influences to incoming edges
            if total_influence == 0:
                for u, v in incoming_edges:
                    self[u][v]['influence'] = random.random()
                total_influence = sum(self[u][v]['influence']
                                      for u, v in incoming_edges)

            # Adjust the influences proportionally
            for u, v in incoming_edges:
                self[u][v]['influence'] /= total_influence

    @classmethod
    def read_network(cls, network_file, opinion_file):
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
                n_neighbours = int(parts[1])
                indices_neighbours = map(int, parts[2:2+n_neighbours])
                weights = map(float, parts[2+n_neighbours:])

                # Add nodes with initial opinion 0, opinion will be updated from opinion_file
                G.add_node(idx_agent, opinion=0)

                # Add edges with weights
                for neighbour, weight in zip(indices_neighbours, weights):
                    G.add_edge(neighbour, idx_agent, influence=weight)

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

        # G.remove_self_loops()
        G.generate_labels()
        G.add_parameters_to_nodes()

        return G

    def write_network(self, network_file, opinion_file, delimiter=','):
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
                f"# idx_agent{delimiter}n_neighbours_in{delimiter}indices_neighbours_in[...]{delimiter}weights_in[...]\n")
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
    def read_json(cls, filename):
        """
        Reads a HariGraph from a JSON file.

        :param filename: The name of the file to read from.
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

    def write_json(self, filename):
        """
        Saves the HariGraph to a JSON file.

        :param filename: The name of the file to write to.
        """
        graph_dict = {
            "nodes": [
                {"id": n, **{prop: self.node_values[prop][n] for prop in self.node_values}} for n in self.nodes()
            ],
            "edges": [{"source": u, "target": v, "influence": self[u][v]["influence"]} for u, v in self.edges()]
        }

        with open(filename, 'w') as file:
            json.dump(graph_dict, file)

    @classmethod
    def guaranteed_connected(cls, n):
        """
        Creates a guaranteed connected HariGraph instance with n nodes.

        :param n: Number of nodes.
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
    def by_deletion(cls, n, factor):
        """
        Creates a HariGraph instance by deleting some of the edges of a fully connected graph.

        :param n: Number of nodes.
        :param factor: Factor representing how many edges to keep.
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
    def strongly_connected_components(cls, cluster_sizes, inter_cluster_edges, mean_opinion=0.5, seed=None):
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

    def copy(self):
        G_copy = super().copy(as_view=False)
        G_copy = HariGraph(G_copy) if not isinstance(
            G_copy, HariGraph) else G_copy
        G_copy.similarity_function = self.similarity_function
        return G_copy

    def check_all_paths_exist(self):
        """
        Checks if there exists a path between every pair of nodes in the HariGraph instance.

        :return: True if a path exists between every pair of nodes, False otherwise.
        """
        for source, target in permutations(self.nodes, 2):
            if not nx.has_path(self, source, target):
                # print(f"No path exists from {source} to {target}")
                return False
        return True

    def dynamics_step(self, t):
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

    @property
    def mean_opinion(self):
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
    def default_similarity_function(vi, vj, size_i, size_j, edge_influence, reverse_edge_influence,
                                    opinion_coef=1., extreme_coef=0.1, influence_coef=1., size_coef=10.):
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

        # Calculate Proximity to 0 or 1 Impact (high if opinions are close to 0 or 1)
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
        :return: Dict, a dictionary with calculated 'max_opinion' and 'min_opinion'.
        """
        if not nodes:
            raise ValueError("The input list of nodes must not be empty.")

        size = sum(node.get('size', len(
            node.get('label', [0, ]))) for node in nodes)

        return {
            'size': size,
            'opinion': sum(node.get('size', len(node.get('label', [0, ]))) * node['opinion'] for node in nodes) / size,
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
        return func(vi, vj, size_i, size_j, edge_influence, reverse_edge_influence)

    def merge_nodes(self, i, j):
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

        # Get opinions and others paramenters using the 'node_parameter_gatherer' method
        parameters = self.node_parameter_gatherer(
            [self.nodes[i], self.nodes[j]])

        # Add a new node to the graph with the calculated opinion, label, and other unpacked parameters
        new_node = max(max(self.nodes), max(
            item for sublist in self.labels for item in sublist)) + 1
        self.add_node(new_node, **parameters)

        # Reconnect edges
        for u, v, data in list(self.edges(data=True)):
            if u == i or u == j:
                if v != i and v != j:  # Avoid connecting the new node to itself
                    influence = self[u][v]['influence']
                    if self.has_edge(new_node, v):
                        # Sum the influences if both original nodes were connected to the same node
                        self[new_node][v]['influence'] += influence
                    else:
                        self.add_edge(new_node, v, influence=influence)
                if self.has_edge(u, v):  # Check if the edge exists before removing it
                    self.remove_edge(u, v)
            if v == i or v == j:
                if u != i and u != j:  # Avoid connecting the new node to itself
                    influence = self[u][v]['influence']
                    if self.has_edge(u, new_node):
                        # Sum the influences if both original nodes were connected to the same node
                        self[u][new_node]['influence'] += influence
                    else:
                        self.add_edge(u, new_node, influence=influence)
                if self.has_edge(u, v):  # Check if the edge exists before removing it
                    self.remove_edge(u, v)

        # Remove the old nodes
        self.remove_node(i)
        self.remove_node(j)

    def find_clusters(self, max_opinion_difference=0.1, min_influence=0.1):
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

                for neighbor in set(self.successors(node)).union(self.predecessors(node)):
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

    def merge_by_intervals(self, intervals):
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
                print(f'{cluster = }')
                clusters.append(set(cluster))

        # Merge the clusters
        if clusters:
            self.merge_clusters(clusters)

    def get_cluster_mapping(self):
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

    def merge_clusters(self, clusters):
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
                item for sublist in self.labels for item in sublist)) + 1
            for i, cluster in enumerate(clusters):
                new_id = new_id_start + i
                for node_id in cluster:
                    assert node_id in self.nodes and node_id not in id_mapping, f"Node {node_id} already exists in the graph or is being merged multiple times."
                    id_mapping[node_id] = new_id
        else:
            raise ValueError(
                "clusters must be a list of sets or a dictionary.")

        for i, cluster in enumerate(clusters_list):
            new_id = id_mapping[next(iter(cluster))]
            parameters = self.node_parameter_gatherer(
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

    def draw(self, pos=None, node_info_mode='none', use_node_color=True,
             use_edge_thickness=True, show_edge_influences=False,
             node_size_multiplier=200,
             arrowhead_length=0.2, arrowhead_width=0.2,
             min_line_width=0.1, max_line_width=3.0,
             seed=None, save_filepath=None, show=True,
             fig=None, ax=None, bottom_right_text=None):
        """
        Visualizes the graph with various customization options.

        :param pos: dict, optional
            Position of nodes as a dictionary of coordinates. If not provided, the spring layout is used to position nodes.

        :param node_info_mode: str, optional
            Determines the information to display on the nodes.
            Options: 'none', 'opinions', 'ids', 'labels', 'size'. Default is 'none'.

        :param use_node_color: bool, optional
            If True, nodes are colored based on their opinions using a colormap. Default is True.

        :param use_edge_thickness: bool, optional
            If True, the thickness of the edges is determined by their influences, scaled between min_line_width and max_line_width. Default is True.

        :param show_edge_influences: bool, optional
            If True, displays the influences of the edges on the plot. Default is False.

        :param node_size_multiplier: int, optional
            Multiplier for node sizes, affecting the visualization scale. Default is 200.

        :param arrowhead_length: float, optional
            Length of the arrowhead for directed edges. Default is 0.2.

        :param arrowhead_width: float, optional
            Width of the arrowhead for directed edges. Default is 0.2.

        :param min_line_width: float, optional
            Minimum line width for edges. Default is 0.1.

        :param max_line_width: float, optional
            Maximum line width for edges. Default is 3.0.

        :param seed: int, optional
            Seed for the spring layout. Affects the randomness in the positioning of the nodes. Default is None.

        :param save_filepath: str, optional
            If provided, saves the plot to the specified filepath. Default is None.

        :param show: bool, optional
            If True, displays the plot immediately. Default is True.

        :param fig: matplotlib.figure.Figure, optional
            Matplotlib Figure object. If None, a new figure is created. Default is None.

        :param ax: matplotlib.axes._axes.Axes, optional
            Matplotlib Axes object. If None, a new axis is created. Default is None.

        :param bottom_right_text: str, optional
            Text to display in the bottom right corner of the plot. Default is None.

        :return: tuple
            A tuple containing the Matplotlib Figure and Axes objects.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 7))

        if pos is None:
            pos = self.position_nodes(seed=seed)

        # Get the node and edge attributes
        node_attributes = nx.get_node_attributes(self, 'opinion')
        edge_attributes = nx.get_edge_attributes(self, 'influence')

        # Prepare Node Labels
        node_labels = {}
        match node_info_mode:
            case 'opinions':
                node_labels = {node: f"{opinion:.2f}" for node,
                               opinion in node_attributes.items()}
            case 'ids':
                node_labels = {node: f"{node}" for node in self.nodes}
            case 'labels':
                for node in self.nodes:
                    label = self.nodes[node].get('label', None)
                    if label is not None:
                        node_labels[node] = ','.join(map(str, label))
                    else:  # If label is not defined, show id instead
                        node_labels[node] = str(node)
            case 'size':
                for node in self.nodes:
                    label_len = len(self.nodes[node].get('label', [node]))
                    node_labels[node] = str(label_len)

        # Prepare Node Colors
        if use_node_color:
            node_colors = [cm.bwr(opinion)
                           for opinion in node_attributes.values()]
        else:
            node_colors = 'lightblue'

        # Prepare Edge Widths
        if use_edge_thickness:

            # Gather edge weights
            edge_weights = list(edge_attributes.values())

            # Scale edge weights non-linearly
            # or np.log1p(edge_weights) for logarithmic scaling
            scaled_weights = np.sqrt(edge_weights)

            # Normalize scaled weights to range [min_line_width, max_line_width]
            max_scaled_weight = max(scaled_weights)
            min_scaled_weight = min(scaled_weights)

            edge_widths = [
                min_line_width + (max_line_width - min_line_width) * (weight -
                                                                      min_scaled_weight) / (max_scaled_weight - min_scaled_weight)
                for weight in scaled_weights
            ]

        else:
            # Default line width applied to all edges
            edge_widths = [1.0] * self.number_of_edges()

        # Prepare Edge Labels
        edge_labels = None
        if show_edge_influences:
            edge_labels = {(u, v): f"{influence:.2f}" for (u, v),
                           influence in edge_attributes.items()}

        # Calculate Node Sizes
        node_sizes = []
        for node in self.nodes:
            label_len = len(self.nodes[node].get('label', [node]))
            size = node_size_multiplier * \
                math.sqrt(label_len)  # Nonlinear scaling
            node_sizes.append(size)

        # Draw Nodes and Edges
        nx.draw_networkx_nodes(
            self, pos, node_color=node_colors, node_size=node_sizes, ax=ax)

        for (u, v), width in zip(self.edges(), edge_widths):
            # Here, node_v_size and node_u_size represent the sizes (or the "radii") of the nodes.
            node_v_size = node_sizes[list(self.nodes).index(v)]
            node_u_size = node_sizes[list(self.nodes).index(u)]

            # Adjust the margins based on node sizes to avoid collision with arrowheads and to avoid unnecessary gaps.
            target_margin = 5*node_v_size / node_size_multiplier
            source_margin = 5*node_u_size / node_size_multiplier

            if self.has_edge(v, u):
                nx.draw_networkx_edges(self, pos, edgelist=[(u, v)], width=width, connectionstyle='arc3,rad=0.3',
                                       arrowstyle=f'->,head_length={arrowhead_length},head_width={arrowhead_width}', min_target_margin=target_margin, min_source_margin=source_margin)
            else:
                nx.draw_networkx_edges(self, pos, edgelist=[(
                    u, v)], width=width, arrowstyle=f'-|>,head_length={arrowhead_length},head_width={arrowhead_width}', min_target_margin=target_margin, min_source_margin=source_margin)

        # Draw Labels
        nx.draw_networkx_labels(self, pos, labels=node_labels)
        if edge_labels:
            nx.draw_networkx_edge_labels(self, pos, edge_labels=edge_labels)

        # Add text in the bottom right corner if provided
        if bottom_right_text:
            ax.text(1, 0, bottom_right_text, horizontalalignment='right',
                    verticalalignment='bottom', transform=ax.transAxes)

        # Save the plot if save_filepath is provided
        if save_filepath:
            plt.savefig(save_filepath)

        # Show the plot if show is True
        if show:
            plt.show()
        return fig, ax

    def plot_opinion_distribution(self):
        """
        Visualizes the distribution of opinions in the graph using a histogram.
        """
        # Plot the histogram
        plt.hist(self.opinions, bins=50, edgecolor='black', alpha=0.75)
        plt.title('Opinion Distribution')
        plt.xlabel('Opinion Value')
        plt.ylabel('Number of Nodes')
        plt.grid(axis='y', alpha=0.75)
        plt.show()

    def get_opinion_neighbor_mean_opinion_pairs(self):
        # Extract opinion values for all nodes
        opinions = nx.get_node_attributes(self, 'opinion')

        x_values = []  # Node's opinion
        y_values = []  # Mean opinion of neighbors

        for node in self.nodes():
            node_opinion = opinions[node]
            neighbors = list(self.neighbors(node))
            if neighbors:  # Ensure the node has neighbors
                mean_neighbor_opinion = sum(
                    opinions[neighbor] for neighbor in neighbors) / len(neighbors)
                x_values.append(node_opinion)
                y_values.append(mean_neighbor_opinion)
        return x_values, y_values
    
    def get_opinion_neighbor_mean_opinion_pairs_dict(self):
        # Extract opinion values for all nodes
        opinions = nx.get_node_attributes(self, 'opinion')

        data={}

        for node in self.nodes():
            node_opinion = opinions[node]
            neighbors = list(self.neighbors(node))
            if neighbors:  # Ensure the node has neighbors
                mean_neighbor_opinion = sum(
                    opinions[neighbor] for neighbor in neighbors) / len(neighbors)
                data[node] = (node_opinion,mean_neighbor_opinion)
        return data

    def plot_neighbor_mean_opinion(self, fig=None, ax=None, save=None, show=True, extent=None, title=None, cmax=None, **kwargs):
        """
        Draws a hexbin plot with nodes' opinion values on the x-axis 
        and the mean opinion value of their neighbors on the y-axis.

        Parameters:
            fig (matplotlib.figure.Figure): Pre-existing figure object. If None, a new figure is created.
            ax (matplotlib.axes._axes.Axes): Pre-existing axes object. If None, new axes are created.
            save (str): Filepath to save the plot. If None, the plot is not saved.
            show (bool): Whether to display the plot. Default is True.
            extent (list): List specifying the range for the plot. If None, it's calculated from the data.
                        If the list has 2 values, they are used to create a square extent. 
                        If the list has 4 values, they are used directly.
            cmax (float or None): Maximum limit for the colorbar. If None, it's calculated from the data.
            **kwargs: Additional keyword arguments passed to plt.hexbin.
        """
        x_values, y_values = self.get_opinion_neighbor_mean_opinion_pairs()

        # Ensure the data returned is valid
        if not (x_values and y_values) or len(x_values) != len(y_values):
            print("Invalid data received. Cannot plot.")
            return

        if fig is None or ax is None:
            fig, ax = plt.subplots()

        # Handle extent parameter
        if extent is None:
            extent = [min(x_values), max(x_values),
                      min(y_values), max(y_values)]
        elif len(extent) == 2:
            extent = [extent[0], extent[1], extent[0], extent[1]]
        elif len(extent) != 4:
            print("Invalid extent value. Please provide None, 2 values, or 4 values.")
            return

        # Create a background filled with the `0` value of the colormap
        ax.imshow([[0, 0], [0, 0]], cmap='inferno', interpolation='nearest',
                  aspect='auto', extent=extent)

        # Create the hexbin plot
        hb = ax.hexbin(x_values, y_values, gridsize=50,
                       cmap='inferno', bins='log', extent=extent, vmax=cmax, **kwargs)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('Log(Number of points in bin)')

        ax.set_title(title or 'Node Opinion vs. Mean Neighbor Opinion')
        ax.set_xlabel('Node Opinion')
        ax.set_ylabel('Mean Neighbor Opinion')

        # Save the plot if save_filepath is provided
        if save:
            plt.savefig(save)

        # Show the plot if show is True
        if show:
            plt.show()

        return fig, ax

    @property
    def labels(self):
        """Returns a list of labels for all nodes in the graph. 
        If a node doesn't have a label, its ID will be used as the default label."""
        return [self.nodes[node]['label'] for node in self.nodes]

    @property
    def cluster_size(self):
        """
        Returns a dictionary with the sizes of the nodes.
        Key is the node ID, and opinion is the size of the node.
        """
        return {node: self.nodes[node].get('size', len(self.nodes[node].get('label', [0, ]))) for node in self.nodes}

    @property
    def importance(self):
        """
        Returns a dictionary with the importance of the nodes.
        Key is the node ID, and value is the ratio of the sum of influences of the node to the size of the node.
        """
        importance_dict = {}
        size_dict = self.cluster_size

        for node in self.nodes:
            influences_sum = sum(data['influence']
                                 for _, _, data in self.edges(node, data=True))
            importance_dict[node] = influences_sum / \
                size_dict[node] if size_dict[node] != 0 else 0

        return importance_dict

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

    @property
    def node_values(self):
        """
        Returns a nested dictionary with the properties of the nodes.
        The first key is the property name, and the value is another dictionary 
        with node IDs as keys and the corresponding property values as values.
        """
        properties_dict = {}
        for node, node_data in self.nodes(data=True):
            for property_name, property_value in node_data.items():
                if property_name not in properties_dict:
                    properties_dict[property_name] = {}
                properties_dict[property_name][node] = property_value
        return properties_dict

    def __str__(self):
        return f"<HariGraph with {self.number_of_nodes()} nodes and {self.number_of_edges()} edges>"

    def __repr__(self):
        return f"<HariGraph object at {id(self)}: {self.number_of_nodes()} nodes, {self.number_of_edges()} edges>"
