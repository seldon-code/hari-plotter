from __future__ import annotations

import json
import os
import random
import re
import warnings
from itertools import combinations, permutations
from typing import Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np

from .distributions import generate_mixture_of_gaussians
from .node_gatherer import DefaultNodeGatherer


class HariGraph(nx.DiGraph):
    """
    HariGraph extends the DiGraph class of Networkx to offer additional functionality.
    It ensures that each node has a label and provides methods to create, save, and load graphs.
    """

    # ---- Initialization Methods ----

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.similarity_function = self.default_similarity_function
        self.gatherer = DefaultNodeGatherer(self)

    def add_parameters_to_nodes(self, nodes: Union[List[Tuple[int]], None] = None):
        """
        Appends or updates node attributes based on predefined criteria.

        Parameters:
            node_ids (Optional[List[Tuple[int]]): List of node IDs to which parameters will be added. 
                If None, all nodes will be considered.
        """
        if nodes is None:
            nodes = list(self.nodes)

        parameters = self.gatherer.gather_everything()
        # Update the node's attributes with the gathered parameters
        for key, value in parameters.items():
            for node in nodes:
                self.nodes[node][key] = value[node]

    def has_self_loops(self):
        """
        Checks if the graph contains any self-loops.

        Returns:
            bool: True if there is at least one self-loop in the graph, False otherwise.
        """
        for node in self.nodes:
            if self.has_edge(node, node):
                return True  # A self-loop exists
        return False  # No self-loops found

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
    def mean_graph(cls, images: List[HariGraph]) -> HariGraph:
        if len(images) == 1:
            return images[0].copy()
        # Initialize a new graph
        mean_graph = cls()

        # Assure that all graphs have the same nodes
        nodes = set(images[0].nodes())
        if not all(set(g.nodes()) == nodes for g in images):
            raise ValueError("Graphs do not have the same nodes")

        # Calculate mean values for node attributes
        for node in nodes:
            mean_opinion = sum(g.nodes[node]['opinion']
                               for g in images) / len(images)
            mean_graph.add_node(node, opinion=mean_opinion)

        # Calculate mean values for edge attributes
        for i, j in combinations(nodes, 2):
            influences = []
            for g in images:
                if g.has_edge(i, j):
                    influences.append(g[i][j]['influence'])
                else:
                    # Assuming 0 influence for missing edges
                    influences.append(0)

            mean_influence = sum(influences) / len(images)
            if mean_influence > 0:  # Add the edge only if the mean influence is positive
                mean_graph.add_edge(i, j, influence=mean_influence)

        return mean_graph

    @classmethod
    def read_network(cls, network_file: str, opinion_file: str) -> 'HariGraph':
        G = cls()  # Instantiate HariGraph

        # Check if '&' is present in the network file
        with open(network_file, 'r') as f:
            content = f.read()
            has_tuples = '&' in content

        # Define parsing function based on presence of '&'
        if has_tuples:
            def parse_node_id(node_id_str):
                if '&' in node_id_str:
                    return tuple(map(int, node_id_str.split('&')))
                else:
                    return (int(node_id_str),)
        else:
            def parse_node_id(node_id_str):
                return (int(node_id_str),)

        # Process the network file
        with open(network_file, 'r') as f:
            next(f)  # Skip header line
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(',')
                idx_agent = parse_node_id(parts[0].strip())
                n_neighbors = int(parts[1])

                for i in range(n_neighbors):
                    neighbor_id = parse_node_id(parts[2 + i].strip())
                    weight = float(parts[2 + n_neighbors + i])
                    G.add_edge(idx_agent, neighbor_id, influence=weight)

        # Process the opinion file
        with open(opinion_file, 'r') as f:
            next(f)  # Skip header line
            for line in f:
                parts = line.split(',')
                idx_agent = parse_node_id(parts[0].strip())
                opinion = float(parts[1])

                # Add node if it doesn't exist
                if not G.has_node(idx_agent):
                    G.add_node(idx_agent)

                # Set node attributes for opinion
                G.nodes[idx_agent]['opinion'] = opinion

                # Optional: set activity level if provided
                if len(parts) > 2:
                    activity = float(parts[2])
                    G.nodes[idx_agent]['activity'] = activity

        return G

    def write_network(self, network_file: str, opinion_file: str, delimiter=','):
        opinions = self.gatherer.gather('opinion')

        with open(network_file, 'w') as f:
            f.write(
                f"# idx_agent{delimiter}n_neighbors_in{delimiter}indices_neighbors_in[...]{delimiter}weights_in[...]\n")
            # Inside the for loop for writing network data to file
            for node in self.nodes:
                neighbors = list(self.predecessors(node))
                weights = [self[neighbor][node]['influence']
                           for neighbor in neighbors]

                # Convert node from tuple to "a&b&c..." format
                node_output = '&'.join(map(str, node))

                # Prepare neighbors' output, converting tuples to "a&b&c..." format
                neighbors_output = ['&'.join(map(str, neighbor))
                                    for neighbor in neighbors]

                f.write(
                    f"{node_output}{delimiter}{len(neighbors)}{delimiter}{delimiter.join(neighbors_output)}{delimiter}{delimiter.join(map(str, weights))}\n")

        with open(opinion_file, 'w') as f:
            f.write(f"# idx_agent{delimiter}opinion[...]\n")
            for node_id in opinions['nodes']:
                opinion = opinions['opinion'][opinions['nodes'].index(node_id)]

                # Convert node_id from tuple to "a&b&c..." format
                node_id_output = '&'.join(map(str, node_id))

                f.write(f"{node_id_output}{delimiter}{opinion}\n")

    @classmethod
    def read_json(cls, filename: str) -> HariGraph:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} does not exist.")

        with open(filename, 'r') as file:
            graph_dict = json.load(file)

        G = cls()
        for node in graph_dict["nodes"]:
            # Convert ID to tuple if necessary
            node_id = tuple(node["id"]) if isinstance(
                node["id"], list) else (node["id"],)
            G.add_node(node_id, opinion=node["opinion"])

        for edge in graph_dict["edges"]:
            # Convert source and target to tuples if necessary
            source_id = tuple(edge["source"]) if isinstance(
                edge["source"], list) else (edge["source"],)
            target_id = tuple(edge["target"]) if isinstance(
                edge["target"], list) else (edge["target"],)
            G.add_edge(source_id, target_id, influence=edge["influence"])

        return G

    def write_json(self, filename: str):
        """
        Saves the HariGraph to a JSON file.

        :param filename (str): The name of the file to write to.
        """
        params_to_save = ['opinion']

        # Check conditions for 'cluster_size' and 'inner_opinions'
        if any(size > 1 for size in self.gatherer.gather('cluster_size')['cluster_size']):
            params_to_save.extend(['cluster_size', 'inner_opinions'])

        # Check condition for 'activity'
        if not np.all(np.isnan(self.gatherer.gather('activity')['activity'])):
            params_to_save.append('activity')

        # Gathering all parameters that met the conditions
        gathered_data = self.gatherer.gather(params_to_save)

        # Convert tuples in 'inner_opinions' to strings if 'inner_opinions' is in the gathered data
        if 'inner_opinions' in gathered_data:
            gathered_data['inner_opinions'] = [
                {key[0]: value for key, value in d.items()} for d in gathered_data['inner_opinions']]

        # Construct node data
        graph_dict = {
            "nodes": [
                {"id": list(node_id) if isinstance(node_id, tuple) else node_id,
                 **{prop: gathered_data[prop][i] for prop in params_to_save}}
                for i, node_id in enumerate(gathered_data['nodes'])
            ],
            "edges": [
                {"source": list(u) if isinstance(u, tuple) else u, "target": list(
                    v) if isinstance(v, tuple) else v, "influence": self[u][v]["influence"]}
                for u, v in self.edges()
            ]
        }

        with open(filename, 'w') as file:
            json.dump(graph_dict, file, ensure_ascii=False, indent=4)

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
            G.add_node((i,), opinion=random.random())
        for i in range(n):
            for j in range(n):
                if i != j:
                    G.add_edge((i,), (j,), influence=random.random())

        edges_to_remove = random.sample(
            G.edges, int(len(G.edges) * (1 - factor)))
        G.remove_edges_from(edges_to_remove)

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
                G.add_node((i,), opinion=opinions[opinion_idx])
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

    def copy(self) -> HariGraph:
        G_copy = super().copy(as_view=False)
        G_copy = HariGraph(G_copy) if not isinstance(
            G_copy, HariGraph) else G_copy
        G_copy.similarity_function = self.similarity_function
        # G_copy.node_parameter_gatherer = self.node_parameter_gatherer
        G_copy.gatherer = type(self.gatherer)(G_copy)
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
        merged_data = self.gatherer.merge([i, j])

        # Generate a new node ID
        new_node_id = tuple(sorted(i+j))

        # Add the new merged node to the graph
        self.add_node(new_node_id, **merged_data)

        # Reconnect edges
        for u, v, data in list(self.edges(data=True)):
            if u in [i, j]:
                if v not in [i, j]:
                    influence = data['influence']
                    self.add_edge(new_node_id, v, influence=influence)
                self.remove_edge(u, v)
            elif v in [i, j]:
                if u not in [i, j]:
                    influence = data['influence']
                    self.add_edge(u, new_node_id, influence=influence)
                self.remove_edge(u, v)

        # Remove the original nodes
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
                    size_i = self.nodes[node].get('cluster_size', len(node))

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
                clusters.append(cluster)

        # Merge the clusters
        if clusters:
            self.merge_clusters(clusters)

    def get_cluster_mapping(self) -> List[List[Tuple[int]]]:
        """
        Generates a list of nodes in the unclustered graph to be clustered to get the current graph
        :return: A list representing the current clusters in the graph.
        """
        return sorted([sorted([(element,) for element in node]) for node in self.nodes])

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
            all_nodes = set(self.nodes)

            # Record all nodes that are part of the specified clusters
            clustered_nodes = set(
                node for cluster in clusters for node in cluster)
            remaining_nodes = all_nodes - clustered_nodes
            if remaining_nodes:
                clusters.append(list(remaining_nodes))
                labels.append(None)

        for cluster, label in zip(clusters, labels):
            new_node_name = tuple(sorted(sum(cluster, ())))
            merged_attributes = self.gatherer.merge(cluster)
            self.add_node(new_node_name, **merged_attributes)
            if label is not None:
                self.nodes[new_node_name]['label'] = label

            # Reconnect edges
            for old_node_id in cluster:
                for successor in list(self.successors(old_node_id)):
                    if successor not in cluster:
                        influence = self[old_node_id][successor]['influence']
                        self.add_edge(new_node_name, successor,
                                      influence=influence)

                for predecessor in list(self.predecessors(old_node_id)):
                    if predecessor not in cluster:
                        influence = self[predecessor][old_node_id]['influence']
                        self.add_edge(predecessor, new_node_name,
                                      influence=influence)

                # Remove old node
                self.remove_node(old_node_id)

    def simplify_graph_one_iteration(self):
        """
        Simplifies the graph by one iteration.

        In each iteration, it finds the pair of nodes with the maximum similarity
        and merges them. 
        If there is only one node left, no further simplification is possible.

        Returns:
            HariGraph: The simplified graph.
        """
        # self.generate_labels()

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

    # @staticmethod
    # def default_node_parameter_gatherer(nodes):
    #     """
    #     Gathers default parameters for a node that is a result of merging given nodes.

    #     :param nodes: List[Dict], a list of node dictionaries, each containing node attributes.
    #     :return: Dict, a dictionary with calculated 'inner_opinions', 'cluster_size', and 'label'.
    #     """
    #     if not nodes:
    #         raise ValueError("The input list of nodes must not be empty.")

    #     size = sum(node.get('cluster_size', len(
    #         node.get('label', [0, ]))) for node in nodes)

    #     # Gather all opinions of the nodes being merged using node labels/identifiers as keys
    #     inner_opinions = {}

    #     for node in nodes:
    #         node_label = node.get('label', None)
    #         if node_label is not None:
    #             # Check if node has 'inner_opinions', if not, create one
    #             if 'inner_opinions' in node:
    #                 inner_opinions.update(node['inner_opinions'])
    #             else:
    #                 if len(node_label) != 1:
    #                     warnings.warn(
    #                         f"The length of the label in the node is higher than one. Assuming that all opinions in this cluster were equal. This is not typical behavior, check that that it corresponds to your intention. Found in node: {node_label}")
    #                 for i in node_label:
    #                     inner_opinions[i] = node['opinion']

    #     return {
    #         'cluster_size': size,
    #         'opinion': sum(node.get('cluster_size', len(node.get('label', [0, ]))) * node['opinion'] for node in nodes) / size,
    #         'label': [id for node in nodes for id in node['label']],
    #         'inner_opinions': inner_opinions
    #     }

    # @staticmethod
    # def min_max_node_parameter_gatherer(nodes):
    #     """
    #     Gathers default parameters for a node that is a result of merging given nodes.

    #     :param nodes: List[Dict], a list of node dictionaries, each containing node attributes.
    #     :return: Dict, a dictionary with calculated 'max_opinion' and 'min_opinion'.
    #     """
    #     if not nodes:
    #         raise ValueError("The input list of nodes must not be empty.")

    #     size = sum(node.get('cluster_size', len(
    #         node.get('label', [0, ]))) for node in nodes)

    #     return {
    #         'cluster_size': size,
    #         'opinion': sum(node.get('cluster_size', len(node.get('label', [0, ]))) * node['opinion'] for node in nodes) / size,
    #         'label': [id for node in nodes for id in node['label']],
    #         'max_opinion': max(node.get('max_opinion', node['opinion']) for node in nodes),
    #         'min_opinion': min(node.get('min_opinion', node['opinion']) for node in nodes)
    #     }

    def compute_similarity(self, i: Tuple[int], j: Tuple[int], similarity_function=None):
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

        size_i = len(i)
        size_j = len(j)

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
    def opinions(self, values: Union[int, float, Dict[Tuple[int]:float]]):
        if isinstance(values, (int, float)):
            for node in self.nodes:
                self.nodes[node]["opinion"] = values
        elif isinstance(values, dict):
            for node, opinion in values.items():
                if node in self.nodes:
                    self.nodes[node]["opinion"] = opinion
                else:
                    raise ValueError(
                        f"Node {node} does not exist in the graph.")
        else:
            raise TypeError(
                f'Values input type for the opinions {type(values)} is not supported')

    # @opinions.setter
    # def opinions(self, values):
    #     """
    #     Sets the opinions of the nodes based on the provided value(s).
    #     Value(s) can be a single float, a list of floats, or a dictionary with node IDs as keys.
    #     """
    #     if isinstance(values, (int, float)):  # Single value provided
    #         for node in self.nodes:
    #             self.nodes[node]["opinion"] = values

    #     elif isinstance(values, list):  # List of values provided
    #         if len(values) != len(self.nodes):
    #             raise ValueError(
    #                 "Length of provided list does not match the number of nodes in the graph.")
    #         for node, opinion in zip(self.nodes, values):
    #             self.nodes[node]["opinion"] = opinion

    #     elif isinstance(values, dict):  # Dictionary provided
    #         for node, opinion in values.items():
    #             if node in self.nodes:
    #                 self.nodes[node]["opinion"] = opinion
    #             else:
    #                 raise ValueError(
    #                     f"Node {node} does not exist in the graph.")

    #     else:
    #         raise TypeError(
    #             "Invalid type for opinions. Expected int, float, list, or dict.")

    def __str__(self):
        return f"<HariGraph with {self.number_of_nodes()} nodes and {self.number_of_edges()} edges>"

    def __repr__(self):
        return f"<HariGraph object at {id(self)}: {self.number_of_nodes()} nodes, {self.number_of_edges()} edges>"
