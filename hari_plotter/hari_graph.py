import json
import math
import os
import random
from itertools import combinations, permutations

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class HariGraph(nx.DiGraph):
    """
    HariGraph extends the DiGraph class of Networkx to offer additional functionality.
    It ensures that each node has a label and provides methods to create, save, and load graphs.
    """

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.generate_labels()
        self.similarity_function = self.default_similarity_function

    def generate_labels(self):
        """Generates labels for the nodes if they don't exist."""
        if not self.nodes:
            return
        if 'label' not in self.nodes[next(iter(self.nodes))]:
            for i in self.nodes:
                self.nodes[i]['label'] = [i]

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
            G.add_node(node["id"], value=node["value"],
                       label=node.get('label', [node["id"]]))

        for edge in graph_dict["edges"]:
            G.add_edge(edge["source"], edge["target"], value=edge["value"])

        return G

    def save_json(self, filename):
        """
        Saves the HariGraph to a JSON file.

        :param filename: The name of the file to write to.
        """
        graph_dict = {
            "nodes": [
                {"id": n, "value": self.nodes[n]["value"], "label": self.nodes[n].get('label', [
                                                                                      n])}
                for n in self.nodes()
            ],
            "edges": [{"source": u, "target": v, "value": self[u][v]["value"]} for u, v in self.edges()]
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
            G.add_node(i, value=random.random())

        nodes = list(G.nodes)
        random.shuffle(nodes)
        for i in range(n - 1):
            G.add_edge(nodes[i], nodes[i + 1], value=random.random())

        additional_edges = random.randint(1, n)
        for _ in range(additional_edges):
            u, v = random.sample(G.nodes, 2)
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v, value=random.random())
                if random.choice([True, False]) and not G.has_edge(v, u):
                    G.add_edge(v, u, value=random.random())

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
            G.add_node(i, value=random.random())
        for i in range(n):
            for j in range(n):
                if i != j:
                    G.add_edge(i, j, value=random.random())

        edges_to_remove = random.sample(
            G.edges, int(len(G.edges) * (1 - factor)))
        G.remove_edges_from(edges_to_remove)

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
                print(f"No path exists from {source} to {target}")
                return False
        return True

    def dynamics_step(self, t):
        """
        Updates the value of each node in the HariGraph instance based on the values of its predecessors.

        :param t: The time step factor influencing the dynamics.
        """
        updated_values = {}  # Temporary dictionary to store updated values

        for i in self.nodes:
            vi = self.nodes[i]['value']

            # Predecessors of a node are the start nodes of its incoming edges.
            for j in self.predecessors(i):
                pij = self[j][i]['value']
                vj = self.nodes[j]['value']
                vi += pij * vj * t  # Calculate updated value based on each incoming edge

            # Clip the updated value to [0, 1]
            vi = max(0, min(vi, 1))

            updated_values[i] = vi

        # Update the values in the graph with the calculated updated values
        for i, vi in updated_values.items():
            self.nodes[i]['value'] = vi

    @property
    def weighted_mean_value(self):
        """
        Calculates the weighted mean value of the nodes in the graph.

        For each node, its value is multiplied by its weight. 
        The weight of a node is the length of its label if defined, 
        otherwise, it is assumed to be 1. The method returns the 
        sum of the weighted values divided by the sum of the weights.

        Returns:
            float: The weighted mean value of the nodes in the graph. 
                Returns 0 if the total weight is 0 to avoid division by zero.
        """
        total_value = 0
        total_weight = 0

        for node in self.nodes:
            value = self.nodes[node]['value']

            # If label is defined, the weight is the length of the label.
            # If not defined, the weight is assumed to be 1.
            weight = len(self.nodes[node].get('label', [node]))

            total_value += value * weight
            total_weight += weight

        if total_weight == 0:  # Prevent division by zero
            return 0

        return total_value / total_weight

    @staticmethod
    def default_similarity_function(vi, vj, size_i, size_j, edge_value, reverse_edge_value,
                                    value_coef=1., extreme_coef=0.1, influence_coef=1., size_coef=10.):
        """
        The default function used to calculate the similarity between two nodes.

        Parameters:
            vi (float): The value attribute of node i.
            vj (float): The value attribute of node j.
            size_i (int): The length of the label of node i.
            size_j (int): The length of the label of node j.
            edge_value (float): The value attribute of the edge from node i to node j, if it exists, else None.
            reverse_edge_value (float): The value attribute of the edge from node j to node i, if it exists, else None.
            value_coef (float): Coefficient for value proximity impact.
            extreme_coef (float): Coefficient for extreme proximity impact.
            influence_coef (float): Coefficient for influence impact.
            size_coef (float): Coefficient for size impact.

        Returns:
            float: The computed similarity value between node i and node j.
        """
        # Calculate Value Proximity Impact (high if values are close)
        value_proximity = value_coef * (1 - abs(vi - vj))

        # Calculate Proximity to 0 or 1 Impact (high if values are close to 0 or 1)
        extreme_proximity = extreme_coef * \
            min(min(vi, vj), min(1 - vi, 1 - vj))

        # Calculate Influence Impact (high if influence is high)
        influence = 0
        label_sum = size_i + size_j

        if edge_value is not None:  # if an edge exists between the nodes
            influence += (edge_value * size_j) / label_sum

        if reverse_edge_value is not None:  # if a reverse edge exists between the nodes
            influence += (reverse_edge_value * size_i) / label_sum

        influence *= influence_coef  # Apply Influence Coefficient

        # Calculate Size Impact (high if size is low)
        size_impact = size_coef * \
            (1 / (1 + size_i) + 1 / (1 + size_j))

        # Combine the impacts
        return value_proximity + extreme_proximity + influence + size_impact

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
            float: The computed similarity value between nodes i and j.
        """
        # Check if there is an edge between nodes i and j
        if not self.has_edge(i, j) and not self.has_edge(j, i):
            raise ValueError(f"No edge exists between nodes {i} and {j}")

        # Extract parameters from node i, node j, and the edge (if exists)
        vi = self.nodes[i]['value']
        vj = self.nodes[j]['value']

        size_i = len(self.nodes[i].get('label', [i]))
        size_j = len(self.nodes[j].get('label', [j]))

        edge_value = self[i][j]['value'] if self.has_edge(i, j) else None
        reverse_edge_value = self[j][i]['value'] if self.has_edge(
            j, i) else None

        # Choose the correct similarity function and calculate the similarity
        func = similarity_function or self.similarity_function
        return func(vi, vj, size_i, size_j, edge_value, reverse_edge_value)

    def merge_nodes(self, i, j):
        """
        Merges two nodes in the graph into a new node.

        The new node's value is a weighted average of the values of 
        the merged nodes, and its label is the concatenation of the labels 
        of the merged nodes. The edges are reconnected to the new node, 
        and the old nodes are removed.

        Parameters:
            i (int): The identifier for the first node to merge.
            j (int): The identifier for the second node to merge.
        """
        # Calculate new value and label
        label_i = self.nodes[i].get('label', [i])
        label_j = self.nodes[j].get('label', [j])
        new_label = label_i + label_j

        vi = self.nodes[i]['value']
        vj = self.nodes[j]['value']
        weight_i = len(label_i)
        weight_j = len(label_j)
        new_value = (vi * weight_i + vj * weight_j) / (weight_i + weight_j)

        # Add a new node to the graph with the calculated value and label
        new_node = max(self.nodes) + 1
        self.add_node(new_node, value=new_value, label=new_label)

        # Reconnect edges
        for u, v, data in list(self.edges(data=True)):
            if u == i or u == j:
                if v != i and v != j:  # Avoid connecting the new node to itself
                    value = self[u][v]['value']
                    if self.has_edge(new_node, v):
                        # Sum the values if both original nodes were connected to the same node
                        self[new_node][v]['value'] += value
                    else:
                        self.add_edge(new_node, v, value=value)
                if self.has_edge(u, v):  # Check if the edge exists before removing it
                    self.remove_edge(u, v)
            if v == i or v == j:
                if u != i and u != j:  # Avoid connecting the new node to itself
                    value = self[u][v]['value']
                    if self.has_edge(u, new_node):
                        # Sum the values if both original nodes were connected to the same node
                        self[u][new_node]['value'] += value
                    else:
                        self.add_edge(u, new_node, value=value)
                if self.has_edge(u, v):  # Check if the edge exists before removing it
                    self.remove_edge(u, v)

        # Remove the old nodes
        self.remove_node(i)
        self.remove_node(j)

    def simplify_graph_one_iteration(self):
        """
        Simplifies the graph by one iteration.

        In each iteration, it finds the pair of nodes with the maximum similarity 
        and merges them. If labels are not initialized, it initializes them. 
        If there is only one node left, no further simplification is possible.

        Returns:
            HariGraph: The simplified graph.
        """
        # Check if labels are initialized, if not, initialize them
        if 'label' not in self.nodes[next(iter(self.nodes))]:
            for i in self.nodes:
                self.nodes[i]['label'] = [i]

        # If there is only one node left, no further simplification is possible
        if self.number_of_nodes() <= 1:
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

    def draw(self, plot_node_info='none', use_node_color=True,
             use_edge_thickness=True, plot_edge_values=False,
             node_size_multiplier=200,
             arrowhead_length=0.2, arrowhead_width=0.2,
             min_line_width=0.1, max_line_width=3.0,
             seed=None, save_filepath=None, show=True,
             fig=None, ax=None):
        """
        Visualizes the graph with various customization options.

        :param plot_node_info: str, optional
            Determines the information to display on the nodes.
            Options: 'none', 'values', 'ids', 'labels', 'size'. Default is 'none'.

        :param use_node_color: bool, optional
            If True, nodes are colored based on their values. Default is True.

        :param use_edge_thickness: bool, optional
            If True, the thickness of the edges is determined by their values. Default is True.

        :param plot_edge_values: bool, optional
            If True, displays the values of the edges on the plot. Default is False.

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

        :return: tuple
            A tuple containing the Matplotlib Figure and Axes objects.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 7))
        pos = nx.spring_layout(self, seed=seed)

        # Get the node and edge attributes
        node_attributes = nx.get_node_attributes(self, 'value')
        edge_attributes = nx.get_edge_attributes(self, 'value')

        # Prepare Node Labels
        node_labels = {}
        if plot_node_info == 'values':
            node_labels = {node: f"{value:.2f}" for node,
                           value in node_attributes.items()}
        elif plot_node_info == 'ids':
            node_labels = {node: f"{node}" for node in self.nodes}
        elif plot_node_info == 'labels':
            for node in self.nodes:
                label = self.nodes[node].get('label', None)
                if label is not None:
                    node_labels[node] = ','.join(map(str, label))
                else:  # If label is not defined, show id instead
                    node_labels[node] = str(node)
        elif plot_node_info == 'size':
            for node in self.nodes:
                label_len = len(self.nodes[node].get('label', [node]))
                node_labels[node] = str(label_len)

        # Prepare Node Colors
        if use_node_color:
            node_colors = [cm.bwr(value) for value in node_attributes.values()]
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
        if plot_edge_values:
            edge_labels = {(u, v): f"{value:.2f}" for (u, v),
                           value in edge_attributes.items()}

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

            # Save the plot if save_filepath is provided
        if save_filepath:
            plt.savefig(save_filepath)

        # Show the plot if show is True
        if show:
            plt.show()
        return fig, ax

    def __str__(self):
        return f"<HariGraph with {self.number_of_nodes()} nodes and {self.number_of_edges()} edges>"

    def __repr__(self):
        return f"<HariGraph object at {id(self)}: {self.number_of_nodes()} nodes, {self.number_of_edges()} edges>"
