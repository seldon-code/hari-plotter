from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Tuple, Optional, Type, Union

import numpy as np

from itertools import combinations


from .cluster import Cluster
from .hari_graph import HariGraph


class Group:

    # Dictionary mapping function names to actual function objects
    common_functions = {
        'mean': np.mean,
        'sum': np.sum,
        'min': np.min,
        'max': np.max,
        'median': np.median,
        'std': np.std,  # Standard Deviation
        'var': np.var,  # Variance
        'ptp': np.ptp,  # Peak to Peak (Max - Min)
        # Add other functions as needed
    }

    def __init__(self, images, time=None, model=None):
        self.images = images

        if time is None:
            self.time = None
        elif isinstance(time, (int, float)):
            self.time = [time] * len(images)
        elif hasattr(time, '__len__'):
            assert len(time) == len(
                images), "Length of time and images must be the same"
            self.time = time
        else:
            raise ValueError("Invalid input for time")

        self.model = model

        self._mean_graph = None

        self.clusters = dict()

    @staticmethod
    def request_to_tuple(request):
        def convert(item):
            if isinstance(item, dict):
                return tuple(sorted((k, convert(v)) for k, v in item.items()))
            elif isinstance(item, list):
                return tuple(convert(v) for v in item)
            else:
                return item

        return convert(request)

    @property
    def mean_graph(self) -> HariGraph:
        if self._mean_graph is not None:
            return self._mean_graph
        self.initialize_mean_graph()
        return self._mean_graph

    def get_mean_graph(self, **settings) -> HariGraph:
        return self.mean_graph

    def initialize_mean_graph(self):
        self._mean_graph = HariGraph.mean_graph(self.images)

    def cluster(self, **cluster_settings):
        # Convert settings to a sorted tuple of pairs to ensure consistent ordering
        cluster_key = Group.request_to_tuple(cluster_settings)

        if cluster_key in self.clusters:
            return self.clusters[cluster_key]['cluster']

        # Create a new cluster instance
        cluster = Cluster.create_cluster(G=self.mean_graph, **cluster_settings)

        # Cache the newly created cluster
        self.clusters[cluster_key] = {'cluster': cluster}
        return cluster

    def get_cluster(self, **settings):
        # print(f'{settings = }')
        if 'cluster_settings' in settings:
            return self.cluster(**(settings['cluster_settings']))
        else:
            return self.cluster()

    def cluster_graph(self, merge_remaining: bool = False,  **cluster_settings) -> HariGraph:

        cluster_key = self.request_to_tuple(cluster_settings)

        if cluster_key in self.clusters and 'graph' in self.clusters[cluster_key]:
            return self.clusters[cluster_key]['graph']

        cluster = self.cluster(**cluster_settings)

        cluster_nodes = cluster.get_cluster_mapping()
        cluster_labels = cluster.cluster_labels

        cluster_graph = self.mean_graph.copy()
        cluster_graph.merge_clusters(
            cluster_nodes, labels=cluster_labels, merge_remaining=merge_remaining)
        # print(f'{cluster_graph = }')
        self.clusters[cluster_key]['graph'] = cluster_graph

        return cluster_graph

    def cluster_graph_values(self, parameters: Tuple[str], cluster_settings: tuple,  **settings) -> Dict[str, np.ndarray]:

        graph = self.cluster_graph(**cluster_settings)
        # print(str(graph))

        params_no_time = [param for param in parameters if param != 'time']
        results = graph.gatherer.gather(params_no_time)
        if 'time' in parameters:
            results['time'] = self.mean_time()

        # print(f'{results = }')

        return results

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        return iter(self.images)

    def __getitem__(self, key):
        return self.images[key]

    def __setitem__(self, key, value):
        self.images[key] = value

    def __delitem__(self, key):
        del self.images[key]

    def append(self, image):
        self.images.append(image)

    def __repr__(self):
        return f"Group(time={self.time}, images={self.images})"

    def calculate_node_values(self, parameters: Tuple[str], **settings) -> dict:
        """
        Calculate the node values based on parameters.

        Args:
            parameters (List[str]): List of parameter names.

        Returns:
            dict: A dictionary containing mean node values.
        """
        params_no_time = [param for param in parameters if param != 'time']
        results = self.mean_graph.gatherer.gather(params_no_time)
        if 'time' in parameters:
            results['time'] = self.mean_time()
        # print(f'{results.keys() = }')
        return results

    def calculate_function_of_node_values(self, parameters: Tuple[str], function='mean', **settings) -> dict:
        """
        Calculate the function of mean node values based on parameters, treating 'time' specially.

        Args:
            parameters (List[str]): List of parameter names.
            function (str): The name of the function to be applied to the mean values of the nodes, except for 'time'.

        Returns:
            dict: A dictionary containing the results of the function applied to mean node values.
        """
        # Get the actual function object from the common_functions dictionary
        if function not in self.common_functions:
            raise ValueError(
                f'{function} is not in a list of common functions: {list(self.common_functions.keys())}')
        func = self.common_functions[function]

        # Separate 'time' from other parameters if it's present
        params_no_time = [param for param in parameters if param != 'time']
        node_data = self.calculate_node_values(params_no_time)
        results = {}

        for param in parameters:
            if param == 'time':
                # Handle 'time' as a special case
                results['time'] = self.mean_time()
            elif param == 'nodes':
                continue
            elif param in node_data:
                # Apply the function to other parameters
                # print(f'{node_data[param] = }')
                results[param] = func(node_data[param])

        return results

    def mean_time(self, **settings):
        return np.mean(self.time)
