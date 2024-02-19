from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

import numpy as np

from .cluster import Clustering
from .hari_graph import HariGraph


class Group:

    # Dictionary mapping function names to actual function objects
    common_functions = {
        'Mean': np.mean,
        'Sum': np.sum,
        'Min': np.min,
        'Max': np.max,
        'Median': np.median,
        'Standard Deviation': np.std,  # Standard Deviation
        'Variance': np.var,  # Variance
        'Peak to Peak': np.ptp,  # Peak to Peak (Max - Min)
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

        self.clusterings = dict()

        self._nodes = None
        self._node_parameters = None

    @property
    def node_parameters(self):
        if not self._node_parameters:
            self._node_parameters = self.images[0].gatherer.parameters
        return self._node_parameters

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

    def clustering(self, **clustering_settings):
        # Convert settings to a sorted tuple of pairs to ensure consistent ordering
        clustering_key = Group.request_to_tuple(clustering_settings)

        if clustering_key in self.clusterings:
            # print(f'+clustering found {clustering_key} ')
            return self.clusterings[clustering_key]['clustering']

        # print(f'-clustering not found {clustering_key} ')
        # print(f'{list(self.clusterings.keys()) = }')

        # Create a new clustering instance
        clustering = Clustering.create_clustering(
            G=self.mean_graph, **clustering_settings)

        # Cache the newly created clustering
        self.clusterings[clustering_key] = {'clustering': clustering}
        return clustering

    def get_clustering(self, **settings):
        # print(f'{settings = }')
        if 'clustering_settings' in settings:
            return self.clustering(**(settings['clustering_settings']))
        else:
            return self.clustering()

    def clustering_graph(self, merge_remaining: bool = False,  **clustering_settings) -> HariGraph:

        clustering_key = self.request_to_tuple(clustering_settings)

        if clustering_key in self.clusterings and 'graph' in self.clusterings[clustering_key]:
            return self.clusterings[clustering_key]['graph']

        clustering = self.clustering(**clustering_settings)

        clustering_nodes = clustering.get_cluster_mapping()
        cluster_labels = clustering.cluster_labels

        clustering_graph = self.mean_graph.copy()
        clustering_graph.merge_clusters(
            clustering_nodes, labels=cluster_labels, merge_remaining=merge_remaining)
        # print(f'{clustering_graph = }')
        self.clusterings[clustering_key]['graph'] = clustering_graph

        return clustering_graph

    def clustering_graph_values(self, parameters: Tuple[str], clustering_settings: tuple,  **settings) -> Dict[str, np.ndarray]:

        graph = self.clustering_graph(**clustering_settings)
        # print(str(graph))

        params_no_time = [param for param in parameters if param != 'Time']
        results = graph.gatherer.gather(params_no_time)
        if 'Time' in parameters:
            results['Time'] = self.mean_time()

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
        return f"Group(id={id(self)}, time={self.time}, images={self.images})"

    @property
    def nodes(self) -> set:
        if self._nodes:
            return self._nodes
        self._nodes = set(self.mean_graph.nodes)
        return self._nodes

    def calculate_node_values(self, parameters: Tuple[str], **settings) -> dict:
        """
        Calculate the node values based on parameters.

        Args:
            parameters (List[str]): List of parameter names.

        Returns:
            dict: A dictionary containing mean node values.
        """
        params_no_time = [param for param in parameters if param != 'Time']
        results = self.mean_graph.gatherer.gather(params_no_time)
        if 'Time' in parameters:
            results['Time'] = self.mean_time()
        # print(f'{results.keys() = }')
        return results

    def calculate_function_of_node_values(self, parameters: Tuple[str], function='Mean', **settings) -> dict:
        """
        Calculate the function of mean node values based on parameters, treating 'Time' specially.

        Args:
            parameters (List[str]): List of parameter names.
            function (str): The name of the function to be applied to the mean values of the nodes, except for 'Time'.

        Returns:
            dict: A dictionary containing the results of the function applied to mean node values.
        """
        # Get the actual function object from the common_functions dictionary
        if function not in self.common_functions:
            raise ValueError(
                f'{function} is not in a list of common functions: {list(self.common_functions.keys())}')
        func = self.common_functions[function]

        # Separate 'Time' from other parameters if it's present
        params_no_time = [param for param in parameters if param != 'Time']
        node_data = self.calculate_node_values(params_no_time)
        results = {}

        for param in parameters:
            if param == 'Time':
                # Handle 'Time' as a special case
                results['Time'] = self.mean_time()
            elif param == 'Nodes':
                continue
            elif param in node_data:
                # Apply the function to other parameters
                # print(f'{node_data[param] = }')
                results[param] = func(node_data[param])

        return results

    def mean_time(self, **settings):
        return np.mean(self.time)
