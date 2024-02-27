from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

import numpy as np

from .cluster import Clustering
from .hari_graph import HariGraph


class Group:
    """
    Represents a group of HariGraph images, allowing for operations like calculating mean graphs,
    performing clustering, and extracting node values. This facilitates analyzing similarities and
    differences among a collection of graph-based images, typically used in complex network analysis or
    similar domains.

    Attributes:
        images (List[HariGraph]): A list of HariGraph instances representing individual images in the group.
        time (Optional[List[float]]): Optional time values associated with each image, used for time-based analyses.
        model (Optional[Any]): An optional model associated with the group, which can be used for further analysis or processing.
        clusterings (Dict): A dictionary to store clustering results with their settings as keys to avoid recomputation.
        _mean_graph (Optional[HariGraph]): Cached mean graph of the group, calculated when needed to optimize performance.
        _nodes (Optional[set]): Cached set of nodes present in the mean graph, used to speed up node-related computations.
        _node_parameters (Optional[Dict]): Cached node parameters from the first image, ensuring consistency across the group.

    Common Functions:
        Group class provides a dictionary mapping statistical function names (e.g., 'Mean', 'Sum') to their corresponding
        numpy function objects, allowing for flexible data aggregation and analysis.
    """

    # Dictionary mapping function names to actual function objects for common statistical operations
    common_functions = {
        'Mean': np.mean,
        'Sum': np.sum,
        'Min': np.min,
        'Max': np.max,
        'Median': np.median,
        'Standard Deviation': np.std,
        'Variance': np.var,
        'Peak to Peak': np.ptp,  # Peak to Peak (Max - Min)
        # Additional functions can be added here as needed
    }

    def __init__(self, images: List[HariGraph], time=None, model=None):
        """
        Initializes a Group instance with a list of images, optional time values, and an optional model.

        Parameters:
            images (List[HariGraph]): A list of HariGraph instances to include in the group.
            time (Optional[List[float] | float]): Time values associated with each image, can be a list of floats,
                a single float (applied to all images), or None if time is not applicable.
            model (Optional[Any]): An optional model to associate with the group for advanced analyses.

        Raises:
            ValueError: If the length of the time list does not match the number of images or an invalid type is provided for time.
        """
        self.images: List[HariGraph] = images

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
            self._node_parameters = self.images[0].node_parameters
        return self._node_parameters

    @staticmethod
    def request_to_tuple(request):
        """
        Converts a request dictionary or list into a sorted, nested tuple to ensure consistent key representation,
        especially useful for caching and retrieving results based on unique request settings.

        Parameters:
            request (Union[Dict, List, Any]): The request to convert, which may include nested dictionaries and lists.

        Returns:
            Tuple: A nested tuple representation of the request, providing a hashable and consistent key for caching.
        """
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
        """
        Lazily computes or retrieves the cached mean graph of the group. The mean graph is a single HariGraph instance
        that represents the average structure and attributes of all graphs in the group.

        Returns:
            HariGraph: The mean graph of the group.
        """
        if self._mean_graph is None:
            self.initialize_mean_graph()
        return self._mean_graph

    def get_mean_graph(self, **settings) -> HariGraph:
        return self.mean_graph

    def initialize_mean_graph(self):
        self._mean_graph = self.images[0].mean_graph(
            [image.get_graph() for image in self.images])

    def clustering(self, **clustering_settings) -> Clustering:
        # Convert settings to a sorted tuple of pairs to ensure consistent ordering
        clustering_key = Group.request_to_tuple(clustering_settings)

        if clustering_key not in self.clusterings:
            # Create a new clustering instance
            clustering = Clustering.create_clustering(
                G=self.mean_graph, **clustering_settings)
            # Cache the newly created clustering
            self.clusterings[clustering_key] = {'clustering': clustering}

        return self.clusterings[clustering_key]['clustering']

    def get_clustering(self, **settings) -> Clustering:
        # print(f'{settings = }')
        if 'clustering_settings' in settings:
            return self.clustering(**(settings['clustering_settings']))
        else:
            return self.clustering()

    def clustering_graph(self, merge_remaining: bool = False,  **clustering_settings) -> HariGraph:

        clustering_key = self.request_to_tuple(clustering_settings)

        if clustering_key not in self.clusterings or 'graph' not in self.clusterings[clustering_key]:

            clustering = self.clustering(**clustering_settings)

            clustering_nodes = clustering.get_cluster_mapping()
            cluster_labels = clustering.cluster_labels

            clustering_graph = self.mean_graph.copy()
            clustering_graph.merge_clusters(
                clustering_nodes, labels=cluster_labels, merge_remaining=merge_remaining)
            self.clusterings[clustering_key]['graph'] = clustering_graph

        return self.clusterings[clustering_key]['graph']

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
