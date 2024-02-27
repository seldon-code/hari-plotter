from __future__ import annotations

from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from collections import defaultdict
from typing import (Any, Callable, Dict, Iterator, List, Optional, Tuple, Type,
                    Union)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import skfuzzy as fuzz
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .hari_graph import HariGraph


class Clustering(ABC):
    """
    Abstract base class representing a cluster. It provides a template for clustering algorithms.

    Attributes:
        clusters (List[np.ndarray]): A list of clusters, where each cluster is represented by a numpy array.
        centroids (np.ndarray): An array of centroids for the clusters.
        labels (np.ndarray): An array indicating the label of each data point.
        parameters (List[str]): A list of parameter names used for clustering.
    """
    _clustering_methods = {}

    def __init__(self, G: HariGraph):
        self.G = G

    @property
    def cluster_labels(self) -> List[str]:
        if self._cluster_labels is None:
            self._cluster_labels = [f'Cluster {i}' for i in range(
                self.get_number_of_clusters())]
        return self._cluster_labels

    def get_cluster_labels(self, **kwargs):
        return self.cluster_labels

    @cluster_labels.setter
    def cluster_labels(self, labels: List[str]):
        if len(labels) != self.get_number_of_clusters():
            raise ValueError(
                f'Labels number {len(labels)} is not equal to the number of clusters {self.get_number_of_clusters()}')
        self._cluster_labels = labels

    def reorder_labels(self, new_order: List[int]):
        current_labels = self.cluster_labels
        self._cluster_labels = [current_labels[i] for i in new_order]

    @classmethod
    def clustering_method(cls, clustering_name):
        def decorator(clustering_func):
            if clustering_name in cls._clustering_methods:
                raise ValueError(
                    f"clustering type {clustering_name} is already defined.")
            cls._clustering_methods[clustering_name] = clustering_func
            return clustering_func
        return decorator

    @abstractclassmethod
    def from_graph(cls, G: HariGraph, **kwargs) -> Clustering:
        raise NotImplementedError(
            "This method must be implemented in subclasses")

    @classmethod
    def create_clustering(cls, G: HariGraph, clustering_method: str = 'K-Means Clustering', **kwargs) -> Clustering:
        """
        Factory method that creates an instance of a subclass of `Clustering` based on the provided method name
        and applies specified scaling functions to the data before clustering.

        Args:
            clustering_method: The name of the clustering method corresponding to a subclass of `Clustering`.
            data: The data to be clustered, structured as a dictionary with the key 'data' and value as another
                dictionary mapping integers to lists of float values.
            scale: An optional dictionary where keys are parameter names and values are functions ('Linear' or 'Tanh')
                to be applied to the parameter values before clustering. If not provided, no scaling is applied.

        Returns:
            An instance of the subclass of `Clustering` that corresponds to the given method name.

        Raises:
            ValueError: If the method name is not recognized (i.e., not found in the `clustering_methods`).
        """
        if clustering_method not in cls._clustering_methods:
            raise ValueError(f"Clustering method '{clustering_method}' not recognized. "
                             f"Available methods: {cls.available_clustering_methods()}")

        # Get the subclass corresponding to the method name
        method_cls = cls._clustering_methods[clustering_method]

        # Create an instance of the subclass from the data, applying any specified scaling functions
        return method_cls.from_graph(G=G,  **kwargs)

    @abstractmethod
    def get_number_of_clusters(self) -> int:
        """
        Abstract method to get the number of clusters.
        """
        pass

    @classmethod
    def available_clustering_methods(self):
        return list(self._clustering_methods.keys())

    def get_values(self, key: Union[str, List[str]]) -> List[np.ndarray]:
        """
        Returns the values corresponding to the given parameter(s) for all points in the clusters.

        Args:
            key (Union[str, List[str]]): The parameter name or list of parameter names.
            keep_scale *bool): For the convenience, some values are kept as the values of the scale function of themselves. You might need it as it is kept or the actual values, bu default, you need the actual values.

        Returns:
            List[np.ndarray]: A list of numpy arrays, where each array corresponds to a cluster
                              and contains the values of the specified parameter(s) for each point in that cluster.
        """
        if isinstance(key, str):
            # Single parameter requested
            key = [key]

        data = self.G.gatherer.gather(param=key)

        return [np.array([[data[k][data['Nodes'].index(node)] for k in key] for node in cluster]) for cluster in self.get_cluster_mapping()]

    def mapping_dict(self):
        labels = self.cluster_labels
        mapping = self.get_cluster_mapping()
        assert len(labels) == len(mapping)
        return {node: label for label, nodes in zip(labels, mapping) for node in nodes}

    def mapping_default_dict(self):
        return defaultdict(lambda: None, self.mapping_dict)


class ParameterBasedClustering(Clustering):
    scale_funcs = {
        'Linear': {'direct': lambda x: x, 'inverse': lambda x: x},
        'Tanh': {'direct': np.tanh, 'inverse': np.arctanh}
    }

    def __init__(self, G: HariGraph, original_labels: np.ndarray, parameters: List[str], scales: List[str], cluster_indexes: np.ndarray):
        """
        Initializes the Cluster object with cluster data.

        Args:
            clustered_data (List[np.ndarray]): Shape NxM where N is number of points and M number of parameters
            cluster_indexes: A numpy array of length N shows what cluster each point from data belongs to
            original_labels (np.ndarray): A numpy array of length N representing original labels for each data point. 
            parameters (List[str]): A list of strings length M, representing the names of the parameters 
                                    or features used in clustering. These names correspond to the 
                                    dimensions/features in the data points. 
            scales (List[str]): A list of strings representing the names of the scales used for clustering. More in scale_funcs
        """
        super().__init__(G)
        self.cluster_indexes = cluster_indexes
        self.original_labels = original_labels
        self.parameters = parameters
        self.scales = scales
        self._cluster_labels = None

    def centroids(self, keep_scale: bool = False):
        centroids = self.unscaled_centroids
        if not keep_scale:
            for i, sc in enumerate(self.scales):
                centroids[:, i] = self.scale_funcs[sc]['inverse'](
                    centroids[:, i])
        return centroids

    @abstractproperty
    def unscaled_centroids(self) -> List[np.ndarray]:
        """
        A numpy array representing the centroids of the clusters.
        Each row in this array corresponds to a centroid.
        """
        raise NotImplementedError(
            "This method must be implemented in subclasses")

    @abstractmethod
    def predict_cluster(self, data_point: List[float]) -> int:
        """
        Abstract method to predict the cluster for a new data point.
        """
        pass

    def degree_of_membership(self, data_point: List[float]) -> List[float]:
        """
        Predicts the 'probability' of belonging to each cluster for a new data point.

        If the clustering method does not provide probabilities, this method
        will return a list with a 1 at the index of the assigned cluster and 0s elsewhere.

        Args:
            data_point: The new data point's parameter values as a list of floats.

        Returns:
            List[float]: A list of zeros and one one, indicating the cluster assignment.
        """
        nearest_cluster_index = self.predict_cluster(data_point)
        return [nearest_cluster_index == i for i in range(self.get_number_of_clusters())]

    @abstractmethod
    def reorder_clusters(self, new_order: List[int]):
        """
        Abstract method to reorder clusters based on a new order.
        Assumes that the new_order list contains the indices of the clusters in their new order.
        """
        pass

    @abstractmethod
    def get_cluster_mapping(self) -> List[list]:
        pass

    def get_indices_from_parameters(self, params: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Returns the indices corresponding to the given parameter(s).

        Args:
            params (Union[str, List[str]]): The parameter name or list of parameter names.

        Returns:
            Union[int, List[int]]: The index or list of indices corresponding to the given parameter(s).
            Returns None if parameter is not present
        """
        if isinstance(params, str):
            return self.parameters.index(params) if params in self.parameters else None
        else:
            return [self.parameters.index(param) if param in self.parameters else None for param in params]

    def get_parameters_from_indices(self, indices: Union[int, List[int]]) -> Union[str, List[str]]:
        """
        Returns the parameter names corresponding to the given index/indices.

        Args:
            indices (Union[int, List[int]]): The index or list of indices.

        Returns:
            Union[str, List[str]]: The parameter name or list of parameter names corresponding to the given index/indices.
        """
        if isinstance(indices, int):
            return self.parameters[indices]
        else:
            return [self.parameters[index] for index in indices]


@Clustering.clustering_method("Interval Clustering")
class ValueIntervalsClustering(ParameterBasedClustering):
    """Value Intervals clustering representation, extending the generic Clustering class."""

    def __init__(self, G: HariGraph, data: np.ndarray, parameter_boundaries: List[List[float]], original_labels: np.ndarray, parameters: List[str], scales: List[str], cluster_indexes: np.ndarray):
        super().__init__(G, original_labels, parameters, scales, cluster_indexes)
        self.parameter_boundaries = parameter_boundaries
        self.data = data
        self.indices_mapping = {}
        self.n_clusters = 0

    def get_number_of_clusters(self) -> int:
        return self.n_clusters  # Assuming self.n_clusters tracks the number of clusters

    def get_cluster_mapping(self) -> List[list]:
        """
        Maps each node to a cluster based on the parameter boundaries.

        Returns:
        - List[list]: A list of lists, where each sublist contains the nodes belonging to that cluster.
        """
        cluster_nodes = [[] for _ in range(self.get_number_of_clusters())]

        for i, point in enumerate(self.data):
            cluster_index = self.find_cluster_index(point)
            if cluster_index is not None:  # Point falls within defined boundaries
                cluster_nodes[cluster_index].append(
                    tuple(self.original_labels[i]))

        return cluster_nodes

    def find_cluster_indices(self, point: np.ndarray) -> np.ndarray:
        """
        Determines the indices of the clusters a point belongs to based on parameter boundaries.

        Args:
        - point: The data point's parameter values as a numpy array.

        Returns:
        - np.ndarray: An array of the indices of the clusters the point belongs to.
        """
        cluster_indices = np.zeros(len(self.parameter_boundaries), dtype=int)
        for i, boundaries in enumerate(self.parameter_boundaries):
            cluster_indices[i] = np.sum(point[i] < np.array(boundaries))
        return tuple(cluster_indices)

    def find_cluster_index(self, point):
        cluster_indices = self.find_cluster_indices(point)
        return self.indices_mapping.get(cluster_indices, None)

    # def flatten_cluster_indices(self, cluster_indices: np.ndarray) -> int:
    #     """
    #     Flattens the array of cluster indices into a single cluster index.

    #     Args:
    #     - cluster_indices: An array of cluster indices for each parameter.

    #     Returns:
    #     - int: The flattened cluster index representing the specific cluster across all parameters.
    #     """
    #     cluster_sizes = np.array(
    #         [len(b) + 1 for b in self.parameter_boundaries])
    #     weights = np.append(1, np.cumprod(cluster_sizes[:-1]))
    #     return np.dot(cluster_indices, weights)

    @classmethod
    def from_graph(cls, G: HariGraph, parameter_boundaries: List[List[float]], clustering_parameters: List[str], scale: Union[List[str], Dict[str, str], None] = None) -> 'ValueIntervalsClustering':
        """
        Creates an instance of valueIntervalsClustering from a HariGraph.

        Args:
            G: HariGraph.
            parameter_boundaries: List of lists, each containing the boundaries for a parameter.
            clustering_parameters: List of parameter names.
            scale: Optional scaling functions for the clustering_parameters.

        Returns:
            valueIntervalsClustering: An instance with nodes assigned to clusters based on the parameter boundaries.

        Raises:
            ValueError: If the number of clustering_parameters does not match the number of parameter boundaries.
        """
        if len(clustering_parameters) != len(parameter_boundaries):
            raise ValueError(
                "The number of clustering_parameters must match the number of parameter boundaries.")

        data = G.gatherer.gather(clustering_parameters)

        # Validate and process scale argument
        if isinstance(scale, dict):
            scale = [scale.get(param, 'Linear')
                     for param in clustering_parameters]
        elif isinstance(scale, list) and len(scale) != len(clustering_parameters):
            raise ValueError('Length mismatch in scale list')
        elif scale is None:
            scale = ['Linear'] * len(clustering_parameters)

        # Prepare data array
        data_array = np.array(
            [data[param] for param in clustering_parameters if param not in ['Time', 'Nodes']]).T

        # Remove NaN values
        valid_indices = ~np.isnan(data_array).any(axis=1)
        data_array = data_array[valid_indices]
        original_labels = np.array(data['Nodes'])[valid_indices]

        if data_array.size == 0:
            raise ValueError(
                "No data points remain after removing NaN values.")

        # Apply scaling to the clustering_parameters
        for i, sc in enumerate(scale):
            data_array[:, i] = cls.scale_funcs[sc]['direct'](data_array[:, i])

        # Initialize the clustering
        clustering = cls(G=G, data=data_array, original_labels=original_labels, parameters=clustering_parameters,
                         scales=scale, cluster_indexes=np.nan*np.zeros(data_array.shape[0]), parameter_boundaries=parameter_boundaries)
        clustering.recluster()
        return clustering

    def recluster(self):
        """
        Recalculates the cluster indices for each data point based on the current parameter boundaries.
        """
        # Iterate over each data point
        self.indices_mapping = {}
        self.n_clusters = 0

        for i, point in enumerate(self.data):
            # Determine the cluster indices for the current point across all parameters
            cluster_indices = self.find_cluster_indices(point)
            if cluster_indices not in self.indices_mapping:
                self.indices_mapping[cluster_indices] = self.n_clusters
                self.n_clusters += 1

            # Flatten these indices into a single cluster index
            # flat_cluster_index = self.flatten_cluster_indices(cluster_indices)

            # Update the cluster index for the current point
            self.cluster_indexes[i] = self.indices_mapping[cluster_indices]

    def reorder_clusters(self, new_order: List[int]):
        # Implement the logic to reorder the clusters
        pass

    @property
    def unscaled_centroids(self) -> np.ndarray:
        # Implement the logic to return centroids or representative points of clusters
        pass

    def predict_cluster(self, data_points: np.ndarray, points_scaled: bool = False) -> np.ndarray:
        """
        Predicts the cluster indices to which new data points belong based on the centroids.

        Args:
            data_points: The new data points' parameter values as a numpy array.

        Returns:
            np.ndarray: An array of indices of the closest cluster centroid to each data point.

        Raises:
            ValueError: If the dimensionality of the data points does not match that of the centroids.
        """
        # Check if the data points are of the correct dimension
        if data_points.shape[1] != len(self.parameters):
            raise ValueError(
                "Data points dimensionality does not match number of parameters")
        scaled_points = np.array(data_points)

        if not points_scaled:
            for i, sc in enumerate(self.scales):
                scaled_points[:, i] = self.scale_funcs[sc]['direct'](
                    scaled_points[:, i])

        cluster_indexes = []

        for i, point in enumerate(data_points):
            # Determine the cluster indices for the current point across all parameters
            cluster_indexes.append(self.find_cluster_index(point))

        return cluster_indexes

    # def get_values(self, key: Union[str, List[str]], keep_scale: bool = False) -> List[np.ndarray]:
    #     """
    #     Returns the values corresponding to the given parameter(s) for all points in the clusters.

    #     Args:
    #         key (Union[str, List[str]]): The parameter name or list of parameter names.
    #         keep_scale *bool): For the convenience, some values are kept as the values of the scale function of themselves. You might need it as it is kept or the actual values, bu default, you need the actual values.

    #     Returns:
    #         List[np.ndarray]: A list of numpy arrays, where each array corresponds to a cluster
    #                           and contains the values of the specified parameter(s) for each point in that cluster.
    #     """
    #     if isinstance(key, str):
    #         # Single parameter requested
    #         key = [key]

    #     if not all(k in self.parameters for k in key):
    #         raise KeyError(
    #             f"One or more requested parameters {key} are not found in the cluster parameters {self.parameters}.")

    #     param_indices = [self.parameters.index(k) for k in key]

    #     scaled_data = self.data[:, param_indices]
    #     if not keep_scale:
    #         for new_index, original_index in enumerate(param_indices):
    #             used_scale = self.scales[original_index]
    #             scaled_data[:, new_index] = self.scale_funcs[used_scale]['inverse'](
    #                 scaled_data[:, new_index])

    #     unique = np.unique(self.cluster_indexes)

    #     return [scaled_data[self.cluster_indexes == u_i, :] for u_i in unique]


@Clustering.clustering_method("K-Means Clustering")
class KMeansClustering(ParameterBasedClustering):
    """A KMeans clustering representation, extending the generic Clustering class."""

    def __init__(self, G: HariGraph, data: np.ndarray, original_labels: np.ndarray, parameters: List[str], scales: List[str], cluster_indexes: np.ndarray):

        super().__init__(G, original_labels, parameters, scales, cluster_indexes)
        self.data = data

    @property
    def unscaled_centroids(self) -> np.ndarray:
        return self._centroids.copy()

    def get_cluster_mapping(self) -> List[list]:
        cluster_nodes = [[] for _ in range(self.get_number_of_clusters())]

        for label, cluster in zip(self.original_labels, self.cluster_indexes):
            cluster_nodes[cluster].append(tuple(label))
        return cluster_nodes

    def get_number_of_clusters(self) -> int:
        """
        Get the number of clusters.

        Returns:
        - int : The number of clusters.
        """
        return len(self._centroids)

    def predict_cluster(self, data_points: np.ndarray, points_scaled: bool = False) -> np.ndarray:
        """
        Predicts the cluster indices to which new data points belong based on the centroids.

        Args:
            data_points: The new data points' parameter values as a numpy array.

        Returns:
            np.ndarray: An array of indices of the closest cluster centroid to each data point.

        Raises:
            ValueError: If the dimensionality of the data points does not match that of the centroids.
        """
        # Check if the data points are of the correct dimension
        centroids = self._centroids
        if data_points.shape[1] != centroids.shape[1]:
            raise ValueError(
                "Data points dimensionality does not match number of features in centroids.")
        scaled_points = np.array(data_points)

        if not points_scaled:
            for i, sc in enumerate(self.scales):
                scaled_points[:, i] = self.scale_funcs[sc]['direct'](
                    scaled_points[:, i])

        # Calculate the distance from each point to each centroid
        distances = cdist(scaled_points, centroids, 'euclidean')

        # Find the indices of the nearest centroid for each data point
        nearest_centroid_indices = np.argmin(distances, axis=1)
        return nearest_centroid_indices

    @classmethod
    def from_graph(cls, G: HariGraph, clustering_parameters:  Union[Tuple[str] | List[str]],  scale: Union[List[str], Dict[str, str], None] = None, n_clusters: int = 2) -> Clustering:
        """
        Creates an instance of KMeansClustering from a structured data dictionary,
        applying specified scaling to each parameter if needed.

        Args:
            G: HariGraph.
            clustering_parameters: list of clustering parameters
            scale: An optional dictionary where keys are parameter names and 
                values are functions ('Linear' or 'Tanh') to be applied to 
                the parameter values before clustering.
            n_clusters: The number of clusters to form.

        Returns:
            KMeansClustering: An instance of KMeansClustering with clusters, centroids, 
                        and labels determined from the data.

        Raises:
            ValueError: If no data points remain after removing NaN values or if
                        an unknown scaling function is specified.
        """

        data = G.gatherer.gather(clustering_parameters)

        # Extract nodes and parameter names
        nodes = data['Nodes']
        parameter_names = clustering_parameters if clustering_parameters is not None else [
            key for key in data.keys() if key != 'Nodes']

        # Validate and process scale argument
        if isinstance(scale, dict):
            scale = [scale.get(param, 'Linear') for param in parameter_names]
        elif isinstance(scale, list) and len(scale) != len(parameter_names):
            raise ValueError('Length mismatch in scale list')
        elif scale is None:
            scale = ['Linear'] * len(parameter_names)

        # print(f'{data.keys() = }')
        # Prepare data array
        data = np.array([data[param] for param in parameter_names if parameter_names not in [
                        'Time', 'Nodes']]).T

        # print(f'{data = }')

        # Remove NaN values
        valid_indices = ~np.isnan(data).any(axis=1)
        data = data[valid_indices]
        original_labels = np.array(nodes)[valid_indices]

        if data.size == 0:
            raise ValueError(
                "No data points remain after removing NaN values.")

        # Apply scaling to the parameters
        for i, sc in enumerate(scale):
            data[:, i] = cls.scale_funcs[sc]['direct'](data[:, i])

        # Perform clustering
        clustering = cls(G=G, data=data, original_labels=original_labels, parameters=parameter_names,
                         scales=scale, cluster_indexes=np.zeros(data.shape[0]))
        clustering.recluster(n_clusters=n_clusters)

        return clustering

    def recluster(self, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++',
                        n_init='auto', random_state=42)
        kmeans.fit(self.data)
        self._centroids = kmeans.cluster_centers_
        self.cluster_indexes = kmeans.labels_

    def reorder_clusters(self, new_order: List[int]):
        """
        Reorders clusters and associated information based on a new order.

        Args:
            new_order: A list containing the indices of the clusters in their new order.

        Raises:
            ValueError: If new_order does not contain all existing cluster indices.
        """
        if set(new_order) != set(range(len(self.centroids))):
            raise ValueError(
                "New order must contain all existing cluster indices.")

        self.centroids = self.centroids[new_order]
        self.clusters = [self.clusters[i] for i in new_order]
        # Create a mapping from old to new labels
        label_mapping = {old: new for new, old in enumerate(new_order)}
        self.labels = np.array([label_mapping[label] for label in self.labels])
        self.reorder_labels(new_order)

    # def get_values(self, key: Union[str, List[str]], keep_scale: bool = False) -> List[np.ndarray]:
    #     """
    #     Returns the values corresponding to the given parameter(s) for all points in the clusters.

    #     Args:
    #         key (Union[str, List[str]]): The parameter name or list of parameter names.
    #         keep_scale *bool): For the convenience, some values are kept as the values of the scale function of themselves. You might need it as it is kept or the actual values, bu default, you need the actual values.

    #     Returns:
    #         List[np.ndarray]: A list of numpy arrays, where each array corresponds to a cluster
    #                           and contains the values of the specified parameter(s) for each point in that cluster.
    #     """
    #     if isinstance(key, str):
    #         # Single parameter requested
    #         key = [key]

    #     if not all(k in self.parameters for k in key):
    #         raise KeyError(
    #             "One or more requested parameters are not found in the cluster parameters.")

    #     param_indices = [self.parameters.index(k) for k in key]

    #     scaled_data = self.data[:, param_indices]
    #     if not keep_scale:
    #         for new_index, original_index in enumerate(param_indices):
    #             used_scale = self.scales[original_index]
    #             scaled_data[:, new_index] = self.scale_funcs[used_scale]['inverse'](
    #                 scaled_data[:, new_index])

    #     unique = np.unique(self.cluster_indexes)

    #     return [scaled_data[self.cluster_indexes == u_i, :] for u_i in unique]
