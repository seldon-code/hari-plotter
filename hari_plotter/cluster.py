from __future__ import annotations

from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from typing import (Any, Callable, Dict, Iterator, List, Optional, Tuple, Type,
                    Union)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import skfuzzy as fuzz
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class Cluster(ABC):
    """
    Abstract base class representing a cluster. It provides a template for clustering algorithms.

    Attributes:
        clusters (List[np.ndarray]): A list of clusters, where each cluster is represented by a numpy array.
        centroids (np.ndarray): An array of centroids for the clusters.
        labels (np.ndarray): An array indicating the label of each data point.
        parameters (List[str]): A list of parameter names used for clustering.
    """

    clusterization_methods = {}
    scale_funcs = {
        'linear': {'direct': lambda x: x, 'inverse': lambda x: x},
        'tanh': {'direct': np.tanh, 'inverse': np.arctanh}
    }

    def __init__(self, data: np.ndarray, original_labels: np.ndarray, parameters: List[str], scales: List[str], cluster_indexes: np.ndarray):
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
        self.data = data
        self.cluster_indexes = cluster_indexes
        self.original_labels = original_labels
        self.parameters = parameters
        self.scales = scales
        self._cluster_labels = None

    @property
    def cluster_labels(self) -> List[str]:
        if self._cluster_labels is None:
            self._cluster_labels = [f'Cluster {i}' for i in range(
                self.get_number_of_clusters())]
        return self._cluster_labels

    @cluster_labels.setter
    def cluster_labels(self, labels: List[str]):
        if len(labels) != self.get_number_of_clusters():
            raise ValueError(
                f'Labels number {len(labels)} is not equal to the number of clusters {self.get_number_of_clusters()}')
        self._cluster_labels = labels

    def reorder_labels(self, new_order: List[int]):
        current_labels = self.cluster_labels
        self._cluster_labels = [current_labels[i] for i in new_order]

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

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Register each subclass in the clusterization_methods dictionary
        cls.clusterization_methods[cls.__name__] = cls

    @abstractclassmethod
    def from_data(cls, data: Dict[str, Dict[int, List[float]]], scale: Dict[str, Callable[[float], float]] = None, clusterization_parameters:  Union[Tuple(str) | None] = None) -> Cluster:
        raise NotImplementedError(
            "This method must be implemented in subclasses")

    @classmethod
    def create_cluster(cls, data: Dict[str, Dict[int, List[float]]], cluster_type: str = 'KMeansCluster',
                       scale: Dict[str, Callable[[float], float]] = None, clusterization_parameters:  Union[Tuple(str) | None] = None, **kwargs) -> Cluster:
        """
        Factory method that creates an instance of a subclass of `Cluster` based on the provided method name
        and applies specified scaling functions to the data before clustering.

        Args:
            cluster_type: The name of the clusterization method corresponding to a subclass of `Cluster`.
            data: The data to be clustered, structured as a dictionary with the key 'data' and value as another
                dictionary mapping integers to lists of float values.
            scale: An optional dictionary where keys are parameter names and values are functions ('linear' or 'tanh')
                to be applied to the parameter values before clustering. If not provided, no scaling is applied.

        Returns:
            An instance of the subclass of `Cluster` that corresponds to the given method name.

        Raises:
            ValueError: If the method name is not recognized (i.e., not found in the `clusterization_methods`).
        """
        if cluster_type not in cls.clusterization_methods:
            raise ValueError(f"Clusterization method '{cluster_type}' not recognized. "
                             f"Available methods: {list(cls.clusterization_methods.keys())}")

        # Get the subclass corresponding to the method name
        method_cls = cls.clusterization_methods[cluster_type]

        # Create an instance of the subclass from the data, applying any specified scaling functions
        return method_cls.from_data(data=data, scale=scale, clusterization_parameters=clusterization_parameters, **kwargs)

    @abstractmethod
    def get_number_of_clusters(self) -> int:
        """
        Abstract method to get the number of clusters.
        """
        pass

    @abstractmethod
    def predict_cluster(self, data_point: List[float]) -> int:
        """
        Abstract method to predict the cluster for a new data point.
        """
        pass

    @abstractmethod
    def degree_of_membership(self, data_point: List[float]) -> List[float]:
        """
        Abstract method to predict the 'probability' of belonging to each cluster for a new data point.
        """
        pass

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

    def get_values(self, key: Union[str, List[str]], keep_scale: bool = False) -> List[np.ndarray]:
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

        if not all(k in self.parameters for k in key):
            raise KeyError(
                "One or more requested parameters are not found in the cluster parameters.")

        param_indices = [self.parameters.index(k) for k in key]

        scaled_data = self.data[:, param_indices]
        if not keep_scale:
            for new_index, original_index in enumerate(param_indices):
                used_scale = self.scales[original_index]
                scaled_data[:, new_index] = self.scale_funcs[used_scale]['inverse'](
                    scaled_data[:, new_index])

        unique = np.unique(self.cluster_indexes)

        return [scaled_data[self.cluster_indexes == u_i, :] for u_i in unique]

    def get_indices_from_parameters(self, params: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Returns the indices corresponding to the given parameter(s).

        Args:
            params (Union[str, List[str]]): The parameter name or list of parameter names.

        Returns:
            Union[int, List[int]]: The index or list of indices corresponding to the given parameter(s).
        """
        if isinstance(params, str):
            return self.parameters.index(params)
        else:
            return [self.parameters.index(param) for param in params]

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


class KMeansCluster(Cluster):
    """A KMeans clustering representation, extending the generic Cluster class."""

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
    def from_data(cls, data: Dict[str, List[float]], scale: Union[List[str], Dict[str, str], None] = None, n_clusters: int = 2, clusterization_parameters:  Union[Tuple(str) | None] = None) -> Cluster:
        """
        Creates an instance of KMeansCluster from a structured data dictionary,
        applying specified scaling to each parameter if needed.

        Args:
            data: A dictionary mapping parameter names to their corresponding list of float values,
                    and 'nodes' to a list of node names/IDs.
            scale: An optional dictionary where keys are parameter names and 
                values are functions ('linear' or 'tanh') to be applied to 
                the parameter values before clustering.
            n_clusters: The number of clusters to form.

        Returns:
            KMeansCluster: An instance of KMeansCluster with clusters, centroids, 
                        and labels determined from the data.

        Raises:
            ValueError: If no data points remain after removing NaN values or if
                        an unknown scaling function is specified.
        """

        # Extract nodes and parameter names
        if 'nodes' not in data:
            raise ValueError("data must include a 'nodes' key.")
        nodes = data['nodes']
        parameter_names = clusterization_parameters if clusterization_parameters is not None else [
            key for key in data.keys() if key != 'nodes']

        # Validate and process scale argument
        if isinstance(scale, dict):
            scale = [scale.get(param, 'linear') for param in parameter_names]
        elif isinstance(scale, list) and len(scale) != len(parameter_names):
            raise ValueError('Length mismatch in scale list')
        elif scale is None:
            scale = ['linear'] * len(parameter_names)

        # print(f'{data.keys() = }')
        # Prepare data array
        data = np.array([data[param] for param in parameter_names if parameter_names not in [
                        'time', 'nodes']]).T

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
        cluster = cls(data=data, original_labels=original_labels, parameters=parameter_names,
                      scales=scale, cluster_indexes=np.zeros(data.shape[0]))
        cluster.recluster(n_clusters=n_clusters)

        return cluster

    def recluster(self, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++',
                        n_init='auto', random_state=42)
        kmeans.fit(self.data)
        self._centroids = kmeans.cluster_centers_
        self.cluster_indexes = kmeans.labels_

    def degree_of_membership(self, data_point: List[float]) -> List[float]:
        """
        Predicts the 'probability' of belonging to each cluster for a new data point.

        Since KMeans does not provide probabilities but hard assignments, this method
        will return a list with a 1 at the index of the assigned cluster and 0s elsewhere.

        Args:
            data_point: The new data point's parameter values as a list of floats.

        Returns:
            List[float]: A list of zeros and one one, indicating the cluster assignment.
        """
        nearest_cluster_index = self.predict_cluster(data_point)
        return [nearest_cluster_index == i for i in range(self.get_number_of_clusters())]

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


# class KMeansCluster(Cluster):
#     """A KMeans clustering representation, extending the generic Cluster class."""

#     def __init__(self, clusters: List[np.ndarray], centroids: np.ndarray, labels: np.ndarray, parameters: List[str]):
#         super().__init__(clusters, centroids, labels, parameters)

#     def get_number_of_clusters(self) -> int:
#         """
#         Get the number of clusters.

#         Returns:
#         - int : The number of clusters.
#         """
#         return len(self.centroids)

#     def predict_cluster(self, data_point: List[float]) -> int:
#         """
#         Predicts the cluster index to which a new data point belongs based on the centroids.

#         Args:
#             data_point: The new data point's parameter values as a list of floats.

#         Returns:
#             int: The index of the closest cluster centroid to the data point.

#         Raises:
#             ValueError: If the dimensionality of the data point does not match that of the centroids.
#         """
#         # Check if the data point is of the correct dimension
#         if len(data_point) != self.centroids.shape[1]:
#             raise ValueError(
#                 "Data point dimensionality does not match number of features in centroids.")

#         # Convert the data point to a numpy array and reshape for cdist
#         data_point_np = np.array(data_point).reshape(1, -1)

#         # Calculate the distance from this point to each centroid
#         distances = cdist(data_point_np, self.centroids, 'euclidean').flatten()

#         # Find the index of the nearest centroid
#         nearest_centroid_index = np.argmin(distances)
#         return nearest_centroid_index

#     @staticmethod
#     def is_single_cluster(points: np.ndarray):
#         """
#         Check if the given data represents a single cluster.

#         Parameters:
#         - points : np.ndarray
#             n-dimensional data points.

#         Returns:
#         - bool : True if data likely represents a single cluster, False otherwise.
#         """
#         kmeans = KMeans(n_clusters=2, init='k-means++',
#                         n_init='auto', random_state=42)
#         kmeans.fit(points)
#         cluster_labels = kmeans.labels_
#         two_cluster_score = silhouette_score(points, cluster_labels)

#         return two_cluster_score < 0.75

#     @staticmethod
#     def merge_close_clusters(centroids, labels, min_distance):
#         """
#         Merges clusters that are closer than a specified minimum distance.

#         Args:
#             centroids: Array containing centroids of clusters.
#             labels: Labels assigned to each data point.
#             min_distance: The minimum distance below which clusters are merged.

#         Returns:
#             Tuple[np.ndarray, np.ndarray]: The updated labels and centroids after merging.
#         """
#         distance_matrix = cdist(centroids, centroids, 'euclidean')
#         np.fill_diagonal(distance_matrix, np.inf)

#         while np.min(distance_matrix) < min_distance:
#             i, j = np.unravel_index(
#                 distance_matrix.argmin(), distance_matrix.shape)
#             labels[labels == j] = i
#             labels[labels > j] -= 1
#             centroids = np.delete(centroids, j, 0)
#             distance_matrix = cdist(centroids, centroids, 'euclidean')
#             np.fill_diagonal(distance_matrix, np.inf)

#         return labels, centroids

#     @staticmethod
#     def optimal_clusters(points: np.ndarray, min_distance=1e-2):
#         """
#         Finds the optimal clustering of data points and merges close clusters.

#         Args:
#             points: A NumPy array of n-dimensional data points.
#             min_distance: The minimum distance between centroids to consider clusters separate.

#         Returns:
#             Tuple[List[np.ndarray], np.ndarray, np.ndarray]: The clusters, centroids, and labels.
#         """
#         if KMeansCluster.is_single_cluster(points):
#             # Assuming points.shape[0] is the number of data points.
#             # Create a single centroid that is the mean of all points
#             # and assign all points to a single cluster with label 0.
#             centroid = np.mean(points, axis=0).reshape(
#                 1, -1)  # single centroid
#             # all points labeled as 0
#             labels = np.zeros(points.shape[0], dtype=int)
#             # all points in one cluster
#             clusters = [list(range(points.shape[0]))]
#             return clusters, centroid, labels

#         wcss = []
#         for i in range(1, 11):
#             kmeans = KMeans(n_clusters=i, init='k-means++',
#                             n_init='auto', random_state=42)
#             kmeans.fit(points)
#             wcss.append(kmeans.inertia_)

#         deltas = np.diff(wcss)
#         n = np.where(deltas == np.min(deltas))[0][0] + 2

#         kmeans = KMeans(n_clusters=n, init='k-means++',
#                         n_init='auto', random_state=42)
#         kmeans.fit(points)
#         labels, centroids = KMeansCluster.merge_close_clusters(
#             kmeans.cluster_centers_, kmeans.labels_, min_distance)

#         unique_labels = np.unique(labels)
#         clusters = [points[labels == label] for label in unique_labels]

#         return clusters, centroids, labels  # Ensure three values are returned here

#     @classmethod
#     def from_data(cls, data_dict: Dict[str, Dict[int, List[float]]],
#                   scale: Dict[str, Callable[[float], float]] = None) -> Cluster:
#         """
#         Creates an instance of KMeansCluster from a structured data dictionary,
#         applying specified scaling to each parameter if needed.

#         Args:
#             data_dict: A dictionary with a 'data' key whose value is another
#                     dictionary mapping integer node numbers to another dictionary of
#                     parameter names and their corresponding list of float values.
#             scale: An optional dictionary where keys are parameter names and
#                 values are functions ('linear' or 'tanh') to be applied to
#                 the parameter values before clustering.

#         Returns:
#             KMeansCluster: An instance of KMeansCluster with clusters, centroids,
#                         and labels determined from the data.

#         Raises:
#             ValueError: If no data points remain after removing NaN values or if
#                         an unknown scaling function is specified.
#         """
#         if scale is None:
#             scale = {}

#         # Define the scaling functions
#         scale_funcs = {
#             'linear': lambda x: x,
#             'tanh': np.tanh
#         }

#         # Ensure scale contains known functions only
#         for param, func in scale.items():
#             if func not in scale_funcs:
#                 raise ValueError(
#                     f"Unknown scale function '{func}' for parameter '{param}'.")

#         # Retrieve parameter names and initialize points array
#         parameter_names = list(next(iter(data_dict['data'].values())).keys())
#         points = []

#         # Extract and flatten data points from the nested dictionary
#         for node_data in data_dict['data'].values():
#             node_values = [node_data[param] for param in parameter_names]
#             points.append(node_values)
#         points = np.array(points).reshape(-1, len(parameter_names))

#         # Remove NaN values
#         if np.isnan(points).any():
#             points = points[~np.isnan(points).any(axis=1)]

#         # Apply scaling to the appropriate parameters
#         for i, parameter_name in enumerate(parameter_names):
#             if parameter_name in scale:
#                 func = scale_funcs[scale[parameter_name]]
#                 points[:, i] = func(points[:, i])

#         # Proceed with clustering if points has data left
#         if points.size > 0:
#             clusters, centroids, labels = cls.optimal_clusters(points)
#             # Create an instance of KMeansCluster with clusters, centroids, labels, and parameter names
#             return cls(clusters, centroids, labels, parameter_names)
#         else:
#             raise ValueError(
#                 "After removing NaN values, no data points remain for clustering.")

#     def degree_of_membership(self, data_point: List[float]) -> List[float]:
#         """
#         Predicts the 'probability' of belonging to each cluster for a new data point.

#         Since KMeans does not provide probabilities but hard assignments, this method
#         will return a list with a 1 at the index of the assigned cluster and 0s elsewhere.

#         Args:
#             data_point: The new data point's parameter values as a list of floats.

#         Returns:
#             List[float]: A list of zeros and one one, indicating the cluster assignment.
#         """
#         nearest_cluster_index = self.predict_cluster(data_point)
#         return [1.0 if i == nearest_cluster_index else 0.0 for i in range(len(self.centroids))]

#     def reorder_clusters(self, new_order: List[int]):
#         """
#         Reorders clusters and associated information based on a new order.

#         Args:
#             new_order: A list containing the indices of the clusters in their new order.

#         Raises:
#             ValueError: If new_order does not contain all existing cluster indices.
#         """
#         if set(new_order) != set(range(len(self.centroids))):
#             raise ValueError(
#                 "New order must contain all existing cluster indices.")

#         self.centroids = self.centroids[new_order]
#         self.clusters = [self.clusters[i] for i in new_order]
#         # Create a mapping from old to new labels
#         label_mapping = {old: new for new, old in enumerate(new_order)}
#         self.labels = np.array([label_mapping[label] for label in self.labels])


# class FuzzyCMeanCluster(Cluster):
#     def __init__(self, clusters: List[np.ndarray], centroids: np.ndarray, labels: np.ndarray, fuzzy_membership: np.ndarray, parameters: List[str]):
#         super().__init__(clusters, centroids, labels, parameters)
#         self.fuzzy_membership = fuzzy_membership

#     def calculate_fpc(self) -> float:
#         """
#         Calculates the Fuzzy Partition Coefficient (FPC) for the clustering.

#         Returns:
#             float: The FPC value.
#         """
#         return np.mean(np.sum(self.fuzzy_membership ** 2, axis=0) / self.fuzzy_membership.shape[1])

#     def get_number_of_clusters(self) -> int:
#         """
#         Get the number of clusters.

#         Returns:
#         - int : The number of clusters.
#         """
#         return len(self.centroids)

#     def degree_of_membership(self, data_point: List[float]) -> List[float]:
#         """
#         Predicts the fuzzy membership values for each cluster for a new data point.

#         Args:
#             data_point: The new data point's parameter values as a list of floats.

#         Returns:
#             List[float]: A list of fuzzy membership values indicating the degree of belonging to each cluster.
#         """
#         # Predict the fuzzy membership values for the data point
#         u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
#             np.array([data_point]).T, self.centroids, 2, error=0.005, maxiter=1000)

#         # Return the membership values for each cluster
#         return u.flatten().tolist()

#     def reorder_clusters(self, new_order: List[int]):
#         """
#         Reorders clusters, centroids, labels, and fuzzy membership matrix based on a new order.

#         Args:
#             new_order: A list containing the indices of the clusters in their new order.

#         Raises:
#             ValueError: If new_order does not contain all existing cluster indices.
#         """
#         if set(new_order) != set(range(len(self.centroids))):
#             raise ValueError(
#                 "New order must contain all existing cluster indices.")

#         # Reorder the centroids and fuzzy memberships
#         self.centroids = self.centroids[new_order, :]
#         self.fuzzy_membership = self.fuzzy_membership[new_order, :]

#         # Update clusters and labels based on the new order
#         self.clusters = [self.clusters[i] for i in new_order]
#         label_mapping = {old: new for new, old in enumerate(new_order)}
#         self.labels = np.array([label_mapping[label] for label in self.labels])

#         # Since fuzzy memberships are not a hard assignment, we do not need to update the fuzzy_membership array

#     @classmethod
#     def from_data(cls, data_dict: Dict[str, Dict[int, List[float]]],
#                   scale: Dict[str, Callable[[float], float]] = None) -> Cluster:
#         """
#         Creates an instance of FuzzyCMeanCluster from a structured data dictionary,
#         applying specified scaling to each parameter if needed.

#         Args:
#             data_dict: A dictionary with a 'data' key whose value is another
#                        dictionary mapping integer node numbers to another dictionary
#                        of parameter names and their corresponding list of float values.
#             scale: An optional dictionary where keys are parameter names and
#                    values are functions ('linear' or 'tanh') to be applied to
#                    the parameter values before clustering.

#         Returns:
#             FuzzyCMeanCluster: An instance of FuzzyCMeanCluster with clusters, centroids,
#                                labels, and fuzzy memberships determined from the data.

#         Raises:
#             ValueError: If no data points remain after removing NaN values or if
#                         an unknown scaling function is specified.
#         """
#         if scale is None:
#             scale = {}

#         # Define the scaling functions
#         scale_funcs = {
#             'linear': lambda x: x,
#             'tanh': np.tanh
#         }

#         # Ensure scale contains known functions only
#         for param, func in scale.items():
#             if func not in scale_funcs:
#                 raise ValueError(
#                     f"Unknown scale function '{func}' for parameter '{param}'.")

#         # Retrieve parameter names and initialize points array
#         parameter_names = list(next(iter(data_dict.values())).keys())
#         points = []

#         # Extract and flatten data points from the nested dictionary
#         for node_data in data_dict.values():
#             node_values = [node_data[param] for param in parameter_names]
#             points.append(node_values)
#         points = np.array(points).reshape(-1, len(parameter_names))

#         # Remove NaN values
#         if np.isnan(points).any():
#             points = points[~np.isnan(points).any(axis=1)]

#         # Apply scaling to the appropriate parameters
#         for i, parameter_name in enumerate(parameter_names):
#             if parameter_name in scale:
#                 func = scale_funcs[scale[parameter_name]]
#                 points[:, i] = func(points[:, i])

#         # Ensure there is data to cluster
#         if not points.size:
#             raise ValueError("No data points remain after preprocessing.")

#         # Perform Fuzzy C-Means clustering, assuming 2 clusters
#         cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
#             points.T, 2, 2, error=0.005, maxiter=1000, init=None
#         )

#         # Determine cluster membership
#         labels = np.argmax(u, axis=0)
#         clusters = [points[labels == k] for k in range(2)]

#         # Create an instance of FuzzyCMeanCluster with clusters, centroids, labels, fuzzy memberships, and parameter names
#         return cls(clusters, cntr, labels, u, parameter_names)

#     def predict_cluster(self, data_point: List[float]) -> int:
#         # In Fuzzy C-Means, the prediction would return the cluster with the highest membership.
#         u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
#             np.array([data_point]).T, self.centroids, 2, error=0.005, maxiter=1000)

#         return np.argmax(u, axis=0)[0]
