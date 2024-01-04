from __future__ import annotations

from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from typing import Any, Callable, Dict, Iterator, List, Optional, Type, Union

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

    def __init__(self, clusters: List[np.ndarray], centroids: np.ndarray, labels: np.ndarray, parameters: List[str]):
        """
        Initializes the Cluster object with cluster data.

        Args:
            clusters (List[np.ndarray]): List of numpy arrays, each representing a cluster. 
                                         Each array contains the indices or representations of 
                                         data points belonging to that cluster.
            centroids (np.ndarray): A numpy array representing the centroids of the clusters.
                                    Each row in this array corresponds to a centroid.
            labels (np.ndarray): A numpy array representing the cluster labels for each data point. 
                                 The value in each position indicates the cluster to which the 
                                 corresponding data point belongs.
            parameters (List[str]): A list of strings representing the names of the parameters 
                                    or features used in clustering. These names correspond to the 
                                    dimensions/features in the data points.
        """
        self.clusters = clusters
        self.centroids = centroids
        self.labels = labels
        self.parameters = parameters

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Register each subclass in the clusterization_methods dictionary
        cls.clusterization_methods[cls.__name__] = cls

    @abstractclassmethod
    def from_data(cls, data_dict: Dict[str, Dict[int, List[float]]]) -> Cluster:
        raise NotImplementedError(
            "This method must be implemented in subclasses")

    @classmethod
    def create_cluster(cls, method_name: str, data: Dict[str, Dict[int, List[float]]],
                       scale: Dict[str, Callable[[float], float]] = None) -> Cluster:
        """
        Factory method that creates an instance of a subclass of `Cluster` based on the provided method name
        and applies specified scaling functions to the data before clustering.

        Args:
            method_name: The name of the clusterization method corresponding to a subclass of `Cluster`.
            data: The data to be clustered, structured as a dictionary with the key 'data' and value as another
                dictionary mapping integers to lists of float values.
            scale: An optional dictionary where keys are parameter names and values are functions ('linear' or 'tanh')
                to be applied to the parameter values before clustering. If not provided, no scaling is applied.

        Returns:
            An instance of the subclass of `Cluster` that corresponds to the given method name.

        Raises:
            ValueError: If the method name is not recognized (i.e., not found in the `clusterization_methods`).
        """
        if method_name not in cls.clusterization_methods:
            raise ValueError(f"Clusterization method '{method_name}' not recognized. "
                             f"Available methods: {list(cls.clusterization_methods.keys())}")

        # Get the subclass corresponding to the method name
        method_cls = cls.clusterization_methods[method_name]

        # Create an instance of the subclass from the data, applying any specified scaling functions
        return method_cls.from_data(data, scale)

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

    def __getitem__(self, key: Union[str, List[str]]) -> List[np.ndarray]:
        """
        Returns the values corresponding to the given parameter(s) for all points in the clusters.

        Args:
            key (Union[str, List[str]]): The parameter name or list of parameter names.

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

        return [cluster[:, param_indices] for cluster in self.clusters]

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

    def __init__(self, clusters: List[np.ndarray], centroids: np.ndarray, labels: np.ndarray, parameters: List[str]):
        super().__init__(clusters, centroids, labels, parameters)

    def get_number_of_clusters(self) -> int:
        """
        Get the number of clusters.

        Returns:
        - int : The number of clusters.
        """
        return len(self.centroids)

    def predict_cluster(self, data_point: List[float]) -> int:
        """
        Predicts the cluster index to which a new data point belongs based on the centroids.

        Args:
            data_point: The new data point's parameter values as a list of floats.

        Returns:
            int: The index of the closest cluster centroid to the data point.

        Raises:
            ValueError: If the dimensionality of the data point does not match that of the centroids.
        """
        # Check if the data point is of the correct dimension
        if len(data_point) != self.centroids.shape[1]:
            raise ValueError(
                "Data point dimensionality does not match number of features in centroids.")

        # Convert the data point to a numpy array and reshape for cdist
        data_point_np = np.array(data_point).reshape(1, -1)

        # Calculate the distance from this point to each centroid
        distances = cdist(data_point_np, self.centroids, 'euclidean').flatten()

        # Find the index of the nearest centroid
        nearest_centroid_index = np.argmin(distances)
        return nearest_centroid_index

    @staticmethod
    def is_single_cluster(points: np.ndarray):
        """
        Check if the given data represents a single cluster.

        Parameters:
        - points : np.ndarray
            n-dimensional data points.

        Returns:
        - bool : True if data likely represents a single cluster, False otherwise.
        """
        kmeans = KMeans(n_clusters=2, init='k-means++',
                        n_init='auto', random_state=42)
        kmeans.fit(points)
        cluster_labels = kmeans.labels_
        two_cluster_score = silhouette_score(points, cluster_labels)

        return two_cluster_score < 0.75

    @staticmethod
    def merge_close_clusters(centroids, labels, min_distance):
        """
        Merges clusters that are closer than a specified minimum distance.

        Args:
            centroids: Array containing centroids of clusters.
            labels: Labels assigned to each data point.
            min_distance: The minimum distance below which clusters are merged.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The updated labels and centroids after merging.
        """
        distance_matrix = cdist(centroids, centroids, 'euclidean')
        np.fill_diagonal(distance_matrix, np.inf)

        while np.min(distance_matrix) < min_distance:
            i, j = np.unravel_index(
                distance_matrix.argmin(), distance_matrix.shape)
            labels[labels == j] = i
            labels[labels > j] -= 1
            centroids = np.delete(centroids, j, 0)
            distance_matrix = cdist(centroids, centroids, 'euclidean')
            np.fill_diagonal(distance_matrix, np.inf)

        return labels, centroids

    @staticmethod
    def optimal_clusters(points: np.ndarray, min_distance=1e-2):
        """
        Finds the optimal clustering of data points and merges close clusters.

        Args:
            points: A NumPy array of n-dimensional data points.
            min_distance: The minimum distance between centroids to consider clusters separate.

        Returns:
            Tuple[List[np.ndarray], np.ndarray, np.ndarray]: The clusters, centroids, and labels.
        """
        if KMeansCluster.is_single_cluster(points):
            # Assuming points.shape[0] is the number of data points.
            # Create a single centroid that is the mean of all points
            # and assign all points to a single cluster with label 0.
            centroid = np.mean(points, axis=0).reshape(
                1, -1)  # single centroid
            # all points labeled as 0
            labels = np.zeros(points.shape[0], dtype=int)
            # all points in one cluster
            clusters = [list(range(points.shape[0]))]
            return clusters, centroid, labels

        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++',
                            n_init='auto', random_state=42)
            kmeans.fit(points)
            wcss.append(kmeans.inertia_)

        deltas = np.diff(wcss)
        n = np.where(deltas == np.min(deltas))[0][0] + 2

        kmeans = KMeans(n_clusters=n, init='k-means++',
                        n_init='auto', random_state=42)
        kmeans.fit(points)
        labels, centroids = KMeansCluster.merge_close_clusters(
            kmeans.cluster_centers_, kmeans.labels_, min_distance)

        unique_labels = np.unique(labels)
        clusters = [points[labels == label] for label in unique_labels]

        return clusters, centroids, labels  # Ensure three values are returned here

    @classmethod
    def from_data(cls, data_dict: Dict[str, Dict[int, List[float]]],
                  scale: Dict[str, Callable[[float], float]] = None) -> Cluster:
        """
        Creates an instance of KMeansCluster from a structured data dictionary,
        applying specified scaling to each parameter if needed.

        Args:
            data_dict: A dictionary with a 'data' key whose value is another 
                    dictionary mapping integer node numbers to another dictionary of 
                    parameter names and their corresponding list of float values.
            scale: An optional dictionary where keys are parameter names and 
                values are functions ('linear' or 'tanh') to be applied to 
                the parameter values before clustering.

        Returns:
            KMeansCluster: An instance of KMeansCluster with clusters, centroids, 
                        and labels determined from the data.

        Raises:
            ValueError: If no data points remain after removing NaN values or if
                        an unknown scaling function is specified.
        """
        if scale is None:
            scale = {}

        # Define the scaling functions
        scale_funcs = {
            'linear': lambda x: x,
            'tanh': np.tanh
        }

        # Ensure scale contains known functions only
        for param, func in scale.items():
            if func not in scale_funcs:
                raise ValueError(
                    f"Unknown scale function '{func}' for parameter '{param}'.")

        # Retrieve parameter names and initialize points array
        parameter_names = list(next(iter(data_dict['data'].values())).keys())
        points = []

        # Extract and flatten data points from the nested dictionary
        for node_data in data_dict['data'].values():
            node_values = [node_data[param] for param in parameter_names]
            points.append(node_values)
        points = np.array(points).reshape(-1, len(parameter_names))

        # Remove NaN values
        if np.isnan(points).any():
            points = points[~np.isnan(points).any(axis=1)]

        # Apply scaling to the appropriate parameters
        for i, parameter_name in enumerate(parameter_names):
            if parameter_name in scale:
                func = scale_funcs[scale[parameter_name]]
                points[:, i] = func(points[:, i])

        # Proceed with clustering if points has data left
        if points.size > 0:
            clusters, centroids, labels = cls.optimal_clusters(points)
            # Create an instance of KMeansCluster with clusters, centroids, labels, and parameter names
            return cls(clusters, centroids, labels, parameter_names)
        else:
            raise ValueError(
                "After removing NaN values, no data points remain for clustering.")

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
        return [1.0 if i == nearest_cluster_index else 0.0 for i in range(len(self.centroids))]

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


class FuzzyCMeanCluster(Cluster):
    def __init__(self, clusters: List[np.ndarray], centroids: np.ndarray, labels: np.ndarray, fuzzy_membership: np.ndarray, parameters: List[str]):
        super().__init__(clusters, centroids, labels, parameters)
        self.fuzzy_membership = fuzzy_membership

    def calculate_fpc(self) -> float:
        """
        Calculates the Fuzzy Partition Coefficient (FPC) for the clustering.

        Returns:
            float: The FPC value.
        """
        return np.mean(np.sum(self.fuzzy_membership ** 2, axis=0) / self.fuzzy_membership.shape[1])

    def get_number_of_clusters(self) -> int:
        """
        Get the number of clusters.

        Returns:
        - int : The number of clusters.
        """
        return len(self.centroids)

    def degree_of_membership(self, data_point: List[float]) -> List[float]:
        """
        Predicts the fuzzy membership values for each cluster for a new data point.

        Args:
            data_point: The new data point's parameter values as a list of floats.

        Returns:
            List[float]: A list of fuzzy membership values indicating the degree of belonging to each cluster.
        """
        # Predict the fuzzy membership values for the data point
        u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
            np.array([data_point]).T, self.centroids, 2, error=0.005, maxiter=1000)

        # Return the membership values for each cluster
        return u.flatten().tolist()

    def reorder_clusters(self, new_order: List[int]):
        """
        Reorders clusters, centroids, labels, and fuzzy membership matrix based on a new order.

        Args:
            new_order: A list containing the indices of the clusters in their new order.

        Raises:
            ValueError: If new_order does not contain all existing cluster indices.
        """
        if set(new_order) != set(range(len(self.centroids))):
            raise ValueError(
                "New order must contain all existing cluster indices.")

        # Reorder the centroids and fuzzy memberships
        self.centroids = self.centroids[new_order, :]
        self.fuzzy_membership = self.fuzzy_membership[new_order, :]

        # Update clusters and labels based on the new order
        self.clusters = [self.clusters[i] for i in new_order]
        label_mapping = {old: new for new, old in enumerate(new_order)}
        self.labels = np.array([label_mapping[label] for label in self.labels])

        # Since fuzzy memberships are not a hard assignment, we do not need to update the fuzzy_membership array

    @classmethod
    def from_data(cls, data_dict: Dict[str, Dict[int, List[float]]],
                  scale: Dict[str, Callable[[float], float]] = None) -> Cluster:
        """
        Creates an instance of FuzzyCMeanCluster from a structured data dictionary,
        applying specified scaling to each parameter if needed.

        Args:
            data_dict: A dictionary with a 'data' key whose value is another 
                       dictionary mapping integer node numbers to another dictionary 
                       of parameter names and their corresponding list of float values.
            scale: An optional dictionary where keys are parameter names and 
                   values are functions ('linear' or 'tanh') to be applied to 
                   the parameter values before clustering.

        Returns:
            FuzzyCMeanCluster: An instance of FuzzyCMeanCluster with clusters, centroids, 
                               labels, and fuzzy memberships determined from the data.

        Raises:
            ValueError: If no data points remain after removing NaN values or if
                        an unknown scaling function is specified.
        """
        if scale is None:
            scale = {}

        # Define the scaling functions
        scale_funcs = {
            'linear': lambda x: x,
            'tanh': np.tanh
        }

        # Ensure scale contains known functions only
        for param, func in scale.items():
            if func not in scale_funcs:
                raise ValueError(
                    f"Unknown scale function '{func}' for parameter '{param}'.")

        # Retrieve parameter names and initialize points array
        parameter_names = list(next(iter(data_dict['data'].values())).keys())
        points = []

        # Extract and flatten data points from the nested dictionary
        for node_data in data_dict['data'].values():
            node_values = [node_data[param] for param in parameter_names]
            points.append(node_values)
        points = np.array(points).reshape(-1, len(parameter_names))

        # Remove NaN values
        if np.isnan(points).any():
            points = points[~np.isnan(points).any(axis=1)]

        # Apply scaling to the appropriate parameters
        for i, parameter_name in enumerate(parameter_names):
            if parameter_name in scale:
                func = scale_funcs[scale[parameter_name]]
                points[:, i] = func(points[:, i])

        # Ensure there is data to cluster
        if not points.size:
            raise ValueError("No data points remain after preprocessing.")

        # Perform Fuzzy C-Means clustering, assuming 2 clusters
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            points.T, 2, 2, error=0.005, maxiter=1000, init=None
        )

        # Determine cluster membership
        labels = np.argmax(u, axis=0)
        clusters = [points[labels == k] for k in range(2)]

        # Create an instance of FuzzyCMeanCluster with clusters, centroids, labels, fuzzy memberships, and parameter names
        return cls(clusters, cntr, labels, u, parameter_names)

    def predict_cluster(self, data_point: List[float]) -> int:
        # In Fuzzy C-Means, the prediction would return the cluster with the highest membership.
        u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
            np.array([data_point]).T, self.centroids, 2, error=0.005, maxiter=1000)

        return np.argmax(u, axis=0)[0]
