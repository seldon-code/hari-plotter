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
    """Abstract base class to represent a cluster."""
    # This dictionary will hold the subclass references with their names
    clusterization_methods = {}

    def __init__(self, clusters: List[np.ndarray], centroids: np.ndarray, labels: np.ndarray):
        self.clusters = clusters
        self.centroids = centroids
        self.labels = labels

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


class KMeansCluster(Cluster):
    """A KMeans clustering representation, extending the generic Cluster class."""

    def __init__(self, clusters: List[np.ndarray], centroids: np.ndarray, labels: np.ndarray):
        super().__init__(clusters, centroids, labels)

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

        parameter_names = next(iter(data_dict['data'].values())).keys()
        node_numbers = data_dict['data'].keys()

        # Convert the nested dictionary into a list of data points
        data_points = []
        for node_number in node_numbers:
            for parameter_name in parameter_names:
                data_points.append(
                    data_dict['data'][node_number][parameter_name])

        # Convert list to numpy array and reshape
        points = np.array(data_points).reshape(-1, len(parameter_names))

        # Check for NaN values and remove any rows that contain NaN
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
            return cls(clusters, centroids, labels)
        else:
            raise ValueError(
                "After removing NaN values, no data points remain for clustering.")


class FuzzyCMeanCluster(Cluster):
    def __init__(self, clusters: List[np.ndarray], centroids: np.ndarray, labels: np.ndarray, fuzzy_membership: np.ndarray):
        super().__init__(clusters, centroids, labels)
        self.fuzzy_membership = fuzzy_membership

    def get_number_of_clusters(self) -> int:
        """
        Get the number of clusters.

        Returns:
        - int : The number of clusters.
        """
        return len(self.centroids)

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

        # Flatten the data and remove NaN values
        flat_data = []
        for node_data in data_dict['data'].values():
            for values in node_data.values():
                flat_data.extend(values)
        flat_data = np.array(flat_data)

        # Reshape flat_data to 2D array (n_samples, n_features)
        num_features = len(next(iter(data_dict['data'].values())))
        points = flat_data.reshape(-1, num_features)

        # Remove NaN values
        if np.isnan(points).any():
            points = points[~np.isnan(points).any(axis=1)]

        # Apply scaling to the appropriate parameters
        for i, (parameter_name, scaling_function) in enumerate(scale.items()):
            if scaling_function in scale_funcs:
                points[:, i] = scale_funcs[scaling_function](points[:, i])
            else:
                raise ValueError(
                    f"Unknown scaling function '{scaling_function}' for parameter '{parameter_name}'.")

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

        return cls(clusters, cntr, labels, u)

    def predict_cluster(self, data_point: List[float]) -> int:
        # In Fuzzy C-Means, the prediction would return the cluster with the highest membership.
        u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
            np.array([data_point]).T, self.centroids, 2, error=0.005, maxiter=1000)

        return np.argmax(u, axis=0)[0]
