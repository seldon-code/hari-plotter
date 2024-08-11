from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import (Any, Union)

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

from .graph import Graph


class Clustering(ABC):
    """
    Abstract base class representing a cluster. It provides a template for clustering algorithms.

    Attributes:
        clusters (list[np.ndarray]): A list of clusters, where each cluster is represented by a numpy array.
        centroids (np.ndarray): An array of centroids for the clusters.
        labels (np.ndarray): An array indicating the label of each data point.
        parameters (list[str]): A list of parameter names used for clustering.
    """
    _clustering_methods = {}

    def __init__(self, G: Graph, node_ids: np.ndarray, cluster_indexes: np.ndarray):
        '''
        node_ids (np.ndarray): A numpy array of length N (length of nodes) representing node ids for each data point.
        cluster_indexes (np.ndarray): A numpy array of length N (length of nodes) shows what cluster each point from data belongs to
        '''
        self.G = G
        self.cluster_indexes: np.ndarray = cluster_indexes
        self.node_ids: np.ndarray = node_ids

    @property
    def cluster_labels(self) -> list[str]:
        if self._cluster_labels is None:
            self._cluster_labels = [f'Cluster {i}' for i in range(
                self.get_number_of_clusters())]
        return self._cluster_labels

    def get_cluster_labels(self, **kwargs) -> list[str]:
        return self.cluster_labels

    @cluster_labels.setter
    def cluster_labels(self, labels: list[str]):
        if len(labels) != self.get_number_of_clusters():
            raise ValueError(
                f'Labels number {len(labels)} is not equal to the number of clusters {self.get_number_of_clusters()}')
        self._cluster_labels = labels

    def label_to_index(self, label: str) -> int:
        return self.cluster_labels.index(label)

    def reorder_labels(self, new_order: list[int]):
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

    @classmethod
    @abstractmethod
    def from_graph(cls, G: Graph, **kwargs) -> Clustering:
        raise NotImplementedError(
            "This method must be implemented in subclasses")

    @classmethod
    def create_clustering(cls, G: Graph, clustering_method: str = 'K-Means Clustering', **kwargs) -> Clustering:
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
    def available_clustering_methods(self) -> list[str]:
        return list(self._clustering_methods.keys())

    def get_values(self, key: Union[str, list[str]]) -> list[np.ndarray]:
        """
        Returns the values corresponding to the given parameter(s) for all points in the clusters.

        Args:
            key (Union[str, list[str]]): The parameter name or list of parameter names.
            keep_scale *bool): For the convenience, some values are kept as the values of the scale function of themselves. You might need it as it is kept or the actual values, bu default, you need the actual values.

        Returns:
            list[np.ndarray]: A list of numpy arrays, where each array corresponds to a cluster
                              and contains the values of the specified parameter(s) for each point in that cluster.
        """
        if isinstance(key, str):
            # Single parameter requested
            key = [key]

        data = self.G.gatherer.gather(param=key)

        return [np.array([[data[k][data['Nodes'].index(node)] for k in key] for node in cluster]) for cluster in self.labels_nodes_dict()]

    def nodes_by_index(self, index: int) -> tuple[tuple[int]]:
        """
        Returns the nodes that are in the cluster with the given label
        """
        return self.node_ids[self.cluster_indexes == index]

    def nodes_by_label(self, label: str) -> tuple[tuple[int]]:
        """
        Returns the nodes that are in the cluster with the given label
        """
        return self.nodes_by_index(self.label_to_index(label))

    def labels_nodes_dict(self) -> dict[str, tuple[tuple[int]]]:
        return {cluster_label: self.nodes_by_label(cluster_label) for cluster_label in self.cluster_labels}

    def nodes_labels_dict(self):
        ln_dict = self.labels_nodes_dict()

        return {tuple(node): label for label, nodes in ln_dict.items() for node in nodes}

    def nodes_labels_default_dict(self):
        return defaultdict(lambda: None, self.nodes_labels_dict())


class ParameterBasedClustering(Clustering):
    scale_funcs = {
        'Linear': {'direct': lambda x: x, 'inverse': lambda x: x},
        'Tanh': {'direct': np.tanh, 'inverse': np.arctanh}
    }

    def __init__(self, G: Graph, node_ids: np.ndarray, cluster_indexes: np.ndarray, parameters: list[str], scales: list[str]):
        """
        Initializes the Cluster object with cluster data.

        Args:
            cluster_indexes: A numpy array of length N shows what cluster each point from data belongs to
            node_ids (np.ndarray): A numpy array of length N representing node ids for each data point. 
            parameters (list[str]): A list of strings length M, representing the names of the parameters 
                                    or features used in clustering. These names correspond to the 
                                    dimensions/features in the data points. 
            scales (list[str]): A list of strings representing the names of the scales used for clustering. More in scale_funcs
        """
        super().__init__(G, cluster_indexes=cluster_indexes, node_ids=node_ids)
        self.parameters: list[str] = parameters
        self.scales: list[str] = scales
        self._cluster_labels: list[str] = None

    def centroids(self, keep_scale: bool = False):
        centroids = self.unscaled_centroids()
        if not keep_scale:
            for i, sc in enumerate(self.scales):
                centroids[:, i] = self.scale_funcs[sc]['inverse'](
                    centroids[:, i])
        return centroids

    @abstractmethod
    def unscaled_centroids(self) -> list[np.ndarray]:
        """
        A numpy array representing the centroids of the clusters.
        Each row in this array corresponds to a centroid.
        """
        raise NotImplementedError(
            "This method must be implemented in subclasses")

    @abstractmethod
    def predict_cluster(self, data_point: list[float], parameters: None | tuple[str] = None) -> int:
        """
        Abstract method to predict the cluster for a new data point.
        """
        pass

    def degree_of_membership(self, data_point: list[float]) -> list[float]:
        """
        Predicts the 'probability' of belonging to each cluster for a new data point.

        If the clustering method does not provide probabilities, this method
        will return a list with a 1 at the index of the assigned cluster and 0s elsewhere.

        Args:
            data_point: The new data point's parameter values as a list of floats.

        Returns:
            list[float]: A list of zeros and one one, indicating the cluster assignment.
        """
        nearest_cluster_index = self.predict_cluster(data_point)
        return [nearest_cluster_index == i for i in range(self.get_number_of_clusters())]

    @abstractmethod
    def reorder_clusters(self, new_order: list[int]):
        """
        Abstract method to reorder clusters based on a new order.
        Assumes that the new_order list contains the indices of the clusters in their new order.
        """
        pass

    def get_indices_from_parameters(self, params: Union[str, list[str]]) -> Union[int, list[int]]:
        """
        Returns the indices corresponding to the given parameter(s).

        Args:
            params (Union[str, list[str]]): The parameter name or list of parameter names.

        Returns:
            Union[int, list[int]]: The index or list of indices corresponding to the given parameter(s).
            Returns None if parameter is not present
        """
        if isinstance(params, str):
            return self.parameters.index(params) if params in self.parameters else None
        else:
            return [self.parameters.index(param) if param in self.parameters else None for param in params]

    def get_parameters_from_indices(self, indices: Union[int, list[int]]) -> Union[str, list[str]]:
        """
        Returns the parameter names corresponding to the given index/indices.

        Args:
            indices (Union[int, list[int]]): The index or list of indices.

        Returns:
            Union[str, list[str]]: The parameter name or list of parameter names corresponding to the given index/indices.
        """
        if isinstance(indices, int):
            return self.parameters[indices]
        else:
            return [self.parameters[index] for index in indices]

    def prepare_data_point_for_prediction(self, data_points, parameters):
        if set(self.parameters) > set(parameters):
            raise ValueError(
                "The provided parameters must include all parameters used for clustering.")

        return data_points[:, [self.get_indices_from_parameters(param) for param in parameters if param in self.parameters]]


@Clustering.clustering_method("Interval Clustering")
class ValueIntervalsClustering(ParameterBasedClustering):
    """Value Intervals clustering representation, extending the generic Clustering class."""

    def __init__(self, G: Graph, data: np.ndarray, parameter_boundaries: list[list[float]], node_ids: np.ndarray, parameters: list[str], scales: list[str], cluster_indexes: np.ndarray):
        super().__init__(G, node_ids=node_ids, parameters=parameters,
                         scales=scales, cluster_indexes=cluster_indexes)
        self.parameter_boundaries = parameter_boundaries
        self.data = data
        self._indices_mapping = {}
        self.n_clusters = 0

    def get_number_of_clusters(self) -> int:
        return self.n_clusters  # Assuming self.n_clusters tracks the number of clusters

    def find_cluster_indices_on_grid(self, point: np.ndarray) -> np.ndarray:
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
        return cluster_indices

    @classmethod
    def from_graph(cls, G: Graph, parameter_boundaries: list[list[float]], clustering_parameters: list[str], scale: Union[list[str], dict[str, str], None] = None) -> 'ValueIntervalsClustering':
        """
        Creates an instance of valueIntervalsClustering from a HariGraph.

        Args:
            G: HariGraph.
            parameter_boundaries: list of lists, each containing the boundaries for a parameter.
            clustering_parameters: list of parameter names.
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
        node_ids = np.array(data['Nodes'])[valid_indices]

        if data_array.size == 0:
            raise ValueError(
                "No data points remain after removing NaN values.")

        # Apply scaling to the clustering_parameters
        for i, sc in enumerate(scale):
            data_array[:, i] = cls.scale_funcs[sc]['direct'](data_array[:, i])

        # Initialize the clustering
        clustering = cls(G=G, data=data_array, node_ids=node_ids, parameters=clustering_parameters,
                         scales=scale, cluster_indexes=np.nan*np.zeros(data_array.shape[0]), parameter_boundaries=parameter_boundaries)
        clustering.recluster()
        return clustering

    def recluster(self):
        """
        Recalculates the cluster indices for each data point based on the current parameter boundaries.
        """
        # Iterate over each data point
        self._indices_mapping = {}
        self.n_clusters = 0

        for i, point in enumerate(self.data):
            # Determine the cluster indices for the current point across all parameters
            cluster_indices = tuple(self.find_cluster_indices_on_grid(point))
            if cluster_indices not in self._indices_mapping:
                self._indices_mapping[cluster_indices] = self.n_clusters
                self.n_clusters += 1

            # Flatten these indices into a single cluster index
            # flat_cluster_index = self.flatten_cluster_indices(cluster_indices)

            # Update the cluster index for the current point
            self.cluster_indexes[i] = self._indices_mapping[cluster_indices]

    def reorder_clusters(self, new_order: list[int]):
        # Implement the logic to reorder the clusters
        pass

    def unscaled_centroids(self) -> np.ndarray:
        return np.array([np.mean(self.data[self.cluster_indexes == i], axis=0) for i in range(self.n_clusters)])

    def predict_cluster(self, data_points: np.ndarray, points_scaled: bool = False, parameters: None | tuple[str] = None) -> np.ndarray:
        """
        Predicts the cluster indices to which new data points belong based on the centroids.

        Args:
            data_points: The new data points' parameter values as a numpy array.

        Returns:
            np.ndarray: An array of indices of the closest cluster centroid to each data point.

        Raises:
            ValueError: If the dimensionality of the data points does not match that of the centroids.
        """

        if parameters is not None:
            data_points = self.prepare_data_point_for_prediction(
                data_points, parameters)

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

        return np.array(cluster_indexes)

    def find_cluster_indices_on_grid(self, point: np.ndarray) -> np.ndarray:
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
        return cluster_indices

    def find_cluster_index(self, point: np.ndarray) -> Union[int, None]:
        """
        Identifies the cluster index for a given point based on the parameter boundaries.

        Args:
        - point: The data point's parameter values as a numpy array.

        Returns:
        - int or None: The cluster index if the cell is a cluster, otherwise None.
        """
        cluster_indices = tuple(self.find_cluster_indices_on_grid(point))
        return self._indices_mapping.get(cluster_indices, None)


@Clustering.clustering_method("K-Means Clustering")
class KMeansClustering(ParameterBasedClustering):
    """A KMeans clustering representation, extending the generic Clustering class."""

    def __init__(self, G: Graph, data: np.ndarray, node_ids: np.ndarray, parameters: list[str], scales: list[str], cluster_indexes: np.ndarray):
        super().__init__(G, node_ids=node_ids, parameters=parameters,
                         scales=scales, cluster_indexes=cluster_indexes)
        self.data = data

    def unscaled_centroids(self) -> np.ndarray:
        return self._centroids.copy()

    def get_number_of_clusters(self) -> int:
        """
        Get the number of clusters.

        Returns:
        - int : The number of clusters.
        """
        return len(self._centroids)

    def predict_cluster(self, data_points: np.ndarray, points_scaled: bool = False, parameters: None | tuple[str] = None) -> np.ndarray:
        """
        Predicts the cluster indices to which new data points belong based on the centroids.

        Args:
            data_points: The new data points' parameter values as a numpy array.

        Returns:
            np.ndarray: An array of indices of the closest cluster centroid to each data point.

        Raises:
            ValueError: If the dimensionality of the data points does not match that of the centroids.
        """

        if parameters is not None:
            data_points = self.prepare_data_point_for_prediction(
                data_points, parameters)

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

    def recluster(self, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++',
                        n_init='auto', random_state=42)
        kmeans.fit(self.data)
        self._centroids = kmeans.cluster_centers_
        self.cluster_indexes = kmeans.labels_

    def reorder_clusters(self, new_order: list[int]):
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

    def calculate_wcss(self, max_clusters: int = 10) -> list[float]:
        """
        Calculate the within-cluster sum of squares (WCSS) for different numbers of clusters.

        Args:
            max_clusters: The maximum number of clusters to consider.

        Returns:
            list[float]: A list of WCSS values for each number of clusters.
        """
        wcss = []
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, init='k-means++',
                            n_init='auto', random_state=42)
            kmeans.fit(self.data)
            wcss.append(kmeans.inertia_)
        return wcss

    def plot_elbow_method(self, max_clusters: int = 10):
        """
        Plot the WCSS values to use the Elbow method for determining the optimal number of clusters.

        Args:
            max_clusters: The maximum number of clusters to consider.
        """
        wcss = self.calculate_wcss(max_clusters)
        plt.plot(range(1, max_clusters + 1), wcss, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.title('Elbow Method')
        plt.show()

    def calculate_silhouette_scores(self, max_clusters: int = 10) -> list[float]:
        """
        Calculate the silhouette scores for different numbers of clusters.

        Args:
            max_clusters: The maximum number of clusters to consider.

        Returns:
            list[float]: A list of silhouette scores for each number of clusters.
        """
        silhouette_scores = []
        for i in range(2, max_clusters + 1):  # Silhouette score is undefined for 1 cluster
            kmeans = KMeans(n_clusters=i, init='k-means++',
                            n_init='auto', random_state=42)
            kmeans.fit(self.data)
            score = silhouette_score(self.data, kmeans.labels_)
            silhouette_scores.append(score)
        return silhouette_scores

    def plot_silhouette_scores(self, max_clusters: int = 10):
        """
        Plot the silhouette scores to help determine the optimal number of clusters.

        Args:
            max_clusters: The maximum number of clusters to consider.
        """
        scores = self.calculate_silhouette_scores(max_clusters)
        plt.plot(range(2, max_clusters + 1), scores, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score Method')
        plt.show()

    def optimal_number_of_clusters(self, method: str = 'silhouette', max_clusters: int = 10) -> int:
        """
        Determine the optimal number of clusters using the specified method.

        Args:
            method: The method to use ('elbow' or 'silhouette').
            max_clusters: The maximum number of clusters to consider.

        Returns:
            int: The optimal number of clusters.
        """
        if method == 'elbow':
            wcss = self.calculate_wcss(max_clusters)
            # Find the "elbow" point where the WCSS decreases at a slower rate
            # A more advanced implementation could use the "knee point" detection algorithm
            diffs = np.diff(wcss)
            second_diffs = np.diff(diffs)
            # Adding 2 because np.diff reduces the array length by 1 each time
            optimal_clusters = np.argmax(second_diffs) + 2
        elif method == 'silhouette':
            scores = self.calculate_silhouette_scores(max_clusters)
            # Adding 2 because silhouette score calculation starts at 2 clusters
            optimal_clusters = np.argmax(scores) + 2
        else:
            raise ValueError("Unknown method. Use 'elbow' or 'silhouette'.")

        return optimal_clusters

    @classmethod
    def from_graph(cls, G: Graph, clustering_parameters: Union[tuple[str] | list[str]], scale: Union[list[str], dict[str, str], None] = None, n_clusters: int = -1, method: str = 'silhouette', max_clusters: int = 10) -> Clustering:
        """
        Creates an instance of KMeansClustering from a structured data dictionary,
        applying specified scaling to each parameter if needed.

        Args:
            G: HariGraph.
            clustering_parameters: list of clustering parameters
            scale: An optional dictionary where keys are parameter names and 
                values are functions ('Linear' or 'Tanh') to be applied to 
                the parameter values before clustering.
            n_clusters: The number of clusters to form. If -1, the optimal number of clusters will be determined.
            method: The method to use for determining the optimal number of clusters ('elbow' or 'silhouette').
            max_clusters: The maximum number of clusters to consider.

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

        # Prepare data array
        data = np.array([data[param] for param in parameter_names if parameter_names not in [
                        'Time', 'Nodes']]).T

        # Remove NaN values
        valid_indices = ~np.isnan(data).any(axis=1)
        data = data[valid_indices]
        node_ids = np.array(nodes)[valid_indices]

        if data.size == 0:
            raise ValueError(
                "No data points remain after removing NaN values.")

        # Apply scaling to the parameters
        for i, sc in enumerate(scale):
            data[:, i] = cls.scale_funcs[sc]['direct'](data[:, i])

        # Determine the number of clusters if not provided
        if n_clusters == -1:
            temp_clustering = cls(G=G, data=data, node_ids=node_ids, parameters=parameter_names,
                                  scales=scale, cluster_indexes=np.zeros(data.shape[0]))
            n_clusters = temp_clustering.optimal_number_of_clusters(
                method=method, max_clusters=max_clusters)

        # Perform clustering
        clustering = cls(G=G, data=data, node_ids=node_ids, parameters=parameter_names,
                         scales=scale, cluster_indexes=np.zeros(data.shape[0]))
        clustering.recluster(n_clusters=n_clusters)

        return clustering


@Clustering.clustering_method("DBSCAN Clustering")
class DBSCANClustering(ParameterBasedClustering):
    """A DBSCAN clustering representation, extending the generic Clustering class."""

    def __init__(self, G: Graph, data: np.ndarray, node_ids: np.ndarray, parameters: list[str], scales: list[str], cluster_indexes: np.ndarray):
        super().__init__(G, node_ids=node_ids, parameters=parameters,
                         scales=scales, cluster_indexes=cluster_indexes)
        self.data = data

    def get_number_of_clusters(self) -> int:
        """
        Get the number of clusters.

        Returns:
        - int : The number of clusters (excluding noise points).
        """
        return len(set(self.cluster_indexes)) - (1 if -1 in self.cluster_indexes else 0)

    @classmethod
    def from_graph(cls, G: Graph, clustering_parameters: Union[tuple[str], list[str]], scale: Union[list[str], dict[str, str], None] = None, eps: float = 0.5, min_samples: int = 5) -> Clustering:
        """
        Creates an instance of DBSCANClustering from a structured data dictionary,
        applying specified scaling to each parameter if needed.

        Args:
            G: HariGraph.
            clustering_parameters: list of clustering parameters
            scale: An optional dictionary where keys are parameter names and 
                values are functions ('Linear' or 'Tanh') to be applied to 
                the parameter values before clustering.
            eps: The maximum distance between two samples for them to be considered as in the same neighborhood.
            min_samples: The number of samples in a neighborhood for a point to be considered as a core point.

        Returns:
            DBSCANClustering: An instance of DBSCANClustering with clusters and labels determined from the data.

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

        # Prepare data array
        data = np.array([data[param] for param in parameter_names if param not in [
                        'Time', 'Nodes']]).T

        # Remove NaN values
        valid_indices = ~np.isnan(data).any(axis=1)
        data = data[valid_indices]
        node_ids = np.array(nodes)[valid_indices]

        if data.size == 0:
            raise ValueError(
                "No data points remain after removing NaN values.")

        # Apply scaling to the parameters
        for i, sc in enumerate(scale):
            data[:, i] = cls.scale_funcs[sc]['direct'](data[:, i])

        # Perform clustering
        clustering = cls(G=G, data=data, node_ids=node_ids, parameters=parameter_names,
                         scales=scale, cluster_indexes=np.zeros(data.shape[0]))
        clustering.recluster(eps=eps, min_samples=min_samples)

        return clustering

    def recluster(self, eps: float, min_samples: int):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(self.data)
        self.cluster_indexes = dbscan.labels_

    def unscaled_centroids(self) -> np.ndarray:
        """
        Calculate centroids for the clusters, excluding noise points.

        Returns:
        - np.ndarray: The centroids of the clusters.
        """
        unique_labels = set(self.cluster_indexes)
        unique_labels.discard(-1)  # Remove noise label
        centroids = np.array([self.data[self.cluster_indexes == label].mean(axis=0)
                              for label in unique_labels])
        return centroids

    def predict_cluster(self, data_points: np.ndarray, points_scaled: bool = False, parameters: None | tuple[str] = None) -> np.ndarray:
        """
        Predicts the cluster indices to which new data points belong based on the clusters formed.

        Args:
            data_points: The new data points' parameter values as a numpy array.
            points_scaled: A boolean indicating whether the data points are already scaled.

        Returns:
            np.ndarray: An array of indices of the closest cluster point to each data point.

        Raises:
            ValueError: If the dimensionality of the data points does not match that of the original data.
        """

        if parameters is not None:
            data_points = self.prepare_data_point_for_prediction(
                data_points, parameters)

        if not points_scaled:
            # Apply scaling to the new data points
            for i, sc in enumerate(self.scales):
                data_points[:, i] = self.scale_funcs[sc]['direct'](
                    data_points[:, i])

        # Check for dimensionality match
        if data_points.shape[1] != self.data.shape[1]:
            raise ValueError(
                "Dimensionality of data points does not match that of the original data.")

        # Compute distances from each new data point to all existing data points
        distances = np.linalg.norm(
            data_points[:, np.newaxis] - self.data, axis=2)

        # Find the nearest existing data point for each new data point
        nearest_indices = np.argmin(distances, axis=1)

        # Assign each new data point to the cluster of the nearest existing data point
        cluster_indices = self.cluster_indexes[nearest_indices]

        return cluster_indices

    def reorder_clusters(self, new_order: list[int]):
        """
        Reorders clusters and associated information based on a new order.

        Args:
            new_order: A list containing the indices of the clusters in their new order.

        Raises:
            ValueError: If new_order does not contain all existing cluster indices.
        """
        unique_labels = set(self.cluster_indexes)
        unique_labels.discard(-1)  # Remove noise label
        if set(new_order) != unique_labels:
            raise ValueError(
                "New order must contain all existing cluster indices.")

        # Create a mapping from old to new labels
        label_mapping = {old: new for new, old in enumerate(new_order)}
        self.cluster_indexes = np.array(
            [label_mapping[label] if label in label_mapping else -1 for label in self.cluster_indexes])
        self.reorder_labels(new_order)
