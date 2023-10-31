"""
Utilities for clustering and visualization of 2D datasets.

This module provides functions to determine the optimal number of clusters in a 2D dataset,
merging close clusters, visualizing the 2D distribution and accompanying 1D distributions,
and plotting hexbin plots.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def is_single_cluster(x_array, y_array):
    """
    Check if the given data represents a single cluster.

    Parameters:
    - x_array : array-like
        Data points for x-axis.
    - y_array : array-like
        Data points for y-axis.

    Returns:
    - bool : True if data likely represents a single cluster, False otherwise.
    """
    points = np.column_stack((x_array, y_array))
    kmeans = KMeans(n_clusters=2, init='k-means++',
                    n_init='auto', random_state=42)
    kmeans.fit(points)
    cluster_labels = kmeans.predict(points)
    two_cluster_score = silhouette_score(points, cluster_labels)

    return two_cluster_score < 0.75


def merge_close_clusters(centroids, labels, min_distance):
    """
    Merge clusters with centroids that are closer than a minimum distance.

    Parameters:
    - centroids : array-like
        Array containing centroids of clusters.
    - labels : array-like
        Labels assigned to data points.
    - min_distance : float
        Minimum distance between centroids to trigger a merge.

    Returns:
    - labels : array-like
        Updated labels after merging clusters.
    - centroids : array-like
        Updated centroids after merging clusters.
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


def optimal_clusters(x_array, y_array, min_distance=1e-2):
    """
    Determine the optimal number of clusters for a given 2D dataset.

    Parameters:
    - x_array : array-like
        Data points for x-axis.
    - y_array : array-like
        Data points for y-axis.
    - min_distance : float, optional (default=1e-2)
        Minimum distance between centroids.

    Returns:
    - int : Optimal number of clusters.
    - list : Lists containing indices of data points in each cluster.
    """
    points = np.column_stack((x_array, y_array))

    if is_single_cluster(x_array, y_array):
        return 1, [list(range(len(x_array)))]

    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        n_init='auto', random_state=42)
        kmeans.fit(points)
        wcss.append(kmeans.inertia_)

    deltas = np.diff(wcss)
    n = np.where(deltas == min(deltas))[0][0] + 2

    kmeans = KMeans(n_clusters=n, init='k-means++',
                    n_init='auto', random_state=42)
    kmeans.fit(points)
    labels, centroids = merge_close_clusters(
        kmeans.cluster_centers_, kmeans.labels_, min_distance)

    unique_labels = np.unique(labels)
    clusters = [list(np.where(labels == i)[0]) for i in unique_labels]

    return clusters
