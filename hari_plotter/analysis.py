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


def set_paper_style(width_fraction=1.0, font_size=10,
                    style='seaborn-whitegrid'):
    """
    Set style for academic papers.

    Parameters:
    - width_fraction: Fraction of the paper width (e.g., 0.25, 0.5, 1.0)
    - font_size: Size of the font to be used in plots
    - style: Style preset from matplotlib
    """

    # Assuming a single-column width of 6 inches for the paper
    single_column_width = 6

    # Set the style
    plt.style.use(style)

    # Set the figure size
    fig_width = single_column_width * width_fraction
    plt.rcParams['figure.figsize'] = (
        fig_width, fig_width * 3 / 4)  # 4:3 aspect ratio

    # Set font size and line width
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.labelsize'] = font_size
    plt.rcParams['axes.titlesize'] = font_size
    plt.rcParams['xtick.labelsize'] = font_size
    plt.rcParams['ytick.labelsize'] = font_size
    plt.rcParams['lines.linewidth'] = 1.5


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


def plot_neighbor_mean_opinion(x_values, y_values, fig=None, ax=None, save=None, show=True, extent=None,
                               title=None, cmax=None, xlabel='Node Opinion', ylabel='Mean Neighbor Opinion', **kwargs):
    """
    Draws a hexbin plot of node opinions versus mean neighbor opinions.

    Parameters:
    - x_values, y_values : array-like
        Data points for x and y axes.
    - fig, ax : matplotlib objects, optional
        Pre-existing figure and axis objects.
    - save : str, optional
        Filepath to save the plot.
    - show : bool, optional (default=True)
        Whether to display the plot.
    - extent : list, optional
        Range for the plot.
    - title : str, optional
        Plot title.
    - cmax : float, optional
        Maximum limit for the colorbar.
    - xlabel, ylabel : str, optional
        Labels for x and y axes.
    - **kwargs : Additional arguments for plt.hexbin.

    Returns:
    - fig, ax : Updated figure and axis objects.
    """
    if x_values is None or y_values is None or len(x_values) != len(y_values):
        print("Invalid data received. Cannot plot.")
        return

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    if extent is None:
        extent = [min(x_values), max(x_values), min(y_values), max(y_values)]
    elif len(extent) == 2:
        extent = [extent[0], extent[1], extent[0], extent[1]]
    elif len(extent) != 4:
        print("Invalid extent value. Please provide None, 2 values, or 4 values.")
        return

    ax.imshow([[0, 0], [0, 0]], cmap='inferno',
              interpolation='nearest', aspect='auto', extent=extent)
    hb = ax.hexbin(x_values, y_values, gridsize=50, cmap='inferno',
                   bins='log', extent=extent, vmax=cmax, **kwargs)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if save:
        plt.savefig(save)

    if show:
        plt.show()

    return fig, ax


def plot_2d_distributions(x_values, y_values, save=None, show=True, extent=None, title=None,
                          cmax=None, xlabel='Node Opinion', ylabel='Mean Neighbor Opinion', paper_style=None, **kwargs):
    """
    Plot 2D hexbin alongside 1D KDE distributions for x and y data.

    Parameters:
    - x_values, y_values : array-like
        Data points for x and y axes.
    - save : str, optional
        Filepath to save the plot.
    - show : bool, optional (default=True)
        Whether to display the plot.
    - extent : list, optional
        Range for the plot.
    - title : str, optional
        Plot title.
    - cmax : float, optional
        Maximum limit for the colorbar.
    - xlabel, ylabel : str, optional
        Labels for x and y axes.
    - **kwargs : Additional arguments for plt.hexbin.

    Returns:
    - fig, axs : Updated figure and axis objects.
    """

    set_paper_style(**paper_style)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={
                            'width_ratios': [4, 1], 'height_ratios': [1, 4]})
    plot_neighbor_mean_opinion(x_values, y_values, fig=fig,
                               ax=axs[1, 0], show=False, extent=extent, title=title, cmax=cmax, **kwargs)
    sns.kdeplot(data=y_values, ax=axs[0, 0], fill=True, color='blue')
    sns.kdeplot(y=x_values, ax=axs[1, 1], fill=True, color='red')

    axs[0, 0].set_title("Distribution of " + xlabel)
    axs[0, 0].set_xlim(axs[1, 0].get_xlim())
    axs[0, 0].set_xticks([])

    axs[1, 1].set_title("Distribution of " + ylabel)
    axs[1, 1].set_ylim(axs[1, 0].get_ylim())
    axs[1, 1].set_yticks([])

    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    for spine in axs[0, 1].spines.values():
        spine.set_visible(False)

    if save:
        plt.savefig(save)

    if show:
        plt.show()

    return fig, axs
