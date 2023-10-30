import os
from typing import List, Optional, Union

import matplotlib.pyplot as plt

from .hari_graph import HariGraph
from .lazy_hari_graph import LazyHariGraph


class HariDynamics:
    def __init__(self):
        self.lazy_hari_graphs = []
        self.groups = []

    @property
    def initialized(self):
        return [index for index, graph in enumerate(
            self.lazy_hari_graphs) if graph.is_initialized]

    def uncluster(self):
        for graph in self.lazy_hari_graphs:
            graph.mapping = None
            graph.uninitialize()

    @classmethod
    def read_network(cls, network_files, opinion_files):
        """
        Reads a list of network files and a list of opinion files to create LazyHariGraph objects
        and appends them to the lazy_hari_graphs list of a HariDynamics instance.

        Parameters:
            network_files (Union[str, List[str]]): Either a single path or a list of paths to the network files.
            opinion_files (List[str]): A list of paths to the opinion files.

        Returns:
            HariDynamics: An instance of HariDynamics with lazy_hari_graphs populated.

        Raises:
            ValueError: If the length of network_files is not equal to the length of opinion_files or other invalid cases.
        """
        # If network_files is a string, convert it to a list
        if isinstance(network_files, str):
            network_files = [network_files]

        if len(network_files) == 1:
            network_files = network_files * len(opinion_files)

        if len(network_files) != len(opinion_files):
            raise ValueError(
                "The number of network files must be equal to the number of opinion files.")

        # Create an instance of HariDynamics
        dynamics_instance = cls()
        dynamics_instance.groups = []  # Initialize groups list

        for idx, (network_file, opinion_file) in enumerate(
                zip(network_files, opinion_files)):
            # Append LazyHariGraph objects to the list, with the class method and parameters needed
            # to create the actual HariGraph instances when they are required.
            dynamics_instance.lazy_hari_graphs.append(
                LazyHariGraph(HariGraph.read_network,
                              network_file, opinion_file)
            )
            dynamics_instance.groups.append([idx])

        return dynamics_instance

    def __getitem__(self, index):
        if index < 0:
            # Adjust the index for negative values
            index += len(self.lazy_hari_graphs)

        if index < 0 or index >= len(self.lazy_hari_graphs):
            raise IndexError(
                "Index out of range of available LazyHariGraph objects.")

        return self.lazy_hari_graphs[index]

    def __iter__(self):
        return iter(self.lazy_hari_graphs)

    def __getattr__(self, name):
        # Try to get the attribute from the first LazyHariGraph object in the list.
        # If it exists, assume it exists on all HariGraph instances in the
        # list.
        if self.lazy_hari_graphs:
            try:
                attr = getattr(self.lazy_hari_graphs[0], name)
            except AttributeError:
                pass  # Handle below
            else:
                if callable(attr):
                    # If the attribute is callable, return a function that
                    # calls it on all HariGraph instances.
                    def forwarded(*args, **kwargs):
                        return [getattr(lazy_graph, name)(*args, **kwargs)
                                for lazy_graph in self.lazy_hari_graphs]
                    return forwarded
                else:
                    # If the attribute is not callable, return a list of its
                    # values from all HariGraph instances.
                    return [getattr(lazy_graph, name)
                            for lazy_graph in self.lazy_hari_graphs]

        # If the attribute does not exist on HariGraph instances, raise an
        # AttributeError.
        raise AttributeError(
            f"'HariDynamics' object and its 'HariGraph' instances have no attribute '{name}'")

    def group(self, num_intervals, interval_size=1, offset=0):
        """
        Groups indices of LazyHariGraphs objects based on provided intervals, interval size, and an offset.
        Indices might show up multiple times or might never show up, depending on the parameters.

        Parameters:
            num_intervals (int): The number of intervals.
            interval_size (int): The size of each interval.
            offset (int): Starting offset for the grouping.
        """
        self.groups.clear()  # Clear the previous grouping

        total_length = len(self.lazy_hari_graphs)

        # Calculate the stride between each interval's starting index so they
        # are evenly spread
        if num_intervals == 1:
            stride = 0
        else:
            stride = (total_length - offset -
                      interval_size) // (num_intervals - 1)

        for i in range(num_intervals):
            start_index = offset + i * stride
            end_index = start_index + interval_size

            # Append the range of indices as a sublist, but ensure they stay
            # within the valid range
            self.groups.append(
                list(range(start_index, min(end_index, total_length))))

    def get_grouped_graphs(self):
        """
        Retrieves grouped LazyHariGraph objects based on the indices in self.groups.

        Returns:
            List[List[LazyHariGraph]]: List of lists where each sublist contains LazyHariGraph objects.
        """
        return [[self.lazy_hari_graphs[i] for i in group_indices]
                for group_indices in self.groups]

    def merge_nodes_by_mapping(self, mapping):
        """
        Merge nodes in each LazyHariGraph based on the provided mapping.
        If a graph was already initialized, uninitialize it first.

        Parameters:
            mapping (Dict[int, int]): A dictionary representing how nodes should be merged.
            skip_indices (set): Either a list of indices representing which LazyHariGraph instances
                                                to skip or a string 'NotInGroups' to skip graphs not in groups.
                                                Defaults to None (no graph is skipped).
        """

        for lazy_graph in self.lazy_hari_graphs:
            lazy_graph.mapping = mapping

    def merge_nodes_by_index(self, index):
        """
        Merges nodes in each LazyHariGraph based on the cluster mapping of the graph at the provided index.
        The graph at the provided index is skipped during merging.

        Parameters:
            index (int): The index of the LazyHariGraph whose cluster mapping should be used for merging nodes.
            skip_indices (Union[List[int], str, None]): Either a list of indices representing which LazyHariGraph instances
                                                        to skip, a string 'NotInGroups' to skip graphs not in groups,
                                                        or None to not skip any. Defaults to None.
        """

        if index < 0:
            # Adjust the index for negative values
            index += len(self.lazy_hari_graphs)

        if index < 0 or index >= len(self.lazy_hari_graphs):
            raise IndexError(
                "Index out of range of available LazyHariGraph objects.")

        target_lazy_graph = self.lazy_hari_graphs[index]

        # Initialize the target graph if not already initialized to access its
        # get_cluster_mapping method
        if not target_lazy_graph.is_initialized:
            target_lazy_graph._initialize()  # Initialize the target graph

        mapping = target_lazy_graph.get_cluster_mapping()
        self.merge_nodes_by_mapping(mapping)

    def plot_neighbor_mean_opinion(self, save: Optional[Union[str, List[str]]] = None,
                                   show_timestamp: bool = False, cmax=None, uninit=False, num_images=None, **kwargs):
        """
        Plots the neighbor mean opinion for the first graph of each group in lazy_hari_graphs.

        :param save: str, list of str, or None, optional
            If str, treated as a directory and each plot is saved with a filename '{i}.png'.
            If list of str, each string in the list is treated as a filename for saving the corresponding plot.
            If None, plots are not saved.
            Default is None.

        :param show_timestamp: bool, optional
            If True, the index of each graph is displayed as a timestamp in title.
            Default is False.

        :param num_images: int or None, optional
            The number of images to generate. If set to a specific number,
            images are spread evenly across the time span.
            Default is None (all images).

        :param kwargs: keyword arguments
            Additional settings that will be provided to the plot_neighbor_mean_opinion method of each graph.

        :return: None
        """

        # Extracting indices of the first image of each group
        indices_to_plot = [group[0] for group in self.groups]

        # If num_images is defined and less than the number of groups, select
        # evenly spaced indices
        if num_images is not None and num_images < len(indices_to_plot):
            step = len(indices_to_plot) / num_images
            indices_to_plot = [
                indices_to_plot[int(i * step)] for i in range(num_images)]

        # Adjust the save_filepaths logic based on the selected indices
        if isinstance(save, str):
            os.makedirs(save, exist_ok=True)
            save_filepaths = [os.path.join(
                save, f'{i}.png') for i in indices_to_plot]
        elif isinstance(save, list):
            assert len(save) >= len(
                indices_to_plot), "The length of 'save' list must be at least the number of selected images."
            save_filepaths = [save[i] for i in indices_to_plot]
        else:
            save_filepaths = [None] * len(indices_to_plot)

        # Plot neighbor_mean_opinion for selected graphs
        for idx, save_filepath in zip(indices_to_plot, save_filepaths):
            graph = self.lazy_hari_graphs[idx]
            fig, ax = plt.subplots(figsize=(10, 7))

            # Use the plot_neighbor_mean_opinion method of the graph
            title_text = f't = {idx}' if show_timestamp else None

            graph.plot_neighbor_mean_opinion(
                fig=fig, ax=ax, save=save_filepath, title=title_text, cmax=cmax, **kwargs)

            plt.close()
            if uninit:
                graph.uninitialize()

    def __str__(self):
        initialized_count = sum(
            1 for lazy_graph in self.lazy_hari_graphs if lazy_graph.is_initialized)
        total_count = len(self.lazy_hari_graphs)
        return f"<HariDynamics object with {total_count} LazyHariGraphs ({initialized_count} initialized)>"
