from __future__ import annotations

import math
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np

from .graph import Graph
from .lazy_graph import LazyGraph


class Dynamics:
    """
    HariDynamics manages a list of LazyHariGraph objects that represent HariGraph instances.

    This class facilitates the batch processing of multiple LazyHariGraph instances. It allows for reading multiple
    networks and opinions, and provides a unified interface to access and manipulate each LazyHariGraph.

    Attributes:
        lazy_hari_graphs (list[LazyHariGraph]): A list of LazyHariGraph objects.
        groups (list[list[int]]): A list where each element is a list of indices representing a group.

    Methods:
        initialized: Returns indices of initialized LazyHariGraph instances.
        uncluster: Resets clustering for all LazyHariGraph instances.
        read_network: Reads network and opinion files to create and add LazyHariGraph objects.
        __getitem__: Allows indexing to access individual LazyHariGraph instances.
        __iter__: Allows iteration over LazyHariGraph instances.
        __getattr__: Allows dynamic attribute access, attempting to get the attribute from LazyHariGraph instances.

    Note:
        Any attribute that isn't directly found on a HariDynamics object will attempt to be retrieved from its
        LazyHariGraph objects. If the attribute is callable, a function will be returned that when called, will 
        apply the method on all LazyHariGraph instances. If the attribute is not callable, a list of its values 
        from all LazyHariGraph instances will be returned.
    """

    def __init__(self):
        self.lazy_hari_graphs: list[LazyGraph] = []
        self.groups: list[list[int]] = []

    @property
    def initialized(self) -> list[int]:
        """
        list of indices of initialized LazyHariGraph instances.

        Iterates over each LazyHariGraph in the list `lazy_hari_graphs` and checks if it's initialized.
        Returns a list containing the indices of all the LazyHariGraphs that are initialized.

        Returns:
            list[int]: Indices of all initialized LazyHariGraph instances.
        """
        return [index for index, graph in enumerate(self.lazy_hari_graphs) if graph.is_initialized]

    def uncluster(self):
        """
        Resets the clustering and uninitializes all LazyHariGraph instances.

        For each LazyHariGraph in `lazy_hari_graphs`, resets its clustering by setting its mapping to None
        and then uninitializes the graph.
        """
        for graph in self.lazy_hari_graphs:
            graph.mapping = None
            graph.uninitialize()

    @classmethod
    def read_network(cls, network_files: Union[str, list[str]], opinion_files: list[str], load_request: dict[str, Any] = {}) -> Dynamics:
        """
        Reads a list of network files and a list of opinion files to create LazyHariGraph objects
        and appends them to the lazy_hari_graphs list of a HariDynamics instance.

        Parameters:
            network_files (Union[str, list[str]]): Either a single path or a list of paths to the network files.
            opinion_files (list[str]): A list of paths to the opinion files.

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
                LazyGraph(Graph.read_network,
                          network_file, opinion_file, **load_request)
            )
            dynamics_instance.groups.append([idx])

        return dynamics_instance

    @classmethod
    def read_json(cls, json_files: Union[str, list[str]], load_request: dict[str, Any] = {}) -> Dynamics:
        """
        Reads a list of network files and a list of opinion files to create LazyHariGraph objects
        and appends them to the lazy_hari_graphs list of a HariDynamics instance.

        Parameters:
            network_files (Union[str, list[str]]): Either a single path or a list of paths to the network files.
            opinion_files (list[str]): A list of paths to the opinion files.

        Returns:
            HariDynamics: An instance of HariDynamics with lazy_hari_graphs populated.

        Raises:
            ValueError: If the length of network_files is not equal to the length of opinion_files or other invalid cases.
        """
        # If json_files is a string, convert it to a list
        if isinstance(json_files, str):
            json_files = [json_files]

        # Create an instance of HariDynamics
        dynamics_instance = cls()
        dynamics_instance.groups = []  # Initialize groups list

        for idx, json_file in enumerate(json_files):
            # Append LazyHariGraph objects to the list, with the class method and parameters needed
            # to create the actual HariGraph instances when they are required.
            dynamics_instance.lazy_hari_graphs.append(
                LazyGraph(Graph.read_json, json_file, **load_request)
            )
            dynamics_instance.groups.append([idx])

        return dynamics_instance

    def __getitem__(self, index):
        # Handle slices
        if isinstance(index, slice):
            return [self.lazy_hari_graphs[i] for i in range(*index.indices(len(self.lazy_hari_graphs)))]

        # Handle lists of integers
        elif isinstance(index, list):
            # Ensure all elements in the list are integers
            if not all(isinstance(i, int) for i in index):
                raise TypeError("All indices must be integers")
            return [self.lazy_hari_graphs[i] for i in index]

        # Handle single integer
        elif isinstance(index, int):
            if index < 0:
                # Adjust the index for negative values
                index += len(self.lazy_hari_graphs)

            if index < 0 or index >= len(self.lazy_hari_graphs):
                raise IndexError(
                    "Index out of range of available LazyHariGraph objects.")

            return self.lazy_hari_graphs[index]

        else:
            raise TypeError(
                "Invalid index type. Must be an integer, slice, or list of integers.")

    def __iter__(self) -> LazyGraph:
        return iter(self.lazy_hari_graphs)

    def __getattr__(self, name: str) -> list:
        """
        Retrieve an attribute from the LazyHariGraph instances in the list.

        If the desired attribute exists on the first LazyHariGraph in `lazy_hari_graphs`,
        it's assumed to exist on all instances in the list.

        If the attribute is callable, this method returns a function that invokes
        the callable attribute on all LazyHariGraph instances and returns the results as a list.

        If the attribute is not callable, this method returns a list of the attribute's
        values from all LazyHariGraph instances.

        If the attribute does not exist on any LazyHariGraph instance, an AttributeError is raised.

        Parameters:
            name (str): Name of the desired attribute.

        Returns:
            list: A list of attribute values or function results for all LazyHariGraph instances,
                depending on the nature of the attribute.

        Raises:
            AttributeError: If the specified attribute doesn't exist on the LazyHariGraph instances.
        """
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

    def group(self, num_intervals: int, interval_size: int = 1, offset: int = 0):
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

    def get_grouped_graphs(self) -> list[list[LazyGraph]]:
        """
        Retrieves grouped LazyHariGraph objects based on the indices in self.groups.

        Returns:
            list[list[LazyHariGraph]]: list of lists where each sublist contains LazyHariGraph objects.
        """
        return [[self.lazy_hari_graphs[i] for i in group_indices]
                for group_indices in self.groups]

    def merge_nodes_by_mapping(self, mapping: tuple[int]):
        """
        Merge nodes in each LazyHariGraph based on the provided mapping.
        If a graph was already initialized, uninitialize it first.

        Parameters:
            mapping (tuple[int]): A dictionary representing how nodes should be merged.
        """

        for lazy_graph in self.lazy_hari_graphs:
            lazy_graph.mapping = mapping

    def merge_nodes_by_index(self, index: int):
        """
        Merges nodes in each LazyHariGraph based on the cluster mapping of the graph at the provided index.
        The graph at the provided index is skipped during merging.

        Parameters:
            index (int): The index of the LazyHariGraph whose cluster mapping should be used for merging nodes.
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

    def plot_initialized(self):
        initialized = np.array(
            [lazy_graph.is_initialized for lazy_graph in self.lazy_hari_graphs])

        # Find the appropriate power of 10 for the last axis dimension
        sqrt_length = math.sqrt(len(initialized))
        last_axis_size = max(
            10 ** int(math.floor(math.log10(sqrt_length))), 10)

        # Calculate the size of the other dimension
        other_axis_size = int(np.ceil(len(initialized) / last_axis_size))

        # Pad the array if necessary to fit the new shape, using NaN for padding
        padded_length = other_axis_size * last_axis_size
        padded_initialized = np.full(padded_length, np.nan)
        padded_initialized[:len(initialized)] = initialized

        # Reshape to 2D array
        initialized_2d = padded_initialized.reshape(
            (other_axis_size, last_axis_size))

        # Determine the label format based on the last axis size
        x_label_digits = int(math.log10(last_axis_size))
        y_label_digits = int(math.log10(padded_length // last_axis_size))

        # Plotting with imshow
        plt.figure(figsize=(12, 6))
        plt.imshow(initialized_2d, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Initialized (False=0, True=1, NaN=Padding)')
        plt.title('Initialization Status of Lazy Hari Graphs')
        plt.xlabel(f'Last {x_label_digits} Digits of Graph ID')
        plt.ylabel(f'Graph ID without Last {x_label_digits} Digits')

        # Ensure axis labels are integers
        plt.xticks(ticks=np.arange(0, last_axis_size, max(1, last_axis_size // 10)),
                   labels=np.arange(0, last_axis_size, max(1, last_axis_size // 10)))
        plt.yticks(ticks=np.arange(0, other_axis_size, max(1, other_axis_size // 10)),
                   labels=np.arange(0, other_axis_size, max(1, other_axis_size // 10)))

        plt.show()

    def __len__(self) -> int:
        return len(self.lazy_hari_graphs)

    def __str__(self):
        initialized_count = sum(
            1 for lazy_graph in self.lazy_hari_graphs if lazy_graph.is_initialized)
        total_count = len(self.lazy_hari_graphs)
        return f"<HariDynamics object with {total_count} LazyHariGraphs ({initialized_count} initialized)>"
