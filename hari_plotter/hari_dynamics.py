import os
from typing import List, Optional, Union

import matplotlib.pyplot as plt

from .hari_graph import HariGraph
from .lazy_hari_graph import LazyHariGraph


class HariDynamics:
    def __init__(self):
        self.lazy_hari_graphs = []  # Initialize an empty list to hold LazyHariGraph objects

    @classmethod
    def read_network(cls, network_file, opinion_files):
        """
        Reads the network file and a list of opinion files to create LazyHariGraph objects
        and appends them to the lazy_hari_graphs list of a HariDynamics instance.

        Parameters:
            network_file (str): The path to the network file.
            opinion_files (List[str]): A list of paths to the opinion files.

        Returns:
            HariDynamics: An instance of HariDynamics with lazy_hari_graphs populated.
        """
        # Create an instance of HariDynamics
        dynamics_instance = cls()

        for opinion_file in opinion_files:
            # Append LazyHariGraph objects to the list, with the class method and parameters needed
            # to create the actual HariGraph instances when they are required.
            dynamics_instance.lazy_hari_graphs.append(
                LazyHariGraph(HariGraph.read_network,
                              network_file, opinion_file)
            )

        return dynamics_instance

    def __getitem__(self, index):
        if index < 0 or index >= len(self.lazy_hari_graphs):
            raise IndexError(
                "Index out of range of available LazyHariGraph objects.")

        return self.lazy_hari_graphs[index]

    def __getattr__(self, name):
        # Try to get the attribute from the first LazyHariGraph object in the list.
        # If it exists, assume it exists on all HariGraph instances in the list.
        if self.lazy_hari_graphs:
            try:
                attr = getattr(self.lazy_hari_graphs[0], name)
            except AttributeError:
                pass  # Handle below
            else:
                if callable(attr):
                    # If the attribute is callable, return a function that calls it on all HariGraph instances.
                    def forwarded(*args, **kwargs):
                        return [getattr(lazy_graph, name)(*args, **kwargs) for lazy_graph in self.lazy_hari_graphs]
                    return forwarded
                else:
                    # If the attribute is not callable, return a list of its values from all HariGraph instances.
                    return [getattr(lazy_graph, name) for lazy_graph in self.lazy_hari_graphs]

        # If the attribute does not exist on HariGraph instances, raise an AttributeError.
        raise AttributeError(
            f"'HariDynamics' object and its 'HariGraph' instances have no attribute '{name}'")

    def merge_nodes_based_on_mapping(self, mapping, skip_indices=None):
        """
        Merge nodes in each LazyHariGraph based on the provided mapping.
        If a graph was already initialized, uninitialize it first.

        Parameters:
            mapping (Dict[int, int]): A dictionary representing how nodes should be merged.
            skip_indices (List[int], optional): A list of indices representing which LazyHariGraph instances 
                                                to skip. Defaults to None (no graph is skipped).
        """
        if skip_indices is None:
            skip_indices = []

        for i, lazy_graph in enumerate(self.lazy_hari_graphs):
            if i in skip_indices:
                continue

            if lazy_graph.is_initialized():
                lazy_graph.uninitialize()

            lazy_graph.merge_clusters(mapping)

    def merge_nodes_based_on_index(self, index):
        """
        Merges nodes in each LazyHariGraph based on the cluster mapping of the graph at the provided index.
        The graph at the provided index is skipped during merging.

        Parameters:
            index (int): The index of the LazyHariGraph whose cluster mapping should be used for merging nodes.
        """
        if index < 0 or index >= len(self.lazy_hari_graphs):
            raise IndexError(
                "Index out of range of available LazyHariGraph objects.")

        target_lazy_graph = self.lazy_hari_graphs[index]

        # Initialize the target graph if not already initialized to access its get_cluster_mapping method
        if not target_lazy_graph.is_initialized():
            target_lazy_graph._initialize()  # Initialize the target graph

        mapping = target_lazy_graph.get_cluster_mapping()

        self.merge_nodes_based_on_mapping(mapping, skip_indices=[index])

    def plot_values(self, reference_index=0):
        assert all(isinstance(graph.node_values, dict)
                   for graph in self.lazy_hari_graphs), "All graph node_values must be dictionaries"

        # Extracting the common keys from the first graph values
        common_keys = set(self.lazy_hari_graphs[0].node_values['value'].keys())

        # Asserting that all the graphs have the same keys in their values, min_values, and max_values
        for graph in self.lazy_hari_graphs:
            assert set(graph.node_values['value'].keys(
            )) == common_keys, "All graph values must have the same keys"
            assert set(graph.node_values['min_value'].keys(
            )) == common_keys, "All graph min_values must have the same keys"
            assert set(graph.node_values['max_value'].keys(
            )) == common_keys, "All graph max_values must have the same keys"

        # Find the value furthest from 0.5 among all values of all nodes in all graphs
        furthest_distance = max(
            abs(value - 0.5)
            for graph in self.lazy_hari_graphs
            for value in graph.node_values['value'].values()
        )

        # Define the colormap range based on the furthest_distance from 0.5
        vmin = 0.5 - furthest_distance
        vmax = 0.5 + furthest_distance

        fig, ax = plt.subplots(figsize=(10, 6))
        x = list(range(len(self.lazy_hari_graphs)))

        for key in common_keys:
            y = [g.node_values['value'][key] for g in self.lazy_hari_graphs]
            min_y = [g.node_values['min_value'][key]
                     for g in self.lazy_hari_graphs]
            max_y = [g.node_values['max_value'][key]
                     for g in self.lazy_hari_graphs]

            # Extract the reference color from the graph at reference_index and map it to the adjusted colormap range
            ref_value = self.lazy_hari_graphs[reference_index].node_values['value'][key]
            color = plt.get_cmap('RdBu')((ref_value - vmin) / (vmax - vmin))

            # Plotting the semitransparent region between min and max values
            ax.fill_between(x, min_y, max_y, color=color, alpha=0.2)

            # Plotting the line for the values
            ax.plot(x, y, color=color, label=f'Node {key}')

        ax.set_title(f"Node Values Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()

    def draw_dynamic_graphs(self, reference_index: int = 0, save: Optional[Union[str, List[str]]] = None,
                            show_timestamp: bool = False, **kwargs):
        """
        Draws the graphs at each time step, using the positions of the nodes as determined by the graph at the reference_index.

        :param reference_index: int, optional
            Index of the graph whose node positions will be used as a reference for drawing all other graphs.
            Default is 0.

        :param save: str, list of str, or None, optional
            If str, treated as a directory and each graph is saved with a filename '{i}.png'.
            If list of str, each string in the list is treated as a filename for saving the corresponding graph.
            If None, graphs are not saved.
            Default is None.

        :param show_timestamp: bool, optional
            If True, the index of each graph is displayed as a timestamp in the bottom right corner of the plot.
            Default is False.

        :param kwargs: keyword arguments
            Additional settings that will be provided to the draw method of each graph.

        :return: None

        """
        # Get the reference graph and its node positions
        pos = self.lazy_hari_graphs[reference_index].position_nodes()

        # Check the save parameter and prepare the save_filepaths list
        if isinstance(save, str):
            os.makedirs(save, exist_ok=True)
            save_filepaths = [os.path.join(
                save, f'{i}.png') for i in range(len(self.lazy_hari_graphs))]
        elif isinstance(save, list):
            assert len(save) == len(
                self.lazy_hari_graphs), "The length of 'save' list must be equal to the number of graphs."
            save_filepaths = save
        else:
            save_filepaths = [None] * len(self.lazy_hari_graphs)

        # Draw each graph with the reference positions
        for i, (graph, save_filepath) in enumerate(zip(self.lazy_hari_graphs, save_filepaths)):
            fig, ax = plt.subplots(figsize=(10, 7))
            bottom_right_text = f't = {i}' if show_timestamp else None
            graph.draw(pos=pos, save_filepath=save_filepath, fig=fig,
                       ax=ax, bottom_right_text=bottom_right_text, **kwargs)
            plt.close()

    def __str__(self):
        initialized_count = sum(
            1 for lazy_graph in self.lazy_hari_graphs if lazy_graph.is_initialized())
        total_count = len(self.lazy_hari_graphs)
        return f"<HariDynamics object with {total_count} LazyHariGraphs ({initialized_count} initialized)>"
