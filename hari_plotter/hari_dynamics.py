import os
from typing import List, Optional, Union

import matplotlib.pyplot as plt

from .hari_graph import HariGraph
from .lazy_hari_graph import LazyHariGraph


class HariDynamics:
    def __init__(self):
        self.lazy_hari_graphs = []  # Initialize an empty list to hold LazyHariGraph objects
        self.groups = []  # New list to hold grouped indices

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

        for idx, (network_file, opinion_file) in enumerate(zip(network_files, opinion_files)):
            # Append LazyHariGraph objects to the list, with the class method and parameters needed
            # to create the actual HariGraph instances when they are required.
            dynamics_instance.lazy_hari_graphs.append(
                LazyHariGraph(HariGraph.read_network,
                              network_file, opinion_file)
            )
            dynamics_instance.groups.append([idx])

        return dynamics_instance

    def __getitem__(self, index):
        if index < 0 or index >= len(self.lazy_hari_graphs):
            raise IndexError(
                "Index out of range of available LazyHariGraph objects.")

        return self.lazy_hari_graphs[index]

    def __iter__(self):
        return iter(self.lazy_hari_graphs)

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

        # Calculate the stride between each interval's starting index so they are evenly spread
        if num_intervals == 1:
            stride = 0
        else:
            stride = (total_length - offset -
                      interval_size) // (num_intervals - 1)

        for i in range(num_intervals):
            start_index = offset + i * stride
            end_index = start_index + interval_size

            # Append the range of indices as a sublist, but ensure they stay within the valid range
            self.groups.append(
                list(range(start_index, min(end_index, total_length))))

    def get_grouped_graphs(self):
        """
        Retrieves grouped LazyHariGraph objects based on the indices in self.groups.

        Returns:
            List[List[LazyHariGraph]]: List of lists where each sublist contains LazyHariGraph objects.
        """
        return [[self.lazy_hari_graphs[i] for i in group_indices] for group_indices in self.groups]

    def merge_nodes_based_on_mapping(self, mapping, skip_indices=None):
        """
        Merge nodes in each LazyHariGraph based on the provided mapping.
        If a graph was already initialized, uninitialize it first.

        Parameters:
            mapping (Dict[int, int]): A dictionary representing how nodes should be merged.
            skip_indices (Union[List[int], str]): Either a list of indices representing which LazyHariGraph instances 
                                                to skip or a string 'NotInGroups' to skip graphs not in groups.
                                                Defaults to None (no graph is skipped).
        """
        if skip_indices == 'NotInGroups':
            all_indices = set(range(len(self.lazy_hari_graphs)))
            in_group_indices = {idx for group in self.groups for idx in group}
            skip_indices = all_indices - in_group_indices
        elif skip_indices is None:
            skip_indices = set()
        else:
            skip_indices = set(skip_indices)

        for i, lazy_graph in enumerate(self.lazy_hari_graphs):
            if i in skip_indices:
                continue

            if lazy_graph.is_initialized():
                lazy_graph.uninitialize()

            lazy_graph.merge_clusters(mapping)

    def merge_nodes_based_on_index(self, index, skip_indices=None):
        """
        Merges nodes in each LazyHariGraph based on the cluster mapping of the graph at the provided index.
        The graph at the provided index is skipped during merging.

        Parameters:
            index (int): The index of the LazyHariGraph whose cluster mapping should be used for merging nodes.
            skip_indices (Union[List[int], str, None]): Either a list of indices representing which LazyHariGraph instances 
                                                        to skip, a string 'NotInGroups' to skip graphs not in groups, 
                                                        or None to not skip any. Defaults to None.
        """
        if skip_indices == 'NotInGroups':
            all_indices = set(range(len(self.lazy_hari_graphs)))
            in_group_indices = {idx for group in self.groups for idx in group}
            skip_indices = all_indices - in_group_indices
        elif skip_indices is None:
            skip_indices = set()
        else:
            skip_indices = set(skip_indices)

        if index < 0 or index >= len(self.lazy_hari_graphs):
            raise IndexError(
                "Index out of range of available LazyHariGraph objects.")

        target_lazy_graph = self.lazy_hari_graphs[index]

        # Initialize the target graph if not already initialized to access its get_cluster_mapping method
        if not target_lazy_graph.is_initialized():
            target_lazy_graph._initialize()  # Initialize the target graph

        mapping = target_lazy_graph.get_cluster_mapping()

        print(f'{mapping = }')

        self.merge_nodes_based_on_mapping(
            mapping, skip_indices=skip_indices.union({index}))

    def plot_opinions(self, reference_index=0, show=True, save=False, minimum_cluster_size=1, colormap='coolwarm'):
        """
        Visualizes the opinions of nodes over time using a line graph and a semitransparent region 
        that spans between the minimum and maximum values of those opinions.

        Parameters:
        -----------
        reference_index : int, optional
            Index for the graph which will be used as a reference for coloring the nodes. Default is 0.

        show : bool, optional
            Whether or not to display the plot immediately after generating it. Default is True.

        save : bool or str, optional
            If given a string (filename), the plot will be saved to the specified filename. 
            If False, the plot will not be saved. Default is False.

        minimum_cluster_size : int, optional
            Minimum cluster size for a node to be considered in the plot. 
            Nodes with sizes less than this value will be excluded. Default is 1.

        colormap : str, optional
            Name of the colormap to use for coloring the nodes. Default is 'coolwarm'.

        Raises:
        -------
        AssertionError
            If the node_values of all graphs are not dictionaries or if opinions, min_opinions, 
            and max_opinions in each graph don't have the same set of keys.

        Description:
        ------------
        The method begins by ensuring the consistency of node_values across all graphs. 
        It then extracts the common keys of nodes from the first graph and verifies that 
        these keys are consistent across the opinions, min_opinions, and max_opinions of all graphs.
        Only the first graph from each group is plotted.

        For nodes that meet the minimum_cluster_size criteria, it plots their opinions over time. 
        A semitransparent region is plotted between the minimum and maximum values of those opinions, 
        allowing for a visualization of the possible variation or uncertainty in the opinions. 
        Each node is colored based on the value of its opinion in the graph at reference_index, 
        mapped to the provided colormap.
        """

        assert all(isinstance(graph.node_values, dict)
                   for graph in self.lazy_hari_graphs), "All graph node_values must be dictionaries"

        # Extracting indices of the first image of each group
        indices_to_plot = [group[0] for group in self.groups]

        # Extracting the common keys from the first graph opinions
        nodes_set = set(self.lazy_hari_graphs[0].nodes)

        # Asserting that all the graphs have the same keys in their opinions, min_opinions, and max_opinions
        for graph in self.lazy_hari_graphs:
            graph_node_values = graph.node_values
            assert set(graph_node_values['opinion'].keys(
            )) == nodes_set, "All graph opinions must have the same keys"
            assert set(graph_node_values['min_opinion'].keys(
            )) == nodes_set, "All graph min_opinions must have the same keys"
            assert set(graph_node_values['max_opinion'].keys(
            )) == nodes_set, "All graph max_opinions must have the same keys"

        new_nodes_set = [key for key in nodes_set if self.lazy_hari_graphs[0].nodes[key].get(
            "size", 1) >= minimum_cluster_size]

        # Find the opinion furthest from 0.5 among all opinions of all nodes in all graphs
        vmax = max(
            abs(graph.nodes[key]['opinion'])
            for i in indices_to_plot
            for key in new_nodes_set
            for graph in [self.lazy_hari_graphs[i]]
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        x = indices_to_plot

        for key in new_nodes_set:
            # Extract node_values only from graphs with indices in indices_to_plot
            node_values_list = [
                self.lazy_hari_graphs[i].node_values for i in indices_to_plot]
            y = [nv['opinion'][key] for nv in node_values_list]
            min_y = [nv['min_opinion'][key] for nv in node_values_list]
            max_y = [nv['max_opinion'][key] for nv in node_values_list]

            # Extract the reference color from the graph at reference_index and map it to the adjusted colormap range
            ref_opinion = self.lazy_hari_graphs[reference_index].node_values['opinion'][key]
            color = plt.get_cmap(colormap)((ref_opinion+vmax) / (2*vmax))

            # Plotting the semitransparent region between min and max opinions
            ax.fill_between(x, min_y, max_y, color=color, alpha=0.2)

            # Plotting the line for the opinions
            ax.plot(x, y, color=color, label=f'Node {key}')

        ax.set_title(f"Node Values Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()

        # If save contains a filename, save the plot to that file
        if save:
            plt.savefig(save)

        # If show is True, display the plot
        if show:
            plt.show()

        # Close the plot to free up resources
        plt.close(fig)

    def draw_dynamic_graphs(self, reference_index: int = 0, save: Optional[Union[str, List[str]]] = None,
                            show_timestamp: bool = False, **kwargs):
        """
        Draws the first graph of each group, using the positions of the nodes as determined by the graph 
        at the reference_index.

        :param reference_index: int, optional
            Index of the graph whose node positions will be used as a reference for drawing all other graphs.
            Default is 0.

        :param save: str, list of str, or None, optional
            If str, treated as a directory and each graph is saved with a filename '{i}.png'.
            If list of str, each string in the list is treated as a filename for saving the corresponding graph.
            If None, graphs are not saved.
            Default is None.

        :param show_timestamp: bool, optional
            If True, the index of each graph (of the group's first graph) is displayed as a timestamp in the 
            bottom right corner of the plot. Default is False.

        :param kwargs: keyword arguments
            Additional settings that will be provided to the draw method of each graph.

        :return: None
        """

        # Get the reference graph and its node positions
        pos = self.lazy_hari_graphs[reference_index].position_nodes()

        # Extracting indices of the first image of each group
        indices_to_plot = [group[0] for group in self.groups]

        # Check the save parameter and prepare the save_filepaths list
        if isinstance(save, str):
            os.makedirs(save, exist_ok=True)
            save_filepaths = [os.path.join(
                save, f'{i}.png') for i in indices_to_plot]
        elif isinstance(save, list):
            assert len(save) == len(
                indices_to_plot), "The length of 'save' list must be equal to the number of group's first graphs."
            save_filepaths = save
        else:
            save_filepaths = [None] * len(indices_to_plot)

        # Draw each graph with the reference positions
        for i, save_filepath in zip(indices_to_plot, save_filepaths):
            graph = self.lazy_hari_graphs[i]
            fig, ax = plt.subplots(figsize=(10, 7))
            bottom_right_text = f't = {i}' if show_timestamp else None
            graph.draw(pos=pos, save_filepath=save_filepath, fig=fig,
                       ax=ax, bottom_right_text=bottom_right_text, **kwargs)
            plt.close()

    def plot_neighbor_mean_opinion(self, save: Optional[Union[str, List[str]]] = None, show_timestamp: bool = False, cmax=None, uninit=False, num_images=None, **kwargs):
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

        # If num_images is defined and less than the number of groups, select evenly spaced indices
        if num_images is not None and num_images < len(indices_to_plot):
            step = len(indices_to_plot) / num_images
            indices_to_plot = [
                indices_to_plot[int(i*step)] for i in range(num_images)]

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
            1 for lazy_graph in self.lazy_hari_graphs if lazy_graph.is_initialized())
        total_count = len(self.lazy_hari_graphs)
        return f"<HariDynamics object with {total_count} LazyHariGraphs ({initialized_count} initialized)>"
