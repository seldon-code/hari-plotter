from __future__ import annotations

import math
import os
import shutil
import tempfile
from typing import Dict, List, Optional, Union, Sequence

import imageio
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from .interface import Interface


class PlotSaver:
    """
    A utility class to handle the saving and display of plots.

    It provides functionality to save individual plots, display them,
    and even create GIFs from a sequence of plots.
    """

    def __init__(self, mode: Union[str, List[str]] = 'show',
                 save_path: Optional[str] = None,
                 gif_path: Optional[str] = None) -> None:
        """
        Initialize the PlotSaver instance.

        Args:
            mode (Union[str, List[str]]): The mode(s) in which to operate. 
                It can be a list or a single string, e.g. ['show', 'save'] or 'gif'.
            save_path (Optional[str]): Path to save individual plots (used if 'save' is in mode).
            gif_path (Optional[str]): Path to save gif (used if 'gif' is in mode).
        """
        # Ensure mode is a list even if a single mode string is provided
        self.mode = mode if isinstance(mode, list) else [mode]
        self.save_path = save_path
        self.gif_path = gif_path
        self.saved_images = []
        self.temp_dir = None

    @staticmethod
    def is_inside_jupyter() -> bool:
        """
        Determine if the current environment is Jupyter Notebook.

        Returns:
            bool: True if inside Jupyter Notebook, False otherwise.
        """
        try:
            get_ipython
            return True
        except NameError:
            return False

    def __enter__(self) -> PlotSaver:
        """
        Entry point for the context manager.

        Returns:
            PlotSaver: The current instance of the PlotSaver.
        """
        return self

    def save(self, fig: matplotlib.figure.Figure) -> None:
        """
        Save and/or display the provided figure based on the specified mode.

        Args:
            fig (matplotlib.figure.Figure): The figure to be saved or displayed.
        """
        plt.tight_layout()

        # Save the figure if 'save' mode is active and save_path is provided
        if 'save' in self.mode and self.save_path:
            path = self.save_path.format(len(self.saved_images))
            fig.savefig(path)
            self.saved_images.append(path)
        # If only 'gif' mode is selected, save figure to a temp directory
        elif 'gif' in self.mode and not self.save_path:
            if not self.temp_dir:
                self.temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(
                self.temp_dir, "tmp_plot_{}.png".format(len(self.saved_images)))
            fig.savefig(temp_path)
            self.saved_images.append(temp_path)

        # Show the figure if 'show' mode is active
        if 'show' in self.mode:
            if self.is_inside_jupyter():
                # In Jupyter, let the figure be displayed automatically
                display(fig)
            else:
                # Outside Jupyter, use fig.show() to display the figure
                fig.show()

        # Close the figure after processing
        plt.close(fig)

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[object]) -> None:
        """
        Exit point for the context manager.

        Args:
            exc_type (Optional[type]): The exception type if raised inside the context.
            exc_val (Optional[Exception]): The exception instance if raised inside the context.
            exc_tb (Optional[object]): The traceback if an exception was raised inside the context.
        """
        # If 'gif' mode is active and gif_path is provided, create a GIF from the saved images
        if 'gif' in self.mode and self.gif_path and self.saved_images:
            with imageio.get_writer(self.gif_path, mode='I') as writer:
                for img_path in self.saved_images:
                    image = imageio.imread(img_path)
                    writer.append_data(image)

        # Cleanup temporary directory if it was used
        if self.temp_dir:
            shutil.rmtree(self.temp_dir)


class Plotter:
    _parameter_dict = {'opinion': 'Node Opinion',
                       'cluster_size': 'Cluster size',
                       'importance': 'Node Importance',
                       'label': 'Node Label',
                       'neighbor_mean_opinion': 'Node Neighbor Mean Opinion',
                       'activity': 'Node Activity',
                       'inner_opinions': 'Node Inner Opinions',
                       'max_opinion': 'Node Max Opinion',
                       'min_opinion': 'Node Min Opinion'}

    def __init__(self, interface: Interface):
        """
        Initialize the Plotter object with the given Interface instance.

        Parameters:
        -----------
        interface : Interface
            Interface instance to be used for plotting.
        """
        self.interface: Interface = interface

    @property
    def available_parameters(self) -> list:
        """
        Retrieves the list of available parameters/methods from the interface.

        Returns:
            list: A list of available parameters or methods.
        """
        return self.interface.available_parameters

    @classmethod
    def create_plotter(cls, data) -> Plotter:
        """
        Class method to create a Plotter instance given data.

        Parameters:
        -----------
        data : Any
            Data used to create the Interface instance.

        Returns:
        --------
        Plotter
            Initialized Plotter object.
        """
        interface = Interface.create_interface(data)
        return cls(interface)

    @staticmethod
    def tanh_axis_labels(ax: plt.Axes, axis: str):
        """
        Adjust axis labels for tanh scaling.

        Parameters:
        -----------
        ax : plt.Axes
            The Axes object to which the label adjustments should be applied.
        axis : str
            Which axis to adjust. Choices: 'x', 'y', or 'both'.
        """
        tickslabels = [-np.inf] + list(np.arange(-2.5, 2.6, 0.5)) + [np.inf]
        ticks = np.tanh(tickslabels)

        tickslabels = [r'-$\infty$' if label == -np.inf else r'$\infty$' if label == np.inf else label if abs(
            label) <= 1.5 else None for label in tickslabels]

        minor_tickslabels = np.arange(-2.5, 2.6, 0.1)
        minor_ticks = np.tanh(minor_tickslabels)

        if axis in ['x', 'both']:
            ax.set_xticks(ticks)
            ax.set_xticklabels(tickslabels)
            ax.set_xticks(minor_ticks, minor=True)
            ax.set_xticklabels([], minor=True)
            ax.set_xlim([-1, 1])

        if axis in ['y', 'both']:
            ax.set_yticks(ticks)
            ax.set_yticklabels(tickslabels)
            ax.set_yticks(minor_ticks, minor=True)
            ax.set_yticklabels([], minor=True)
            ax.set_ylim([-1, 1])

    @staticmethod
    def plot_histogram(ax: plt.Axes,
                       values: List[float],
                       scale: Optional[str] = 'linear',
                       rotated: Optional[bool] = False,
                       extent: Optional[Sequence[float] | None] = None):
        """
        Plot a histogram on the given ax with the provided values data.

        Parameters:
        -----------
        ax : plt.Axes
            Axes object where the histogram will be plotted.
        values : list[float]
            List containing opinion values.
        scale : str, optional
            The scale for the x-axis. Options: 'linear' or 'tanh'.
        rotated : bool, optional
            If True, the histogram is rotated to be horizontal.
        extent : Optional[Sequence[float] | None]
            Limits of the histogram
        """

        values = np.array(values)
        valid_indices = ~np.isnan(values)
        values = values[valid_indices]

        if scale == 'tanh':
            values = np.tanh(values)

        if extent is None:
            extent = [-1, 1] if scale == 'tanh' else [
                np.nanmin(values), np.nanmax(values)]

        if rotated:
            sns.kdeplot(y=values, ax=ax, fill=True)
            sns.histplot(y=values, kde=False, ax=ax,
                         binrange=extent, element="step", fill=False, stat="density")
            if scale == 'tanh':
                Plotter.tanh_axis_labels(ax=ax, axis='y')
        else:
            sns.kdeplot(data=values, ax=ax, fill=True)
            sns.histplot(data=values, kde=False, ax=ax,
                         binrange=extent, element="step", fill=False, stat="density")
            if scale == 'tanh':
                Plotter.tanh_axis_labels(ax=ax, axis='x')

        if rotated:
            ax.set_ylim(extent[0], extent[1])
        else:
            ax.set_xlim(extent[0], extent[1])

    @staticmethod
    def plot_hexbin(ax: plt.Axes,
                    x_values: List[float], y_values: List[float],
                    extent: Optional[Sequence[float] | None] = None,
                    colormap: Optional[str] = 'inferno',
                    cmax: Optional[float | None] = None,
                    scale: Optional[Sequence | None] = None,
                    show_colorbar: Optional[bool] = False):
        """
        Plot a hexbin on the given ax with the provided x and y values.

        Parameters:
        -----------
        ax : plt.Axes
            Axes object where the hexbin will be plotted.
        x_values : list[float]
            List containing x-values.
        y_values : list[float]
            List containing y-values.
        extent : list[float], optional
            The bounding box in data coordinates that the hexbin should fill.
        colormap : str, optional
            The colormap to be used for hexbin coloring.
        cmax : float, optional
            The maximum number of counts in a hexbin for colormap scaling.
        scale : list, optional
            Scale for the plot values (x and y). Options: 'linear' or 'tanh'. Default is 'linear' for both.
        show_colorbar : bool, optional
        """

        scale = ['linear', 'linear'] if scale is None else scale

        x_values = np.array(x_values)
        y_values = np.array(y_values)
        # Find indices where neither x_values nor y_values are NaN
        valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)

        # Filter the values using these indices
        x_values = x_values[valid_indices]
        y_values = y_values[valid_indices]

        tanh_axis = 'both' if scale[0] == 'tanh' and scale[1] == 'tanh' else 'x' if scale[
            0] == 'tanh' else 'y' if scale[1] == 'tanh' else None

        if scale[0] == 'tanh':
            x_values = np.tanh(x_values)

        if scale[1] == 'tanh':
            y_values = np.tanh(y_values)

        if extent is None:
            x_extent = [-1, 1] if scale[0] == 'tanh' else [
                np.nanmin(x_values), np.nanmax(x_values)]
            y_extent = [-1, 1] if scale[1] == 'tanh' else [
                np.nanmin(y_values), np.nanmax(y_values)]

            extent = x_extent+y_extent

        delta_x = 0.1*(extent[1]-extent[0])
        x_field_extent = [extent[0]-delta_x, extent[1]+delta_x]

        delta_y = 0.1*(extent[3]-extent[2])
        y_field_extent = [extent[2]-delta_y, extent[3]+delta_y]

        field_extent = x_field_extent + y_field_extent

        ax.imshow([[0, 0], [0, 0]], cmap=colormap,
                  interpolation='nearest', aspect='auto', extent=field_extent)

        hb = ax.hexbin(x_values, y_values, gridsize=50, cmap=colormap,
                       bins='log', extent=extent, vmax=cmax)

        if tanh_axis is not None:
            Plotter.tanh_axis_labels(ax=ax, axis=tanh_axis)

        # Create a background filled with the `0` value of the colormap
        ax.imshow([[0, 0], [0, 0]], cmap=colormap,
                  interpolation='nearest', aspect='auto', extent=extent)
        # Create the hexbin plot

        hb = ax.hexbin(x_values, y_values, gridsize=50, cmap=colormap,
                       bins='log', extent=extent, vmax=cmax)

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        if show_colorbar:
            plt.colorbar(hb, ax=ax)

    def draw(self,
             mode: str = 'show',
             save_dir: Optional[str] = None,
             gif_path: Optional[str] = None,
             pos: Optional[Dict[Union[int, str], tuple]] = None,
             node_info_mode: str = 'none',
             use_node_color: bool = True,
             use_edge_thickness: bool = True,
             show_edge_influences: bool = False,
             node_size_multiplier: int = 200,
             arrowhead_length: float = 0.2,
             arrowhead_width: float = 0.2,
             min_line_width: float = 0.1,
             max_line_width: float = 3.0,
             seed: Optional[int] = None,
             show_time: bool = False,
             name: str = 'graph') -> None:
        """
        Draws a graphical representation of the graphs in the interface. Node positions are determined 
        by the graph at the reference index.

        Parameters:
        -----------
        mode : str
            How to display the graph. If 'show', graphs are displayed directly. The default is 'show'.

        save_dir : Optional[str]
            Directory to save drawn graphs. If provided, each graph is saved as '{i}.png'.

        gif_path : Optional[str]
            Path to save animation as gif, if needed.

        pos : Optional[Dict[Union[int, str], tuple]]
            Dictionary of node positions as coordinates. If not provided, spring layout positions nodes.

        node_info_mode : str
            Determines node information display. Options: 'opinions', 'ids', 'labels', 'cluster_size'. Default is 'none'.

        use_node_color : bool
            If True, colors nodes based on attributes. Default is True.

        use_edge_thickness : bool
            If True, edge thickness varies based on influence attributes. Default is True.

        show_edge_influences : bool
            If True, shows influence values on edges. Default is False.

        node_size_multiplier : int
            Multiplier for node size to enhance visibility. Default is 200.

        arrowhead_length : float
            Arrowhead length. Default is 0.2.

        arrowhead_width : float
            Arrowhead width. Default is 0.2.

        min_line_width : float
            Minimum line width for edges. Default is 0.1.

        max_line_width : float
            Maximum line width for edges. Default is 3.0.

        seed : Optional[int]
            Seed for reproducibility in node positioning.

        show_time : bool
            If True, displays timestamp of each graph in the plot's bottom-right corner. Default is False.

        name : str
            Prefix for saved filenames. Default is 'graph'.

        Returns:
        --------
        None
        """

        with PlotSaver(mode=mode, save_path=f"{save_dir}/{name}_" + "{}.png", gif_path=gif_path) as saver:
            for data in self.interface.images():
                image = data['image']
                fig, ax = plt.subplots(figsize=(10, 7))

                if pos is None:
                    pos = image.position_nodes(seed=seed)

                node_attributes = nx.get_node_attributes(image, 'opinion')
                edge_attributes = nx.get_edge_attributes(image, 'influence')

                # Prepare Node Labels
                node_labels = {}
                match node_info_mode:
                    case 'opinion':
                        node_labels = {node: f"{opinion:.2f}" for node,
                                       opinion in node_attributes.items()}
                    case 'ids':
                        node_labels = {node: f"{node}" for node in image.nodes}
                    case 'labels':
                        for node in image.nodes:
                            label = image.nodes[node].get('label', None)
                            if label is not None:
                                node_labels[node] = ','.join(map(str, label))
                            else:  # If label is not defined, show id instead
                                node_labels[node] = str(node)
                    case 'cluster_size':
                        for node in image.nodes:
                            label_len = len(
                                image.nodes[node].get('label', [node]))
                            node_labels[node] = str(label_len)

                # Prepare Node Colors
                if use_node_color:
                    node_colors = [cm.bwr(opinion)
                                   for opinion in node_attributes.values()]
                else:
                    node_colors = 'lightblue'

                # Prepare Edge Widths
                if use_edge_thickness:

                    # Gather edge weights
                    edge_weights = list(edge_attributes.values())

                    # Scale edge weights non-linearly
                    # or np.log1p(edge_weights) for logarithmic scaling
                    scaled_weights = np.sqrt(edge_weights)

                    # Normalize scaled weights to range [min_line_width,
                    # max_line_width]
                    max_scaled_weight = max(scaled_weights)
                    min_scaled_weight = min(scaled_weights)

                    edge_widths = [
                        min_line_width + (max_line_width - min_line_width) * (weight -
                                                                              min_scaled_weight) / (max_scaled_weight - min_scaled_weight)
                        for weight in scaled_weights
                    ]

                else:
                    # Default line width applied to all edges
                    edge_widths = [1.0] * image.number_of_edges()

                # Prepare Edge Labels
                edge_labels = None
                if show_edge_influences:
                    edge_labels = {(u, v): f"{influence:.2f}" for (u, v),
                                   influence in edge_attributes.items()}

                # Calculate Node Sizes
                node_sizes = []
                for node in image.nodes:
                    label_len = len(image.nodes[node].get('label', [node]))
                    size = node_size_multiplier * \
                        math.sqrt(label_len)  # Nonlinear scaling
                    node_sizes.append(size)

                # Draw Nodes and Edges
                nx.draw_networkx_nodes(
                    image, pos, node_color=node_colors, node_size=node_sizes, ax=ax)

                for (u, v), width in zip(image.edges(), edge_widths):
                    # Here, node_v_size and node_u_size represent the sizes (or the
                    # "radii") of the nodes.
                    node_v_size = node_sizes[list(image.nodes).index(v)]
                    node_u_size = node_sizes[list(image.nodes).index(u)]

                    # Adjust the margins based on node sizes to avoid collision with
                    # arrowheads and to avoid unnecessary gaps.
                    target_margin = 5 * node_v_size / node_size_multiplier
                    source_margin = 5 * node_u_size / node_size_multiplier

                    if image.has_edge(v, u):
                        nx.draw_networkx_edges(image, pos, edgelist=[(u, v)], width=width, connectionstyle='arc3,rad=0.3',
                                               arrowstyle=f'->,head_length={arrowhead_length},head_width={arrowhead_width}', min_target_margin=target_margin, min_source_margin=source_margin)
                    else:
                        nx.draw_networkx_edges(image, pos, edgelist=[(
                            u, v)], width=width, arrowstyle=f'-|>,head_length={arrowhead_length},head_width={arrowhead_width}', min_target_margin=target_margin, min_source_margin=source_margin)

                # Draw Labels
                nx.draw_networkx_labels(image, pos, labels=node_labels)
                if edge_labels:
                    nx.draw_networkx_edge_labels(
                        image, pos, edge_labels=edge_labels)
                if show_time:
                    ax.set_title(f't = {data["time"]}')

                saver.save(fig)

    def plot_1D_distribution(self, x_parameter: str,
                             mode: str = 'show',
                             save_dir: Optional[str] = None,
                             gif_path: Optional[str] = None,
                             show_time: bool = False,
                             name: str = 'opinion_histogram',
                             scale: str = 'linear') -> None:
        """
        Plots the histogram of x_parameter in the graphs.

        Parameters:
        -----------
        x_parameter : str
            Parameter to show on x axis
        mode : str
            How to display the histogram. If 'show', it's displayed directly. Default is 'show'.

        save_dir : Optional[str]
            Directory to save the histogram. If provided, the histogram is saved as '{i}.png'.

        gif_path : Optional[str]
            Path to save animation as gif, if needed.

        show_time : bool
            If True, includes the time in the histogram's title. Default is False.

        name : str
            Prefix for saved filenames. Default is 'opinion_histogram'.

        scale : str
            The x-axis scale. Can be 'linear' or 'tanh'. Default is 'linear'.

        Returns:
        --------
        None
        """

        with PlotSaver(mode=mode, save_path=f"{save_dir}/{name}_" + "{}.png", gif_path=gif_path) as saver:
            for group_data in self.interface.mean_group_values_iterator([x_parameter]):
                fig, ax = plt.subplots(figsize=(10, 7))

                Plotter.plot_histogram(
                    ax, group_data['data'][x_parameter], scale=scale)

                ax.set_xlabel(self._parameter_dict.get(
                    x_parameter, x_parameter))
                ax.set_ylabel('Number of Nodes')
                group_data["time"] = group_data["time"]

                if show_time:
                    if isinstance(group_data["time"], float):
                        formatted_time = "{:.2f}".format(group_data["time"])
                    else:
                        formatted_time = str(group_data["time"])

                    ax.set_title(f't = {formatted_time}')

                saver.save(fig)

    def plot_opinions(self,
                      mode: str = 'show',
                      save_dir: Optional[str] = None,
                      gif_path: Optional[str] = None,
                      reference_index: int = -1,
                      minimum_cluster_size: int = 1,
                      colormap: str = 'coolwarm',
                      name: str = 'opinions_dynamics',
                      width_threshold: float = 1,
                      scale: str = 'linear',
                      show_legend: bool = True):
        """
        Plot the opinions of nodes over time.

        Parameters:
        - mode (str): Mode of the plot. Default is 'show'.
        - save_dir (str, optional): Directory to save the plot. Default is None.
        - gif_path (str, optional): Path to save the gif. Default is None.
        - reference_index (int): Index to refer for plotting. Default is -1.
        - minimum_cluster_size (int): Minimum size of the cluster for plotting. Default is 1.
        - colormap (str): Colormap for the plot. Default is 'coolwarm'.
        - name (str): Name of the plot. Default is 'opinions_dynamics'.
        - width_threshold (float): Threshold for the width of the plot. Default is 1.
        - scale (str): Scale for the plot values. Options: 'linear' or 'tanh'. Default is 'linear'.
        - show_legend (bool): Whether to show the legend. Default is True.
        """
        all_nodes_data = {}
        time_array = []

        for group_data in self.interface.mean_group_values_iterator(['opinion', 'cluster_size', 'min_opinion', 'max_opinion']):
            time_array.append(group_data['time'])

            for node, opinion, size, max_opinion, min_opinion in zip(
                    group_data['data']['node'], group_data['data']['opinion'], group_data['data']['cluster_size'], group_data['data']['max_opinion'], group_data['data']['min_opinion']):
                if node not in all_nodes_data:

                    all_nodes_data[node] = {
                        'opinion': [],
                        'cluster_size': [],
                        'max_opinion': [],
                        'min_opinion': []
                    }

                all_nodes_data[node]['opinion'].append(opinion)
                all_nodes_data[node]['cluster_size'].append(size)
                all_nodes_data[node]['max_opinion'].append(max_opinion)
                all_nodes_data[node]['min_opinion'].append(min_opinion)

        # Filter nodes where the size is always below the threshold
        nodes_to_remove = [node for node, data in all_nodes_data.items() if all(
            size < minimum_cluster_size for size in data['cluster_size'])]

        # Remove those nodes from all_nodes_data
        for node in nodes_to_remove:
            del all_nodes_data[node]

        # Initialize values with extreme opposites for comparison
        smallest_opinion = float('inf')
        highest_opinion = float('-inf')

        for node, data in all_nodes_data.items():
            current_min = min(data['opinion'])
            current_max = max(data['opinion'])

            # Update the global smallest and highest values if needed
            if current_min < smallest_opinion:
                smallest_opinion = current_min
            if current_max > highest_opinion:
                highest_opinion = current_max

        vmax = max(abs(smallest_opinion), abs(highest_opinion))
        absolute_width_threshold = 0.01 * width_threshold * vmax

        with PlotSaver(mode=mode, save_path=f"{save_dir}/{name}_" + "{}.png", gif_path=gif_path) as saver:
            fig, ax = plt.subplots(figsize=(10, 7))

            for key, data in all_nodes_data.items():
                if scale == 'tanh':
                    vmax_tanh = np.tanh(vmax)
                    y = np.tanh(data['opinion'])
                    min_y = np.tanh(data['min_opinion'])
                    max_y = np.tanh(data['max_opinion'])

                    ref_opinion = np.tanh(y[reference_index])
                    color = plt.get_cmap(colormap)(
                        (ref_opinion + vmax_tanh) / (2 * vmax_tanh))

                    # Check if the width difference exceeds the threshold anywhere
                    widths = [m - n for m, n in zip(max_y, min_y)]
                    if any(w > absolute_width_threshold for w in widths):
                        # Plotting the semitransparent region between min and max
                        # opinions
                        ax.fill_between(time_array, min_y, max_y,
                                        color=color, alpha=0.2)

                    # Plotting the line for the opinions
                    ax.plot(time_array, y, color=color, label=f'Node {key}')

                    Plotter.tanh_axis_labels(ax=ax, axis='y')
                else:
                    y = data['opinion']
                    min_y = data['min_opinion']
                    max_y = data['max_opinion']

                    ref_opinion = y[reference_index]
                    color = plt.get_cmap(colormap)(
                        (ref_opinion + vmax) / (2 * vmax))

                    # Check if the width difference exceeds the threshold anywhere
                    widths = [m - n for m, n in zip(max_y, min_y)]
                    if any(w > absolute_width_threshold for w in widths):
                        # Plotting the semitransparent region between min and max
                        # opinions
                        ax.fill_between(time_array, min_y, max_y,
                                        color=color, alpha=0.2)

                    # Plotting the line for the opinions
                    ax.plot(time_array, y, color=color, label=f'Node {key}')

            ax.set_title(f"Opinions Over Time")
            ax.set_xlabel("Time")
            ax.set_ylabel("Opinion")
            if show_legend:
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

            ax.set_xlim([min(time_array), max(time_array)])

            saver.save(fig)

    def plot_2D_distribution(self, x_parameter: str, y_parameter: str,
                             mode: str = 'show',
                             save_dir: Optional[str] = None,
                             gif_path: Optional[str] = None,
                             show_time: bool = False,
                             extent: Optional[Union[list, tuple]] = None,
                             cmax: Optional[float] = None,
                             colormap: str = 'inferno',
                             scale: list = None,
                             name: str = '2D_distribution'):
        """
        Plot the mean opinion of the neighbors of nodes.

        Parameters:
        - x_parameter (str): Parameter to show on x axis
        - y_parameter (str): Parameter to show on y axis
        - mode (str): Mode of the plot. Default is 'show'.
        - save_dir (str, optional): Directory to save the plot. Default is None.
        - gif_path (str, optional): Path to save the gif. Default is None.
        - show_time (bool): Whether to show time in the plot. Default is False.
        - extent (list/tuple, optional): Range of the plot. Default is None.
        - cmax (float, optional): Maximum value for the color scale. Default is None.
        - colormap (str): Colormap for the plot. Default is 'inferno'.
        - scale (List): Scale for the plot values (x and y). Options: 'linear' or 'tanh'. Default is 'linear' for both.
        - name (str): Name of the plot. Default is '2D_distribution'.
        """

        if scale is None:
            scale = ['linear', 'linear']

        with PlotSaver(mode=mode, save_path=f"{save_dir}/{name}_" + "{}.png", gif_path=gif_path) as saver:
            for group_data in self.interface.group_values_iterator([x_parameter, y_parameter]):

                x_values = []
                y_values = []
                for node, node_data in group_data['data'].items():
                    x_values.extend(node_data[x_parameter])
                    y_values.extend(node_data[y_parameter])

                fig, ax = plt.subplots(figsize=(10, 7))
                Plotter.plot_hexbin(ax=ax, x_values=x_values, y_values=y_values, extent=extent,
                                    colormap=colormap, cmax=cmax, scale=scale)

                if show_time:
                    if isinstance(group_data["time"], float):
                        formatted_time = "{:.2f}".format(group_data["time"])
                    else:
                        formatted_time = str(group_data["time"])

                    ax.set_title(f't = {formatted_time}')

                ax.set_xlabel(self._parameter_dict.get(
                    x_parameter, x_parameter))
                ax.set_ylabel(self._parameter_dict.get(
                    y_parameter, y_parameter))

                saver.save(fig)

    def plot_2D_distribution_extended(self, x_parameter: str, y_parameter: str,
                                      mode: str = 'show',
                                      save_dir: Optional[str | None] = None,
                                      gif_path: Optional[str | None] = None,
                                      show_time: Optional[bool] = False,
                                      extent: Optional[Union[list,
                                                             tuple] | None] = None,
                                      cmax: Optional[float | None] = None,
                                      colormap: Optional[str] = 'inferno',
                                      scale: Optional[list | None] = None,
                                      name: Optional[str] = '2D_distribution_extended'):
        """
        Extended plot of the mean opinion of the neighbors of nodes.

        Parameters:
        - x_parameter (str): Parameter to show on x axis
        - y_parameter (str): Parameter to show on y axis
        - mode (str): Mode of the plot. Default is 'show'.
        - save_dir (str, optional): Directory to save the plot. Default is None.
        - gif_path (str, optional): Path to save the gif. Default is None.
        - show_time (bool): Whether to show time in the plot. Default is False.
        - extent (list/tuple, optional): Range of the plot. Default is None.
        - cmax (float, optional): Maximum value for the color scale. Default is None.
        - colormap (str): Colormap for the plot. Default is 'inferno'.
        - scale (List): Scale for the plot values (x and y). Options: 'linear' or 'tanh'. Default is 'linear' for both.
        - name (str): Name of the plot. Default is '2D_distribution_extended'.
        """

        scale = ['linear', 'linear'] if scale is None else scale

        with PlotSaver(mode=mode, save_path=f"{save_dir}/{name}_" + "{}.png", gif_path=gif_path) as saver:
            for group_data in self.interface.group_values_iterator([x_parameter, y_parameter]):

                x_values = []
                y_values = []
                for node, node_data in group_data['data'].items():
                    x_values.extend(node_data[x_parameter])
                    y_values.extend(node_data[y_parameter])

                x_values = np.array(x_values)
                y_values = np.array(y_values)
                # Find indices where neither x_values nor y_values are NaN
                valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)

                # Filter the values using these indices
                x_values = x_values[valid_indices]
                y_values = y_values[valid_indices]

                if extent is None:
                    x_extent = [-1, 1] if scale[0] == 'tanh' else [
                        np.nanmin(x_values), np.nanmax(x_values)]
                    y_extent = [-1, 1] if scale[1] == 'tanh' else [
                        np.nanmin(y_values), np.nanmax(y_values)]

                    extent = x_extent+y_extent

                fig, axs = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={
                                        'width_ratios': [4, 1], 'height_ratios': [1, 4]})

                Plotter.plot_hexbin(
                    ax=axs[1, 0], x_values=x_values, y_values=y_values, extent=extent, cmax=cmax, scale=scale, colormap=colormap)
                Plotter.plot_histogram(
                    values=x_values, ax=axs[0, 0], scale=scale[0], extent=[extent[0], extent[1]])
                Plotter.plot_histogram(
                    values=y_values, ax=axs[1, 1], scale=scale[1], extent=[extent[2], extent[3]], rotated=True)

                axs[0, 0].set_xlim(axs[1, 0].get_xlim())
                axs[0, 0].set_xticks([])

                axs[1, 1].set_ylim(axs[1, 0].get_ylim())
                axs[1, 1].set_yticks([])

                axs[0, 1].set_xticks([])
                axs[0, 1].set_yticks([])
                for spine in axs[0, 1].spines.values():
                    spine.set_visible(False)

                if show_time:
                    if isinstance(group_data["time"], float):
                        formatted_time = "{:.2f}".format(
                            group_data["time"])
                    else:
                        formatted_time = str(group_data["time"])

                    axs[0, 0].set_title(f't = {formatted_time}')

                axs[1, 0].set_xlabel(self._parameter_dict.get(
                    x_parameter, x_parameter))
                axs[1, 0].set_ylabel(self._parameter_dict.get(
                    y_parameter, y_parameter))

                saver.save(fig)
