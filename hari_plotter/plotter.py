from __future__ import annotations

import copy
import inspect
import math
import os
import shutil
import tempfile
import warnings
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from collections import defaultdict
from typing import (Any, Dict, Iterator, List, Optional, Sequence, Tuple, Type,
                    Union)

import imageio
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from .cluster import Cluster
from .hari_graph import HariGraph
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
    _parameter_dict = {'time': 'Time',
                       'opinion': 'Node Opinion',
                       'cluster_size': 'Cluster size',
                       'importance': 'Node Importance',
                       'label': 'Node Label',
                       'neighbor_mean_opinion': 'Node Neighbor Mean Opinion',
                       'activity': 'Node Activity',
                       'inner_opinions': 'Node Inner Opinions',
                       'max_opinion': 'Node Max Opinion',
                       'min_opinion': 'Node Min Opinion'}

    _plot_methods = {}

    def __init__(self, interface: Interface, figsize=None):
        """
        Initialize the Plotter object with the given Interface instance.

        Parameters:
        -----------
        interface : Interface
            Interface instance to be used for plotting.
        """
        self.interface: Interface = interface
        self.plots = [[[]]]
        self._figsize = figsize
        self.num_rows = 1
        self.num_cols = 1
        self.size_ratios = [[1], [1]]

    @property
    def figsize(self):
        """
        Property to get the figure size. If a size was set during initialization, it returns that size.
        Otherwise, it calculates the size based on the sum of size_ratios.

        Returns:
            List[int]: The size of the figure as [width, height].
        """
        if self._figsize is not None:
            return self._figsize
        else:
            # Calculate size based on the sum of size ratios
            width = np.sum(self.size_ratios[0])*2
            height = np.sum(self.size_ratios[1])*2
            return [width, height]

    @figsize.setter
    def figsize(self, value):
        """
        Property setter to set the figure size.

        Parameters:
            value (List[int]): The size of the figure as [width, height].
        """
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(
                "fig_size must be a list or tuple of two elements: [width, height]")
        self._figsize = value

    @classmethod
    def plot_type(cls, plot_name):
        """
        Decorator to register a plot method.

        Parameters:
        plot_name (str): Name of the plot type.
        instructions (dict): Provides the information about the plot type and function that is used for creating the plot.
        """
        def decorator(plot_func):
            if plot_name in cls._plot_methods:
                raise ValueError(f"Plot type {plot_name} is already defined.")

            # sig = inspect.signature(plot_func)
            # method_parameters = {name: param.default for name,
            #                      param in sig.parameters.items() if name != 'ax'}

            cls._plot_methods[plot_name] = plot_func
            # cls._plot_methods[plot_name] = {plot_func,
            #     'method_parameters': method_parameters,
            #     'instructions': instructions
            # }
            return plot_func
        return decorator

    def expand_plot_grid(self, new_row, new_col):
        """
        Expand the plot grid and adjust size ratios to at least the specified number of rows and columns.
        Existing plots are preserved, and new spaces are filled with individual empty lists.
        """

        # Expand rows if needed
        while len(self.plots) <= new_row:
            self.plots.append([[] for _ in range(len(self.plots[0]))])
            self.size_ratios[1].append(1)  # Adjust height ratios

        # Expand columns if needed
        for row in self.plots:
            while len(row) <= new_col:
                row.append([])

        # Adjust width ratios for columns
        if self.plots and len(self.plots[0]) > len(self.size_ratios[0]):
            additional_cols = len(self.plots[0]) - len(self.size_ratios[0])
            self.size_ratios[0].extend([1] * additional_cols)

        # Update the stored number of rows and columns
        self.num_rows = len(self.plots)
        self.num_cols = len(self.plots[0]) if self.plots else 0

    def add_plot(self, plot_type, plot_arguments, row: int = 0, col: int = 0):
        self.expand_plot_grid(row, col)
        plot = self._plot_methods[plot_type](**plot_arguments)

        # Initialize the cell with an empty list if it's None
        if self.plots[row][col] is None:
            self.plots[row][col] = []

        # Append the new plot to the cell's list
        self.plots[row][col].append(plot)

    def plot(self, mode: str = 'show', save_dir: Optional[str] = None, gif_path: Optional[str] = None, name: str = 'opinion_histogram') -> None:
        """
        Create and display the plots based on the stored configurations.
        """
        with PlotSaver(mode=mode, save_path=f"{save_dir}/{name}_" + "{}.png", gif_path=gif_path) as saver:

            # Determine the rendering order
            render_order = self._determine_plot_order()
            # print(f'{render_order = }')

            # Data fetching for static plots
            static_data_cache_requests = [item for i in range(self.num_rows) for j in range(
                self.num_cols) for plot in self.plots[i][j] for item in plot.get_static_plot_requests()]
            dynamic_data_cache_requests = [item for i in range(self.num_rows) for j in range(
                self.num_cols) for plot in self.plots[i][j] for item in plot.get_dynamic_plot_requests()]

            # print(f'{static_data_cache_requests = }')
            # print(f'{dynamic_data_cache_requests = }')

            static_data_cache = {}

            for request in static_data_cache_requests:
                method_name = request['method']
                settings = request['settings']
                data_key = Plotter.request_to_tuple(request)
                if data_key not in static_data_cache:
                    # Fetch and cache the data for each group
                    data_for_all_groups = []
                    for group in self.interface.groups():
                        data = getattr(group, method_name)(**settings)
                        # print(f'{data = }')
                        data_for_all_groups.append(data)

                    static_data_cache[data_key] = data_for_all_groups

            # print(f'{static_data_cache = }')

            # static cache postprocessing

            # for key, values in static_data_cache.items():
            #     if 'calculate_node_values' in key:
            #         static_data_cache[key] = transform_data(values)

            # print(f'{static_data_cache = }')

            for group in self.interface.groups():

                dynamic_data_cache = {}
                axis_limits = dict()

                for request in dynamic_data_cache_requests:
                    method_name = request['method']
                    settings = request['settings']
                    # print(f'{request = }')
                    data_key = Plotter.request_to_tuple(request)
                    if data_key not in dynamic_data_cache:
                        dynamic_data_cache[data_key] = getattr(
                            group, method_name)(**settings)

                # print(f'{dynamic_data_cache = }')

                fig, axs = plt.subplots(self.num_rows, self.num_cols, figsize=self.figsize, gridspec_kw={
                    'width_ratios': self.size_ratios[0], 'height_ratios': self.size_ratios[1]})

                # Ensure axs is a 2D array for consistency
                if self.num_rows == 1 and self.num_cols == 1:
                    axs = [[axs]]  # Single plot
                elif self.num_rows == 1:
                    axs = [axs]  # Single row, multiple columns
                elif self.num_cols == 1:
                    axs = [[ax] for ax in axs]  # Multiple rows, single column

                for (i, j) in render_order:
                    # print(f'{i,j = }')
                    ax = axs[i][j]
                    for plot in self.plots[i][j]:
                        plot.plot(ax=ax, dynamic_data_cache=dynamic_data_cache,
                                  static_data_cache=static_data_cache, axis_limits=axis_limits)

                        # Store axis limits for future reference
                        axis_limits[(i, j)] = (ax.get_xlim(), ax.get_ylim())
                    if not self.plots[i][j]:
                        ax.grid(False)
                        ax.axis('off')
                        ax.set_visible(False)

    def _determine_plot_order(self):
        """
        Determine the order in which plots should be rendered based on dependencies.

        Returns:
            List of tuples: Each tuple represents the row and column indices of a plot,
            sorted in the order they should be rendered.
        """
        dependency_graph = nx.DiGraph()

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                cell_plots = self.plots[i][j]
                plot_node = (i, j)
                dependency_graph.add_node(plot_node)
                if cell_plots is not None:
                    for plot in cell_plots:
                        # Use (i, j) as the node

                        dependencies = plot.plot_dependencies()

                        for dependency in dependencies['after']:
                            dependency_graph.add_edge(dependency, plot_node)

        # print("Nodes:")
        # for node in dependency_graph.nodes():
        #     print(node)

        # # Print the edges
        # print("\nEdges:")
        # for edge in dependency_graph.edges():
        #     print(edge)

        sorted_order = self._topological_sort(dependency_graph)

        return sorted_order

    def info(self):
        info_str = ''
        if len(self.plots) == 1 and len(self.plots[0]) == 1 and len(self.plots[0][0]) == 0:
            return 'Plot is empty'
        for i, row in enumerate(self.plots):
            for j, cell in enumerate(row):
                info_str += f'\n{i,j}\n'
                if len(cell) > 0:
                    for plot in cell:
                        info_str += str(plot) + '\n'
                else:
                    info_str += 'Empty cell'
        return info_str

    def clear_plot(self, row: int, col: int):
        if row < len(self.plots) and col < len(self.plots[row]):
            self.plots[row][col] = []

    def _topological_sort(self, dependency_graph):
        """
        Perform a topological sort on the dependency graph.

        Args:
            dependency_graph (networkx.DiGraph): The dependency graph.

        Returns:
            List of tuples: Sorted order of plots.
        """
        # Check for cycles
        if not nx.is_directed_acyclic_graph(dependency_graph):
            raise ValueError("A cycle was detected in the plot dependencies.")

        # Perform topological sort
        return list(nx.topological_sort(dependency_graph))

    @property
    def available_parameters(self) -> list:
        """
        Retrieves the list of available parameters/methods from the interface.

        Returns:
            list: A list of available parameters or methods.
        """
        return self.interface.available_parameters

    @property
    def available_plot_types(self) -> list:
        """
        Retrieves the list of available parameters/methods from the interface.

        Returns:
            list: A list of available parameters or methods.
        """
        return list(self._plot_methods.keys())

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
    def tanh_axis_labels(ax: plt.Axes, scale: List[str]):
        """
        Adjust axis labels for tanh scaling.

        Parameters:
        -----------
        ax : plt.Axes
            The Axes object to which the label adjustments should be applied.
        scale : List[str]
            Which axis to adjust. Choices: 'x', 'y', or 'both'.
        """
        tickslabels = [-np.inf] + list(np.arange(-2.5, 2.6, 0.5)) + [np.inf]
        ticks = np.tanh(tickslabels)

        tickslabels = [r'-$\infty$' if label == -np.inf else r'$\infty$' if label == np.inf else label if abs(
            label) <= 1.5 else None for label in tickslabels]

        minor_tickslabels = np.arange(-2.5, 2.6, 0.1)
        minor_ticks = np.tanh(minor_tickslabels)

        if scale[0] == 'tanh':
            ax.set_xticks(ticks)
            ax.set_xticklabels(tickslabels)
            ax.set_xticks(minor_ticks, minor=True)
            ax.set_xticklabels([], minor=True)
            ax.set_xlim([-1, 1])

        if scale[1] == 'tanh':
            ax.set_yticks(ticks)
            ax.set_yticklabels(tickslabels)
            ax.set_yticks(minor_ticks, minor=True)
            ax.set_yticklabels([], minor=True)
            ax.set_ylim([-1, 1])

    @staticmethod
    def request_to_tuple(request):
        def convert(item):
            if isinstance(item, dict):
                return tuple(sorted((k, convert(v)) for k, v in item.items()))
            elif isinstance(item, list):
                return tuple(convert(v) for v in item)
            else:
                return item

        return convert(request)


class Plot(ABC):

    def _parse_axis_limit_reference(self, reference_str):
        """
        Parse the axis limit reference string.

        Args:
            reference_str (str): The reference string (e.g., 'x@1,0').

        Returns:
            tuple: A tuple containing the axis ('x' or 'y'), row index, and column index.
        """
        ref_axis, ref_indices = reference_str.split('@')
        ref_row, ref_col = map(int, ref_indices.split(','))
        return ref_axis, ref_row, ref_col

    def plot_dependencies(self):
        dependencies = {'before': [], 'after': []}

        for value in [self.x_lim, self.y_lim]:
            if isinstance(value, str):
                # Assuming format is 'x(y)@row,col'
                ref_plot = tuple(map(int, value[2:].split(',')))
                # Add edge with (row, col) only
                dependencies['after'].append(ref_plot)
        return dependencies

    def get_limits(self, axis_limits: dict):
        final_limits = []
        for i_lim in (self.x_lim, self.y_lim):
            if isinstance(i_lim, str):
                ref_axis, ref_row, ref_col = self._parse_axis_limit_reference(
                    i_lim)
                if (ref_row, ref_col) in axis_limits:
                    final_limits.append(
                        axis_limits[(ref_row, ref_col)][0 if ref_axis == 'x' else 1])
                else:
                    raise ValueError('Render order failure')
            else:
                final_limits.append(i_lim)

        return final_limits

    @staticmethod
    def transform_data(data_list):
        # Extract unique node IDs and sort them
        nodes = sorted(
            {node for data in data_list for node in data['nodes']})
        node_index = {node: i for i, node in enumerate(nodes)}

        # Extract time steps
        time_steps = [data['time'] for data in data_list]

        # Initialize parameters dictionary
        params = {key: np.full((len(nodes), len(time_steps)), np.nan)
                  for key in data_list[0] if key not in ['nodes', 'time']}

        # Fill in the parameter values
        for t, data in enumerate(data_list):
            for param in params:
                if param in data:
                    # Map each node's value to the corresponding row in the parameter's array
                    for node, value in zip(data['nodes'], data[param]):
                        idx = node_index[node]
                        params[param][idx, t] = value

        return {
            'time': np.array(time_steps),
            'nodes': nodes,
            **params
        }


@Plotter.plot_type("Histogram")
class plot_histogram(Plot):
    def __init__(self, parameters: tuple[str],
                 scale: Optional[str | None] = None,
                 rotated: Optional[bool] = False,
                 show_x_label: bool = True, show_y_label: bool = True,
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None):
        self.parameters = tuple(parameters)
        self.scale = tuple(('linear', 'linear') if scale is None else scale)
        self.rotated = rotated
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self.x_lim = x_lim
        self.y_lim = y_lim

        # print(f'{self.get_dynamic_plot_requests()[0] = }')

        self.data_key = Plotter.request_to_tuple(
            self.get_dynamic_plot_requests()[0])

    def get_static_plot_requests(self):
        return []

    def get_dynamic_plot_requests(self):
        return [{'method': 'calculate_node_values', 'settings': {'parameters': self.parameters, 'scale': self.scale}}]

    def plot(self, ax: plt.Axes, dynamic_data_cache: dict, static_data_cache: dict, axis_limits: dict):
        """
        Plot a histogram on the given ax with the provided data data.

        Parameters:
        -----------
        ax : plt.Axes
            Axes object where the histogram will be plotted.
        data : list[float]
            List containing parameter values.
        scale : str, optional
            The scale for the x-axis. Options: 'linear' or 'tanh'.
        rotated : bool, optional
            If True, the histogram is rotated to be horizontal.
        x_lim : Optional[Sequence[float] | None]
            Limits of the x-axis.
        y_lim : Optional[Sequence[float] | None]
            Limits of the y-axis.
        """

        if len(self.parameters) != 1:
            raise ValueError('Histogram expects only one parameter')

        x_lim, y_lim = self.get_limits(axis_limits)
        data = dynamic_data_cache[self.data_key]

        parameter = self.parameters[0]
        values = np.array(data[parameter])
        valid_indices = ~np.isnan(values)
        values = values[valid_indices]

        if self.rotated:
            if self.scale[1] == 'tanh':
                values = np.tanh(values)

            if y_lim is None:
                y_lim = [-1, 1] if self.scale[1] == 'tanh' else [
                    np.nanmin(values), np.nanmax(values)]

            values = values[(values >= y_lim[0]) & (values <= y_lim[1])]

            sns.kdeplot(y=values, ax=ax, fill=True)
            sns.histplot(y=values, kde=False, ax=ax,
                         binrange=y_lim, element="step", fill=False, stat="density")

            if self.show_y_label:
                ax.set_ylabel(Plotter._parameter_dict.get(
                    parameter, parameter))
            if self.show_x_label:
                ax.set_xlabel('Density')

        else:

            if self.scale[0] == 'tanh':
                values = np.tanh(values)

            if x_lim is None:
                x_lim = [-1, 1] if self.scale[0] == 'tanh' else [
                    np.nanmin(values), np.nanmax(values)]

            values = values[(values >= x_lim[0]) & (values <= x_lim[1])]

            sns.kdeplot(data=values, ax=ax, fill=True)
            sns.histplot(data=values, kde=False, ax=ax,
                         binrange=x_lim, element="step", fill=False, stat="density")

            if self.show_x_label:
                ax.set_xlabel(Plotter._parameter_dict.get(
                    parameter, parameter))
            if self.show_y_label:
                ax.set_ylabel('Density')

        Plotter.tanh_axis_labels(ax=ax, scale=self.scale)
        if x_lim is not None:
            ax.set_xlim(*x_lim)
        if y_lim is not None:
            ax.set_ylim(*y_lim)


@Plotter.plot_type("Hexbin")
class plot_hexbin(Plot):
    def __init__(self, parameters: tuple[str],
                 scale: Optional[str | None] = None,
                 rotated: Optional[bool] = False,
                 show_x_label: bool = True, show_y_label: bool = True,
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None, colormap: str = 'coolwarm', show_colorbar: bool = False):
        self.parameters = tuple(parameters)
        self.scale = tuple(('linear', 'linear') if scale is None else scale)
        self.rotated = rotated
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.colormap = colormap
        self.show_colorbar = show_colorbar

        self.data_key = Plotter.request_to_tuple(
            self.get_dynamic_plot_requests()[0])

    def get_static_plot_requests(self):
        return []

    def get_dynamic_plot_requests(self):
        return [{'method': 'calculate_node_values', 'settings': {'parameters': self.parameters, 'scale': self.scale}}]

    def plot(self, ax: plt.Axes, dynamic_data_cache: dict, static_data_cache: dict, axis_limits: dict):
        """
        Plot a hexbin on the given ax with the provided x and y values.

        Parameters:
        -----------
        ax : plt.Axes
            Axes object where the hexbin will be plotted.
        x_values, y_values : list[float]
            Lists containing x-values and y-values
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

        x_lim, y_lim = self.get_limits(axis_limits)

        data = dynamic_data_cache[self.data_key]

        x_parameter = self.parameters[0]
        y_parameter = self.parameters[1]
        x_values = np.array(data[x_parameter])
        y_values = np.array(data[y_parameter])

        # Find indices where neither x_values nor y_values are NaN
        valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)

        # Filter the values using these indices
        x_values = x_values[valid_indices]
        y_values = y_values[valid_indices]

        if self.scale[0] == 'tanh':
            x_values = np.tanh(x_values)

        if self.scale[1] == 'tanh':
            y_values = np.tanh(y_values)

        if x_lim is None:
            x_extent = [-1, 1] if self.scale[0] == 'tanh' else [
                np.nanmin(x_values), np.nanmax(x_values)]
        if y_lim is None:
            y_extent = [-1, 1] if self.scale[1] == 'tanh' else [
                np.nanmin(y_values), np.nanmax(y_values)]

        extent = x_extent+y_extent

        delta_x = 0.1*(extent[1]-extent[0])
        x_field_extent = [extent[0]-delta_x, extent[1]+delta_x]

        delta_y = 0.1*(extent[3]-extent[2])
        y_field_extent = [extent[2]-delta_y, extent[3]+delta_y]

        field_extent = x_field_extent + y_field_extent

        ax.imshow([[0, 0], [0, 0]], cmap=self.colormap,
                  interpolation='nearest', aspect='auto', extent=field_extent)

        hb = ax.hexbin(x_values, y_values, gridsize=50,
                       bins='log', extent=extent, cmap=self.colormap)

        # Create a background filled with the `0` value of the colormap
        ax.imshow([[0, 0], [0, 0]], cmap=self.colormap,
                  interpolation='nearest', aspect='auto', extent=extent)
        # Create the hexbin plot

        hb = ax.hexbin(x_values, y_values, gridsize=50, cmap=self.colormap,
                       bins='log', extent=extent)

        Plotter.tanh_axis_labels(ax=ax, scale=self.scale)

        if x_lim is not None:
            ax.set_xlim(*x_lim)
        if y_lim is not None:
            ax.set_ylim(*y_lim)

        if self.show_colorbar:
            plt.colorbar(hb, ax=ax)

        if self.show_x_label:
            ax.set_xlabel(Plotter._parameter_dict.get(
                self.parameters[0], self.parameters[0]))
        if self.show_y_label:
            ax.set_ylabel(Plotter._parameter_dict.get(
                self.parameters[1], self.parameters[1]))


@Plotter.plot_type("Scatter")
class plot_scatter(Plot):
    def __init__(self, parameters: tuple[str],
                 scale: Optional[str | None] = None,
                 rotated: Optional[bool] = False,
                 show_x_label: bool = True, show_y_label: bool = True,
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None, color: Optional[str] = 'blue', marker: str = 'o',):
        self.parameters = tuple(parameters)
        self.scale = tuple(('linear', 'linear') if scale is None else scale)
        self.rotated = rotated
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.color = color
        self.marker = marker

        self.data_key = Plotter.request_to_tuple(
            self.get_dynamic_plot_requests()[0])

    def get_static_plot_requests(self):
        return []

    def get_dynamic_plot_requests(self):
        return [{'method': 'calculate_node_values', 'settings': {'parameters': self.parameters, 'scale': self.scale}}]

    def plot(self, ax: plt.Axes, dynamic_data_cache: dict, static_data_cache: dict, axis_limits: dict):
        """
        Plot a scatter plot on the given ax with the provided x and y values.

        Parameters:
        -----------
        ax : plt.Axes
            Axes object where the scatter plot will be plotted.
        data : defaultdict[List[float]]
            A dictionary containing lists of x and y values.
        parameters : tuple[str]
            A tuple containing the names of the parameters to be plotted.
        x_lim, y_lim : Optional[Sequence[float]]
            The limits for the x and y axes.
        color : Optional[str]
            The color of the markers.
        marker : str
            The shape of the marker.
        show_x_label, show_y_label : bool
            Flags to show or hide the x and y labels.
        """

        x_lim, y_lim = self.get_limits(axis_limits)

        data = dynamic_data_cache[self.data_key]

        x_parameter, y_parameter = self.parameters
        x_values = np.array(data[x_parameter])
        y_values = np.array(data[y_parameter])

        # Remove NaN values
        valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
        x_values = x_values[valid_indices]
        y_values = y_values[valid_indices]

        if self.scale[0] == 'tanh':
            x_values = np.tanh(x_values)

        if self.scale[1] == 'tanh':
            y_values = np.tanh(y_values)

        ax.scatter(x_values, y_values, color=self.color, marker=self.marker)

        # Setting the plot limits
        if x_lim is not None:
            ax.set_xlim(*x_lim)
        if y_lim is not None:
            ax.set_ylim(*y_lim)

        Plotter.tanh_axis_labels(ax=ax, scale=self.scale)

        if self.show_x_label:
            ax.set_xlabel(Plotter._parameter_dict.get(
                x_parameter, x_parameter))

        if self.show_y_label:
            ax.set_ylabel(Plotter._parameter_dict.get(
                y_parameter, y_parameter))


@Plotter.plot_type("Cluster: Centroids")
class plot_cluster_centroids(Plot):
    def __init__(self, parameters: tuple[str], cluster_settings: dict = {},
                 scale: Optional[str | None] = None,
                 rotated: Optional[bool] = False,
                 show_x_label: bool = True, show_y_label: bool = True,
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None, resolution: int = 100):
        self.parameters = tuple(parameters)
        self.cluster_settings = cluster_settings
        if 'clusterization_parameters' not in self.cluster_settings:
            self.cluster_settings['clusterization_parameters'] = self.parameters

        # print(f'{self.cluster_settings = }')
        self.scale = tuple(('linear', 'linear') if scale is None else scale)
        self.rotated = rotated
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.resolution = resolution

        self.data_key = Plotter.request_to_tuple(
            self.get_dynamic_plot_requests()[0])

    def get_static_plot_requests(self):
        return []

    def get_dynamic_plot_requests(self):
        return [{'method': 'get_cluster', 'settings': {'parameters': self.parameters, 'scale': self.scale, 'cluster_settings': self.cluster_settings}}]

    def plot(self, ax: plt.Axes, dynamic_data_cache: dict, static_data_cache: dict, axis_limits: dict):
        """
        Plots the decision boundaries for a 2D slice of the cluster object's data.

        Args:
        - values (Cluster): A Cluster object with fitted clusters.
        - x_feature_index (int): The index of the feature to be plotted on the x-axis.
        - y_feature_index (int): The index of the feature to be plotted on the y-axis.
        - plot_limits (tuple): A tuple containing the limits of the plot: (x_min, x_max, y_min, y_max).
        - resolution (int): The number of points to generate in the mesh for the plot.

        Returns:
        None
        """

        x_lim, y_lim = self.get_limits(axis_limits)
        cluster = dynamic_data_cache[self.data_key]

        x_feature_name, y_feature_name = self.parameters

        x_feature_index, y_feature_index = cluster.get_indices_from_parameters(
            [x_feature_name, y_feature_name])

        # Plot centroids if they are 2D
        centroids = cluster.centroids()
        if centroids.shape[1] == 2:
            centroids_x = centroids[:, x_feature_index]
            centroids_y = centroids[:, y_feature_index]
            if self.scale[0] == 'tanh':
                centroids_x = np.tanh(centroids_x)
            if self.scale[1] == 'tanh':
                centroids_y = np.tanh(centroids_y)

            ax.scatter(centroids_x, centroids_y,
                       color="red",
                       label="Centroids",
                       marker="X",
                       )
        else:
            warnings.warn(
                f'centroids.shape[1] != 2, it is {centroids.shape[1]}. Cluster centroids are not shown on a plot')

        if self.show_x_label:
            ax.set_xlabel(Plotter._parameter_dict.get(
                self.parameters[0], self.parameters[0]))
        if self.show_y_label:
            ax.set_ylabel(Plotter._parameter_dict.get(
                self.parameters[1], self.parameters[1]))


@Plotter.plot_type("Cluster: Scatter")
class plot_cluster_scatter(Plot):
    def __init__(self, parameters: tuple[str], cluster_settings: dict = {},
                 scale: Optional[str | None] = None,
                 show_x_label: bool = True, show_y_label: bool = True,
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None, resolution: int = 100):
        self.parameters = tuple(parameters)
        self.cluster_settings = cluster_settings
        if 'clusterization_parameters' not in self.cluster_settings:
            self.cluster_settings['clusterization_parameters'] = self.parameters

        self.scale = tuple(('linear', 'linear') if scale is None else scale)
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.resolution = resolution

        self.data_key = Plotter.request_to_tuple(
            self.get_dynamic_plot_requests()[0])

    def get_static_plot_requests(self):
        return []

    def get_dynamic_plot_requests(self):
        return [{'method': 'get_cluster', 'settings': {'parameters': self.parameters, 'scale': self.scale, 'cluster_settings': self.cluster_settings}}]

    def plot(self, ax: plt.Axes, dynamic_data_cache: dict, static_data_cache: dict, axis_limits: dict):
        """
        Plots the decision scatter for a 2D slice of the cluster object's data.
        """

        x_lim, y_lim = self.get_limits(axis_limits)
        cluster = dynamic_data_cache[self.data_key]

        x_feature_name, y_feature_name = self.parameters

        x_feature_index, y_feature_index = cluster.get_indices_from_parameters(
            [x_feature_name, y_feature_name])

        data = cluster.get_values([x_feature_name, y_feature_name])

        for i, points in enumerate(data):
            x_points = points[:, 0]
            y_points = points[:, 1]
            if self.scale[0] == 'tanh':
                x_points = np.tanh(x_points)
            if self.scale[1] == 'tanh':
                y_points = np.tanh(y_points)

            ax.scatter(x_points, y_points, label=f"Cluster {i}")

        # Setting the plot limits
        if self.x_lim is not None:
            ax.set_xlim(*self.x_lim)
        if self.y_lim is not None:
            ax.set_ylim(*self.y_lim)

        if self.show_x_label:
            ax.set_xlabel(Plotter._parameter_dict.get(
                self.parameters[0], self.parameters[0]))
        if self.show_y_label:
            ax.set_ylabel(Plotter._parameter_dict.get(
                self.parameters[1], self.parameters[1]))


@Plotter.plot_type("Cluster: Fill")
class plot_cluster_fill(Plot):
    def __init__(self, parameters: tuple[str], cluster_settings: dict = {},
                 scale: Optional[str | None] = None,
                 show_x_label: bool = True, show_y_label: bool = True,
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None, resolution: int = 100):
        self.parameters = tuple(parameters)
        self.cluster_settings = cluster_settings
        if 'clusterization_parameters' not in self.cluster_settings:
            self.cluster_settings['clusterization_parameters'] = self.parameters

        self.scale = tuple(('linear', 'linear') if scale is None else scale)
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.resolution = resolution

        self.data_key = Plotter.request_to_tuple(
            self.get_dynamic_plot_requests()[0])

    def get_static_plot_requests(self):
        return []

    def get_dynamic_plot_requests(self):
        return [{'method': 'get_cluster', 'settings': {'parameters': self.parameters, 'scale': self.scale, 'cluster_settings': self.cluster_settings}}]

    def plot(self, ax: plt.Axes, dynamic_data_cache: dict, static_data_cache: dict, axis_limits: dict):
        x_lim, y_lim = self.get_limits(axis_limits)
        cluster = dynamic_data_cache[self.data_key]

        x_feature_name, y_feature_name = self.parameters
        x_feature_index, y_feature_index = cluster.get_indices_from_parameters(
            [x_feature_name, y_feature_name])

        xx, yy = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], self.resolution), np.linspace(
                y_lim[0], y_lim[1], self.resolution)
        )

        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_points_scaled = np.array(mesh_points)
        if self.scale[0] == 'tanh':
            mesh_points_scaled[:, 0] = np.arctanh(mesh_points_scaled[:, 0])
        if self.scale[1] == 'tanh':
            mesh_points_scaled[:, 1] = np.arctanh(mesh_points_scaled[:, 1])

        Z = cluster.predict_cluster(mesh_points_scaled)
        Z = Z.reshape(xx.shape)

        im = ax.imshow(Z, extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]],
                       origin='lower', aspect='auto', alpha=0.4, interpolation='nearest')

        if self.show_x_label:
            ax.set_xlabel(Plotter._parameter_dict.get(
                self.parameters[0], self.parameters[0]))
        if self.show_y_label:
            ax.set_ylabel(Plotter._parameter_dict.get(
                self.parameters[1], self.parameters[1]))

    def get_limits(self, axis_limits):
        default_x_lim = [-1, 1] if self.scale[0] == 'tanh' else [0, 1]
        default_y_lim = [-1, 1] if self.scale[1] == 'tanh' else [0, 1]

        x_lim = self.x_lim if self.x_lim is not None else axis_limits.get(
            'x', default_x_lim)
        y_lim = self.y_lim if self.y_lim is not None else axis_limits.get(
            'y', default_y_lim)

        return x_lim, y_lim


@Plotter.plot_type("Cluster: Degree of Membership")
class plot_cluster_degree_of_membership(Plot):
    def __init__(self, parameters: tuple[str], cluster_settings: dict = {},
                 scale: Optional[str | None] = None,
                 show_x_label: bool = True, show_y_label: bool = True,
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None, resolution: int = 100):
        self.parameters = tuple(parameters)
        self.cluster_settings = cluster_settings
        if 'clusterization_parameters' not in self.cluster_settings:
            self.cluster_settings['clusterization_parameters'] = self.parameters

        self.scale = tuple(('linear', 'linear') if scale is None else scale)
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.resolution = resolution

        self.data_key = Plotter.request_to_tuple(
            self.get_dynamic_plot_requests()[0])

    def get_static_plot_requests(self):
        return []

    def get_dynamic_plot_requests(self):
        return [{'method': 'get_cluster', 'settings': {'parameters': self.parameters, 'scale': self.scale, 'cluster_settings': self.cluster_settings}}]

    def plot(self, ax: plt.Axes, dynamic_data_cache: dict, static_data_cache: dict, axis_limits: dict):
        x_lim, y_lim = self.get_limits(axis_limits)
        cluster = dynamic_data_cache[self.data_key]

        xx, yy = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], self.resolution), np.linspace(
                y_lim[0], y_lim[1], self.resolution)
        )

        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_points_scaled = np.array(mesh_points)
        if self.scale[0] == 'tanh':
            mesh_points_scaled[:, 0] = np.arctanh(mesh_points_scaled[:, 0])
        if self.scale[1] == 'tanh':
            mesh_points_scaled[:, 1] = np.arctanh(mesh_points_scaled[:, 1])

        Z = np.array(cluster.degree_of_membership(mesh_points_scaled))
        Z = Z.max(axis=0)
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.5,
                    levels=np.linspace(0, 1, 11), cmap='viridis')

        if self.show_x_label:
            ax.set_xlabel(Plotter._parameter_dict.get(
                self.parameters[0], self.parameters[0]))
        if self.show_y_label:
            ax.set_ylabel(Plotter._parameter_dict.get(
                self.parameters[1], self.parameters[1]))

    def get_limits(self, axis_limits):
        default_x_lim = [-1, 1] if self.scale[0] == 'tanh' else [0, 1]
        default_y_lim = [-1, 1] if self.scale[1] == 'tanh' else [0, 1]

        x_lim = self.x_lim if self.x_lim is not None else axis_limits.get(
            'x', default_x_lim)
        y_lim = self.y_lim if self.y_lim is not None else axis_limits.get(
            'y', default_y_lim)

        return x_lim, y_lim


@Plotter.plot_type("Cluster: sns")
class plot_cluster_sns(Plot):
    def __init__(self, parameters: tuple[str], cluster_settings: dict = {},
                 scale: Optional[str | None] = None,
                 show_x_label: bool = True, show_y_label: bool = True,
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None, resolution: int = 100):
        self.parameters = tuple(parameters)
        self.cluster_settings = cluster_settings
        if 'clusterization_parameters' not in self.cluster_settings:
            self.cluster_settings['clusterization_parameters'] = self.parameters

        self.scale = tuple(('linear', 'linear') if scale is None else scale)
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.resolution = resolution

        self.data_key = Plotter.request_to_tuple(
            self.get_dynamic_plot_requests()[0])

    def get_static_plot_requests(self):
        return []

    def get_dynamic_plot_requests(self):
        return [{'method': 'get_cluster', 'settings': {'parameters': self.parameters, 'scale': self.scale, 'cluster_settings': self.cluster_settings}}]

    def plot(self, ax: plt.Axes, dynamic_data_cache: dict, static_data_cache: dict, axis_limits: dict):
        x_lim, y_lim = self.get_limits(axis_limits)
        cluster = dynamic_data_cache[self.data_key]

        xx, yy = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], self.resolution), np.linspace(
                y_lim[0], y_lim[1], self.resolution)
        )

        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_points_scaled = np.array(mesh_points)
        if self.scale[0] == 'tanh':
            mesh_points_scaled[:, 0] = np.arctanh(mesh_points_scaled[:, 0])
        if self.scale[1] == 'tanh':
            mesh_points_scaled[:, 1] = np.arctanh(mesh_points_scaled[:, 1])

        Z = np.array(cluster.degree_of_membership(mesh_points_scaled))
        Z = Z.reshape(-1, *xx.shape)
        Z_index = Z.argmax(axis=0)
        Z_flat = Z.max(axis=0).ravel()

        xx_flat = xx.ravel()
        yy_flat = yy.ravel()
        Z_index_flat = Z_index.ravel()

        sns.kdeplot(
            ax=ax,
            x=xx_flat,
            y=yy_flat,
            hue=Z_index_flat,
            weights=Z_flat,
            levels=5,
            thresh=0.5,
            alpha=0.5,
            cmap="mako"
        )

        if self.show_x_label:
            ax.set_xlabel(Plotter._parameter_dict.get(
                self.parameters[0], self.parameters[0]))
        if self.show_y_label:
            ax.set_ylabel(Plotter._parameter_dict.get(
                self.parameters[1], self.parameters[1]))

    def get_limits(self, axis_limits):
        default_x_lim = [-1, 1] if self.scale[0] == 'tanh' else [0, 1]
        default_y_lim = [-1, 1] if self.scale[1] == 'tanh' else [0, 1]

        x_lim = self.x_lim if self.x_lim is not None else axis_limits.get(
            'x', default_x_lim)
        y_lim = self.y_lim if self.y_lim is not None else axis_limits.get(
            'y', default_y_lim)

        return x_lim, y_lim

# Needs to be fixed!


@Plotter.plot_type("Draw")
class draw(Plot):
    def __init__(self, parameters: Union[tuple[str], None] = None,
                 pos: Optional[Dict[Union[int, str], tuple]] = None,
                 node_attributes: str = "opinion",
                 edge_attributes: str = 'importance',
                 node_info_mode: str = 'none',
                 use_node_color: bool = True,
                 use_edge_thickness: bool = True,
                 show_edge_influences: bool = False,
                 node_size_multiplier: int = 200,
                 arrowhead_length: float = 0.2,
                 arrowhead_width: float = 0.2,
                 min_line_width: float = 0.1,
                 max_line_width: float = 3.0,
                 seed: Optional[int] = None):
        self.parameters = tuple(parameters) if parameters is not None else ()

        self.pos = pos
        self.node_attributes = node_attributes
        self.edge_attributes = edge_attributes
        self.node_info_mode = node_info_mode
        self.use_node_color = use_node_color
        self.use_edge_thickness = use_edge_thickness
        self.show_edge_influences = show_edge_influences
        self.node_size_multiplier = node_size_multiplier
        self.arrowhead_length = arrowhead_length
        self.arrowhead_width = arrowhead_width
        self.min_line_width = min_line_width
        self.max_line_width = max_line_width
        self.seed = seed

        self.data_key = Plotter.request_to_tuple(
            self.get_dynamic_plot_requests()[0])

        self.x_lim = None
        self.y_lim = None

    def get_static_plot_requests(self):
        return []

    def get_dynamic_plot_requests(self):
        return [{'method': 'get_mean_graph', 'settings': {}}]

    def plot(self, ax: plt.Axes, dynamic_data_cache: dict, static_data_cache: dict, axis_limits: dict):
        # Fetch the HariGraph instance using the data_key
        image = dynamic_data_cache[self.data_key]

        # Use the specified or default positions for nodes
        if self.pos is None:
            self.pos = image.position_nodes(seed=self.seed)

        # Prepare Node Labels
        node_labels = self._prepare_node_labels(image)

        # Prepare Node Colors
        node_colors = self._prepare_node_colors(image, self.use_node_color)

        # Prepare Edge Widths and Labels
        edge_widths, edge_labels = self._prepare_edge_attributes(image)

        # Calculate Node Sizes
        node_sizes = self._calculate_node_sizes(image)

        # Draw Nodes
        nx.draw_networkx_nodes(
            image, self.pos, node_color=node_colors, node_size=node_sizes, ax=ax)

        # Draw Edges
        self._draw_edges(image, ax, edge_widths)

        # Draw Labels
        nx.draw_networkx_labels(image, self.pos, labels=node_labels, ax=ax)
        if self.show_edge_influences and edge_labels:
            nx.draw_networkx_edge_labels(
                image, self.pos, edge_labels=edge_labels, ax=ax)

    def _prepare_node_labels(self, image):
        node_labels = {}
        match self.node_info_mode:
            case 'opinion':
                node_labels = {node: f"{opinion:.2f}" for node, opinion in image.gatherer.gather(
                    self.node_attributes).items()}
            case 'ids':
                node_labels = {node: str(node) for node in image.nodes()}
            case 'labels':
                node_labels = {node: ','.join(
                    map(str, label)) for node, label in image.gatherer.gather("label").items()}
            case 'cluster_size':
                node_labels = {node: str(
                    len(label)) for node, label in image.gatherer.gather("label").items()}
        return node_labels

    def _prepare_node_colors(self, image, use_node_color):
        node_colors = []
        default_color = 'lightblue'  # Default color for nodes without specified colors
        for node in image.nodes():
            if use_node_color:
                color = cm.bwr(image.gatherer.gather(
                    self.node_attributes).get(node, 0.5))
            else:
                color = default_color
            node_colors.append(color)
        return node_colors

    def _prepare_edge_attributes(self, image):
        edge_weights = list(image.gatherer.gather(
            self.edge_attributes).values())
        scaled_weights = np.sqrt(edge_weights)  # Non-linear scaling
        max_scaled_weight = np.max(scaled_weights)
        min_scaled_weight = np.min(scaled_weights)
        edge_widths = [(self.min_line_width + (self.max_line_width - self.min_line_width) * (weight -
                        min_scaled_weight) / (max_scaled_weight - min_scaled_weight)) for weight in scaled_weights]
        edge_labels = {(u, v): f"{influence:.2f}" for (u, v), influence in image.gatherer.gather(
            self.edge_attributes).items()} if self.show_edge_influences else None
        return edge_widths, edge_labels

    def _calculate_node_sizes(self, image):
        node_sizes = []
        # Default size for nodes without labels
        default_size = self.node_size_multiplier
        for node in image.nodes():
            label = image.nodes[node].get('label')
            if label:
                size = self.node_size_multiplier * \
                    math.sqrt(len(label))  # Nonlinear scaling
            else:
                size = default_size
            node_sizes.append(size)
        return node_sizes

    def _draw_edges(self, image, ax, edge_widths):
        # Initialize style with a default value
        default_style = 'arc3,rad=0.3'  # or any other default style you prefer

        for (u, v), width in zip(image.edges(), edge_widths):
            # Here you might have some logic to determine the style for each edge
            # For example:
            # if some_condition:
            #     style = 'some_specific_style'
            # else:
            #     style = default_style

            # If no specific style is set, use the default
            style = default_style

            # Now draw the edge with the defined style
            nx.draw_networkx_edges(image, self.pos, edgelist=[(u, v)], width=width, ax=ax,
                                   arrowstyle=f'-|>,head_length={self.arrowhead_length},head_width={self.arrowhead_width}',
                                   connectionstyle=style)


@Plotter.plot_type("Static: Time line")
class plot_time_line(Plot):
    def __init__(self, parameters: tuple[str],
                 scale: Optional[str | None] = None,
                 show_x_label: bool = True, show_y_label: bool = True,
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None):
        self.parameters = tuple(parameters)
        self.scale = tuple(('linear', 'linear') if scale is None else scale)
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self.x_lim = x_lim
        self.y_lim = y_lim

        self.data_key = Plotter.request_to_tuple(
            self.get_dynamic_plot_requests()[0])

    def get_static_plot_requests(self):
        return []

    def get_dynamic_plot_requests(self):
        return [{'method': 'mean_time', 'settings': {}}]

    def plot(self, ax: plt.Axes, dynamic_data_cache: dict, static_data_cache: dict, axis_limits: dict):
        x_lim, y_lim = self.get_limits(axis_limits)

        data = dynamic_data_cache[self.data_key]

        x_parameter, y_parameter = self.parameters

        if x_parameter == 'time':
            # Time is on the x-axis, draw a vertical line
            ax.axvline(x=data, color='r', linestyle='--')
        elif y_parameter == 'time':
            # Time is on the y-axis, draw a horizontal line
            ax.axhline(y=data, color='r', linestyle='--')

        if x_lim is not None:
            ax.set_xlim(*x_lim)
        if y_lim is not None:
            ax.set_ylim(*y_lim)

        if self.show_x_label:
            ax.set_xlabel(Plotter._parameter_dict.get(
                x_parameter, x_parameter))
        if self.show_y_label:
            ax.set_ylabel(Plotter._parameter_dict.get(
                y_parameter, y_parameter))


@Plotter.plot_type("Static: Node lines")
class plot_node_lines(Plot):
    def __init__(self, parameters: tuple[str],
                 scale: Optional[str | None] = None,
                 show_x_label: bool = True, show_y_label: bool = True,
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None, colormap: str = 'coolwarm'):
        self.parameters = tuple(parameters)
        self.scale = tuple(('linear', 'linear') if scale is None else scale)
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.colormap = colormap

        self._static_data = None

        self.data_key = Plotter.request_to_tuple(
            self.get_static_plot_requests()[0])

    def get_static_plot_requests(self):
        return [{'method': 'calculate_node_values', 'settings': {'parameters': self.parameters, 'scale': self.scale}}]

    def get_dynamic_plot_requests(self):
        return []

    def data(self, static_data_cache: List[dict]):
        if self._static_data is not None:
            return self._static_data

        data_list = static_data_cache[self.data_key]
        self._static_data = self.transform_data(data_list)
        return self._static_data

    def plot(self, ax: plt.Axes, dynamic_data_cache: dict, static_data_cache: List[dict], axis_limits: dict):

        x_lim, y_lim = self.get_limits(axis_limits)

        data = self.data(static_data_cache)

        # print(f'{data = }')
        # print(f'{data = }')
        # print(f'{data.keys() = }')
        # print(f'{data["time"] = }')

        x_parameter, y_parameter = self.parameters

        if not (x_parameter == 'time' or y_parameter == 'time'):
            raise ValueError('One of the parameters should be time.')

        # Determine which axis time is on
        time_is_x_axis = x_parameter == 'time'

        x_values = data[x_parameter]
        y_values = data[y_parameter]

        if self.scale[0 if time_is_x_axis else 1] == 'tanh':
            x_values = np.tanh(x_values)
        if self.scale[1 if time_is_x_axis else 0] == 'tanh':
            y_values = np.tanh(y_values)

        # Color map for final state values
        cmap = plt.get_cmap(self.colormap)
        final_values = y_values[:, -1] if time_is_x_axis else x_values[:, -1]
        colors = cmap(final_values / max(final_values))

        # Plotting
        for i, color in enumerate(colors):
            if time_is_x_axis:
                ax.plot(x_values, y_values[i], color=color)
            else:
                ax.plot(y_values[i], x_values, color=color)

        # Set limits
        if x_lim is not None:
            ax.set_xlim(*x_lim)
        if y_lim is not None:
            ax.set_ylim(*y_lim)

        if self.show_x_label:
            ax.set_xlabel(Plotter._parameter_dict.get(
                self.parameters[0], self.parameters[0]))
        if self.show_y_label:
            ax.set_ylabel(Plotter._parameter_dict.get(
                self.parameters[1], self.parameters[1]))


@Plotter.plot_type("Static: Graph line")
class plot_graph_line(Plot):
    def __init__(self, parameters: tuple[str],
                 scale: Optional[str | None] = None,
                 show_x_label: bool = True, show_y_label: bool = True,
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None, function: str = 'mean'):
        self.parameters = tuple(parameters)
        self.scale = tuple(('linear', 'linear') if scale is None else scale)
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.function = function

        self._static_data = None

        self.data_key = Plotter.request_to_tuple(
            self.get_static_plot_requests()[0])

    def get_static_plot_requests(self):
        return [{'method': 'calculate_function_of_node_values', 'settings': {'parameters': self.parameters, 'scale': self.scale, 'function': self.function}}]

    def get_dynamic_plot_requests(self):
        return []

    def data(self, static_data_cache: List[dict]):
        if self._static_data is not None:
            return self._static_data

        data = static_data_cache[self.data_key]

        keys = list(data[0].keys())

        self._static_data = {key: [] for key in keys}
        for frame in data:
            for key in keys:
                self._static_data[key].append(frame[key])

        return self._static_data

    def plot(self, ax: plt.Axes, dynamic_data_cache: dict, static_data_cache: List[dict], axis_limits: dict):
        x_lim, y_lim = self.get_limits(axis_limits)

        data = self.data(static_data_cache)

        x_parameter, y_parameter = self.parameters

        x_values = data[x_parameter]
        y_values = data[y_parameter]

        if self.scale[0] == 'tanh':
            x_values = np.tanh(x_values)
        if self.scale[1] == 'tanh':
            y_values = np.tanh(y_values)

        ax.plot(x_values, y_values)

        if x_lim is not None:
            ax.set_xlim(*x_lim)
        if y_lim is not None:
            ax.set_ylim(*y_lim)

        if self.show_x_label:
            ax.set_xlabel(Plotter._parameter_dict.get(
                x_parameter, x_parameter))
        if self.show_y_label:
            ax.set_ylabel(Plotter._parameter_dict.get(
                y_parameter, y_parameter))


@Plotter.plot_type("Static: Graph Range")
class plot_fill_between(Plot):
    def __init__(self, parameters: tuple[str], functions: Optional[List[str]] = None,
                 scale: Optional[str | None] = None,
                 show_x_label: bool = True, show_y_label: bool = True,
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None):
        self.parameters = tuple(parameters)
        self.functions = functions if functions is not None else ['min', 'max']
        self.scale = tuple(('linear', 'linear') if scale is None else scale)
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self.x_lim = x_lim
        self.y_lim = y_lim

        self._static_data = None

        # Generate data keys for both functions
        self.data_key_min = Plotter.request_to_tuple(
            self.get_static_plot_requests()[0])
        self.data_key_max = Plotter.request_to_tuple(
            self.get_static_plot_requests()[1])

    def get_static_plot_requests(self):
        return [
            {'method': 'calculate_function_of_node_values', 'settings': {
                'parameters': self.parameters, 'scale': self.scale, 'function': self.functions[0]}},
            {'method': 'calculate_function_of_node_values', 'settings': {
                'parameters': self.parameters, 'scale': self.scale, 'function': self.functions[1]}}
        ]

    def get_dynamic_plot_requests(self):
        return []

    def data(self, static_data_cache: List[dict], function_key):
        data = static_data_cache[function_key]

        keys = list(data[0].keys())

        _static_data = {key: [] for key in keys}
        for frame in data:
            for key in keys:
                _static_data[key].append(frame[key])

        return _static_data

    def plot(self, ax: plt.Axes, dynamic_data_cache: dict, static_data_cache: List[dict], axis_limits: dict):
        x_lim, y_lim = self.get_limits(axis_limits)

        data_min = self.data(static_data_cache, self.data_key_min)
        data_max = self.data(static_data_cache, self.data_key_max)

        x_parameter, y_parameter = self.parameters

        x_values = data_min[x_parameter]
        y_values_min = data_min[y_parameter]
        y_values_max = data_max[y_parameter]

        if self.scale[0] == 'tanh':
            x_values = np.tanh(x_values)
        if self.scale[1] == 'tanh':
            y_values_min = np.tanh(y_values_min)
            y_values_max = np.tanh(y_values_max)

        # Fill the area between the min and max curves
        ax.fill_between(x_values, y_values_min, y_values_max, alpha=0.5)

        if x_lim is not None:
            ax.set_xlim(*x_lim)
        if y_lim is not None:
            ax.set_ylim(*y_lim)

        if self.show_x_label:
            ax.set_xlabel(Plotter._parameter_dict.get(
                x_parameter, x_parameter))
        if self.show_y_label:
            ax.set_ylabel(Plotter._parameter_dict.get(
                y_parameter, y_parameter))


@Plotter.plot_type("Static: Cluster Line")
class plot_cluster_line(Plot):
    def __init__(self, parameters: tuple[str], cluster_settings: dict = {},
                 scale: Optional[str | None] = None,
                 show_x_label: bool = True, show_y_label: bool = True,
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None):
        self.parameters = tuple(parameters)
        self.cluster_settings = cluster_settings
        self.scale = tuple(('linear', 'linear') if scale is None else scale)
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self.x_lim = x_lim
        self.y_lim = y_lim

        self._static_data = None

        self.data_key = Plotter.request_to_tuple(
            self.get_static_plot_requests()[0])

    def get_static_plot_requests(self):
        return [{'method': 'cluster_graph_values', 'settings': {'parameters': self.parameters, 'scale': self.scale, 'cluster_settings': self.cluster_settings}}]

    def get_dynamic_plot_requests(self):
        return []

    def data(self, static_data_cache: List[dict]):
        if self._static_data is not None:
            return self._static_data

        data = static_data_cache[self.data_key]

        data = self.transform_data(data)

        x_parameter, y_parameter = self.parameters

        # Transform data to suitable format for plotting
        x_values = np.array(data[x_parameter])
        y_values = np.array(data[y_parameter])

        if self.scale[0] == 'tanh':
            x_values = np.tanh(x_values)
        if self.scale[1] == 'tanh':
            y_values = np.tanh(y_values)

        self._static_data = {'x': x_values, 'y': y_values}
        return self._static_data

    def plot(self, ax: plt.Axes, dynamic_data_cache: dict, static_data_cache: List[dict], axis_limits: dict):
        x_lim, y_lim = self.get_limits(axis_limits)

        data = self.data(static_data_cache)

        # print(f'{data = }')

        ax.plot(data['x'], data['y'].T)

        if x_lim is not None:
            ax.set_xlim(*x_lim)
        if y_lim is not None:
            ax.set_ylim(*y_lim)

        if self.show_x_label:
            ax.set_xlabel(Plotter._parameter_dict.get(
                self.parameters[0], self.parameters[0]))
        if self.show_y_label:
            ax.set_ylabel(Plotter._parameter_dict.get(
                self.parameters[1], self.parameters[1]))


@Plotter.plot_type("Static: Cluster Range")
class plot_fill_between_cluster(Plot):
    def __init__(self, parameters: tuple[str], cluster_settings: dict = {},
                 scale: Optional[str | None] = None,
                 show_x_label: bool = True, show_y_label: bool = True,
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None):
        assert len(
            parameters) == 3, "Three parameters are required, with the last or first being 'time'."

        self.parameters = tuple(parameters)
        self.cluster_settings = cluster_settings
        self.scale = tuple(('linear', 'linear') if scale is None else scale)
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self.x_lim = x_lim
        self.y_lim = y_lim

        self._static_data = None

        self.data_key = Plotter.request_to_tuple(
            self.get_static_plot_requests()[0])

    def get_static_plot_requests(self):
        return [{'method': 'cluster_graph_values', 'settings': {'parameters': self.parameters, 'scale': self.scale, 'cluster_settings': self.cluster_settings}}]

    def get_dynamic_plot_requests(self):
        return []

    def data(self, static_data_cache: List[dict]):
        if self._static_data is not None:
            return self._static_data

        data = static_data_cache[self.data_key]

        data = self.transform_data(data)

        x_parameter, y1_parameter, y2_parameter = self.parameters

        # Transform data to suitable format for plotting
        x_values = np.array(data[x_parameter])
        y1_values = np.array(data[y1_parameter])
        y2_values = np.array(data[y2_parameter])

        if self.scale[0] == 'tanh':
            x_values = np.tanh(x_values)
        if self.scale[1] == 'tanh':
            y1_values = np.tanh(y1_values)
            y2_values = np.tanh(y2_values)

        self._static_data = {'x': x_values, 'y1': y1_values, 'y2': y2_values}
        return self._static_data

    def plot(self, ax: plt.Axes, dynamic_data_cache: dict, static_data_cache: List[dict], axis_limits: dict):
        x_lim, y_lim = self.get_limits(axis_limits)

        data = self.data(static_data_cache)

        # Assuming x_values are common for all intervals
        x_values = data['x']

        # Iterate over each set of intervals
        # Assuming data['y1'] and data['y2'] have the same first dimension
        for i in range(data['y1'].shape[0]):
            y1_values = data['y1'][i, :]
            y2_values = data['y2'][i, :]

            # Use tanh scaling if specified
            if self.scale[1] == 'tanh':
                y1_values = np.tanh(y1_values)
                y2_values = np.tanh(y2_values)

            # Fill the area between y1 and y2 for this set of intervals
            ax.fill_between(x_values, y1_values, y2_values, alpha=0.5)

        # Setting x and y limits if provided
        if x_lim is not None:
            ax.set_xlim(*x_lim)
        if y_lim is not None:
            ax.set_ylim(*y_lim)

        # Setting labels
        if self.show_x_label:
            ax.set_xlabel(Plotter._parameter_dict.get(
                self.parameters[0], self.parameters[0]))
        if self.show_y_label:
            ax.set_ylabel(Plotter._parameter_dict.get(
                self.parameters[1], self.parameters[1]))


#     # class plot_opinions(self,
#     #                       mode: str = 'show',
#     #                       save_dir: Optional[str] = None,
#     #                       gif_path: Optional[str] = None,
#     #                       reference_index: int = -1,
#     #                       minimum_cluster_size: int = 1,
#     #                       colormap: str = 'coolwarm',
#     #                       name: str = 'opinions_statics',
#     #                       width_threshold: float = 1,
#     #                       scale: str = 'linear',
#     #                       show_legend: bool = True):
#     #         """
#     #         Plot the opinions of nodes over time.

#     #         Parameters:
#     #         - mode (str): Mode of the plot. Default is 'show'.
#     #         - save_dir (str, optional): Directory to save the plot. Default is None.
#     #         - gif_path (str, optional): Path to save the gif. Default is None.
#     #         - reference_index (int): Index to refer for plotting. Default is -1.
#     #         - minimum_cluster_size (int): Minimum size of the cluster for plotting. Default is 1.
#     #         - colormap (str): Colormap for the plot. Default is 'coolwarm'.
#     #         - name (str): Name of the plot. Default is 'opinions_statics'.
#     #         - width_threshold (float): Threshold for the width of the plot. Default is 1.
#     #         - scale (str): Scale for the plot values. Options: 'linear' or 'tanh'. Default is 'linear'.
#     #         - show_legend (bool): Whether to show the legend. Default is True.
#     #         """
#     #         all_nodes_data = {}
#     #         time_array = []

#     #         for group_data in self.interface.mean_group_values_iterator(['opinion', 'cluster_size', 'min_opinion', 'max_opinion']):
#     #             time_array.append(group_data['time'])

#     #             for node, opinion, size, max_opinion, min_opinion in zip(
#     #                     group_data['data']['node'], group_data['data']['opinion'], group_data['data']['cluster_size'], group_data['data']['max_opinion'], group_data['data']['min_opinion']):
#     #                 if node not in all_nodes_data:

#     #                     all_nodes_data[node] = {
#     #                         'opinion': [],
#     #                         'cluster_size': [],
#     #                         'max_opinion': [],
#     #                         'min_opinion': []
#     #                     }

#     #                 all_nodes_data[node]['opinion'].append(opinion)
#     #                 all_nodes_data[node]['cluster_size'].append(size)
#     #                 all_nodes_data[node]['max_opinion'].append(max_opinion)
#     #                 all_nodes_data[node]['min_opinion'].append(min_opinion)

#     #         # Filter nodes where the size is always below the threshold
#     #         nodes_to_remove = [node for node, data in all_nodes_data.items() if all(
#     #             size < minimum_cluster_size for size in data['cluster_size'])]

#     #         # Remove those nodes from all_nodes_data
#     #         for node in nodes_to_remove:
#     #             del all_nodes_data[node]

#     #         # Initialize values with extreme opposites for comparison
#     #         smallest_opinion = float('inf')
#     #         highest_opinion = float('-inf')

#     #         for node, data in all_nodes_data.items():
#     #             current_min = min(data['opinion'])
#     #             current_max = max(data['opinion'])

#     #             # Update the global smallest and highest values if needed
#     #             if current_min < smallest_opinion:
#     #                 smallest_opinion = current_min
#     #             if current_max > highest_opinion:
#     #                 highest_opinion = current_max

#     #         vmax = max(abs(smallest_opinion), abs(highest_opinion))
#     #         absolute_width_threshold = 0.01 * width_threshold * vmax

#     #         with PlotSaver(mode=mode, save_path=f"{save_dir}/{name}_" + "{}.png", gif_path=gif_path) as saver:
#     #             fig, ax = plt.subplots(figsize=(10, 7))

#     #             for key, data in all_nodes_data.items():
#     #                 if scale == 'tanh':
#     #                     vmax_tanh = np.tanh(vmax)
#     #                     y = np.tanh(data['opinion'])
#     #                     min_y = np.tanh(data['min_opinion'])
#     #                     max_y = np.tanh(data['max_opinion'])

#     #                     ref_opinion = np.tanh(y[reference_index])
#     #                     color = plt.get_cmap(colormap)(
#     #                         (ref_opinion + vmax_tanh) / (2 * vmax_tanh))

#     #                     # Check if the width difference exceeds the threshold anywhere
#     #                     widths = [m - n for m, n in zip(max_y, min_y)]
#     #                     if any(w > absolute_width_threshold for w in widths):
#     #                         # Plotting the semitransparent region between min and max
#     #                         # opinions
#     #                         ax.fill_between(time_array, min_y, max_y,
#     #                                         color=color, alpha=0.2)

#     #                     # Plotting the line for the opinions
#     #                     ax.plot(time_array, y, color=color, label=f'Node {key}')

#     #                     Plotter.tanh_axis_labels(ax=ax, axis='y')
#     #                 else:
#     #                     y = data['opinion']
#     #                     min_y = data['min_opinion']
#     #                     max_y = data['max_opinion']

#     #                     ref_opinion = y[reference_index]
#     #                     color = plt.get_cmap(colormap)(
#     #                         (ref_opinion + vmax) / (2 * vmax))

#     #                     # Check if the width difference exceeds the threshold anywhere
#     #                     widths = [m - n for m, n in zip(max_y, min_y)]
#     #                     if any(w > absolute_width_threshold for w in widths):
#     #                         # Plotting the semitransparent region between min and max
#     #                         # opinions
#     #                         ax.fill_between(time_array, min_y, max_y,
#     #                                         color=color, alpha=0.2)

#     #                     # Plotting the line for the opinions
#     #                     ax.plot(time_array, y, color=color, label=f'Node {key}')

#     #             ax.set_title(f"Opinions Over Time")
#     #             ax.set_xlabel("Time")
#     #             ax.set_ylabel("Opinion")
#     #             if show_legend:
#     #                 ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

#     #             ax.set_xlim([min(time_array), max(time_array)])

#     #             saver.save(fig)
