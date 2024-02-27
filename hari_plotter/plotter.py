from __future__ import annotations

import math
import os
import shutil
import tempfile
import warnings
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from typing import (Any, Dict, Iterator, List, Optional, Sequence, Tuple, Type,
                    Union)

import imageio
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from .cluster import Clustering
from .color_scheme import ColorScheme
from .hari_graph import HariGraph
from .interface import Interface

plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0


class Plotter:
    _parameter_dict = {'Time': 'Time',
                       'Opinion': 'Node Opinion',
                       'Cluster size': 'Cluster size',
                       'Importance': 'Node Importance',
                       'Label': 'Node Label',
                       'Neighbor mean opinion': 'Node Neighbor Mean Opinion',
                       'Activity': 'Node Activity',
                       'Inner opinions': 'Node Inner Opinions',
                       'Max opinion': 'Node Max Opinion',
                       'Min opinion': 'Node Min Opinion'}

    _plot_types = {}

    class PlotSaver:
        """
        A utility class to handle the saving and display of plots.

        It provides functionality to save individual plots, display them,
        and even create GIFs from a sequence of plots.
        """

        def __init__(self, mode: Union[str, List[str]] = 'show',
                     save_path: Optional[str] = None,
                     save_format: Optional[str] = 'image_{}',
                     gif_path: Optional[str] = None) -> None:
            """
            Initialize the PlotSaver instance.

            Args:
                mode (Union[str, List[str]]): The mode(s) in which to operate. 
                    It can be a list or a single string, e.g. ['show', 'save'] or 'gif'.
                save_path (Optional[str]): Path to save individual plots (used if 'save' is in mode)
                save_format (Optional[str]): string with {} for formatting in the number
                gif_path (Optional[str]): Path to save gif (used if 'gif' is in mode).
            """
            # Ensure mode is a list even if a single mode string is provided
            self.mode = mode if isinstance(mode, list) else [mode]
            if not os.path.exists(save_path):
                warnings.warn(f"Path {save_path} does not exist. Creating it.")
                # Create the directory, including any necessary parent directories
                os.makedirs(save_path, exist_ok=True)
            self.save_path = save_path if save_path[-1] == '/' else save_path+'/'
            self.save_format = save_format
            self.gif_path = gif_path
            self.saved_images = []
            self.temp_dir = None
            self.color_palette = ColorScheme()

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

        def __enter__(self) -> Plotter.PlotSaver:
            """
            Entry point for the context manager.

            Returns:
                Plotter.PlotSaver: The current instance of the PlotSaver.
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
                path = self.save_path + \
                    self.save_format.format(len(self.saved_images))
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

    def __init__(self, interface: Interface = None, figsize=None):
        """
        Initialize the Plotter object with the given Interface instance.

        Parameters:
        -----------
        interface : Interface
            Interface instance to be used for plotting.
        """
        self.interface: Interface = interface
        self.color_scheme = ColorScheme(self.interface)
        self.plots = [[[]]]
        self._figsize = figsize
        self.num_rows = 1
        self.num_cols = 1
        self.size_ratios = [[1], [1]]

    def update_interface(self, new_interface):
        self.interface = new_interface
        self.color_scheme = ColorScheme(new_interface)

    @property
    def is_initialized(self):
        return self.interface is not None

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
            if plot_name in cls._plot_types:
                raise ValueError(f"Plot type {plot_name} is already defined.")
            cls._plot_types[plot_name] = plot_func
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
        if plot_type not in self.available_plot_types:
            raise ValueError(f"Plot type '{plot_type}' not recognized. "
                             f"Available methods: {self.available_plot_types}")

        # Check if 'colorscheme' is not in plot_arguments
        if 'color_scheme' not in plot_arguments:
            # Add self.color_scheme to plot_arguments with the key 'colorscheme'
            plot_arguments['color_scheme'] = self.color_scheme

        plot = self._plot_types[plot_type](**plot_arguments)

        # Initialize the cell with an empty list if it's None
        self.plots[row][col] = self.plots[row][col] or []

        # Append the new plot to the cell's list
        self.plots[row][col].append(plot)

    def create_fig_and_axs(self):
        fig, axs = plt.subplots(self.num_rows, self.num_cols, figsize=self.figsize, gridspec_kw={
            'width_ratios': self.size_ratios[0], 'height_ratios': self.size_ratios[1]})
        # Ensure axs is a 2D array for consistency
        if self.num_rows == 1 and self.num_cols == 1:
            axs = [[axs]]  # Single plot
        elif self.num_rows == 1:
            axs = [axs]  # Single row, multiple columns
        elif self.num_cols == 1:
            axs = [[ax] for ax in axs]  # Multiple rows, single column

        return fig, axs

    def _plot(self, fig, axs: List[list], group_number: int):
        # Data fetching for static plots
        track_clusters_requests = [item for i in range(self.num_rows) for j in range(
            self.num_cols) for plot in self.plots[i][j] for item in plot.get_track_clusterings_requests()]
        self.interface.cluster_tracker.track_clusters(track_clusters_requests)

        # Determine the rendering order
        render_order = self._determine_plot_order()
        # print(f'{render_order = }')

        axis_limits = dict()

        for (i, j) in render_order:
            # print(f'{i,j = }')
            ax = axs[i][j]
            for plot in self.plots[i][j]:
                plot.plot(ax=ax, group_number=group_number, dynamic_data_cache=self.interface.dynamic_data_cache[group_number],
                          static_data_cache=self.interface.static_data_cache, axis_limits=axis_limits)

                # Store axis limits for future reference
                axis_limits[(i, j)] = (ax.get_xlim(), ax.get_ylim())
            if not self.plots[i][j]:
                ax.grid(False)
                ax.axis('off')
                ax.set_visible(False)

    def plot(self, group_number: int):
        fig, axs = self.create_fig_and_axs()

        self._plot(fig, axs, group_number)

    def plot_dynamics(self, mode: Union[str, List[str]] = 'show', save_dir: Optional[str] = None, gif_path: Optional[str] = None, name: str = 'opinion_histogram', preview: bool = False) -> None:
        """
        Create and display the plots based on the stored configurations.
        mode : ["show", "save", "gif"]
        """
        with Plotter.PlotSaver(mode=mode, save_path=save_dir, save_format=f"{name}_" + "{}.png", gif_path=gif_path) as saver:

            # self.interface.static_data_cache_requests = [item for i in range(self.num_rows) for j in range(
            #     self.num_cols) for plot in self.plots[i][j] for item in plot.get_static_plot_requests()]
            # self.interface.dynamic_data_cache_requests = [item for i in range(self.num_rows) for j in range(
            #     self.num_cols) for plot in self.plots[i][j] for item in plot.get_dynamic_plot_requests()]

            # self.interface.collect_static_data()

            for group_number in range(len(self.interface.groups)):

                # self.interface.collect_dynamic_data(group_number)

                fig, axs = self.create_fig_and_axs()

                self._plot(fig, axs, group_number)

                saver.save(fig)

                if preview:
                    break

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
    def node_parameters(self) -> list:
        """
        Retrieves the list of available parameters/methods from the interface.

        Returns:
            list: A list of available parameters or methods.
        """
        return self.interface.node_parameters

    @property
    def existing_plot_types(self) -> list:
        """
        Retrieves the list of available parameters/methods from the interface.

        Returns:
            list: A list of available parameters or methods.
        """
        return list(self._plot_types.keys())

    @property
    def available_plot_types(self) -> list:
        """
        Retrieves the list of available parameters/methods from the interface.

        Returns:
            list: A list of available parameters or methods.
        """
        return [method_name for method_name, method in self._plot_types.items() if method.is_available(self.interface)[0]]

    def get_plot_class(self, plot_type: str):
        return self._plot_types[plot_type]

    def get_plot_name(self, plot_class) -> str:
        for key, value in self._plot_types.items():
            if value == plot_class:
                return key
        warnings.warn(f'{plot_class} was not found in plot types')
        return ''

    @property
    def available_plot_types_hint(self) -> list:
        """
        Retrieves the list of available parameters/methods from the interface.

        Returns:
            list: A list of available parameters or methods.
        """
        info_string = ''
        for method_name, method in self._plot_types.items():
            is_available, comment = method.is_available(self.interface)
            info_string += method_name + ': '
            info_string += '+ ' if is_available else '- '
            info_string += comment
            info_string += '\n'

        return info_string

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

        if scale[0] == 'Tanh':
            ax.set_xticks(ticks)
            ax.set_xticklabels(tickslabels)
            ax.set_xticks(minor_ticks, minor=True)
            ax.set_xticklabels([], minor=True)
            ax.set_xlim([-1, 1])

        if scale[1] == 'Tanh':
            ax.set_yticks(ticks)
            ax.set_yticklabels(tickslabels)
            ax.set_yticks(minor_ticks, minor=True)
            ax.set_yticklabels([], minor=True)
            ax.set_ylim([-1, 1])

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

    def to_code(self) -> str:
        if len(self.plots) == 1 and len(self.plots[0]) == 1 and len(self.plots[0][0]) == 0:
            return ''

        final_code = ''
        for i, row in enumerate(self.plots):
            for j, cell in enumerate(row):
                if len(cell) > 0:
                    for plot in cell:
                        final_code += 'plotter.add_plot('
                        final_code += f'\'{self.get_plot_name(type(plot))}\''
                        final_code += ',{' + plot.settings_to_code() + '},'
                        final_code += f'row={i},col={j},)\n'
        return final_code
