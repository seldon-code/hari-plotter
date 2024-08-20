from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .plot import Plot

import imageio
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .color_scheme import ColorScheme
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

    _plot_types: dict[str, type[Plot]] = {}

    class PlotSaver:
        """
        A utility class to handle the saving and display of plots.

        It provides functionality to save individual plots, display them,
        and even create GIFs from a sequence of plots.
        """

        def __init__(self, mode: str | list[str] = 'show',
                     save_path: Optional[str] = None,
                     save_format: Optional[str] = 'image_{}',
                     animation_path: Optional[str] = None) -> None:
            """
            Initialize the PlotSaver instance.

            Args:
                mode (str | list[str]): The mode(s) in which to operate. 
                    It can be a list or a single string, e.g. ['show', 'save'] or 'gif'. Available modes: ["show", "save", "gif", "mp4"]
                save_path (Optional[str]): Path to save individual plots (used if 'save' is in mode)
                save_format (Optional[str]): string with {} for formatting in the number
                animation_path (Optional[str]): Path to save gif (used if 'gif' is in mode).
            """
            # Ensure mode is a list even if a single mode string is provided
            self.mode = mode if isinstance(mode, list) else [mode]
            if not os.path.exists(save_path):
                warnings.warn(f"Path {save_path} does not exist. Creating it.")
                # Create the directory, including any necessary parent directories
                os.makedirs(save_path, exist_ok=True)
            self.save_path = os.Path(
                save_path) if save_path[-1] == '/' else save_path+'/'
            self.save_format = save_format
            self.animation_path = animation_path
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
            # If only 'gif' or 'mp4' mode is selected, save figure to a temp directory
            elif ('gif' in self.mode or 'mp4' in self.mode) and not self.save_path:
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
            # If 'gif' mode is active and animation_path is provided, create a GIF from the saved images
            if 'gif' in self.mode and self.animation_path and self.saved_images:
                with imageio.get_writer(self.animation_path+'.gif', mode='I') as writer:
                    for img_path in self.saved_images:
                        image = imageio.imread(img_path)
                        writer.append_data(image)

            # # Create MP4 animation if mode is selected
            if 'mp4' in self.mode and self.animation_path and self.saved_images:
                with imageio.get_writer(self.animation_path+'.mp4', mode='I', fps=5, codec='libx264') as writer:
                    for img_path in self.saved_images:
                        image = imageio.imread(img_path)
                        writer.append_data(image)

            # Cleanup temporary directory if it was used
            if self.temp_dir:
                shutil.rmtree(self.temp_dir)

    class PlotLattice:
        def __init__(self, size_ratios: tuple[list[float], list[float]] = ([1.], [1.]), figsize=None) -> None:
            """
            Initialize the PlotLattice class.

            Parameters:
            size_ratios (tuple[list[float], list[float]], optional): The size ratios for the rows and columns of the lattice. Defaults to ([1.], [1.]).
            figsize (tuple[float, float], optional): The size of the figure. Defaults to None.
            """
            self._fig = None
            self._axs = None
            self._figsize = figsize
            self._size_ratios: tuple[list[float], list[float]] = size_ratios

        @property
        def size_ratios(self) -> tuple[list[float], list[float]]:
            """
            Get the size ratios for the rows and columns of the lattice.

            Returns:
            tuple[list[float], list[float]]: The size ratios.
            """
            return self._size_ratios

        @size_ratios.setter
        def size_ratios(self, value: tuple[list[float], list[float]]):
            """
            Set the size ratios for the rows and columns of the lattice.

            Parameters:
            value (tuple[list[float], list[float]]): The size ratios.
            """
            print(f'{value=}')
            if not len(value) == 2:
                raise ValueError('Size ratios must be a tuple of two lists')
            if not all(isinstance(ratio, (tuple, list)) for ratio in value):
                raise ValueError(
                    'Size ratios must be a tuple of two tuples/lists')
            if not len(value[0]) == len(self._size_ratios[0]) or not len(value[1]) == len(self._size_ratios[1]):
                raise ValueError(
                    'Size ratios must have the same length as the current size ratios')
            self._size_ratios = value

        def get_figsize(self) -> tuple[float, float]:
            """
            Get the figure size. If a size was set during initialization, it returns that size.
            Otherwise, it calculates the size based on the sum of size_ratios.

            Returns:
            tuple[float, float]: The size of the figure as (height, width).
            """
            if self._figsize is not None:
                return self._figsize
            else:
                # Calculate size based on the sum of size ratios
                width = np.sum(self.size_ratios[1])
                height = np.sum(self.size_ratios[0])

            # Adjust the width and height to ensure the smallest dimension is higher than 3
            if width < height:
                width = max(width, 4*len(self.size_ratios[1]))
                height = width * \
                    (np.sum(self.size_ratios[0])/np.sum(self.size_ratios[1]))
            else:
                height = max(height, 4*len(self.size_ratios[0]))
                width = height * \
                    (np.sum(self.size_ratios[1])/np.sum(self.size_ratios[0]))

            self._figsize = (width, height)

            return self._figsize

        def set_figsize(self, value):
            """
            Set the figure size.

            Parameters:
            value (tuple[float, float]): The size of the figure as (height, width).
            """
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(
                    "fig_size must be a list or tuple of two elements: [height, width]")
            self._figsize = value

        @property
        def num_rows(self) -> int:
            """
            Get the number of rows in the lattice.

            Returns:
            int: The number of rows.
            """
            return len(self.size_ratios[0])

        @property
        def num_cols(self) -> int:
            """
            Get the number of columns in the lattice.

            Returns:
            int: The number of columns.
            """
            return len(self.size_ratios[1])

        def fig(self) -> Figure:
            """
            Get the figure object.

            Returns:
            Figure: The figure object.
            """
            if self._fig is None:
                self.create_fig_and_axs()
            return self._fig

        def axs(self) -> list[list[Axes]]:
            """
            Get the axes objects.

            Returns:
            list[list[Axes]]: The axes objects.
            """
            if self._fig is None:
                self.create_fig_and_axs()
            return self._axs

        def fig_axs(self) -> tuple[Figure, list[list[Axes]]]:
            """
            Get the figure and axes objects.

            Returns:
            tuple[Figure, list[list[Axes]]]: The figure and axes objects.
            """
            if self._fig is None:
                self.create_fig_and_axs()
            return self._fig, self._axs

        def update_size_ratios(self, row: int, column: int):
            """
            Update the size ratios based on the given row and column indices.

            Parameters:
            row (int): The row index.
            column (int): The column index.
            """
            if row >= self.num_rows:
                self._size_ratios[0].extend([1.]*(row-self.num_rows+1))
            if column >= self.num_cols:
                self._size_ratios[1].extend([1.]*(column-self.num_cols+1))

        def convert_parameters_to_index(self, column: int, row: int) -> tuple[int, int]:
            """
            Convert the column and row indices to the corresponding index in the axes list.

            Parameters:
            column (int): The column index.
            row (int): The row index.

            Returns:
            tuple[int, int]: The corresponding index in the axes list.
            """
            self.update_size_ratios(row, column)  # Fixed the argument order
            return (row, column)

        def get_ax_by_index(self, index: tuple[int, int]) -> Axes:
            """
            Get the axes object based on the given index.

            Parameters:
            index (tuple[int, int]): The index in the axes list.

            Returns:
            Axes: The axes object.
            """
            self.update_size_ratios(index[0], index[1])
            return self.axs()[index[0]][index[1]]

        def create_fig_and_axs(self) -> tuple[Figure, Axes]:
            """
            Create the figure and axes objects.

            Returns:
            tuple[Figure, Axes]: The figure and axes objects.
            """
            size_ratios = self.size_ratios
            figsize = self.get_figsize()
            self._fig, self._axs = plt.subplots(self.num_rows, self.num_cols, figsize=figsize, gridspec_kw={
                'width_ratios': size_ratios[1], 'height_ratios': size_ratios[0]})
            # Ensure axs is a 2D array for consistency
            if self.num_rows == 1 and self.num_cols == 1:
                self._axs = [[self._axs]]  # Single plot
            elif self.num_rows == 1:
                self._axs = [self._axs]  # Single row, multiple columns
            elif self.num_cols == 1:
                # Multiple rows, single column
                self._axs = [[ax] for ax in self._axs]

    def __init__(self, interfaces:  Interface | list[Interface] | None = None):
        """
        Initialize the Plotter object with the given Interface instance.

        Parameters:
        -----------
        interface : Interface
            Interface instance to be used for plotting.
        """
        assert interfaces is None or isinstance(
            interfaces, (Interface, list)), "Interface must be an Interface instance or a list of Interface instances"
        self._interfaces:  list[Interface] | None = [interfaces] if isinstance(
            interfaces, Interface) else interfaces
        self.default_color_scheme: ColorScheme = ColorScheme()
        self.color_schemes: dict[Interface, ColorScheme] = {}
        self.plots: dict[tuple[int, int, Interface], list[Plot]] = {}
        self.plot_grid: Plotter.PlotLattice = self.PlotLattice()

    @property
    def number_of_interfaces(self) -> int:
        if self._interfaces is None:
            return 0
        return len(self._interfaces)

    def update_interface(self, new_interface):
        self._interfaces = new_interface
        self.color_scheme = ColorScheme(new_interface)

    @property
    def is_initialized(self) -> bool:
        return self._interfaces is not None

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

    def add_plot_for_interface(self, interface: Interface, plot_type: str, plot_arguments: dict[str, str | list[str] | dict[str, Any]], row: int = 0, col: int = 0):
        if plot_type not in self.available_plot_types:
            raise ValueError(f"Plot type '{plot_type}' not recognized. "
                             f"Available methods: {self.available_plot_types}")

        is_available, comment = self._plot_types[plot_type].is_available(
            interface)
        if not is_available:
            raise ValueError(f"Plot type '{plot_type}' is not available for the given interface {interface}."
                             f"Reason: {comment}")

        self.plot_grid.update_size_ratios(row, col)

        if 'color_scheme' not in plot_arguments:

            if interface not in self.color_schemes:
                self.color_schemes[interface] = self.default_color_scheme.new_interface(
                    interface)
            # Add self.color_scheme to plot_arguments with the key 'color_scheme'
            plot_arguments['color_scheme'] = self.color_schemes[interface]

        plot_arguments['interface'] = interface

        plot = self._plot_types[plot_type](**plot_arguments)

        if (row, col, interface) in self.plots:
            self.plots[(row, col, interface)].append(plot)
        else:
            self.plots[(row, col, interface)] = [plot]

    def number_of_groups(self) -> int:
        return max(len(interface.groups) for interface in self._interfaces)

    def regroup(self, num_intervals: int, interval_size: int = 1, offset: int = 0):
        for interface in self._interfaces:
            interface.regroup(num_intervals, interval_size)

    def add_plot(self, plot_type: str, plot_arguments: dict[str, str | list[str] | dict[str, Any]], row: int = 0, col: int = 0):
        """
        Example:
            plotter.add_plot(
                "Scatter",
                {
                    "parameters": ["Opinion", "Activity"],
                    "scale": ["Linear", "Linear"],
                    "color": {
                        "mode": "Parameter Colormap",
                        "settings": {
                            "parameter": "Opinion density",
                            "scale": ("Linear", )
                        },
                    },
                },
                row=2,
                col=0,
            )
        """

        for interface in self._interfaces:
            self.add_plot_for_interface(
                interface, plot_type, plot_arguments, row, col)

    def _plot(self, fig, group_number: int) -> Figure:
        if self._interfaces is None:
            warnings.warn("Interface not initialized. Cannot plot.")
            return fig

        # Data fetching for static plots
        track_clusters_requests = [request for cell in self.plots.values(
        ) for plot in cell if cell for request in plot.get_track_clusterings_requests() if request]
        for interface in self._interfaces:
            interface.cluster_tracker.track_clusters(track_clusters_requests)

        # Determine the rendering order
        render_order = self._determine_plot_order()

        axis_limits = dict()

        for index in render_order:
            ax = self.plot_grid.get_ax_by_index(index)
            for plot in self.plots[index]:
                plot.plot(ax=ax, group_number=group_number,
                          axis_limits=axis_limits)

                # Store axis limits for future reference
                axis_limits[index] = (ax.get_xlim(), ax.get_ylim())
            if not self.plots[index]:
                ax.grid(False)
                ax.axis('off')
                ax.set_visible(False)

        return fig

    def plot(self, group_number: int) -> tuple[Figure, Axes]:
        fig = self.plot_grid.fig()
        self._plot(fig, group_number)

        return fig

    def plot_dynamics(self, mode: str | list[str] = 'show', save_dir: Optional[str] = None, animation_path: Optional[str] = None, name: str = 'opinion_histogram', preview: bool = False) -> None:
        """
        Create and display the plots based on the stored configurations.
        mode : ["show", "save", "gif", "mp4"]
        """
        if self._interfaces is None:
            raise ValueError("Interface not initialized. Cannot plot.")
        with Plotter.PlotSaver(mode=mode, save_path=save_dir, save_format=f"{name}_" + "{}.png", animation_path=animation_path) as saver:

            for group_number in range(self.number_of_groups()):

                fig = self.plot(group_number)
                saver.save(fig)

                if preview:
                    break

    def _determine_plot_order(self) -> list[tuple]:
        """
        Determine the order in which plots should be rendered based on dependencies.

        Returns:
            list of tuples: Each tuple represents the row and column indices of a plot,
            sorted in the order they should be rendered.
        """
        dependency_graph = nx.DiGraph()

        for index, plots in self.plots.items():
            dependency_graph.add_node(index)
            if plots is not None:
                for plot in plots:
                    dependencies = plot.plot_dependencies()

                    for dependency in dependencies['after']:
                        dependency_graph.add_edge(dependency, index)

        sorted_order = self._topological_sort(dependency_graph)

        return sorted_order

    def clear_plot(self, row: int, col: int):
        '''
        Clears axis on a given position in a grid  
        '''
        cell = self.plots.get((row, col))
        if cell:
            cell.clear()

    def _topological_sort(self, dependency_graph: nx.DiGraph):
        """
        Perform a topological sort on the dependency graph.

        Args:
            dependency_graph (networkx.DiGraph): The dependency graph.

        Returns:
            list of tuples: Sorted order of plots.
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
        if self._interfaces is None:
            return None
        parameters = [
            interface.node_parameters for interface in self._interfaces]
        # TODO figure out what does in return after the [interface]
        return parameters

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
        return [method_name for method_name, method in self._plot_types.items() if all(method.is_available(interface)[0] for interface in self._interfaces)]

    @classmethod
    def get_plot_class(cls, plot_type: str) -> type[Plot]:
        ''' Converts name of the class to the class'''
        return cls._plot_types[plot_type]

    @classmethod
    def get_plot_name(cls, plot_class: type[Plot]) -> str:
        ''' Converts the class to its name'''
        for key, value in cls._plot_types.items():
            if value == plot_class:
                return key
        warnings.warn(f'{plot_class} was not found in plot types')
        return ''

    @property
    def available_plot_types_hint(self) -> str:
        """
        Retrieves the list of available parameters/methods from the interface.

        Returns:
            str: A list of available parameters or methods.
        """
        info_string = ''
        for method_name, method in self._plot_types.items():
            is_available_final = True
            comments = []
            for interface in self._interfaces:
                is_available, comment = method.is_available(interface)
                is_available_final &= is_available
                comments.append(comment)

            final_comment = ', '.join(comments)

            info_string += method_name + ': '
            info_string += '+ ' if is_available else '- '
            info_string += final_comment
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

    def info(self) -> str:
        '''
        String with the information about the plotter setup
        '''
        info_str = ''
        if len(self.plots) == 0:
            return 'Plot is empty'
        for (i, j, interface), cell in self.plots.items():
            info_str += f'\n{i, j} {interface}\n'
            if len(cell) > 0:
                for plot in cell:
                    info_str += str(plot) + '\n'
            else:
                info_str += 'Empty cell'
        return info_str

    def to_code(self) -> str:
        '''
        Generates a string that can be used to create the current plotter setup. Can be used in GUI as well.
        '''
        if len(self.plots) == 0:
            return ''

        # TODO: Add the case when plots are different for different interfaces

        base_interface = self._interfaces[0]

        final_code = ''
        for (i, j, interface), cell in self.plots.items():
            if interface == base_interface:
                if len(cell) > 0:
                    for plot in cell:
                        final_code += 'plotter.add_plot('
                        final_code += f'\'{Plotter.get_plot_name(type(plot))}\''
                        final_code += ',{' + plot.settings_to_code() + '},'
                        final_code += f'row={i},col={j},)\n'
        return final_code
