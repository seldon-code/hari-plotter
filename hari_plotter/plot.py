from __future__ import annotations

import math
import warnings
from abc import ABC
from typing import (Any, Dict, Iterator, List, Optional, Sequence, Tuple, Type,
                    Union)

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap, NoNorm

from .cluster import Clustering, ParameterBasedClustering
from .color_scheme import ColorScheme
from .interface import Interface
from .parameters import (BoolParameter, FloatParameter, ListParameter,
                         NoneOrFloatParameter, NoneRangeParameter, Parameter)
from .plotter import Plotter


class Plot(ABC):
    def __init__(self):
        self.interface: Interface
        self.color_scheme: ColorScheme
        self.parameters: Tuple[str]
        self.scale: Tuple[str]
        self.show_x_label: bool
        self.show_y_label: bool
        self._x_lim: Sequence[float] | None
        self._y_lim: Sequence[float] | None

    @staticmethod
    def settings(interface: Interface):
        return []

    @classmethod
    def from_qt(cls, qt_settings: dict):
        return cls(**qt_settings)

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

        for value in [self._x_lim, self._y_lim]:
            if isinstance(value, str):
                # Assuming format is 'x(y)@row,col'
                ref_plot = tuple(map(int, value[2:].split(',')))
                # Add edge with (row, col) only
                dependencies['after'].append(ref_plot)
        return dependencies

    def get_limits(self, axis_limits: dict) -> List[Tuple[float | None]]:
        final_limits = []
        parameters = list(self.parameters) + [None] * \
            max(2 - len(self.parameters), 0)
        for i_lim, scale, parameter in zip((self._x_lim, self._y_lim), self.scale, parameters):
            if parameter == 'Time':
                final_limits.append((None, None))
            elif scale == 'Tanh':
                final_limits.append((-1., 1.))
            elif i_lim is None:
                final_limits.append((None, None))
            elif isinstance(i_lim, str):
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
    def transform_data(data_list, transform_parameter: str = 'Nodes'):
        # Extract unique transform_parameter's and sort them
        transform_parameter_values = {
            transform_parameter_value for data in data_list for transform_parameter_value in data[transform_parameter]}
        transform_parameter_values = {
            elem for elem in transform_parameter_values if elem is not None}
        transform_parameter_values = sorted(transform_parameter_values)
        transform_parameter_value_index = {transform_parameter_value: i for i,
                                           transform_parameter_value in enumerate(transform_parameter_values)}

        # Extract time steps
        time_steps = [data['Time'] for data in data_list]

        # Initialize parameters dictionary
        params = {key: np.full((len(transform_parameter_values), len(time_steps)), np.nan)
                  for key in data_list[0] if key not in [transform_parameter, 'Time', 'Nodes', 'Type']}
        if 'Type' in data_list[0]:
            params['Type'] = np.full(
                (len(transform_parameter_values), len(time_steps)), '')

        # Fill in the parameter values
        for t, data in enumerate(data_list):
            for param in params:
                if param in data and param != 'Nodes':
                    # Map each transform_parameter_value's value to the corresponding row in the parameter's array
                    for transform_parameter_value, value in zip(data[transform_parameter], data[param]):
                        if transform_parameter_value is not None:
                            idx = transform_parameter_value_index[transform_parameter_value]
                            params[param][idx, t] = value

        return {
            'Time': np.array(time_steps),
            transform_parameter: transform_parameter_values,
            **params
        }

    def get_static_plot_requests(self):
        return []

    def get_dynamic_plot_requests(self):
        return []

    def get_track_clusterings_requests(self) -> List[Dict[str, Any]]:
        return []

    @staticmethod
    def is_available(interface: Interface) -> Tuple[bool, str]:
        ''' Returns True if available for this interface and comment why'''
        return True, ''

    def settings_to_code(self) -> str:
        return ''

    def is_single_color(color: str | float | np.ndarray) -> bool:
        return isinstance(color, (str, float)) or color.shape == (4,) or color.shape == (3,)

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


@Plotter.plot_type("Histogram")
class plot_histogram(Plot):
    def __init__(self, interface: Interface, color_scheme: ColorScheme, parameter: str,
                 scale: Optional[Tuple[str] | None] = None,
                 rotated: Optional[bool] = False,
                 show_x_label: bool = True, show_y_label: bool = True,
                 exclude_types: Tuple[str] = (),
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None,
                 histogram_color: str | dict | float | None = None):
        self.interface: Interface = interface
        self.color_scheme: ColorScheme = color_scheme
        self.parameter: str = parameter
        self.parameters: Tuple[str] = (parameter,)
        self.scale: Tuple[str] = tuple(scale or ('Linear', 'Linear'))
        self.exclude_types: Tuple[str] = exclude_types
        self.rotated: bool = rotated
        self.show_x_label: bool = show_x_label
        self.show_y_label: bool = show_y_label
        self._x_lim: Sequence[float] | None = x_lim
        self._y_lim: Sequence[float] | None = y_lim

        def histogram_color_to_histogram_color_settings(histogram_color) -> dict:
            if isinstance(histogram_color, dict):
                # check if only 'mode' and 'settings' in dict
                if not all(key in {'mode', 'settings'} for key in histogram_color.keys()):
                    raise ValueError(
                        'Histogram color is incorrectly formatted')
                return histogram_color
            if isinstance(histogram_color, (str, float)):
                return {'mode': 'Constant Color', 'settings': {'color': histogram_color}}
            else:
                return {'mode': 'Constant Color'}

        self.histogram_color_settings: dict = histogram_color_to_histogram_color_settings(
            histogram_color)

        if self.histogram_color_settings['mode'] not in self.color_scheme.method_logger['Distribution Color']['modes']:
            raise ValueError('Histogram color is incorrectly formatted')

    def get_track_clusterings_requests(self):
        return [self.color_scheme.requires_tracking(self.histogram_color_settings)]

    def settings_to_code(self) -> str:
        return ('\'parameter\':'+self.parameter +
                ',\'scale\':'+str(self.scale) +
                ',\'rotated\':'+str(self.rotated) +
                ',\'show_x_label\':'+str(self.show_x_label) +
                ',\'show_y_label\':'+str(self.show_y_label) +
                ',\'x_lim\':'+str(self._x_lim) +
                ',\'y_lim\':'+str(self._y_lim) +
                ',\'histogram_color\':'+str(self.histogram_color_settings))

    @staticmethod
    def settings(interface: Interface) -> List[Parameter]:
        return [ListParameter(name='Parameter', parameter_name='parameter', arguments=interface.node_parameters, comment='Parameter of the histogram'),
                ListParameter(name='Scale', parameter_name='scale', arguments=[
                              'Linear', 'Tanh'], comment='Scale of the parameter'),
                BoolParameter(name='Rotate', parameter_name='rotated', default_value=False,
                              comment='Should the histogram be rotated?'),
                BoolParameter(
                    name='Show X Label', parameter_name='show_x_label', default_value=True, comment=''),
                BoolParameter(
                    name='Show Y Label', parameter_name='show_y_label', default_value=True, comment=''),
                NoneRangeParameter(name='X Limit', parameter_name='x_lim',
                                   default_min_value=None, default_max_value=None, limits=(None, None), comment=''),
                NoneRangeParameter(name='Y Limit', parameter_name='y_lim',
                                   default_min_value=None, default_max_value=None, limits=(None, None), comment=''),
                ]

    @staticmethod
    def qt_to_settings(qt_settings: dict) -> dict:
        '''
        Transforms dict of settings from PyQT GUI to the dict that will be used for class init.
        '''
        settings = qt_settings.copy()

        # Extract the value of 'parameter' and remove it from settings
        parameter_value = settings.pop('parameter', None)

        settings['parameter'] = parameter_value

        settings['scale'] = (settings['scale'], 'Linear')

        return settings

    def get_dynamic_plot_requests(self) -> List[dict]:
        return [{'method': 'calculate_node_values', 'settings': {'parameters': (self.parameter, 'Type'), 'scale': self.scale}}]

    def plot(self, ax: plt.Axes, group_number: int,  axis_limits: dict):
        """
        Plot a histogram on the given ax with the provided data data.

        Parameters:
        -----------
        ax : plt.Axes
            Axes object where the histogram will be plotted.
        data : list[float]
            List containing parameter values.
        scale : str, optional
            The scale for the x-axis. Options: 'Linear' or 'Tanh'.
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
        data = self.interface.dynamic_data_cache[group_number][self.get_dynamic_plot_requests()[
            0]]

        nodes = data['Nodes']
        types = data['Type']
        acceptable_types = np.array(
            [t not in self.exclude_types for t in types])

        self.parameter = self.parameters[0]
        values = np.array(data[self.parameter])
        valid_indices = (~np.isnan(values)) & acceptable_types
        values = values[valid_indices]

        histogram_color = self.color_scheme.distribution_color(
            nodes=nodes, group_number=group_number, **self.histogram_color_settings)

        if self.rotated:
            if self.scale[1] == 'Tanh':
                values = np.tanh(values)

            y_lim = [np.nanmin(values) if y_lim[0] is None else y_lim[0], np.nanmax(
                values) if y_lim[1] is None else y_lim[1]]

            values = values[(values >= y_lim[0]) & (values <= y_lim[1])]

            sns.kdeplot(y=values, ax=ax, fill=True, color=histogram_color)
            sns.histplot(y=values, kde=False, ax=ax,
                         binrange=y_lim, element="step", fill=False, stat="density", color=histogram_color)

            if self.show_y_label:
                ax.set_ylabel(Plotter._parameter_dict.get(
                    self.parameter, self.parameter))
            if self.show_x_label:
                ax.set_xlabel('Density')

        else:
            if self.scale[0] == 'Tanh':
                values = np.tanh(values)

            x_lim = [np.nanmin(values) if x_lim[0] is None else x_lim[0], np.nanmax(
                values) if x_lim[1] is None else x_lim[1]]

            values = values[(values >= x_lim[0]) & (values <= x_lim[1])]

            sns.kdeplot(data=values, ax=ax, fill=True, color=histogram_color)
            sns.histplot(data=values, kde=False, ax=ax,
                         binrange=x_lim, element="step", fill=False, stat="density", color=histogram_color)

            if self.show_x_label:
                ax.set_xlabel(Plotter._parameter_dict.get(
                    self.parameter, self.parameter))
            if self.show_y_label:
                ax.set_ylabel('Density')

        Plot.tanh_axis_labels(ax=ax, scale=self.scale)
        if x_lim is not None:
            ax.set_xlim(*x_lim)
        if y_lim is not None:
            ax.set_ylim(*y_lim)


@Plotter.plot_type("Hexbin")
class plot_hexbin(Plot):
    def __init__(self, interface: Interface, color_scheme: ColorScheme, parameters: tuple[str],
                 scale: Optional[Tuple[str] | None] = None,
                 rotated: Optional[bool] = False,
                 show_x_label: bool = True, show_y_label: bool = True,
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None, colormap: str | None = None, show_colorbar: bool = False):
        self.interface: Interface = interface
        self.color_scheme = color_scheme
        self.parameters = tuple(parameters)
        self.scale = tuple(scale or ('Linear', 'Linear'))
        self.rotated = rotated
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self._x_lim = x_lim
        self._y_lim = y_lim
        self.show_colorbar = show_colorbar

        def colormap_to_colormap_settings(colormap) -> dict:
            if isinstance(colormap, dict):
                # check if only 'mode' and 'settings' in dict
                if not all(key in {'mode', 'settings'} for key in colormap.keys()):
                    raise ValueError(
                        'Colormap is incorrectly formatted')
                return colormap
            if isinstance(colormap, (str, float)):
                return {'mode': 'Independent Colormap', 'settings': {'colormap': colormap}}
            else:
                return {'mode': 'Independent Colormap'}

        self.colormap_settings = colormap_to_colormap_settings(colormap)

        if self.colormap_settings['mode'] not in self.color_scheme.method_logger['Color Map']['modes']:
            raise ValueError(
                f"Colormap is incorrectly formatted: Mode {self.colormap_settings['mode']} not in known modes {self.color_scheme.method_logger['Color Map']['modes']}")

    def get_track_clusterings_requests(self):
        return [self.color_scheme.requires_tracking(self.colormap_settings)]

    def settings_to_code(self) -> str:
        return ('\'parameters\':'+str(self.parameters) +
                ',\'scale\':'+str(self.scale) +
                ',\'rotated\':'+str(self.rotated) +
                ',\'show_x_label\':'+str(self.show_x_label) +
                ',\'show_y_label\':'+str(self.show_y_label) +
                ',\'x_lim\':'+str(self._x_lim) +
                ',\'y_lim\':'+str(self._y_lim) +
                ',\'show_colorbar\':'+str(self.show_colorbar))

    @staticmethod
    def settings(interface: Interface) -> List[Parameter]:
        return [ListParameter(name='X parameter', parameter_name='x_parameter', arguments=interface.node_parameters, comment=''),
                ListParameter(name='X scale', parameter_name='x_scale', arguments=[
                              'Linear', 'Tanh'], comment=''),
                ListParameter(name='Y parameter', parameter_name='y_parameter',
                              arguments=interface.node_parameters, comment=''),
                ListParameter(name='Y scale', parameter_name='y_scale', arguments=[
                              'Linear', 'Tanh'], comment=''),
                BoolParameter(name='Rotate', parameter_name='rotated', default_value=False,
                              comment='Should the plot be rotated?'),
                BoolParameter(
                    name='Show X Label', parameter_name='show_x_label', default_value=True, comment=''),
                BoolParameter(
                    name='Show Y Label', parameter_name='show_y_label', default_value=True, comment=''),
                NoneRangeParameter(name='X Limit', parameter_name='x_lim',
                                   default_min_value=None, default_max_value=None, limits=(None, None), comment=''),
                NoneRangeParameter(name='Y Limit', parameter_name='y_lim',
                                   default_min_value=None, default_max_value=None, limits=(None, None), comment=''),
                BoolParameter(
                    name='Show Colorbar', parameter_name='show_colorbar', default_value=False, comment=''),
                ]

    @staticmethod
    def qt_to_settings(qt_settings: dict) -> dict:
        # Copy qt_settings to avoid modifying the original dictionary
        settings = qt_settings.copy()

        # Extract the value of 'parameter' and remove it from settings
        x_parameter_value = settings.pop('x_parameter', None)
        y_parameter_value = settings.pop('y_parameter', None)

        x_scale = settings.pop('x_scale', None)
        y_scale = settings.pop('y_scale', None)

        settings['parameters'] = (x_parameter_value, y_parameter_value)

        settings['scale'] = (x_scale, y_scale)

        return settings

    def get_dynamic_plot_requests(self):
        return [{'method': 'calculate_node_values', 'settings': {'parameters': self.parameters, 'scale': self.scale}}]

    def plot(self, ax: plt.Axes, group_number: int,  axis_limits: dict):
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
            Scale for the plot values (x and y). Options: 'Linear' or 'Tanh'. Default is 'Linear' for both.
        show_colorbar : bool, optional
        """

        x_lim, y_lim = self.get_limits(axis_limits)

        data = self.interface.dynamic_data_cache[group_number][self.get_dynamic_plot_requests()[
            0]]

        x_parameter = self.parameters[0]
        y_parameter = self.parameters[1]
        x_values = np.array(data[x_parameter])
        y_values = np.array(data[y_parameter])
        nodes = data['Nodes']
        colormap = self.color_scheme.colorbar(
            nodes=nodes, group_number=group_number, **self.colormap_settings)

        # Find indices where neither x_values nor y_values are NaN
        valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)

        # Filter the values using these indices
        x_values = x_values[valid_indices]
        y_values = y_values[valid_indices]

        if self.scale[0] == 'Tanh':
            x_values = np.tanh(x_values)

        if self.scale[1] == 'Tanh':
            y_values = np.tanh(y_values)

        if x_lim == (None, None):
            x_lim = [-1, 1] if self.scale[0] == 'Tanh' else [
                np.nanmin(x_values), np.nanmax(x_values)]
        if y_lim == (None, None):
            y_lim = [-1, 1] if self.scale[1] == 'Tanh' else [
                np.nanmin(y_values), np.nanmax(y_values)]

        extent = x_lim+y_lim

        delta_x = 0.1*(extent[1]-extent[0])
        x_field_extent = [extent[0]-delta_x, extent[1]+delta_x]

        delta_y = 0.1*(extent[3]-extent[2])
        y_field_extent = [extent[2]-delta_y, extent[3]+delta_y]

        field_extent = x_field_extent + y_field_extent

        ax.imshow([[0, 0], [0, 0]], cmap=colormap,
                  interpolation='nearest', aspect='auto', extent=field_extent)

        hb = ax.hexbin(x_values, y_values, gridsize=50,
                       bins='log', extent=extent, cmap=colormap)

        # Create a background filled with the `0` value of the colormap
        ax.imshow([[0, 0], [0, 0]], cmap=colormap,
                  interpolation='nearest', aspect='auto', extent=extent)
        # Create the hexbin plot

        hb = ax.hexbin(x_values, y_values, gridsize=50, cmap=colormap,
                       bins='log', extent=extent)

        Plot.tanh_axis_labels(ax=ax, scale=self.scale)

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
    def __init__(self, interface: Interface, color_scheme: ColorScheme, parameters: tuple[str],
                 scale: Optional[Tuple[str] | None] = None,
                 rotated: Optional[bool] = False,
                 show_x_label: bool = True, show_y_label: bool = True,
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None,
                 color: Optional[str | None] = None, marker: Optional[str | None] = None):
        self.interface: Interface = interface
        self.parameters = tuple(parameters)
        self.scale = tuple(scale or ('Linear', 'Linear'))
        self.rotated = rotated
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self._x_lim = x_lim
        self._y_lim = y_lim
        self.color_scheme = color_scheme

        def scatter_color_to_scatter_color_settings(scatter_color) -> dict:
            if isinstance(scatter_color, dict):
                # check if only 'mode' and 'settings' in dict
                if not all(key in {'mode', 'settings'} for key in scatter_color.keys()):
                    raise ValueError(
                        'Scatter color is incorrectly formatted')
                return scatter_color
            if isinstance(scatter_color, (str, float)):
                return {'mode': 'Constant Color', 'settings': {'color': scatter_color}}
            else:
                return {'mode': 'Constant Color'}

        self.scatter_color_settings: dict = scatter_color_to_scatter_color_settings(
            color)

        if self.scatter_color_settings['mode'] not in self.color_scheme.method_logger['Scatter Color']['modes']:
            raise ValueError(
                f"Scatter color is incorrectly formatted: Mode {self.scatter_color_settings['mode']} not in known modes {self.color_scheme.method_logger['Scatter Color']['modes']}")

        def scatter_marker_to_scatter_marker_settings(scatter_marker) -> dict:
            if isinstance(scatter_marker, dict):
                # check if only 'mode' and 'settings' in dict
                if not all(key in {'mode', 'settings'} for key in scatter_marker.keys()):
                    raise ValueError(
                        'Scatter marker is incorrectly formatted')
                return scatter_marker
            if isinstance(scatter_marker, (str, float)):
                return {'mode': 'Constant Marker', 'settings': {'marker': scatter_marker}}
            else:
                return {'mode': 'Constant Marker'}

        self.scatter_marker_settings: dict = scatter_marker_to_scatter_marker_settings(
            marker)

    def get_track_clusterings_requests(self):
        return [self.color_scheme.requires_tracking(self.scatter_color_settings), self.color_scheme.requires_tracking(self.scatter_marker_settings)]

    def get_dynamic_plot_requests(self):
        return [{'method': 'calculate_node_values', 'settings': {'parameters': self.parameters, 'scale': self.scale}}]

    def plot(self, ax: plt.Axes, group_number: int,  axis_limits: dict):
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

        data = self.interface.dynamic_data_cache[group_number][self.get_dynamic_plot_requests()[
            0]]

        x_parameter, y_parameter = self.parameters
        x_values = np.array(data[x_parameter])
        y_values = np.array(data[y_parameter])
        nodes = data['Nodes']

        # Remove NaN values
        valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
        x_values = x_values[valid_indices]
        y_values = y_values[valid_indices]
        valid_nodes = [nodes[i] for i in np.where(valid_indices)[0]]

        if self.scale[0] == 'Tanh':
            x_values = np.tanh(x_values)

        if self.scale[1] == 'Tanh':
            y_values = np.tanh(y_values)

        colors = self.color_scheme.scatter_colors_nodes(
            nodes=valid_nodes, group_number=group_number, **self.scatter_color_settings)
        markers = self.color_scheme.scatter_markers_nodes(
            nodes=valid_nodes, group_number=group_number, **self.scatter_marker_settings)

        if isinstance(markers, (str, type(None))):
            ax.scatter(x_values, y_values, color=colors, marker=markers)
        else:
            # Convert markers to a NumPy array for efficient processing
            markers = np.array(markers)
            unique_markers = np.unique(markers)
            colors = np.array(colors)
            colors_is_array = not (Plot.is_single_color(colors))
            if colors_is_array:
                # Ensure colors is a NumPy array of individual colors for each point
                colors = np.array(colors)
            else:
                # Single color definition, use it directly for all points
                c = colors

            # Plot each group of points with the same marker individually
            for marker in unique_markers:
                # Find indices of points with the current marker
                indices = np.where(markers == marker)[0]

                if colors_is_array:
                    # Extract the colors for the selected indices for individual colors per point
                    c = colors[indices, :]

                # Plot these points as a separate scatter plot
                ax.scatter(x_values[indices],
                           y_values[indices], color=c, marker=marker)

        # Setting the plot limits
        if x_lim is not None:
            ax.set_xlim(*x_lim)
        if y_lim is not None:
            ax.set_ylim(*y_lim)

        Plot.tanh_axis_labels(ax=ax, scale=self.scale)

        if self.show_x_label:
            ax.set_xlabel(Plotter._parameter_dict.get(
                x_parameter, x_parameter))

        if self.show_y_label:
            ax.set_ylabel(Plotter._parameter_dict.get(
                y_parameter, y_parameter))


@Plotter.plot_type("Clustering: Centroids")
class plot_clustering_centroids(Plot):
    def __init__(self, interface: Interface, color_scheme: ColorScheme, parameters: tuple[str], clustering_settings: dict,
                 scale: Optional[Tuple[str] | None] = None,
                 rotated: Optional[bool] = False,
                 show_x_label: bool = True, show_y_label: bool = True,
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None,
                 color: Optional[str | None] = None, marker: Optional[str | None] = None):
        self.interface: Interface = interface
        self.parameters = tuple(parameters)
        self.clustering_settings = clustering_settings
        self.scale = tuple(scale or ('Linear', 'Linear'))
        self.rotated = rotated
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self._x_lim = x_lim
        self._y_lim = y_lim
        self.color_scheme = color_scheme

        def centroid_color_to_centroid_color_settings(centroid_color) -> dict:
            if isinstance(centroid_color, dict):
                # check if only 'mode' and 'settings' in dict
                if not all(key in {'mode', 'settings'} for key in centroid_color.keys()):
                    raise ValueError(
                        'Centroid color is incorrectly formatted')
                return centroid_color
            if isinstance(centroid_color, (str, float)):
                return {'mode': 'Constant Color', 'settings': {'color': centroid_color}}
            else:
                return {'mode': 'Constant Color'}

        self.centroid_color_settings: dict = centroid_color_to_centroid_color_settings(
            color)

        if self.centroid_color_settings['mode'] not in self.color_scheme.method_logger['Scatter Color']['modes']:
            raise ValueError(
                f"Centroid color is incorrectly formatted: Mode {self.centroid_color_settings['mode']} not in known modes {self.color_scheme.method_logger['Centroid Color']['modes']}")

        def centroid_marker_to_centroid_marker_settings(centroid_marker) -> dict:
            if isinstance(centroid_marker, dict):
                # check if only 'mode' and 'settings' in dict
                if not all(key in {'mode', 'settings'} for key in centroid_marker.keys()):
                    raise ValueError(
                        'Centroid marker is incorrectly formatted')
                return centroid_marker
            if isinstance(centroid_marker, (str, float)):
                return {'mode': 'Constant Marker', 'settings': {'marker': centroid_marker}}
            else:
                return {'mode': 'Constant Marker'}

        self.centroid_marker_settings: dict = centroid_marker_to_centroid_marker_settings(
            marker)

        if self.centroid_marker_settings['mode'] not in self.color_scheme.method_logger['Centroid Marker']['modes']:
            raise ValueError(
                f"Centroid marker is incorrectly formatted: Mode {self.centroid_marker_settings['mode']} not in known modes {self.color_scheme.method_logger['Centroid Marker']['modes']}")

    def get_track_clusterings_requests(self):
        return [self.color_scheme.requires_tracking(self.centroid_color_settings), self.color_scheme.requires_tracking(self.centroid_marker_settings)]

    def get_dynamic_plot_requests(self):
        return [{'method': 'get_clustering', 'settings': {'parameters': self.parameters, 'scale': self.scale, 'clustering_settings': self.clustering_settings}}]

    def plot(self, ax: plt.Axes, group_number: int,  axis_limits: dict):
        """
        Plots the decision boundaries for a 2D slice of the clustering object's data.

        Args:
        - x_feature_index (int): The index of the feature to be plotted on the x-axis.
        - y_feature_index (int): The index of the feature to be plotted on the y-axis.
        - plot_limits (tuple): A tuple containing the limits of the plot: (x_min, x_max, y_min, y_max).
        - resolution (int): The number of points to generate in the mesh for the plot.

        Returns:
        None
        """

        x_lim, y_lim = self.get_limits(axis_limits)
        clustering: ParameterBasedClustering = self.interface.dynamic_data_cache[group_number][self.get_dynamic_plot_requests()[
            0]]

        x_feature_name, y_feature_name = self.parameters

        x_feature_index, y_feature_index = clustering.get_indices_from_parameters(
            [x_feature_name, y_feature_name])

        # Plot centroids if they are 2D
        centroids = clustering.centroids()
        labels = clustering.cluster_labels

        markers = self.color_scheme.centroid_markers(
            clusters=labels, group_number=group_number, **self.centroid_marker_settings)
        colors = self.color_scheme.centroid_colors(
            clusters=labels, group_number=group_number, **self.centroid_color_settings)

        if centroids.shape[1] != 2:
            raise ValueError(f"Centroids ({centroids}) shape is incorrect")

        x_values = centroids[:, x_feature_index]
        y_values = centroids[:, y_feature_index]
        if self.scale[0] == 'Tanh':
            x_values = np.tanh(x_values)
        if self.scale[1] == 'Tanh':
            y_values = np.tanh(y_values)

        if isinstance(markers, (str, type(None))):
            ax.scatter(x_values, y_values, color=colors, marker=markers)
        else:
            # Convert markers to a NumPy array for efficient processing
            markers = np.array(markers)
            unique_markers = np.unique(markers)

            # Check the type of colors to handle single color definitions
            colors = np.array(colors)
            colors_is_array = not (Plot.is_single_color(colors))
            if colors_is_array:
                # Ensure colors is a NumPy array of individual colors for each point
                colors = np.array(colors)
            else:
                # Single color definition, use it directly for all points
                c = colors

            # Plot each group of points with the same marker individually
            for marker in unique_markers:
                # Find indices of points with the current marker
                indices = np.where(markers == marker)[0]

                if colors_is_array:
                    # Extract the colors for the selected indices for individual colors per point
                    c = colors[indices, :]

                # Plot these points as a separate scatter plot
                ax.scatter(x_values[indices],
                           y_values[indices], color=c, marker=marker)

        # Setting the plot limits
        if x_lim is not None:
            ax.set_xlim(*x_lim)
        if y_lim is not None:
            ax.set_ylim(*y_lim)

        Plot.tanh_axis_labels(ax=ax, scale=self.scale)

        if self.show_x_label:
            ax.set_xlabel(Plotter._parameter_dict.get(
                self.parameters[0], self.parameters[0]))
        if self.show_y_label:
            ax.set_ylabel(Plotter._parameter_dict.get(
                self.parameters[1], self.parameters[1]))


@Plotter.plot_type("Clustering: Fill")
class plot_clustering_fill(Plot):
    def __init__(self, interface: Interface, color_scheme: ColorScheme, parameters: tuple[str], clustering_settings: dict = {},
                 scale: Optional[Tuple[str] | None] = None,
                 show_x_label: bool = True, show_y_label: bool = True,
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None,
                 fill_color: dict = None, alpha: float = 0.2, resolution: int = 100):
        self.interface: Interface = interface
        self.parameters = tuple(parameters)
        self.color_scheme = color_scheme
        self.clustering_settings = clustering_settings

        self.scale = tuple(scale or ('Linear', 'Linear'))
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self._x_lim = x_lim
        self._y_lim = y_lim
        self.resolution = resolution
        self.alpha = alpha

        def fill_color_to_fill_color_settings(fill_color) -> dict:
            if isinstance(fill_color, dict):
                # check if only 'mode' and 'settings' in dict
                if not all(key in {'mode', 'settings'} for key in fill_color.keys()):
                    raise ValueError(
                        'Histogram color is incorrectly formatted')
                return fill_color
            else:
                return {'mode': 'Cluster Color', 'settings': {'clustering_settings': self.clustering_settings}}

        self.fill_color_settings: dict = fill_color_to_fill_color_settings(
            fill_color)

    def get_track_clusterings_requests(self):
        return [self.color_scheme.requires_tracking(self.fill_color_settings)]

    def get_dynamic_plot_requests(self):
        return [{'method': 'get_clustering', 'settings': {'parameters': self.parameters, 'scale': self.scale, 'clustering_settings': self.clustering_settings}}]

    def plot(self, ax: plt.Axes, group_number: int,  axis_limits: dict):
        x_lim, y_lim = self.get_limits(axis_limits)
        clustering: Clustering = self.interface.dynamic_data_cache[group_number][self.get_dynamic_plot_requests()[
            0]]

        labels = clustering.cluster_labels

        colors = self.color_scheme.fill_colors(
            clusters=labels, group_number=group_number, **self.fill_color_settings)

        x_feature_name, y_feature_name = self.parameters
        x_feature_index, y_feature_index = clustering.get_indices_from_parameters(
            [x_feature_name, y_feature_name])

        if np.any(x_lim) == None:
            x_lim = [np.nanmin(clustering.data[:, x_feature_index]), np.nanmax(
                clustering.data[:, x_feature_index])]

        if np.any(y_lim) == None:
            y_lim = [np.nanmin(clustering.data[:, y_feature_index]), np.nanmax(
                clustering.data[:, y_feature_index])]

        xx, yy = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], self.resolution), np.linspace(
                y_lim[0], y_lim[1], self.resolution)
        )

        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_points_scaled = np.array(mesh_points)
        if self.scale[0] == 'Tanh':
            mesh_points_scaled[:, 0] = np.arctanh(mesh_points_scaled[:, 0])
        if self.scale[1] == 'Tanh':
            mesh_points_scaled[:, 1] = np.arctanh(mesh_points_scaled[:, 1])

        Z = clustering.predict_cluster(mesh_points_scaled)
        Z = Z.reshape(xx.shape)

        # Create a ListedColormap with your colors
        cmap = ListedColormap(colors)

        # Custom function to check if a value is None or NaN
        def is_nan_or_none(value):
            return value is None or (isinstance(value, float) and np.isnan(value))

        # Replace None with np.nan
        Z = np.where(Z == None, np.nan, Z)

        # Convert the array to float
        Z = Z.astype(float)

        # Use the custom colormap in imshow, with NoNorm to avoid normalization of Z values
        im = ax.imshow(Z, extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]],
                       origin='lower', aspect='auto', alpha=self.alpha, interpolation='nearest',
                       cmap=cmap, norm=NoNorm())

        Plot.tanh_axis_labels(ax=ax, scale=self.scale)

        if self.show_x_label:
            ax.set_xlabel(Plotter._parameter_dict.get(
                self.parameters[0], self.parameters[0]))
        if self.show_y_label:
            ax.set_ylabel(Plotter._parameter_dict.get(
                self.parameters[1], self.parameters[1]))


@Plotter.plot_type("Clustering: Degree of Membership")
class plot_clustering_degree_of_membership(Plot):
    def __init__(self, interface: Interface, color_scheme: ColorScheme, parameters: tuple[str], clustering_settings: dict = {},
                 scale: Optional[Tuple[str] | None] = None,
                 show_x_label: bool = True, show_y_label: bool = True,
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None,
                 colormap=None, alpha: float = 0.2, resolution: int = 100):
        self.interface: Interface = interface
        self.parameters = tuple(parameters)
        self.color_scheme = color_scheme
        self.clustering_settings = clustering_settings
        if 'clustering_parameters' not in self.clustering_settings:
            self.clustering_settings['clustering_parameters'] = self.parameters

        self.scale = tuple(scale or ('Linear', 'Linear'))
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self._x_lim = x_lim
        self._y_lim = y_lim
        self.resolution = resolution
        self.alpha = alpha

        def colormap_to_colormap_settings(colormap) -> dict:
            if isinstance(colormap, dict):
                # check if only 'mode' and 'settings' in dict
                if not all(key in {'mode', 'settings'} for key in colormap.keys()):
                    raise ValueError(
                        'Colormap is incorrectly formatted')
                return colormap
            if isinstance(colormap, (str, float)):
                return {'mode': 'Independent Colormap', 'settings': {'colormap': colormap}}
            else:
                return {'mode': 'Independent Colormap'}

        self.colormap_settings = colormap_to_colormap_settings(colormap)

        if self.colormap_settings['mode'] not in self.color_scheme.method_logger['Color Map']['modes']:
            raise ValueError(
                f"Colormap is incorrectly formatted: Mode {self.colormap_settings['mode']} not in known modes {self.color_scheme.method_logger['Color Map']['modes']}")

    def get_track_clusterings_requests(self):
        return [self.color_scheme.requires_tracking(self.colormap_settings)]

    def get_dynamic_plot_requests(self):
        return [{'method': 'get_clustering', 'settings': {'parameters': self.parameters, 'scale': self.scale, 'clustering_settings': self.clustering_settings}}]

    def plot(self, ax: plt.Axes, group_number: int,  axis_limits: dict):
        x_lim, y_lim = self.get_limits(axis_limits)
        clustering: Clustering = self.interface.dynamic_data_cache[group_number][
            self.get_dynamic_plot_requests()[0]]
        colormap = self.color_scheme.colorbar(
            group_number=group_number, **self.colormap_settings)

        xx, yy = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], self.resolution), np.linspace(
                y_lim[0], y_lim[1], self.resolution)
        )

        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_points_scaled = np.array(mesh_points)
        if self.scale[0] == 'Tanh':
            mesh_points_scaled[:, 0] = np.arctanh(mesh_points_scaled[:, 0])
        if self.scale[1] == 'Tanh':
            mesh_points_scaled[:, 1] = np.arctanh(mesh_points_scaled[:, 1])

        Z = np.array(clustering.degree_of_membership(mesh_points_scaled))
        Z = Z.max(axis=0)
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=self.alpha,
                    levels=np.linspace(0, 1, 11), cmap=colormap)

        Plot.tanh_axis_labels(ax=ax, scale=self.scale)

        if self.show_x_label:
            ax.set_xlabel(Plotter._parameter_dict.get(
                self.parameters[0], self.parameters[0]))
        if self.show_y_label:
            ax.set_ylabel(Plotter._parameter_dict.get(
                self.parameters[1], self.parameters[1]))


@Plotter.plot_type("Clustering: Density Plot")
class plot_clustering_density(Plot):
    def __init__(self, interface: Interface, color_scheme: ColorScheme, parameters: tuple[str], clustering_settings: dict = {},
                 scale: Optional[Tuple[str] | None] = None,
                 show_x_label: bool = True, show_y_label: bool = True,
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None,
                 fill_color: dict = None, alpha: float = 0.5, levels: int = 5, thresh: float = 0.5, resolution: int = 100):
        self.interface: Interface = interface
        self.parameters = tuple(parameters)
        self.color_scheme = color_scheme
        self.clustering_settings = clustering_settings
        if 'clustering_parameters' not in self.clustering_settings:
            self.clustering_settings['clustering_parameters'] = self.parameters

        self.scale = tuple(scale or ('Linear', 'Linear'))
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self._x_lim = x_lim
        self._y_lim = y_lim
        self.resolution = resolution

        self.alpha = alpha
        self.levels = levels
        self.thresh = thresh

        def fill_color_to_fill_color_settings(fill_color) -> dict:
            if isinstance(fill_color, dict):
                # check if only 'mode' and 'settings' in dict
                if not all(key in {'mode', 'settings'} for key in fill_color.keys()):
                    raise ValueError(
                        'Histogram color is incorrectly formatted')
                return fill_color
            else:
                return {'mode': 'Cluster Color', 'settings': {'clustering_settings': self.clustering_settings}}

        self.fill_color_settings: dict = fill_color_to_fill_color_settings(
            fill_color)

    def get_track_clusterings_requests(self):
        return [self.clustering_settings, self.color_scheme.requires_tracking(self.fill_color_settings)]

    def get_dynamic_plot_requests(self):
        return [{'method': 'get_clustering', 'settings': {'parameters': self.parameters, 'scale': self.scale, 'clustering_settings': self.clustering_settings}}]

    def plot(self, ax: plt.Axes, group_number: int,  axis_limits: dict):
        x_lim, y_lim = self.get_limits(axis_limits)
        clustering: Clustering = self.interface.dynamic_data_cache[group_number][self.get_dynamic_plot_requests()[
            0]]
        labels = clustering.cluster_labels

        colors = self.color_scheme.fill_colors(
            clusters=labels, group_number=group_number, **self.fill_color_settings)

        xx, yy = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], self.resolution), np.linspace(
                y_lim[0], y_lim[1], self.resolution)
        )

        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_points_scaled = np.array(mesh_points)
        if self.scale[0] == 'Tanh':
            mesh_points_scaled[:, 0] = np.arctanh(mesh_points_scaled[:, 0])
        if self.scale[1] == 'Tanh':
            mesh_points_scaled[:, 1] = np.arctanh(mesh_points_scaled[:, 1])

        Z = np.array(clustering.degree_of_membership(mesh_points_scaled))
        Z = Z.reshape(-1, *xx.shape)
        Z_index = Z.argmax(axis=0)
        Z_flat = Z.max(axis=0).ravel()

        xx_flat = xx.ravel()
        yy_flat = yy.ravel()
        Z_index_flat = Z_index.ravel()

        # Create a mapping from integer categories to custom labels
        Z_index_flat_categorical = pd.Categorical(Z_index_flat)
        category_mapping = {i: label for i, label in enumerate(labels)}

        # Map the categories in Z_index_flat_categorical to custom labels
        Z_index_flat_mapped = Z_index_flat_categorical.map(category_mapping)

        sns.kdeplot(
            ax=ax,
            x=xx_flat,
            y=yy_flat,
            hue=Z_index_flat_mapped,  # Use the mapped data for hue
            weights=Z_flat,
            levels=self.levels,
            thresh=self.thresh,
            alpha=self.alpha,
            palette=colors  # Use the 'colors' list as a palette
        )

        Plot.tanh_axis_labels(ax=ax, scale=self.scale)

        if self.show_x_label:
            ax.set_xlabel(Plotter._parameter_dict.get(
                self.parameters[0], self.parameters[0]))
        if self.show_y_label:
            ax.set_ylabel(Plotter._parameter_dict.get(
                self.parameters[1], self.parameters[1]))


@Plotter.plot_type("Clustered Histogram")
class plot_clustered_histogram(Plot):
    def __init__(self, interface: Interface, color_scheme: ColorScheme, parameter: str,
                 clustering_settings: Optional[dict] = None,
                 scale: Optional[Tuple[str] | None] = None,
                 rotated: Optional[bool] = False,
                 show_x_label: bool = True, show_y_label: bool = True,
                 exclude_types: Tuple[str] = (),
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None,
                 fill_color: str | dict | float | None = None):
        self.interface: Interface = interface
        self.color_scheme: ColorScheme = color_scheme
        self.parameter: str = parameter
        self.parameters: Tuple[str] = (parameter,)
        self.clustering_settings = clustering_settings
        self.scale: Tuple[str] = tuple(scale or ('Linear', 'Linear'))
        self.exclude_types: Tuple[str] = exclude_types
        self.rotated: bool = rotated
        self.show_x_label: bool = show_x_label
        self.show_y_label: bool = show_y_label
        self._x_lim: Sequence[float] | None = x_lim
        self._y_lim: Sequence[float] | None = y_lim

        def fill_color_to_fill_color_settings(fill_color) -> dict:
            if isinstance(fill_color, dict):
                # check if only 'mode' and 'settings' in dict
                if not all(key in {'mode', 'settings'} for key in fill_color.keys()):
                    raise ValueError(
                        'Histogram color is incorrectly formatted')
                return fill_color
            else:
                return {'mode': 'Cluster Color', 'settings': {'clustering_settings': self.clustering_settings}}

        self.fill_color_settings: dict = fill_color_to_fill_color_settings(
            fill_color)

    def get_track_clusterings_requests(self):
        return [self.color_scheme.requires_tracking(self.fill_color_settings)]

    def settings_to_code(self) -> str:
        return ('\'parameter\':' + self.parameter +
                ',\'scale\':' + str(self.scale) +
                ',\'rotated\':' + str(self.rotated) +
                ',\'show_x_label\':' + str(self.show_x_label) +
                ',\'show_y_label\':' + str(self.show_y_label) +
                ',\'x_lim\':' + str(self._x_lim) +
                ',\'y_lim\':' + str(self._y_lim) +
                ',\'fill_color\':' + str(self.fill_color_settings))

    @staticmethod
    def settings(interface: Interface) -> List[Parameter]:
        return [ListParameter(name='Parameter', parameter_name='parameter', arguments=interface.node_parameters, comment='Parameter of the histogram'),
                ListParameter(name='Scale', parameter_name='scale', arguments=[
                              'Linear', 'Tanh'], comment='Scale of the parameter'),
                BoolParameter(name='Rotate', parameter_name='rotated',
                              default_value=False, comment='Should the histogram be rotated?'),
                BoolParameter(
                    name='Show X Label', parameter_name='show_x_label', default_value=True, comment=''),
                BoolParameter(
                    name='Show Y Label', parameter_name='show_y_label', default_value=True, comment=''),
                NoneRangeParameter(name='X Limit', parameter_name='x_lim', default_min_value=None,
                                   default_max_value=None, limits=(None, None), comment=''),
                NoneRangeParameter(name='Y Limit', parameter_name='y_lim', default_min_value=None,
                                   default_max_value=None, limits=(None, None), comment=''),
                ]

    @staticmethod
    def qt_to_settings(qt_settings: dict) -> dict:
        settings = qt_settings.copy()
        parameter_value = settings.pop('parameter', None)
        settings['parameter'] = parameter_value
        settings['scale'] = (settings['scale'], 'Linear')
        return settings

    def get_dynamic_plot_requests(self):
        return [{'method': 'get_clustering', 'settings': {'parameters': self.parameters, 'scale': self.scale, 'clustering_settings': self.clustering_settings}},
                {'method': 'calculate_node_values', 'settings': {'parameters': (self.parameter, 'Type'), 'scale': self.scale}}]

    def plot(self, ax: plt.Axes, group_number: int,  axis_limits: dict):
        if len(self.parameters) != 1:
            raise ValueError('Histogram expects only one parameter')

        clustering: Clustering = self.interface.dynamic_data_cache[group_number][self.get_dynamic_plot_requests()[
            0]]
        labels = clustering.cluster_labels

        data = self.interface.dynamic_data_cache[group_number][self.get_dynamic_plot_requests()[
            1]]

        self.parameter = self.parameters[0]
        values = np.array(data[self.parameter])

        # Now get the limits and set default values if necessary
        x_lim, y_lim = self.get_limits(axis_limits)

        colors = self.color_scheme.fill_colors(
            clusters=labels, group_number=group_number, **self.fill_color_settings)

        cluster_labels = clustering.cluster_labels

        # Iterate through each cluster and plot its histogram
        for i, (cluster, color) in enumerate(zip(cluster_labels, colors)):
            cluster_values = values[np.where(clustering.cluster_indexes == i)]

            if self.rotated:
                sns.histplot(y=cluster_values, kde=False, ax=ax, element="step", fill=False,
                             stat="density", label=f'Cluster {cluster}', color=color)
            else:
                sns.histplot(cluster_values, kde=False, ax=ax, element="step", fill=False,
                             stat="density", label=f'Cluster {cluster}', color=color)

        ax.legend()

        Plot.tanh_axis_labels(ax=ax, scale=self.scale)

        if x_lim is not None:
            ax.set_xlim(*x_lim)
        if y_lim is not None:
            ax.set_ylim(*y_lim)

        if self.show_x_label:
            ax.set_xlabel(Plotter._parameter_dict.get(
                self.parameter, self.parameter))
        if self.show_y_label:
            ax.set_ylabel('Density')


@Plotter.plot_type("Static: Time line")
class plot_time_line(Plot):
    def __init__(self, interface: Interface, color_scheme: ColorScheme, parameters: tuple[str],
                 scale: Optional[Tuple[str] | None] = None,
                 show_x_label: bool = True, show_y_label: bool = True,
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None,
                 linestyle: str | None = None, color: str | None = None):
        self.interface: Interface = interface
        self.parameters = tuple(parameters)
        self.color_scheme = color_scheme
        self.scale = tuple(scale or ('Linear', 'Linear'))
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self._x_lim = x_lim
        self._y_lim = y_lim

        def time_color_to_time_color_settings(time_color) -> dict:
            if isinstance(time_color, dict):
                # check if only 'mode' and 'settings' in dict
                if not all(key in {'mode', 'settings'} for key in time_color.keys()):
                    raise ValueError(
                        'Time color is incorrectly formatted')
                return time_color
            if isinstance(time_color, (str, float)):
                return {'mode': 'Constant Color', 'settings': {'color': time_color}}
            else:
                return {'mode': 'Constant Color'}

        self.time_color_settings: dict = time_color_to_time_color_settings(
            color)

        if self.time_color_settings['mode'] not in self.color_scheme.method_logger['Timeline Color']['modes']:
            raise ValueError(
                f"Time color is incorrectly formatted: Mode {self.time_color_settings['mode']} not in known modes {self.color_scheme.method_logger['Timeline Color']['modes']}")

        def time_style_to_time_style_settings(time_style) -> dict:
            if isinstance(time_style, dict):
                # check if only 'mode' and 'settings' in dict
                if not all(key in {'mode', 'settings'} for key in time_style.keys()):
                    raise ValueError(
                        'Time style is incorrectly formatted')
                return time_style
            if isinstance(time_style, (str, float)):
                return {'mode': 'Constant Style', 'settings': {'style': time_style}}
            else:
                return {'mode': 'Constant Style'}

        self.time_style_settings: dict = time_style_to_time_style_settings(
            linestyle)

        if self.time_style_settings['mode'] not in self.color_scheme.method_logger['Timeline Style']['modes']:
            raise ValueError(
                f"Time style is incorrectly formatted: Mode {self.time_style_settings['mode']} not in known modes {self.color_scheme.method_logger['Timeline Style']['modes']}")

    def get_dynamic_plot_requests(self):
        return [{'method': 'mean_time', 'settings': {}}]

    def plot(self, ax: plt.Axes, group_number: int,  axis_limits: dict):
        x_lim, y_lim = self.get_limits(axis_limits)

        data = self.interface.dynamic_data_cache[group_number][self.get_dynamic_plot_requests()[
            0]]
        time_range = self.interface.group_time_range()

        color = self.color_scheme.timeline_color(
            group_number=group_number, **self.time_color_settings)
        linestyle = self.color_scheme.timeline_linestyle(
            group_number=group_number, **self.time_style_settings)

        x_parameter, y_parameter = self.parameters

        if x_parameter == 'Time':
            # Time is on the x-axis, draw a vertical line
            ax.axvline(x=data, color=color, linestyle=linestyle)
            ax.set_xlim(time_range)
        elif y_parameter == 'Time':
            # Time is on the y-axis, draw a horizontal line
            ax.axhline(y=data, color=color, linestyle=linestyle)
            ax.set_ylim(time_range)

        if self.show_x_label:
            ax.set_xlabel(Plotter._parameter_dict.get(
                x_parameter, x_parameter))
        if self.show_y_label:
            ax.set_ylabel(Plotter._parameter_dict.get(
                y_parameter, y_parameter))


@Plotter.plot_type("Static: Node lines")
class plot_node_lines(Plot):
    def __init__(self, interface: Interface, color_scheme: ColorScheme, parameters: tuple[str],
                 scale: Optional[Tuple[str] | None] = None,
                 exclude_types: Tuple[str] = (),
                 show_x_label: bool = True, show_y_label: bool = True,
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None,
                 color: dict | None = None, linestyle: str | None = None, ):
        self.interface: Interface = interface
        self.parameters = tuple(parameters)
        self.color_scheme = color_scheme
        self.scale = tuple(scale or ('Linear', 'Linear'))
        self.exclude_types: Tuple[str] = exclude_types
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self._x_lim = x_lim
        self._y_lim = y_lim

        def line_color_to_line_color_settings(line_color) -> dict:
            if isinstance(line_color, dict):
                # check if only 'mode' and 'settings' in dict
                if not all(key in {'mode', 'settings'} for key in line_color.keys()):
                    raise ValueError(
                        'Line color is incorrectly formatted')
                return line_color
            if isinstance(line_color, (str, float)):
                return {'mode': 'Constant Color', 'settings': {'color': line_color}}
            else:
                return {'mode': 'Constant Color'}

        self.line_color_settings: dict = line_color_to_line_color_settings(
            color)

        if self.line_color_settings['mode'] not in self.color_scheme.method_logger['Line Color']['modes']:
            raise ValueError(
                f"Line color is incorrectly formatted: Mode {self.line_color_settings['mode']} not in known modes {self.color_scheme.method_logger['Line Color']['modes']}")

        def line_style_to_line_style_settings(line_style) -> dict:
            if isinstance(line_style, dict):
                # check if only 'mode' and 'settings' in dict
                if not all(key in {'mode', 'settings'} for key in line_style.keys()):
                    raise ValueError(
                        'Line style is incorrectly formatted')
                return line_style
            if isinstance(line_style, (str, float)):
                return {'mode': 'Constant Style', 'settings': {'style': line_style}}
            else:
                return {'mode': 'Constant Style'}

        self.line_style_settings: dict = line_style_to_line_style_settings(
            linestyle)

        if self.line_style_settings['mode'] not in self.color_scheme.method_logger['Line Style']['modes']:
            raise ValueError(
                f"Line style is incorrectly formatted: Mode {self.line_style_settings['mode']} not in known modes {self.color_scheme.method_logger['Line Style']['modes']}")

        self._static_data = None

    def get_track_clusterings_requests(self):
        return [self.color_scheme.requires_tracking(self.line_style_settings), self.color_scheme.requires_tracking(self.line_color_settings)]

    def get_static_plot_requests(self):
        parameters = self.parameters + ('Type',)
        return [{'method': 'calculate_node_values', 'settings': {'parameters': parameters, 'scale': self.scale}}]

    def data(self):
        if self._static_data is not None:
            return self._static_data

        data_list = self.interface.static_data_cache[self.get_static_plot_requests()[
            0]]
        data = self.transform_data(data_list)
        mask = ~np.isin(data['Type'], self.exclude_types)
        flat_mask = np.any(mask, axis=-1)
        masked_data = {key: value[flat_mask] for key,
                       value in data.items() if key not in ('Time', 'Nodes')}
        masked_data['Time'] = data['Time']
        masked_data['Nodes'] = [node for node,
                                m in zip(data['Nodes'], flat_mask) if m]

        self._static_data = masked_data
        return self._static_data

    def plot(self, ax: plt.Axes, group_number: int,  axis_limits: dict):

        x_lim, y_lim = self.get_limits(axis_limits)

        data = self.data()
        nodes = data['Nodes']

        colors = np.array(self.color_scheme.line_color(nodes=nodes,
                                                       group_number=group_number, **self.line_color_settings))
        linestyle = self.color_scheme.line_linestyle(
            group_number=group_number, **self.line_style_settings)

        x_parameter, y_parameter = self.parameters

        if not (x_parameter == 'Time' or y_parameter == 'Time'):
            raise ValueError('One of the parameters should be Time.')

        # Determine which axis time is on
        time_is_x_axis = x_parameter == 'Time'

        x_values = data[x_parameter]
        y_values = data[y_parameter]

        if self.scale[0 if time_is_x_axis else 1] == 'Tanh':
            x_values = np.tanh(x_values)
        if self.scale[1 if time_is_x_axis else 0] == 'Tanh':
            y_values = np.tanh(y_values)

        # Plotting
        if Plot.is_single_color(colors):
            for node_values in y_values:
                ax.plot(x_values, node_values,
                        color=colors, linestyle=linestyle)
        else:
            for node_values, color in zip(y_values, colors):
                if time_is_x_axis:
                    ax.plot(x_values, node_values,
                            color=color, linestyle=linestyle)
                else:
                    ax.plot(node_values, x_values,
                            color=color, linestyle=linestyle)

        # Set limits
        if x_lim is not None:
            ax.set_xlim(*x_lim)
        if y_lim is not None:
            ax.set_ylim(*y_lim)

        Plot.tanh_axis_labels(ax=ax, scale=self.scale)

        if self.show_x_label:
            ax.set_xlabel(Plotter._parameter_dict.get(
                self.parameters[0], self.parameters[0]))
        if self.show_y_label:
            ax.set_ylabel(Plotter._parameter_dict.get(
                self.parameters[1], self.parameters[1]))


@Plotter.plot_type("Static: Graph line")
class plot_graph_line(Plot):
    def __init__(self, interface: Interface, color_scheme: ColorScheme, parameters: tuple[str],
                 scale: Optional[Tuple[str] | None] = None,
                 show_x_label: bool = True, show_y_label: bool = True,
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None, function: str = 'Mean',
                 color: dict | None = None, linestyle: str | None = None, ):
        '''
        Graph parameters functions list can be accessed in Group.common_functions.keys()
        '''
        self.interface: Interface = interface
        self.parameters = tuple(parameters)
        self.color_scheme = color_scheme
        self.scale = tuple(scale or ('Linear', 'Linear'))
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self._x_lim = x_lim
        self._y_lim = y_lim
        self.function = function

        self._static_data = None

        def line_color_to_line_color_settings(line_color) -> dict:
            if isinstance(line_color, dict):
                # check if only 'mode' and 'settings' in dict
                if not all(key in {'mode', 'settings'} for key in line_color.keys()):
                    raise ValueError(
                        'Line color is incorrectly formatted')
                return line_color
            if isinstance(line_color, (str, float)):
                return {'mode': 'Constant Color', 'settings': {'color': line_color}}
            else:
                return {'mode': 'Constant Color'}

        self.line_color_settings: dict = line_color_to_line_color_settings(
            color)

        if self.line_color_settings['mode'] not in self.color_scheme.method_logger['Graph Line Color']['modes']:
            raise ValueError(
                f"Line color is incorrectly formatted: Mode {self.line_color_settings['mode']} not in known modes {self.color_scheme.method_logger['Graph Line Color']['modes']}")

        def line_style_to_line_style_settings(line_style) -> dict:
            if isinstance(line_style, dict):
                # check if only 'mode' and 'settings' in dict
                if not all(key in {'mode', 'settings'} for key in line_style.keys()):
                    raise ValueError(
                        'Line style is incorrectly formatted')
                return line_style
            if isinstance(line_style, (str, float)):
                return {'mode': 'Constant Style', 'settings': {'style': line_style}}
            else:
                return {'mode': 'Constant Style'}

        self.line_style_settings: dict = line_style_to_line_style_settings(
            linestyle)

        if self.line_style_settings['mode'] not in self.color_scheme.method_logger['Line Style']['modes']:
            raise ValueError(
                f"Line style is incorrectly formatted: Mode {self.line_style_settings['mode']} not in known modes {self.color_scheme.method_logger['Line Style']['modes']}")

        self._static_data = None

    def get_track_clusterings_requests(self):
        return [self.color_scheme.requires_tracking(self.line_style_settings), self.color_scheme.requires_tracking(self.line_color_settings)]

    def get_static_plot_requests(self):
        return [{'method': 'calculate_function_of_node_values', 'settings': {'parameters': self.parameters, 'scale': self.scale, 'function': self.function}}]

    def data(self):
        if self._static_data is not None:
            return self._static_data

        data = self.interface.static_data_cache[self.get_static_plot_requests()[
            0]]

        keys = list(data[0].keys())

        self._static_data = {key: [] for key in keys}
        for frame in data:
            for key in keys:
                self._static_data[key].append(frame[key])

        return self._static_data

    def plot(self, ax: plt.Axes, group_number: int,  axis_limits: dict):
        x_lim, y_lim = self.get_limits(axis_limits)

        data = self.data()
        colors = np.array(self.color_scheme.graph_line_color(
            group_number=group_number, **self.line_color_settings))
        linestyle = self.color_scheme.line_linestyle(
            group_number=group_number, **self.line_style_settings)

        x_parameter, y_parameter = self.parameters

        x_values = data[x_parameter]
        y_values = data[y_parameter]

        if self.scale[0] == 'Tanh':
            x_values = np.tanh(x_values)
        if self.scale[1] == 'Tanh':
            y_values = np.tanh(y_values)

        ax.plot(x_values, y_values, color=colors, linestyle=linestyle)

        if x_lim is not None:
            ax.set_xlim(*x_lim)
        if y_lim is not None:
            ax.set_ylim(*y_lim)

        Plot.tanh_axis_labels(ax=ax, scale=self.scale)

        if self.show_x_label:
            ax.set_xlabel(Plotter._parameter_dict.get(
                x_parameter, x_parameter))
        if self.show_y_label:
            ax.set_ylabel(Plotter._parameter_dict.get(
                y_parameter, y_parameter))


@Plotter.plot_type("Static: Graph Range")
class plot_fill_between(Plot):
    def __init__(self, interface: Interface, color_scheme: ColorScheme, parameters: tuple[str], functions: Optional[List[str]] = None,
                 scale: Optional[Tuple[str] | None] = None,
                 show_x_label: bool = True, show_y_label: bool = True,
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None,
                 color: dict | None = None):
        self.interface: Interface = interface
        self.parameters = tuple(parameters)
        self.color_scheme = color_scheme
        self.functions = functions or ('Min', 'Max')
        self.scale = tuple(scale or ('Linear', 'Linear'))
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self._x_lim = x_lim
        self._y_lim = y_lim

        self._static_data = None

        def line_color_to_line_color_settings(line_color) -> dict:
            if isinstance(line_color, dict):
                # check if only 'mode' and 'settings' in dict
                if not all(key in {'mode', 'settings'} for key in line_color.keys()):
                    raise ValueError(
                        'Line color is incorrectly formatted')
                return line_color
            if isinstance(line_color, (str, float)):
                return {'mode': 'Constant Color', 'settings': {'color': line_color}}
            else:
                return {'mode': 'Constant Color'}

        self.line_color_settings: dict = line_color_to_line_color_settings(
            color)

        if self.line_color_settings['mode'] not in self.color_scheme.method_logger['Graph Line Color']['modes']:
            raise ValueError(
                f"Line color is incorrectly formatted: Mode {self.line_color_settings['mode']} not in known modes {self.color_scheme.method_logger['Graph Line Color']['modes']}")

    def get_static_plot_requests(self):
        return [
            {'method': 'calculate_function_of_node_values', 'settings': {
                'parameters': self.parameters, 'scale': self.scale, 'function': self.functions[0]}},
            {'method': 'calculate_function_of_node_values', 'settings': {
                'parameters': self.parameters, 'scale': self.scale, 'function': self.functions[1]}}
        ]

    def data(self, function_key):
        data = self.interface.static_data_cache[function_key]

        keys = list(data[0].keys())

        _static_data = {key: [] for key in keys}
        for frame in data:
            for key in keys:
                _static_data[key].append(frame[key])

        return _static_data

    def plot(self, ax: plt.Axes, group_number: int,  axis_limits: dict):
        x_lim, y_lim = self.get_limits(axis_limits)

        request_min, request_max = self.get_static_plot_requests()

        data_min = self.data(request_min)
        data_max = self.data(request_max)

        colors = np.array(self.color_scheme.graph_line_color(
            group_number=group_number, **self.line_color_settings))

        x_parameter, y_parameter = self.parameters

        x_values = data_min[x_parameter]
        y_values_min = data_min[y_parameter]
        y_values_max = data_max[y_parameter]

        if self.scale[0] == 'Tanh':
            x_values = np.tanh(x_values)
        if self.scale[1] == 'Tanh':
            y_values_min = np.tanh(y_values_min)
            y_values_max = np.tanh(y_values_max)

        # Fill the area between the min and max curves
        ax.fill_between(x_values, y_values_min, y_values_max,
                        color=colors)

        if x_lim is not None:
            ax.set_xlim(*x_lim)
        if y_lim is not None:
            ax.set_ylim(*y_lim)

        Plot.tanh_axis_labels(ax=ax, scale=self.scale)

        if self.show_x_label:
            ax.set_xlabel(Plotter._parameter_dict.get(
                x_parameter, x_parameter))
        if self.show_y_label:
            ax.set_ylabel(Plotter._parameter_dict.get(
                y_parameter, y_parameter))


@Plotter.plot_type("Static: Clustering Line")
class plot_clustering_line(Plot):
    def __init__(self, interface: Interface, color_scheme: ColorScheme, parameter: str, clustering_settings: dict = {},
                 scale: Optional[Tuple[str] | None] = None,
                 show_x_label: bool = True, show_y_label: bool = True,
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None,
                 color: dict | None = None, linestyle: str | None = None, show_legend: bool = True):
        self.interface: Interface = interface
        self.parameter = parameter
        self.parameters = ('Time', parameter)
        self.color_scheme = color_scheme
        self.clustering_settings = clustering_settings
        self.scale = tuple(scale or ('Linear', 'Linear'))
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self._x_lim = x_lim
        self._y_lim = y_lim
        self.show_legend = show_legend

        self._static_data = None

        def line_color_to_line_color_settings(line_color) -> dict:
            if isinstance(line_color, dict):
                # check if only 'mode' and 'settings' in dict
                if not all(key in {'mode', 'settings'} for key in line_color.keys()):
                    raise ValueError(
                        'Line color is incorrectly formatted')
                return line_color
            if isinstance(line_color, (str, float)):
                return {'mode': 'Constant Color', 'settings': {'color': line_color}}
            else:
                return {'mode': 'Constant Color'}

        self.line_color_settings: dict = line_color_to_line_color_settings(
            color)

        if self.line_color_settings['mode'] not in self.color_scheme.method_logger['Cluster Line Color']['modes']:
            raise ValueError(
                f"Line color is incorrectly formatted: Mode {self.line_color_settings['mode']} not in known modes {self.color_scheme.method_logger['Cluster Line Color']['modes']}")

        def line_style_to_line_style_settings(line_style) -> dict:
            if isinstance(line_style, dict):
                # check if only 'mode' and 'settings' in dict
                if not all(key in {'mode', 'settings'} for key in line_style.keys()):
                    raise ValueError(
                        'Line style is incorrectly formatted')
                return line_style
            if isinstance(line_style, (str, float)):
                return {'mode': 'Constant Style', 'settings': {'style': line_style}}
            else:
                return {'mode': 'Constant Style'}

        self.line_style_settings: dict = line_style_to_line_style_settings(
            linestyle)

        if self.line_style_settings['mode'] not in self.color_scheme.method_logger['Line Style']['modes']:
            raise ValueError(
                f"Line style is incorrectly formatted: Mode {self.line_style_settings['mode']} not in known modes {self.color_scheme.method_logger['Line Style']['modes']}")

    def get_track_clusterings_requests(self):
        return [self.clustering_settings, self.color_scheme.requires_tracking(self.line_color_settings), self.color_scheme.requires_tracking(self.line_style_settings)]

    def get_static_plot_requests(self):
        return [{'method': 'clustering_graph_values', 'settings': {'parameters': (self.parameter, 'Label', 'Time'), 'scale': self.scale, 'clustering_settings': self.clustering_settings}}]

    def data(self):
        if self._static_data is not None:
            return self._static_data

        data = self.interface.static_data_cache[self.get_static_plot_requests()[
            0]]

        data = self.transform_data(data, transform_parameter='Label')

        x_parameter, y_parameter = self.parameters

        # Transform data to suitable format for plotting
        x_values = np.array(data[x_parameter])
        y_values = np.array(data[y_parameter])

        if self.scale[0] == 'Tanh':
            x_values = np.tanh(x_values)
        if self.scale[1] == 'Tanh':
            y_values = np.tanh(y_values)

        labels = data['Label']

        self._static_data = {'x': x_values,
                             'y': list(y_values), 'labels': labels}
        return self._static_data

    def plot(self, ax: plt.Axes, group_number: int,  axis_limits: dict):
        x_lim, y_lim = self.get_limits(axis_limits)

        data = self.data()
        x = data['x']
        labels = data['labels']

        linestyle = self.color_scheme.line_linestyle(
            clusters=labels, group_number=group_number, **self.line_style_settings)
        colors = np.array(self.color_scheme.cluster_line_colors(
            clusters=labels, group_number=group_number, **self.line_color_settings))

        if Plot.is_single_color(colors):
            for values, label in zip(data['y'], labels):
                if self.show_legend:
                    ax.plot(x, values, label=label,
                            linestyle=linestyle, color=colors)
                else:
                    ax.plot(x, values,
                            linestyle=linestyle, color=colors)
        else:
            for values, label, color in zip(data['y'], labels, colors):
                if self.show_legend:
                    ax.plot(x, values, label=label,
                            linestyle=linestyle, color=color)
                else:
                    ax.plot(x, values,
                            linestyle=linestyle, color=color)

        if x_lim is not None:
            ax.set_xlim(*x_lim)
        if y_lim is not None:
            ax.set_ylim(*y_lim)

        Plot.tanh_axis_labels(ax=ax, scale=self.scale)

        if self.show_x_label:
            ax.set_xlabel(Plotter._parameter_dict.get(
                self.parameters[0], self.parameters[0]))
        if self.show_y_label:
            ax.set_ylabel(Plotter._parameter_dict.get(
                self.parameters[1], self.parameters[1]))

        if self.show_legend:
            ax.legend()


@Plotter.plot_type("Static: Clustering Range")
class plot_fill_between_clustering(Plot):
    def __init__(self, interface: Interface, color_scheme: ColorScheme, parameter: str, range_parameter: str, clustering_settings: dict = {},
                 scale: Optional[Tuple[str] | None] = None,
                 show_x_label: bool = True, show_y_label: bool = True,
                 x_lim: Optional[Sequence[float] | None] = None, y_lim: Optional[Sequence[float] | None] = None,
                 color: dict | None = None, show_legend: bool = True, alpha: float = 0.3):
        self.interface: Interface = interface
        self.parameter = parameter
        self.parameters = ('Time', 'Parameter')
        self.color_scheme = color_scheme
        self.range_parameter = range_parameter
        self.clustering_settings = clustering_settings
        self.scale = tuple(scale or ('Linear', 'Linear'))
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self._x_lim = x_lim
        self._y_lim = y_lim
        self.show_legend = show_legend
        self.alpha = alpha

        self._static_data = None

        def line_color_to_line_color_settings(line_color) -> dict:
            if isinstance(line_color, dict):
                # check if only 'mode' and 'settings' in dict
                if not all(key in {'mode', 'settings'} for key in line_color.keys()):
                    raise ValueError(
                        'Line color is incorrectly formatted')
                return line_color
            if isinstance(line_color, (str, float)):
                return {'mode': 'Constant Color', 'settings': {'color': line_color}}
            else:
                return {'mode': 'Constant Color'}

        self.line_color_settings: dict = line_color_to_line_color_settings(
            color)

        if self.line_color_settings['mode'] not in self.color_scheme.method_logger['Cluster Line Color']['modes']:
            raise ValueError(
                f"Line color is incorrectly formatted: Mode {self.line_color_settings['mode']} not in known modes {self.color_scheme.method_logger['Cluster Line Color']['modes']}")

    def get_track_clusterings_requests(self):
        return [self.clustering_settings, self.color_scheme.requires_tracking(self.line_color_settings)]

    def get_static_plot_requests(self):
        return [{'method': 'clustering_graph_values', 'settings': {'parameters': (self.parameter, self.range_parameter, 'Label', 'Time'), 'scale': self.scale, 'clustering_settings': self.clustering_settings}}]

    def data(self):
        if self._static_data is not None:
            return self._static_data

        data = self.interface.static_data_cache[self.get_static_plot_requests()[
            0]]

        data = self.transform_data(data, transform_parameter='Label')

        # Transform data to suitable format for plotting
        x_values = np.array(data['Time'])
        value = np.array(data[self.parameter])

        labels = data['Label']

        value_delta = np.array(data[self.range_parameter])
        min_value = value-value_delta
        max_value = value+value_delta

        if self.scale[1] == 'Tanh':
            min_value = np.tanh(min_value)
            value = np.tanh(value)
            max_value = np.tanh(max_value)

        self._static_data = {'x': x_values,
                             'y': list(value), 'y_min': list(min_value), 'y_max': list(max_value), 'labels': labels}
        return self._static_data

    def plot(self, ax: plt.Axes, group_number: int,  axis_limits: dict):
        x_lim, y_lim = self.get_limits(axis_limits)

        data = self.data()
        x = data['x']
        labels = data['labels']

        colors = np.array(self.color_scheme.cluster_line_colors(
            clusters=labels, group_number=group_number, **self.line_color_settings))

        if Plot.is_single_color(colors):
            for values, values_min, values_max, label in zip(data['y'], data['y_min'], data['y_max'], labels):
                if self.show_legend:
                    ax.fill_between(x, values_min, values_max, alpha=self.alpha,
                                    label=label, color=colors)
                else:
                    ax.fill_between(x, values_min, values_max,
                                    alpha=self.alpha, color=colors)
        else:
            for values, values_min, values_max, label, color in zip(data['y'], data['y_min'], data['y_max'], labels, colors):
                if self.show_legend:
                    ax.fill_between(x, values_min, values_max, alpha=self.alpha,
                                    label=label, color=color)
                else:
                    ax.fill_between(x, values_min, values_max,
                                    alpha=self.alpha, color=color)

        if x_lim is not None:
            ax.set_xlim(*x_lim)
        if y_lim is not None:
            ax.set_ylim(*y_lim)

        Plot.tanh_axis_labels(ax=ax, scale=self.scale)

        if self.show_x_label:
            ax.set_xlabel(Plotter._parameter_dict.get(
                self.parameters[0], self.parameters[0]))
        if self.show_y_label:
            ax.set_ylabel(Plotter._parameter_dict.get(
                self.parameters[1], self.parameters[1]))

        if self.show_legend:
            ax.legend()
