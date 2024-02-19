from __future__ import annotations

import copy
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

from .interface import Interface


class ColorScheme:
    class MethodLogger:
        def __init__(self) -> None:
            self.methods = {}

        def __call__(self, method_name, modes):
            """
            Makes the MethodLogger instance callable. It can be used as a decorator to register methods.

            Methods:
            -----------
            property_name : str
                The name of the property that the method provides.
            """
            def decorator(class_method):
                if method_name in self.methods:
                    raise ValueError(
                        f"Property {method_name} is already defined.")

                self.methods[method_name] = {
                    'method': class_method, 'modes': modes}
                return class_method
            return decorator

        def copy(self):
            # Create a new instance of MethodLogger
            new_instance = ColorScheme.MethodLogger()
            new_instance.methods = copy.deepcopy(
                self.methods)  # Deep copy the dictionary
            return new_instance

        def keys(self):
            return self.methods.keys()

        def __getitem__(self, key):
            # Returns None if key is not found
            return self.methods.get(key, None)

        def __setitem__(self, key, value):
            self.methods[key] = value

        def __contains__(self, key):
            return key in self.methods

        def __str__(self):
            return str(self.methods)

        def __len__(self):
            return len(self.methods)

        def __iter__(self):
            return iter(self.methods)

    method_logger = MethodLogger()

    def __init__(self, interface: Interface = None) -> None:
        self.interface = interface
        self.default_scatter_marker = 'o'
        self.default_distribution_color = 'blue'
        self.default_scatter_color = 'blue'
        self.default_color_map = 'coolwarm'
        self.available_colormaps = plt.colormaps()
        self.available_markers = list(Line2D.markers.keys())

    @property
    def methods(self) -> list:
        """Returns a list of property names that have been registered."""
        return list(self._methods.keys())

    @method_logger('Distribution Color', modes=('Constant Color',))
    def distribution_color(self, nodes, mode: str = None, settings=None):
        mode = mode or 'Constant Color'
        if mode == 'Constant Color':
            if settings is not None and 'Color' in settings:
                return settings['Color']
            return self.default_distribution_color

    @method_logger('Scatter Marker', modes=('Constant Marker',))
    def scatter_markers_nodes(self, nodes, mode: str = None, settings=None):
        mode = mode or 'Constant Marker'
        if mode == 'Constant Marker':
            if settings is not None and 'Marker' in settings:
                return settings['Marker']
            return self.default_scatter_marker

    @method_logger('Scatter Color', modes=('Constant Color',))
    def scatter_colors_nodes(self, nodes, mode: str = None, settings=None):
        mode = mode or 'Constant Color'
        if mode == 'Constant Color':
            if settings is not None and 'Color' in settings:
                return settings['Color']
            return self.default_scatter_color

    @method_logger('Color Map', modes=('Independent Colormap',))
    def colorbar(self, nodes, mode: str = None, settings=None):
        mode = mode or 'Independent Colormap'
        if mode == 'Independent Colormap':
            if settings is not None and 'Color Pallet' in settings:
                return plt.get_cmap(settings['Color Pallet'])
            return plt.get_cmap(self.default_color_map)

    def copy(self):
        return ColorScheme(self.interface)

    def apply_changes(self, changes):
        pass

    def variation(self, changes):
        copy = self.copy()
        copy.apply_changes(changes)
        return copy

    def __getitem__(self, key: str):
        return self.method_logger[key]
