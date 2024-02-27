from __future__ import annotations

import copy
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

from .cluster import Clustering
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
        self.default_distribution_color = 'blue'
        self.default_scatter_color = 'blue'
        self.default_none_color = 'green'
        self.default_color_map = 'coolwarm'
        self.available_colormaps = plt.colormaps()
        self.available_markers = list(Line2D.markers.keys())
        self.default_markers = ['o', 'X', '4', '5', '6', '7',  '<', '>', '^', 'v', '*', '+',
                                'D', 'H', 'P', 'd', 'h',  'p', 's', '_', '|', ',', '.',]
        self.default_scatter_marker = self.default_markers[0]
        self.default_none_marker = 'x'

        self._cluster_color_cache = {}
        self._node_color_cache = {}
        self._cluster_marker_cache = {}
        self._node_marker_cache = {}

    def clear(self):
        self._cluster_color_cache = {}
        self._node_color_cache = {}
        self._cluster_marker_cache = {}
        self._node_marker_cache = {}

    def get_image(self, settings, group_number: int):
        image = 'Current' if settings is None else settings.get(
            'Image', group_number)
        if image == 'Current':
            return group_number
        image = int(image)
        if image < 0:
            image += len(self.interface.groups)
        return image

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

    @property
    def methods(self) -> list:
        """Returns a list of property names that have been registered."""
        return list(self._methods.keys())

    @method_logger('Distribution Color', modes=('Constant Color',))
    def distribution_color(self, nodes, group_number: int, mode: str = None, settings: Union[dict, None] = None):
        mode = mode or 'Constant Color'
        if mode == 'Constant Color':
            if settings is not None and 'Color' in settings:
                return settings['Color']
            return self.default_distribution_color

    @method_logger('Scatter Marker', modes=('Constant Marker', 'Cluster Marker'))
    def scatter_markers_nodes(self, nodes, group_number: int,  mode: str = None, settings: Union[dict, None] = None) -> Union[str, List[str]]:
        mode = mode or 'Constant Marker'
        image = self.get_image(settings, group_number)
        if mode == 'Constant Marker':
            if settings is not None and 'Marker' in settings:
                return settings['Marker']
            return self.default_scatter_marker
        elif mode == 'Cluster Marker':
            if settings is None or "clustering_settings" not in settings:
                raise ValueError(
                    f'Settings ({settings}) are not formatted correctly. "clustering_settings" key is expected')
            # marker for nodes that are not in the cluster
            none_marker = settings.get('None Marker', self.default_none_marker)
            cluster_request: Dict[str, Any] = {
                'clustering_settings': settings['clustering_settings'], 'none_marker': none_marker}
            cluster_request_tuple = ColorScheme.request_to_tuple(
                cluster_request)
            if cluster_request_tuple not in self._cluster_marker_cache:
                cluster_names: list[str] = self.interface.cluster_tracker.get_unique_clusters(
                    settings['clustering_settings'])[0]
                l = len(self.default_markers)
                cluster_markers = {
                    cluster_name: self.default_markers[i % l] for i, cluster_name in enumerate(cluster_names)}
                cluster_markers[None] = none_marker
                self._cluster_marker_cache[cluster_request_tuple] = cluster_markers

            request: Dict[str, Any] = {'image': image,
                                       'clustering_settings': settings['clustering_settings'], 'none_marker': none_marker}
            request_tuple = ColorScheme.request_to_tuple(request)
            if request_tuple not in self._node_marker_cache:
                clustering: Clustering = self.interface.groups[image].clustering(
                    **settings['clustering_settings'])
                cluster_mapping = clustering.mapping_dict()
                cluster_to_marker_mapping = self._cluster_marker_cache[
                    cluster_request_tuple]
                marker_mapping = {
                    node: cluster_to_marker_mapping[cluster] for node, cluster in cluster_mapping.items()}
                self._node_marker_cache[request_tuple] = defaultdict(
                    lambda: none_marker, marker_mapping)

            return [self._node_marker_cache[request_tuple][node] for node in nodes]

    @method_logger('Scatter Color', modes=('Constant Color', 'State Colormap', 'Cluster Color'))
    def scatter_colors_nodes(self, nodes, group_number: int,  mode: str = None, settings: Union[dict, None] = None) -> Union[str, list, List[list]]:
        mode = mode or 'Constant Color'
        image = self.get_image(settings, group_number)
        if mode == 'Constant Color':
            if settings is not None and 'Color' in settings:
                return settings['Color']
            return self.default_scatter_color
        elif mode == 'State Colormap':
            if settings is None or 'parameter' not in settings:
                raise ValueError(
                    f'Settings ({settings}) are not formatted correctly. "parameter" key is expected')
            request_settings = {'parameters': (settings['parameter'],)}
            request_settings['scale'] = settings.get('scale', 'Linear')
            colormap = settings.get('colormap', self.default_color_map)
            none_color = settings.get('None Marker', self.default_none_color)
            request = {**request_settings,
                       'colormap': colormap, 'image': image, 'none_color': none_color}
            request_tuple = ColorScheme.request_to_tuple(request)
            if request_tuple not in self._node_color_cache:
                self.interface
                data = self.interface.dynamic_data_cache[image][{
                    'method': 'calculate_node_values', 'group_number': group_number, 'settings': request_settings}]
                floats = data[settings['parameter']]
                norm = colors.Normalize(vmin=min(floats), vmax=max(floats))
                cm = plt.get_cmap(colormap)
                self._node_color_cache[request_tuple] = defaultdict(lambda: none_color, {node: cm(
                    norm(value)) for node, value in zip(data['Nodes'], floats)})
            return [self._node_color_cache[request_tuple][node] for node in nodes]
        elif mode == 'Cluster Color':
            if settings is None or "clustering_settings" not in settings:
                raise ValueError(
                    f'Settings ({settings}) are not formatted correctly. "clustering_settings" key is expected')
            colormap = settings.get('colormap', self.default_color_map)
            # color for nodes that are not in the cluster
            none_color = settings.get('None Color', self.default_none_color)
            cluster_request: Dict[str, Any] = {
                'clustering_settings': settings['clustering_settings'], 'colormap': colormap, 'none_color': none_color}
            cluster_request_tuple = ColorScheme.request_to_tuple(
                cluster_request)
            if cluster_request_tuple not in self._cluster_color_cache:
                cluster_names: list[str] = self.interface.cluster_tracker.get_unique_clusters(
                    settings['clustering_settings'])[0]
                cm = plt.get_cmap(colormap)
                color_indices = np.linspace(0, 1, len(cluster_names))

                # Map each number to a color in the colormap
                cluster_colors = {cluster_name: cm(
                    color_index) for cluster_name, color_index in zip(cluster_names, color_indices)}
                cluster_colors[None] = none_color
                self._cluster_color_cache[cluster_request_tuple] = cluster_colors

            request: Dict[str, Any] = {'image': image,
                                       'clustering_settings': settings['clustering_settings'], 'colormap': colormap, 'none_color': none_color}
            request_tuple = ColorScheme.request_to_tuple(request)
            if request_tuple not in self._node_color_cache:
                clustering: Clustering = self.interface.groups[image].clustering(
                    **settings['clustering_settings'])
                cluster_mapping = clustering.mapping_dict()
                cluster_to_color_mapping = self._cluster_color_cache[
                    cluster_request_tuple]
                color_mapping = {
                    node: cluster_to_color_mapping[cluster] for node, cluster in cluster_mapping.items()}
                self._node_color_cache[request_tuple] = defaultdict(
                    lambda: none_color, color_mapping)

            return [self._node_color_cache[request_tuple][node] for node in nodes]

    @method_logger('Color Map', modes=('Independent Colormap',))
    def colorbar(self, nodes, group_number: int,  mode: str = None, settings=None):
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
