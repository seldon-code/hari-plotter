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
from matplotlib.colors import Colormap, to_rgba
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

        def copy(self) -> ColorScheme.MethodLogger:
            # Create a new instance of MethodLogger
            new_instance = ColorScheme.MethodLogger()
            new_instance.methods = copy.deepcopy(
                self.methods)  # Deep copy the dictionary
            return new_instance

        def keys(self) -> List[str]:
            return list(self.methods.keys())

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
        self.default_distribution_color = ColorScheme.to_rgba('blue')
        self.default_scatter_color = ColorScheme.to_rgba('blue')
        self.default_none_color = ColorScheme.to_rgba((0, 255, 0))
        self.default_color_map = 'coolwarm'
        self.available_colormaps = plt.colormaps()
        self.available_markers = list(Line2D.markers.keys())
        self.default_markers = ['o', 'X', '4', '5', '6', '7',  '<', '>', '^', 'v', '*', '+',
                                'D', 'H', 'P', 'd', 'h',  'p', 's', '_', '|', ',', '.',]
        self.default_scatter_marker = self.default_markers[0]
        self.default_none_marker = 'x'
        self.default_centroid_marker = '*'
        self.default_centroid_color = ColorScheme.to_rgba('red')

        self.default_timeline_color = ColorScheme.to_rgba('grey')
        self.default_timeline_linestyle = '--'

        self.default_line_color = ColorScheme.to_rgba('black')
        self.default_line_linestyle = '-'

        self._cluster_color_cache = {}
        self._node_color_cache = {}
        self._cluster_marker_cache = {}
        self._node_marker_cache = {}
        self._graph_color_cache = {}

    def clear(self) -> None:
        self._cluster_color_cache = {}
        self._node_color_cache = {}
        self._cluster_marker_cache = {}
        self._node_marker_cache = {}
        self._graph_color_cache = {}

    def get_image(self, settings, group_number: int) -> int:
        image = 'Current' if settings is None else settings.get(
            'group_number', group_number)
        if image == 'Current':
            return group_number
        image = int(image)
        if image < 0:
            image += len(self.interface.groups)
        return image

    @staticmethod
    def request_to_tuple(request) -> tuple:
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
        return list(self.method_logger.keys())

    def requires_tracking(self, settings) -> Union[bool, dict]:
        mode = settings.get('mode', None)
        if mode in ('Cluster Marker', 'Cluster Color', 'Cluster Parameter Color'):
            if 'settings' not in settings or 'clustering_settings' not in settings['settings']:
                raise ValueError(
                    f'{mode} requires tracking, but no clustering settings is provided in settings {settings}')
            return settings['settings']['clustering_settings']
        return False

#############################
# Markers
#############################

    def get_cluster_marker(self, clustering_settings: Dict[str, Any], none_marker: str) -> dict:
        request: Dict[str, Any] = {
            'clustering_settings': clustering_settings, 'none_marker': none_marker}
        request_tuple = ColorScheme.request_to_tuple(request)
        if request_tuple not in self._cluster_marker_cache:
            cluster_names: list[str] = self.interface.cluster_tracker.get_unique_clusters(
                clustering_settings)[0]
            l = len(self.default_markers)
            cluster_markers = {
                cluster_name: self.default_markers[i % l] for i, cluster_name in enumerate(cluster_names)}
            cluster_markers[None] = none_marker
            self._cluster_marker_cache[request_tuple] = cluster_markers
        return self._cluster_marker_cache[request_tuple]

    def get_cluster_node_marker(self, image: int, clustering_settings: Dict[str, Any], none_marker: str) -> dict:
        request: Dict[str, Any] = {
            'image': image, 'clustering_settings': clustering_settings, 'none_marker': none_marker}
        request_tuple = ColorScheme.request_to_tuple(request)

        if request_tuple not in self._node_marker_cache:
            clustering: Clustering = self.interface.groups[request['image']].clustering(
                **(clustering_settings))
            cluster_mapping = clustering.nodes_labels_dict()
            cluster_to_marker_mapping = self.get_cluster_marker(
                clustering_settings=clustering_settings, none_marker=none_marker)
            marker_mapping = {
                node: cluster_to_marker_mapping[cluster] for node, cluster in cluster_mapping.items()}
            self._node_marker_cache[request_tuple] = defaultdict(
                lambda: request['none_marker'], marker_mapping)

        return self._node_marker_cache[request_tuple]

    @method_logger('Scatter Marker', modes=('Constant Marker', 'Cluster Marker'))
    def scatter_markers_nodes(self, nodes: Union[List[Tuple[int]], None] = None, clusters: Union[List[str], None] = None, group_number: Union[int, None] = None,
                              mode: str = None, settings: Union[dict, None] = None) -> Union[str, List[str]]:
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

            data = self.get_cluster_node_marker(
                image=image, clustering_settings=settings['clustering_settings'], none_marker=none_marker)
            return [data[node] for node in nodes]

    @method_logger('Centroid Marker', modes=('Constant Marker', 'Cluster Marker'))
    def centroid_markers(self, nodes: Union[List[Tuple[int]], None] = None, clusters: Union[List[str], None] = None,
                         group_number: Union[int, None] = None,  mode: str = None, settings: Union[dict, None] = None) -> Union[str, List[str]]:
        mode = mode or 'Constant Marker'
        image = self.get_image(settings, group_number)

        if mode == 'Constant Marker':
            if settings is not None and 'Marker' in settings:
                return settings['Marker']
            return self.default_centroid_marker

        elif mode == 'Cluster Marker':
            if settings is None or "clustering_settings" not in settings:
                raise ValueError(
                    f'Settings ({settings}) are not formatted correctly. "clustering_settings" key is expected')
            # marker for nodes that are not in the cluster
            none_marker = settings.get('None Marker', self.default_none_marker)

            data = self.get_cluster_marker(
                clustering_settings=settings['clustering_settings'], none_marker=none_marker)
            return [data[cluster] for cluster in clusters]

    @method_logger('Timeline Style', modes=('Constant Style',))
    def timeline_linestyle(self, nodes: Union[List[Tuple[int]], None] = None, clusters: Union[List[str], None] = None, group_number: Union[int, None] = None,
                           mode: str = None, settings: Union[dict, None] = None) -> Union[str, List[str]]:
        mode = mode or 'Constant Style'
        if mode == 'Constant Style':
            if settings is not None and 'linestyle' in settings:
                return settings['linestyle']
            return self.default_timeline_linestyle

    @method_logger('Line Style', modes=('Constant Style',))
    def line_linestyle(self, nodes: Union[List[Tuple[int]], None] = None, clusters: Union[List[str], None] = None, group_number: Union[int, None] = None,
                       mode: str = None, settings: Union[dict, None] = None) -> Union[str, List[str]]:
        mode = mode or 'Constant Style'
        if mode == 'Constant Style':
            if settings is not None and 'linestyle' in settings:
                return settings['linestyle']
            return self.default_line_linestyle

#############################
# Colors
#############################

    def get_cluster_color(self, clustering_settings: Dict[str, Any], colormap: str, none_color: str) -> dict:
        cluster_request: Dict[str, Any] = {
            'clustering_settings': clustering_settings, 'colormap': colormap, 'none_color': none_color}
        cluster_request_tuple = ColorScheme.request_to_tuple(
            cluster_request)
        if cluster_request_tuple not in self._cluster_color_cache:
            cluster_names: list[str] = self.interface.cluster_tracker.get_unique_clusters(
                clustering_settings)[0]
            cm = plt.get_cmap(colormap)
            color_indices = np.linspace(0, 1, len(cluster_names))

            # Map each number to a color in the colormap
            cluster_colors = {cluster_name: cm(
                color_index) for cluster_name, color_index in zip(cluster_names, color_indices)}
            cluster_colors[None] = none_color
            self._cluster_color_cache[cluster_request_tuple] = cluster_colors

        return self._cluster_color_cache[cluster_request_tuple]

    def get_cluster_node_color(self, image: int, clustering_settings: Dict[str, Any], colormap: str, none_color: str) -> dict:
        request: Dict[str, Any] = {'image': image,
                                   'clustering_settings': clustering_settings, 'colormap': colormap, 'none_color': none_color}
        request_tuple = ColorScheme.request_to_tuple(request)
        if request_tuple not in self._node_color_cache:
            clustering: Clustering = self.interface.groups[image].clustering(
                **clustering_settings)
            print(f'colorscheme {id(clustering) = }')
            cluster_mapping = clustering.nodes_labels_dict()
            cluster_to_color_mapping = self.get_cluster_color(
                clustering_settings=clustering_settings, colormap=colormap, none_color=none_color)
            color_mapping = {
                node: cluster_to_color_mapping[cluster] for node, cluster in cluster_mapping.items()}
            self._node_color_cache[request_tuple] = defaultdict(
                lambda: none_color, color_mapping)

        return self._node_color_cache[request_tuple]

    def get_parameter_based_cluster_color(self, parameter: str, clustering_settings: Dict[str, Any], colormap: str, none_color: str) -> dict:
        cluster_request: Dict[str, Any] = {'parameters': (
            parameter,), 'clustering_settings': clustering_settings, 'colormap': colormap, 'none_color': none_color}
        cluster_request_tuple = ColorScheme.request_to_tuple(
            cluster_request)
        if cluster_request_tuple not in self._cluster_color_cache:
            cluster_values: Dict[str, float] = self.interface.cluster_tracker.get_final_value(
                clustering_settings, parameter)

            floats = cluster_values.values()
            norm = colors.Normalize(vmin=min(floats), vmax=max(floats))
            cm = plt.get_cmap(colormap)
            cluster_colors = {cluster_name: cm(
                norm(value)) for cluster_name, value in cluster_values.items()}
            cluster_colors[None] = none_color
            self._cluster_color_cache[cluster_request_tuple] = cluster_colors
        return self._cluster_color_cache[cluster_request_tuple]

    def get_parameter_based_cluster_node_color(self, image: int, parameter: str, clustering_settings: Dict[str, Any], colormap: str, none_color: str) -> dict:
        request: Dict[str, Any] = {'image': image,
                                   'clustering_settings': clustering_settings, 'colormap': colormap, 'none_color': none_color}
        request_tuple = ColorScheme.request_to_tuple(request)
        if request_tuple not in self._node_color_cache:
            clustering: Clustering = self.interface.groups[image].clustering(
                **clustering_settings)
            cluster_mapping = clustering.nodes_labels_dict()
            cluster_to_color_mapping = self.get_parameter_based_cluster_color(
                parameter=parameter, clustering_settings=clustering_settings, colormap=colormap, none_color=none_color)
            color_mapping = {
                node: cluster_to_color_mapping[cluster] for node, cluster in cluster_mapping.items()}
            self._node_color_cache[request_tuple] = defaultdict(
                lambda: none_color, color_mapping)
        return self._node_color_cache[request_tuple]

    @method_logger('Scatter Color', modes=('Constant Color', 'Parameter Colormap', 'Cluster Color', 'Cluster Parameter Color'))
    def scatter_colors_nodes(self, nodes: Union[List[Tuple[int]], None] = None, clusters: Union[List[str], None] = None, group_number: Union[int, None] = None,
                             mode: str = None, settings: Union[dict, None] = None) -> Union[str, list, List[list]]:
        mode = mode or 'Constant Color'
        image = self.get_image(settings, group_number)

        if mode == 'Constant Color':
            if settings is not None and 'Color' in settings:
                return ColorScheme.to_rgba(settings['Color'])
            return self.default_scatter_color

        elif mode == 'Parameter Colormap':
            if settings is None or 'parameter' not in settings:
                raise ValueError(
                    f'Settings ({settings}) are not formatted correctly. "parameter" key is expected')
            request_settings = {'parameters': (settings['parameter'],)}
            request_settings['scale'] = settings.get('scale', 'Linear')
            colormap = settings.get('colormap', self.default_color_map)
            none_color = ColorScheme.to_rgba(
                settings.get('None Color', self.default_none_color))
            request = {**request_settings,
                       'colormap': colormap, 'image': image, 'none_color': none_color}
            request_tuple = ColorScheme.request_to_tuple(request)

            if request_tuple not in self._node_color_cache:
                data = self.interface.dynamic_data_cache[image][{
                    'method': 'calculate_node_values', 'settings': request_settings}]
                floats = data[settings['parameter']]
                norm = colors.Normalize(vmin=min(floats), vmax=max(floats))
                cm = plt.get_cmap(colormap)
                self._node_color_cache[request_tuple] = defaultdict(lambda: none_color, {node: cm(
                    norm(value)) for node, value in zip(data['Nodes'], floats)})

            data = self._node_color_cache[request_tuple]
            return [data[node] for node in nodes]

        elif mode == 'Cluster Color':
            if settings is None or "clustering_settings" not in settings:
                raise ValueError(
                    f'Settings ({settings}) are not formatted correctly. "clustering_settings" key is expected')
            colormap = settings.get('colormap', self.default_color_map)
            # color for nodes that are not in the cluster
            none_color = ColorScheme.to_rgba(
                settings.get('None Color', self.default_none_color))

            data = self.get_cluster_node_color(
                image=image, clustering_settings=settings['clustering_settings'], colormap=colormap, none_color=none_color)
            return [data[node] for node in nodes]

        elif mode == 'Cluster Parameter Color':
            if settings is None or "clustering_settings" not in settings:
                raise ValueError(
                    f'Settings ({settings}) are not formatted correctly. "clustering_settings" key is expected')
            if settings is None or 'parameter' not in settings:
                raise ValueError(
                    f'Settings ({settings}) are not formatted correctly. "parameter" key is expected')
            # request_settings['scale'] = settings.get('scale', 'Linear')
            colormap = settings.get('colormap', self.default_color_map)
            # color for nodes that are not in the cluster
            none_color = ColorScheme.to_rgba(
                settings.get('None Color', self.default_none_color))

            data = self.get_parameter_based_cluster_node_color(
                image=image, parameter=settings['parameter'], clustering_settings=settings['clustering_settings'], colormap=colormap, none_color=none_color)
            return [data[node] for node in nodes]

    @method_logger('Centroid Color', modes=('Constant Color', 'Cluster Color', 'Cluster Parameter Color'))
    def centroid_colors(self, nodes: Union[List[Tuple[int]], None] = None, clusters: Union[List[str], None] = None, group_number: Union[int, None] = None,
                        mode: str = None, settings: Union[dict, None] = None) -> Union[str, list, List[list]]:
        mode = mode or 'Constant Color'
        image = self.get_image(settings, group_number)

        if mode == 'Constant Color':
            if settings is not None and 'Color' in settings:
                return ColorScheme.to_rgba(settings['Color'])
            return self.default_centroid_color

        elif mode == 'Cluster Color':
            if settings is None or "clustering_settings" not in settings:
                raise ValueError(
                    f'Settings ({settings}) are not formatted correctly. "clustering_settings" key is expected')
            colormap = settings.get('colormap', self.default_color_map)
            # color for nodes that are not in the cluster
            none_color = ColorScheme.to_rgba(
                settings.get('None Color', self.default_none_color))

            data = self.get_cluster_color(
                clustering_settings=settings['clustering_settings'], colormap=colormap, none_color=none_color)
            return [data[cluster] for cluster in clusters]

        elif mode == 'Cluster Parameter Color':
            if settings is None or "clustering_settings" not in settings:
                raise ValueError(
                    f'Settings ({settings}) are not formatted correctly. "clustering_settings" key is expected')
            if settings is None or 'parameter' not in settings:
                raise ValueError(
                    f'Settings ({settings}) are not formatted correctly. "parameter" key is expected')
            # request_settings['scale'] = settings.get('scale', 'Linear')
            colormap = settings.get('colormap', self.default_color_map)
            # color for nodes that are not in the cluster
            none_color = ColorScheme.to_rgba(
                settings.get('None Color', self.default_none_color))

            data = self.get_parameter_based_cluster_color(parameter=settings['parameter'],
                                                          clustering_settings=settings['clustering_settings'], colormap=colormap, none_color=none_color)
            return [data[cluster] for cluster in clusters]

    @method_logger('Fill Color', modes=('Cluster Color', 'Cluster Parameter Color'))
    def fill_colors(self, nodes: Union[List[Tuple[int]], None] = None, clusters: Union[List[str], None] = None, group_number: Union[int, None] = None,
                    mode: str = None, settings: Union[dict, None] = None) -> Union[str, list, List[list]]:
        mode = mode or 'Cluster Color'
        image = self.get_image(settings, group_number)

        if mode == 'Cluster Color':
            if settings is None or "clustering_settings" not in settings:
                raise ValueError(
                    f'Settings ({settings}) are not formatted correctly. "clustering_settings" key is expected')
            colormap = settings.get('colormap', self.default_color_map)
            # color for nodes that are not in the cluster
            none_color = ColorScheme.to_rgba(
                settings.get('None Color', self.default_none_color))

            data = self.get_cluster_color(
                clustering_settings=settings['clustering_settings'], colormap=colormap, none_color=none_color)
            return [data[cluster] for cluster in clusters]

        elif mode == 'Cluster Parameter Color':
            if settings is None or "clustering_settings" not in settings:
                raise ValueError(
                    f'Settings ({settings}) are not formatted correctly. "clustering_settings" key is expected')
            if settings is None or 'parameter' not in settings:
                raise ValueError(
                    f'Settings ({settings}) are not formatted correctly. "parameter" key is expected')
            # request_settings['scale'] = settings.get('scale', 'Linear')
            colormap = settings.get('colormap', self.default_color_map)
            # color for nodes that are not in the cluster
            none_color = ColorScheme.to_rgba(
                settings.get('None Color', self.default_none_color))

            data = self.get_parameter_based_cluster_color(parameter=settings['parameter'],
                                                          clustering_settings=settings['clustering_settings'], colormap=colormap, none_color=none_color)
            return [data[cluster] for cluster in clusters]

    @method_logger('Distribution Color', modes=('Constant Color',))
    def distribution_color(self, nodes: Union[List[Tuple[int]], None] = None, clusters: Union[List[str], None] = None, group_number: Union[int, None] = None,
                           mode: str = None, settings: Union[dict, None] = None) -> Union[str, List[str]]:
        mode = mode or 'Constant Color'
        if mode == 'Constant Color':
            if settings is not None and 'Color' in settings:
                return ColorScheme.to_rgba(settings['Color'])
            return self.default_distribution_color

    @method_logger('Timeline Color', modes=('Constant Color',))
    def timeline_color(self, nodes: Union[List[Tuple[int]], None] = None, clusters: Union[List[str], None] = None, group_number: Union[int, None] = None,
                       mode: str = None, settings: Union[dict, None] = None) -> Union[str, List[str]]:
        mode = mode or 'Constant Color'
        if mode == 'Constant Color':
            if settings is not None and 'Color' in settings:
                return ColorScheme.to_rgba(settings['Color'])
            return self.default_timeline_color

    @method_logger('Line Color', modes=('Constant Color', 'Parameter Colormap', 'Cluster Color', 'Cluster Parameter Color'))
    def line_color(self, nodes: Union[List[Tuple[int]], None] = None, clusters: Union[List[str], None] = None, group_number: Union[int, None] = None,
                   mode: str = None, settings: Union[dict, None] = None) -> Union[str, List[str]]:
        mode = mode or 'Constant Color'
        image = self.get_image(settings, group_number)
        print(f'{image = }')
        if mode == 'Constant Color':
            if settings is not None and 'Color' in settings:
                return ColorScheme.to_rgba(settings['Color'])
            return self.default_line_color
        elif mode == 'Parameter Colormap':
            if settings is None or 'parameter' not in settings:
                raise ValueError(
                    f'Settings ({settings}) are not formatted correctly. "parameter" key is expected')
            request_settings = {'parameters': (settings['parameter'],)}
            request_settings['scale'] = settings.get('scale', 'Linear')
            colormap = settings.get('colormap', self.default_color_map)
            none_color = ColorScheme.to_rgba(
                settings.get('None Color', self.default_none_color))
            request = {**request_settings,
                       'colormap': colormap, 'image': image, 'none_color': none_color}
            request_tuple = ColorScheme.request_to_tuple(request)

            if request_tuple not in self._node_color_cache:
                data = self.interface.dynamic_data_cache[image][{
                    'method': 'calculate_node_values', 'settings': request_settings}]
                floats = data[settings['parameter']]
                norm = colors.Normalize(vmin=min(floats), vmax=max(floats))
                cm = plt.get_cmap(colormap)
                self._node_color_cache[request_tuple] = defaultdict(lambda: none_color, {node: cm(
                    norm(value)) for node, value in zip(data['Nodes'], floats)})

            data = self._node_color_cache[request_tuple]
            return [data[node] for node in nodes]

        elif mode == 'Cluster Color':
            if settings is None or "clustering_settings" not in settings:
                raise ValueError(
                    f'Settings ({settings}) are not formatted correctly. "clustering_settings" key is expected')
            colormap = settings.get('colormap', self.default_color_map)
            # color for nodes that are not in the cluster
            none_color = ColorScheme.to_rgba(
                settings.get('None Color', self.default_none_color))

            data = self.get_cluster_node_color(
                image=image, clustering_settings=settings['clustering_settings'], colormap=colormap, none_color=none_color)
            return [data[node] for node in nodes]

        elif mode == 'Cluster Parameter Color':
            if settings is None or "clustering_settings" not in settings:
                raise ValueError(
                    f'Settings ({settings}) are not formatted correctly. "clustering_settings" key is expected')
            if settings is None or 'parameter' not in settings:
                raise ValueError(
                    f'Settings ({settings}) are not formatted correctly. "parameter" key is expected')
            # request_settings['scale'] = settings.get('scale', 'Linear')
            colormap = settings.get('colormap', self.default_color_map)
            # color for nodes that are not in the cluster
            none_color = ColorScheme.to_rgba(
                settings.get('None Color', self.default_none_color))

            data = self.get_parameter_based_cluster_node_color(
                image=image, parameter=settings['parameter'], clustering_settings=settings['clustering_settings'], colormap=colormap, none_color=none_color)
            return [data[node] for node in nodes]

    @method_logger('Graph Line Color', modes=('Constant Color', 'Graph Parameter'))
    def graph_line_color(self, nodes: Union[List[Tuple[int]], None] = None, clusters: Union[List[str], None] = None, group_number: Union[int, None] = None,
                         mode: str = None, settings: Union[dict, None] = None) -> Union[str, List[str]]:
        '''
        Graph parameters functions list can be accessed in Group.common_functions.keys()
        '''
        mode = mode or 'Constant Color'
        image = self.get_image(settings, group_number)
        if mode == 'Constant Color':
            if settings is not None and 'Color' in settings:
                return ColorScheme.to_rgba(settings['Color'])
            return self.default_line_color
        elif mode == 'Graph Parameter':
            if settings is None or 'parameter' not in settings:
                raise ValueError(
                    f'Settings ({settings}) are not formatted correctly. "parameter" key is expected')
            colormap = settings.get('colormap', self.default_color_map)
            none_color = ColorScheme.to_rgba(
                settings.get('None Color', self.default_none_color))
            value_limits = settings.get('value_limits', (0., 1.))
            request_settings = {'parameters': (
                settings['parameter'],), }
            request_settings['function'] = settings.get('function', 'Mean')
            request = {**request_settings,
                       'colormap': colormap, 'image': image, 'none_color': none_color, 'value_limits': value_limits}
            request_tuple = ColorScheme.request_to_tuple(request)

            if request_tuple not in self._graph_color_cache:
                data = self.interface.dynamic_data_cache[image][{
                    'method': 'calculate_function_of_node_values', 'settings': request_settings}]
                value = data[settings['parameter']]
                print(f'{value = }')
                norm = colors.Normalize(
                    vmin=value_limits[0], vmax=value_limits[1])
                cm = plt.get_cmap(colormap)
                self._graph_color_cache[request_tuple] = cm(norm(value))

            data = self._graph_color_cache[request_tuple]
            return data

    @method_logger('Cluster Line Color', modes=('Constant Color', 'Cluster Color', 'Cluster Parameter Color'))
    def cluster_line_colors(self, nodes: Union[List[Tuple[int]], None] = None, clusters: Union[List[str], None] = None, group_number: Union[int, None] = None,
                            mode: str = None, settings: Union[dict, None] = None) -> Union[str, list, List[list]]:
        mode = mode or 'Constant Color'
        image = self.get_image(settings, group_number)

        if mode == 'Constant Color':
            if settings is not None and 'Color' in settings:
                return ColorScheme.to_rgba(settings['Color'])
            return self.default_line_color

        elif mode == 'Cluster Color':
            if settings is None or "clustering_settings" not in settings:
                raise ValueError(
                    f'Settings ({settings}) are not formatted correctly. "clustering_settings" key is expected')
            colormap = settings.get('colormap', self.default_color_map)
            # color for nodes that are not in the cluster
            none_color = ColorScheme.to_rgba(
                settings.get('None Color', self.default_none_color))

            data = self.get_cluster_color(
                clustering_settings=settings['clustering_settings'], colormap=colormap, none_color=none_color)
            return [data[cluster] for cluster in clusters]

        elif mode == 'Cluster Parameter Color':
            if settings is None or "clustering_settings" not in settings:
                raise ValueError(
                    f'Settings ({settings}) are not formatted correctly. "clustering_settings" key is expected')
            if settings is None or 'parameter' not in settings:
                raise ValueError(
                    f'Settings ({settings}) are not formatted correctly. "parameter" key is expected')
            # request_settings['scale'] = settings.get('scale', 'Linear')
            colormap = settings.get('colormap', self.default_color_map)
            # color for nodes that are not in the cluster
            none_color = ColorScheme.to_rgba(
                settings.get('None Color', self.default_none_color))

            data = self.get_parameter_based_cluster_color(parameter=settings['parameter'],
                                                          clustering_settings=settings['clustering_settings'], colormap=colormap, none_color=none_color)
            return [data[cluster] for cluster in clusters]

    @method_logger('Color Map', modes=('Independent Colormap',))
    def colorbar(self, nodes: Union[List[Tuple[int]], None] = None, clusters: Union[List[str], None] = None, group_number: Union[int, None] = None,
                 mode: str = None, settings: Union[Dict[str, Any], None] = None) -> Colormap:
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

    @staticmethod
    def to_rgba(color):
        print(f'{color = }')
        if isinstance(color, str):
            if color == '':
                rgba = (0, 0, 0, 0)
            else:
                # Convert string colors (names or hex) to RGBA using matplotlib
                rgba = to_rgba(color)
        elif isinstance(color, (list, tuple)):
            if len(color) == 3:
                # Assuming RGB, append alpha value 1.0 to make it RGBA
                rgba = tuple(color) + (1.0,)
            elif len(color) == 4:
                rgba = tuple(color)
            else:
                raise ValueError(
                    "Array color must be length 3 (RGB) or 4 (RGBA)")
        elif isinstance(color, float) or isinstance(color, int):
            # Assuming grayscale value, replicate it across R, G, B; set alpha to 1.0
            if 0 <= color <= 1:
                rgba = (color, color, color, 1.0)
            else:
                raise ValueError("Float color must be in the range [0, 1]")
        else:
            raise TypeError("Color must be a string, an array, or a float")

        # Ensure all values are in the [0, 1] range
        rgba_normalized = tuple(max(0, min(1, c)) for c in rgba)
        return rgba_normalized
