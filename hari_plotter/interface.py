from __future__ import annotations

import warnings
from abc import ABC, abstractmethod, abstractproperty
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Type, Union

import networkx as nx
import numpy as np

from .cluster import Clustering
from .group import Group
from .hari_dynamics import HariDynamics
from .hari_graph import HariGraph
from .simulation import Simulation


class Interface(ABC):
    """Abstract base class to define interface behaviors.

    Attributes:
        REQUIRED_TYPE: Expected type for the data attribute.
        available_classes: Dictionary mapping REQUIRED_TYPEs to their corresponding classes.
    """

    REQUIRED_TYPE: Optional[Type[Any]] = None
    available_classes: Dict[Type[Any], Type['Interface']] = {}

    def __init__(self, data: Any, group_length: int = 0):
        """
        Initialize the Interface instance.

        Args:
            data: The underlying data object to which the interface applies.
            group_length[int]: length of groups

        Raises:
            ValueError: If the data is not an instance of REQUIRED_TYPE.
        """
        if not isinstance(data, self.REQUIRED_TYPE):
            raise ValueError(
                f"data must be an instance of {self.REQUIRED_TYPE}")
        self.data = data
        self._group_cache = [None]*group_length
        self.groups = self.GroupIterable(self)
        self.static_data_cache = self.StaticDataCache(self)
        self.dynamic_data_cache = [self.DynamicDataCache(
            self, i) for i in range(len(self.groups))]
        self.cluster_tracker = self.ClusterTracker(self)

        self._nodes = None
        self._node_parameters = None

    @property
    def node_parameters(self):
        if not self._node_parameters:
            self._node_parameters = self.groups[0].node_parameters
        return self._node_parameters

    @property
    def nodes(self):
        if self._nodes:
            return self._nodes
        self._nodes = set().union(*[group.nodes for group in self.groups])
        return self._nodes

    class StaticDataCache:
        def __init__(self, interface_instance):
            self._interface = interface_instance
            self.cache = {}

        def __getitem__(self, request):
            method_name = request['method']
            settings = request['settings']
            data_key = Interface.request_to_tuple(request)
            if data_key not in self.cache:
                data_for_all_groups = []
                for group in self._interface.groups:
                    data = getattr(group, method_name)(**settings)
                    data_for_all_groups.append(data)
                self.cache[data_key] = data_for_all_groups
            return self.cache[data_key]

        def clean(self):
            self.cache = {}

    class DynamicDataCache:
        def __init__(self, interface_instance, i: int):
            self._interface = interface_instance
            self.i = i
            self.cache = {}

        def __getitem__(self, request):
            method_name = request['method']
            settings = request['settings']
            data_key = Interface.request_to_tuple(request)
            if data_key not in self.cache:
                self.cache[data_key] = getattr(
                    self._interface.groups[self.i], method_name)(**settings)
            return self.cache[data_key]

        def clean(self):
            self.cache = {}

    def clean_cache(self):
        self.cluster_tracker.clean()
        self.static_data_cache.clean()
        for c in self.dynamic_data_cache:
            c.clean()

    class GroupIterable:
        def __init__(self, interface_instance: Interface):
            self._interface: Interface = interface_instance

        def __getitem__(self, index):
            # Handle slices
            if isinstance(index, slice):
                return [self._get_group(i) for i in range(*index.indices(len(self._interface._group_cache)))]

            # Handle lists of integers
            elif isinstance(index, list):
                # Ensure all elements in the list are integers
                if not all(isinstance(i, int) for i in index):
                    raise TypeError("All indices must be integers")
                return [self._get_group(i) for i in index]

            # Handle single integer
            elif isinstance(index, int):
                return self._get_group(index)

            else:
                raise TypeError(
                    "Invalid index type. Must be an integer, slice, or list of integers.")

        def _get_group(self, i) -> Group:
            # Adjust index for negative values
            if i < 0:
                i += len(self._interface._group_cache)

            if 0 <= i < len(self._interface._group_cache):
                if self._interface._group_cache[i] is None:
                    self._interface._group_cache[i] = self._interface._initialize_group(
                        i)
                return self._interface._group_cache[i]
            else:
                raise IndexError("Group index out of range")

        def __iter__(self):
            for i in range(len(self._interface._group_cache)):
                yield self[i]

        def __len__(self) -> int:
            return len(self._interface._group_cache)

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """Auto-register subclasses in available_classes based on their REQUIRED_TYPE."""
        super().__init_subclass__(**kwargs)
        if cls.REQUIRED_TYPE:
            Interface.available_classes[cls.REQUIRED_TYPE] = cls

    @classmethod
    def create_interface(cls, data: Any) -> Interface:
        """Create an interface for the given data.

        Args:
            data: The underlying data object to which the interface applies.

        Returns:
            Instance of a subclass of Interface based on data's type.

        Raises:
            ValueError: If no matching interface is found for the data type.
        """
        interface_class = cls.available_classes.get(type(data))
        if interface_class:
            return interface_class(data)
        else:
            raise ValueError("Invalid data type, no matching interface found.")

    @abstractmethod
    def _initialize_group(self, i: int) -> Group:
        """Abstract method to define the behavior for data grouping."""
        raise NotImplementedError(
            "This method must be implemented in subclasses")

    @abstractmethod
    def _regroup_dynamics(self, num_intervals: int, interval_size: int = 1):
        """Abstract method to regroup dynamics."""
        raise NotImplementedError(
            "This method must be implemented in subclasses")

    def regroup(self, num_intervals: int, interval_size: int = 1, offset: int = 0):
        self.clean_cache()
        self._group_cache = [None]*num_intervals
        self._regroup_dynamics(num_intervals, interval_size, offset)

    @classmethod
    def info(cls) -> str:
        """Return a string representation of the available classes and their mapping.

        Returns:
            A string detailing the available classes and their mappings.
        """
        mappings = ', '.join(
            [f"{key.__name__} -> {value.__name__}" for key, value in cls.available_classes.items()])
        return f"Available Classes: {mappings}"

    @abstractproperty
    def available_parameters(self) -> list:
        raise NotImplementedError(
            "This method must be implemented in subclasses")

    class ClusterTracker:
        def __init__(self, interface):
            # Initialize with an instance of the enclosing class if needed to access its attributes and methods
            self._interface: Interface = interface
            self._track_clusters_cache = {}

        def clean(self):
            self._track_clusters_cache = {}

        def is_tracked(self, clusterization_settings: Union[dict, List[dict]]) -> List[bool]:
            """
            Tracks the clusters across frames based on the provided clusterization settings.

            :param clusterization_settings: The settings for clusterization, either a single dictionary for all frames or a list of dictionaries for each frame.
            :return: A list of dictionaries where each dictionary maps frame indices to lists of cluster labels.
            """
            clusterization_settings_list = [clusterization_settings] if isinstance(
                clusterization_settings, dict) else clusterization_settings
            is_tracked_list = []
            for clust_settings in clusterization_settings_list:
                # Extract the dynamics of clusters across frames
                cluster_tuple = Interface.request_to_tuple(clust_settings)
                is_tracked_list.append(
                    cluster_tuple in self._track_clusters_cache)
            return is_tracked_list

        def cluster_graph(self, cluster_settings: Dict[str, Any], clusters_dynamics: List[List[Clustering]] = None) -> nx.DiGraph:
            """
            Constructs a graph representing the cluster dynamics, optionally using provided cluster dynamics.

            :param cluster_settings: Settings for clusterization.
            :param clusters_dynamics: A list of lists where each sub-list represents the clusters in a frame.

            :return: A directed graph where nodes represent clusters and edges represent the temporal evolution of clusters.
            """
            clusters_dynamics: List[List[Clustering]] = clusters_dynamics or [group.clustering(
                **cluster_settings).labels_nodes_dict() for group in self._interface.groups]

            # Initialize the final labels from the last frame
            final_labels = self._interface.groups[-1].clustering(**
                                                                 cluster_settings).cluster_labels
            G = nx.DiGraph()

            # Build the graph
            for frame_index, frame in enumerate(clusters_dynamics):
                for cluster_index, cluster in enumerate(frame):
                    node_id = f"{frame_index}-{cluster_index}"
                    G.add_node(node_id, frame=frame_index, cluster=cluster,
                               Label=None)  # Initially, labels are None

                    if frame_index > 0:  # Connect clusters to their predecessors
                        # Simplify connections if the number of clusters remains constant
                        if len(frame) == len(clusters_dynamics[frame_index - 1]):
                            # Direct connection from each cluster to its corresponding cluster in the next frame
                            prev_node_id = f"{frame_index - 1}-{cluster_index}"
                            G.add_edge(prev_node_id, node_id)
                        else:
                            for prev_cluster_index, prev_cluster in enumerate(clusters_dynamics[frame_index - 1]):
                                prev_node_id = f"{frame_index - 1}-{prev_cluster_index}"
                                # Implement logic to determine connections (e.g., based on overlap)
                                overlap = len(
                                    set(cluster).intersection(set(prev_cluster)))
                                if overlap > 0:  # Adjust this threshold as necessary
                                    G.add_edge(prev_node_id, node_id)

            # Label Propagation
            for cluster_index, label in enumerate(final_labels):
                node_id = f"{len(clusters_dynamics)-1}-{cluster_index}"
                # Assign final labels to the last frame's clusters
                G.nodes[node_id]['Label'] = label

            def propagate_labels_with_splits(G, clusters_dynamics):
                for frame_index in range(len(clusters_dynamics) - 1, 0, -1):
                    for cluster_index in range(len(clusters_dynamics[frame_index])):
                        node_id = f"{frame_index}-{cluster_index}"
                        current_label = G.nodes[node_id]['Label']

                        predecessors = list(G.predecessors(node_id))
                        # If the current node has multiple predecessors, it's a split in reverse
                        if len(predecessors) > 1:
                            # Assign new labels to each predecessor node based on the split
                            for i, pred in enumerate(predecessors):
                                # Create new labels for splits as "original.i"
                                new_label = f"{current_label}.{i+1}"
                                G.nodes[pred]['Label'] = new_label
                        # If there's only one predecessor, propagate the label directly
                        elif len(predecessors) == 1:
                            G.nodes[predecessors[0]]['Label'] = current_label

            # Call the function to propagate labels with the handling of splits
            propagate_labels_with_splits(G, clusters_dynamics)

            return G

        def track_clusters(self, clusterization_settings: Union[dict, List[dict]]) -> List[Dict[int, Dict[str:str]]]:
            """
            Tracks the clusters across frames based on the provided clusterization settings.

            :param clusterization_settings: The settings for clusterization, either a single dictionary for all frames or a list of dictionaries for each frame.
            :return: A list of dictionaries where each dictionary maps frame indices to lists of cluster labels.
            """
            clusterization_settings_list = [clusterization_settings] if isinstance(
                clusterization_settings, dict) else clusterization_settings

            updated_labels_list = []
            for clust_settings in clusterization_settings_list:
                # Extract the dynamics of clusters across frames
                cluster_tuple = Interface.request_to_tuple(clust_settings)
                if cluster_tuple not in self._track_clusters_cache:
                    clusters_dynamics = [group.clustering(
                        **clust_settings).labels_nodes_dict() for group in self._interface.groups]

                    G = self.cluster_graph(clust_settings,
                                           clusters_dynamics=clusters_dynamics)
                    # Extract updated labels
                    # updated labels - dict for each group in dynamics
                    updated_labels = {}
                    for frame_index in range(len(clusters_dynamics)):
                        # dict - old_label:new_label
                        frame_labels = {}
                        for cluster_index in range(len(clusters_dynamics[frame_index])):
                            node_id = f"{frame_index}-{cluster_index}"
                            new_label = G.nodes[node_id]['Label']+'*'
                            frame_labels[G.nodes[node_id]
                                         ['cluster']] = new_label
                        updated_labels[frame_index] = frame_labels

                    # Apply updated labels
                    for frame_index, labels in updated_labels.items():
                        group: Group = self._interface.groups[frame_index]
                        print(
                            f'{group.clustering(**clust_settings).cluster_labels = }\n{labels = }')
                        group.clustering(
                            **clust_settings).cluster_labels = list(labels.values())
                        print(
                            f'Updated {group.clustering(**clust_settings).cluster_labels}')
                        if group.is_clustering_graph_initialized(**clust_settings):
                            warnings.warn(
                                'Clustering tracked after clustering graph was initialized. This is unexpected behavior, trying to handle.')
                            group.clustering_graph(
                                reinitialize=True, **clust_settings)

                    self._track_clusters_cache[cluster_tuple] = updated_labels

                updated_labels_list.append(
                    self._track_clusters_cache[cluster_tuple])
            return updated_labels_list

        def get_unique_clusters(self, clusterization_settings: Union[dict, List[dict]]) -> List[List[str]]:
            """
            Retrieves a list of lists, each containing unique clusters based on the given clusterization settings.
            Each inner list corresponds to one clustering type if multiple types are provided.

            :param clusterization_settings: The settings used for clusterization, either a single dictionary or a list of dictionaries for multiple types.
            :return: A list of lists, each containing unique cluster names for one type of clustering.
            """
            tracked_clusters_list = self.track_clusters(
                clusterization_settings)
            unique_clusters_list = []

            for tracked_clusters in tracked_clusters_list:
                unique_clusters = set()
                for clusters in tracked_clusters.values():
                    unique_clusters.update(clusters.values())
                unique_clusters_list.append(list(unique_clusters))

            return unique_clusters_list

        def get_cluster_presence(self, clusterization_settings: Union[dict, List[dict]]) -> List[Dict[str, List[int]]]:
            """
            Retrieves a list of dictionaries, each mapping unique clusters to the frames in which they appear, based on the given clusterization settings.
            Each dictionary corresponds to one clustering type if multiple types are provided.

            :param clusterization_settings: The settings used for clusterization, either a single dictionary or a list of dictionaries for multiple types.
            :return: A list of dictionaries, each mapping unique cluster names to frame presence for one type of clustering.
            """
            tracked_clusters_list = self.track_clusters(
                clusterization_settings)
            cluster_presence_list = []

            for tracked_clusters in tracked_clusters_list:
                cluster_presence = {}
                for frame_index, clusters in tracked_clusters.items():
                    for cluster in clusters.values():
                        if cluster not in cluster_presence:
                            cluster_presence[cluster] = []
                        cluster_presence[cluster].append(frame_index)
                cluster_presence_list.append(cluster_presence)

            return cluster_presence_list

        def get_final_value(self, clusterization_settings: Union[dict, List[dict]], parameter) -> Dict[str, float]:
            '''
            Returns the value of the parameter in the last group cluster appeared  
            '''
            presence = self.get_cluster_presence(clusterization_settings)[0]
            final_image = {cluster: images[-1]
                           for cluster, images in presence.items()}
            final_values = {}
            for cluster, group_number in final_image.items():
                data = self._interface.groups[group_number].clustering_graph_values(
                    parameters=(parameter, 'Label'), clustering_settings=clusterization_settings)
                final_values[cluster] = data[parameter][data['Label'].index(
                    cluster)]
            return final_values

    @abstractmethod
    def __len__(self):
        pass

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


class HariGraphInterface(Interface):
    """Interface specifically designed for the HariGraph class."""

    REQUIRED_TYPE = HariGraph

    def __init__(self, data):
        super().__init__(data=data, group_length=1)
        self.data: HariGraph
        self._group_size = 1

    def __len__(self):
        return 1

    def _initialize_group(self, i: int = 0) -> Group:
        if i != 0 or self._group_size != 1:
            Warning.warn(
                'An attempt to create the dynamics from a singe image')
        return Group([self.data]*self._group_size, time=np.array([0]))

    def _regroup_dynamics(self, num_intervals: int, interval_size: int = 1, offset: int = 0):
        if num_intervals != 1 and interval_size != 1:
            Warning.warn(
                'An attempt to create the dynamics from a singe image')
        self._group_size = interval_size

    @property
    def available_parameters(self) -> list:
        """
        Retrieves the list of available parameters/methods from the data gatherer.

        Returns:
            list: A list of available parameters or methods.
        """
        return self.data.gatherer.methods


class HariDynamicsInterface(Interface):
    """Interface specifically designed for the HariDynamics class."""

    REQUIRED_TYPE = HariDynamics

    def __init__(self, data):
        super().__init__(data=data, group_length=len(data.groups))
        self.data: HariDynamics

    def __len__(self):
        return len(self.data.groups)

    def _initialize_group(self, i: int) -> Group:
        group = self.data.groups[i]
        return Group([self.data[j] for j in group], time=np.array(group))

    def _regroup_dynamics(self, num_intervals: int, interval_size: int = 1, offset: int = 0):
        self.data.group(num_intervals, interval_size, offset)

    @property
    def available_parameters(self) -> list:
        """
        Retrieves the list of available parameters/methods from the data gatherer.

        Returns:
            list: A list of available parameters or methods.
        """
        return self.data[0].gatherer.methods


class SimulationInterface(Interface):
    """Interface specifically designed for the Simulation class."""

    REQUIRED_TYPE = Simulation

    def __init__(self, data):
        super().__init__(data=data, group_length=len(data.dynamics.groups))
        self.data: Simulation

    def __len__(self):
        return len(self.data.dynamics.groups)

    def _initialize_group(self, i: int) -> Group:
        group = self.data.dynamics.groups[i]
        return Group([self.data.dynamics[j] for j in group], time=np.array(group) * self.data.dt)

    def _regroup_dynamics(self, num_intervals: int, interval_size: int = 1, offset: int = 0):
        self.data.dynamics.group(num_intervals, interval_size, offset)

    @property
    def available_parameters(self) -> list:
        """
        Retrieves the list of available parameters/methods from the data gatherer.

        Returns:
            list: A list of available parameters or methods.
        """
        return self.data.dynamics[0].gatherer.methods
