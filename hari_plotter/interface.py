from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Type, Union

import networkx as nx
import numpy as np

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

        # self.track_clusters_requests = []
        # self.static_data_cache_requests = []
        # self.dynamic_data_cache_requests = []

        self.track_clusters_cache = {}
        self.static_data_cache = self.StaticDataCache(self)
        self.dynamic_data_cache = [self.DynamicDataCache(
            self, i) for i in range(len(self.groups))]

        self._nodes = None

    # def collect_static_data(self):
    #     self.static_data_cache = {}
    #     for request in self.static_data_cache_requests:
    #         method_name = request['method']
    #         settings = request['settings']
    #         data_key = Interface.request_to_tuple(request)
    #         if data_key not in self.static_data_cache:
    #             # Fetch and cache the data for each group
    #             data_for_all_groups = []
    #             for group in self.groups:
    #                 data = getattr(group, method_name)(**settings)
    #                 data_for_all_groups.append(data)
    #             self.static_data_cache[data_key] = data_for_all_groups

    # def collect_dynamic_data(self, i: int):
    #     self.dynamic_data_cache = {}
    #     for request in self.dynamic_data_cache_requests:
    #         method_name = request['method']
    #         settings = request['settings']
    #         # print(f'{request = }')
    #         data_key = Interface.request_to_tuple(request)
    #         if data_key not in self.dynamic_data_cache:
    #             self.dynamic_data_cache[data_key] = getattr(
    #                 self.groups[i], method_name)(**settings)

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
        self.static_data_cache.clean()
        for c in self.dynamic_data_cache:
            c.clean()

    class GroupIterable:
        def __init__(self, interface_instance):
            self._interface = interface_instance

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

    def cluster_graph(self, clusters_dynamics=None, **cluster_settings):
        # Extract the dynamics of clusters across frames
        clusters_dynamics = clusters_dynamics or [list(group.clustering(
            **cluster_settings).get_cluster_mapping()) for group in self.groups]

        # Initialize the final labels from the last frame
        final_labels = self.groups[-1].clustering(**
                                                  cluster_settings).cluster_labels
        G = nx.DiGraph()

        # Build the graph
        for frame_index, frame in enumerate(clusters_dynamics):
            for cluster_index, cluster in enumerate(frame):
                node_id = f"{frame_index}-{cluster_index}"
                G.add_node(node_id, frame=frame_index, cluster=cluster,
                           label=None)  # Initially, labels are None

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
            G.nodes[node_id]['label'] = label

        def propagate_labels_with_splits(G, clusters_dynamics):
            for frame_index in range(len(clusters_dynamics) - 1, 0, -1):
                for cluster_index in range(len(clusters_dynamics[frame_index])):
                    node_id = f"{frame_index}-{cluster_index}"
                    current_label = G.nodes[node_id]['label']

                    predecessors = list(G.predecessors(node_id))
                    # If the current node has multiple predecessors, it's a split in reverse
                    if len(predecessors) > 1:
                        # Assign new labels to each predecessor node based on the split
                        for i, pred in enumerate(predecessors):
                            # Create new labels for splits as "original.i"
                            new_label = f"{current_label}.{i+1}"
                            G.nodes[pred]['label'] = new_label
                    # If there's only one predecessor, propagate the label directly
                    elif len(predecessors) == 1:
                        G.nodes[predecessors[0]]['label'] = current_label

        # Call the function to propagate labels with the handling of splits
        propagate_labels_with_splits(G, clusters_dynamics)

        return G

    def track_clusters(self, clusterization_settings: Union[dict, List[dict]]):

        clusterization_settings_list = [clusterization_settings] if isinstance(
            clusterization_settings, dict) else clusterization_settings

        updated_labels_list = []
        for clust_settings in clusterization_settings_list:
            # Extract the dynamics of clusters across frames
            # print(f'{clust_settings = }')
            cluster_tuple = Interface.request_to_tuple(clust_settings)
            if cluster_tuple not in self.track_clusters_cache:
                clusters_dynamics = [list(group.clustering(
                    **clust_settings).get_cluster_mapping()) for group in self.groups]

                G = self.cluster_graph(
                    clusters_dynamics=clusters_dynamics, **clust_settings)
                # Step 5: Extract updated labels
                updated_labels = {}
                for frame_index in range(len(clusters_dynamics)):
                    frame_labels = []
                    for cluster_index in range(len(clusters_dynamics[frame_index])):
                        node_id = f"{frame_index}-{cluster_index}"
                        label = G.nodes[node_id]['label']
                        frame_labels.append(label)
                    updated_labels[frame_index] = frame_labels

                # Apply updated labels
                for frame_index, labels in updated_labels.items():
                    group = self.groups[frame_index]
                    group.clustering(**clust_settings).cluster_labels = labels

                self.track_clusters_cache[cluster_tuple] = updated_labels

            updated_labels_list.append(
                self.track_clusters_cache[cluster_tuple])
        return updated_labels_list

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

    def __len__(self):
        return 1

    def _initialize_group(self, i: int = 0) -> Group:
        assert i == 0
        return Group([self.data], time=np.array([0]))

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

    def __len__(self):
        return len(self.data.groups)

    def _initialize_group(self, i: int) -> Group:
        group = self.data.groups[i]
        return Group([self.data[j] for j in group], time=np.array(group))

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

    def __len__(self):
        return len(self.data.dynamics.groups)

    def _initialize_group(self, i: int) -> Group:
        group = self.data.dynamics.groups[i]
        return Group([self.data.dynamics[j] for j in group], time=np.array(group) * self.data.model.params.get("dt", 1))

    @property
    def available_parameters(self) -> list:
        """
        Retrieves the list of available parameters/methods from the data gatherer.

        Returns:
            list: A list of available parameters or methods.
        """
        return self.data.dynamics[0].gatherer.methods
