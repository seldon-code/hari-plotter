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

        def _get_group(self, i):
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

        def __len__(self):
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

    # @abstractmethod
    # def images(self):
    #     """Return an iterator of image data."""
    #     raise NotImplementedError(
    #         "This method must be implemented in subclasses")

    # def _calculate_mean_node_values(self, group: Group, params: List[str]) -> dict:
    #     """
    #     Calculate the mean node values based on parameters.

    #     Args:
    #         group (Group): Group of images of the dynamics
    #         params (List[str]): List of parameter names.

    #     Returns:
    #         dict: A dictionary containing mean node values.
    #     """
    #     node_values_accumulator = defaultdict(lambda: defaultdict(list))
    #     results = defaultdict(list)

    #     # Process each image in the group
    #     for image in group:
    #         node_values = self.get_node_values(image, params)

    #         # Accumulate the parameter values for each node
    #         for node, parameters in node_values.items():
    #             for param, value in parameters.items():
    #                 node_values_accumulator[node][param].append(value)

    #     # Calculate results for each node and parameter
    #     for node, parameters in node_values_accumulator.items():
    #         results['node'].append(node)
    #         for param, values in parameters.items():
    #             if param.startswith("max_"):
    #                 # Compute max value for parameters starting with 'max_'
    #                 results[param].append(max(values))
    #             elif param.startswith("min_"):
    #                 # Compute min value for parameters starting with 'min_'
    #                 results[param].append(min(values))
    #             else:
    #                 # Compute mean value for other parameters
    #                 results[param].append(np.mean(values))

    #     return results

    # def _calculate_node_values(self, group: Group, params: List[str]) -> dict:
    #     """
    #     Calculate node values based on parameters.

    #     Args:
    #         group (Group): Group of images of the dynamics
    #         params (List[str]): List of parameter names.

    #     Returns:
    #         dict: A dictionary containing node values.
    #     """
    #     node_values_accumulator = defaultdict(lambda: defaultdict(list))
    #     results = defaultdict(lambda: defaultdict(list))

    #     # Process each image in the group
    #     for image in group:
    #         node_values = self.get_node_values(image, params)

    #         # print(f'{len(node_values.keys()) = }')
    #         # print(f'{node_values[0].keys() = }')
    #         # print(f'{node_values[0]["neighbor_mean_opinion"] = }')

    #         # Accumulate the parameter values for each node
    #         for node, parameters in node_values.items():
    #             for param, value in parameters.items():
    #                 node_values_accumulator[node][param].append(value)

    #     # Transfer values from the accumulator to the results
    #     for node, parameters in node_values_accumulator.items():
    #         for param, values in parameters.items():
    #             results[node][param] = values

    #     return results

    # @abstractmethod
    # def mean_group_values_iterator(self, params: List[str]) -> Iterator[dict]:
    #     """
    #     Abstract method to fetch mean node values based on parameters.

    #     Args:
    #         params (List[str]): List of parameter names.

    #     Returns:
    #         Iterator[dict]: An iterator containing dictionaries of mean node values.
    #     """
    #     raise NotImplementedError(
    #         "This method must be implemented in subclasses")

    # @abstractmethod
    # def group_values_iterator(self, params: List[str]) -> Iterator[dict]:
    #     """
    #     Abstract method to fetch node values based on parameters.

    #     Args:
    #         params (List[str]): List of parameter names.

    #     Returns:
    #         Iterator[dict]: An iterator containing dictionaries of node values.
    #     """
    #     raise NotImplementedError(
    #         "This method must be implemented in subclasses")

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

    def track_clusters(self, **cluster_settings):
        # Extract the dynamics of clusters across frames
        clusters_dynamics = [list(group.clustering(
            **cluster_settings).get_cluster_mapping()) for group in self.groups]

        G = self.cluster_graph(
            clusters_dynamics=clusters_dynamics, **cluster_settings)
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
            group.clustering(**cluster_settings).cluster_labels = labels

        return updated_labels

    @abstractmethod
    def __len__(self):
        pass


class HariGraphInterface(Interface):
    """Interface specifically designed for the HariGraph class."""

    REQUIRED_TYPE = HariGraph

    def __init__(self, data):
        super().__init__(data=data, group_length=1)

    def __len__(self):
        return 1

    # def images(self) -> Iterator[dict]:
    #     """
    #     Return an iterator of image data for the HariGraph.

    #     Yields:
    #         dict: The image data for the HariGraph with an assigned time of 0.
    #     """
    #     image = self.data.copy()
    #     time = 0
    #     yield {'image': image, 'time': time}

    def _initialize_group(self, i: int = 0) -> Group:
        assert i == 0
        return Group([self.data], time=np.array([0]))

    # def groups(self) -> List[Group]:
    #     return [self.group(0)]

    # def get_node_values(self, image: Any, params: List[str]) -> dict:
    #     """
    #     Fetch the values for the given nodes based on provided parameters.

    #     Args:
    #         image (Any): The image data or identifier for which node values are fetched.
    #         params (List[str]): List of parameter names.

    #     Returns:
    #         dict: A dictionary containing node values based on provided parameters.
    #     """
    #     return self.data.gatherer.gather(params)

    # def mean_group_values(self, params: List[str]) -> dict:
    #     """
    #     Fetch the mean values for nodes based on provided parameters for the HariGraph.

    #     Args:
    #         params (List[str]): List of parameter names.

    #     Returns:
    #         dict: A dictionary containing mean node values and the time stamp.
    #     """

    #     data = {'data': self._calculate_mean_node_values(
    #         [None], params)}  # No group for single image
    #     data['time'] = 0
    #     return data

    # def group_values(self, params: List[str]) -> dict:
    #     """
    #     Fetch the node values based on provided parameters for the HariGraph.

    #     Args:
    #         params (List[str]): List of parameter names.

    #     Returns:
    #         dict: A dictionary containing node values and the time stamp.
    #     """
    #     data = {'data': self._calculate_node_values(
    #         [None], params)}  # No group for single image
    #     data['time'] = 0
    #     return data

    # def mean_group_values_iterator(self, params: List[str]) -> Iterator[dict]:
    #     """
    #     Fetch the mean values for groups based on provided parameters for the HariGraph.

    #     Args:
    #         params (List[str]): List of parameter names.

    #     Yields:
    #         dict: A dictionary containing mean node values and the time stamp.
    #     """
    #     data = {'data': self._calculate_mean_node_values(
    #         [None], params)}  # No group for single image
    #     data['time'] = 0
    #     yield data

    # def group_values_iterator(self, params: List[str]) -> Iterator[dict]:
    #     """
    #     Fetch the node values based on provided parameters for the HariGraph.

    #     Args:
    #         params (List[str]): List of parameter names.

    #     Yields:
    #         dict: A dictionary containing node values and the time stamp.
    #     """
    #     data = {'data': self._calculate_node_values(
    #         [None], params)}  # No group for single image
    #     data['time'] = 0
    #     yield data

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

    # def images(self) -> Iterator[dict]:
    #     """
    #     Return an iterator of image data for the HariDynamics.

    #     Iterates over the groups present in the data and yields the last image
    #     from each group along with its corresponding time.

    #     Yields:
    #         dict: Dictionary containing the image data and its associated time.
    #     """
    #     for group in self.data.groups:
    #         image = self.data[group[-1]].copy()
    #         time = group[-1]
    #         yield {'image': image, 'time': time}

    def _initialize_group(self, i: int) -> Group:
        group = self.data.groups[i]
        return Group([self.data[j] for j in group], time=np.array(group))

    # def groups(self) -> List[Group]:
    #     return [self.data.group(i) for i in range(len(self.data.groups))]

    # def get_node_values(self, image: Any, params: List[str]) -> dict:
    #     """
    #     Fetch the values for the given nodes based on provided parameters.

    #     Args:
    #         image (Any): The image data or identifier for which node values are fetched.
    #         params (List[str]): List of parameter names.

    #     Returns:
    #         dict: A dictionary containing node values based on provided parameters.
    #     """
    #     return self.data[image].gatherer.gather(params)

    # def mean_group_values(self, params: List[str], i: int) -> dict:
    #     """
    #     Fetch the mean values for nodes based on provided parameters for a specific group in the HariDynamics.

    #     Args:
    #         params (List[str]): List of parameter names.
    #         i (int): Index of the group.

    #     Returns:
    #         dict: A dictionary containing mean node values and the time stamp for the specified group.
    #     """
    #     group = self.data.groups[i]
    #     data = {'data': self._calculate_mean_node_values(group, params)}
    #     data['time'] = group[-1]
    #     return data

    # def group_values(self, params: List[str], i: int) -> dict:
    #     """
    #     Fetch the node values based on provided parameters for a specific group in the HariDynamics.

    #     Args:
    #         params (List[str]): List of parameter names.
    #         i (int): Index of the group.

    #     Returns:
    #         dict: A dictionary containing node values and the time stamp for the specified group.
    #     """
    #     group = self.data.groups[i]
    #     data = {'data': self._calculate_node_values(group, params)}
    #     data['time'] = group[-1]
    #     return data

    # def mean_group_values_iterator(self, params: List[str]) -> Iterator[dict]:
    #     """
    #     Fetch the mean values for nodes based on provided parameters for the HariDynamics.

    #     Iterates over the groups present in the data, calculates mean values for each
    #     group and yields the results.

    #     Args:
    #         params (List[str]): List of parameter names.

    #     Yields:
    #         dict: A dictionary containing mean node values and the time stamp.
    #     """
    #     for group in self.data.groups:
    #         data = {'data': self._calculate_mean_node_values(group, params)}
    #         data['time'] = group[-1]
    #         yield data

    # def group_values_iterator(self, params: List[str]) -> Iterator[dict]:
    #     """
    #     Fetch the node values based on provided parameters for the HariDynamics.

    #     Iterates over the groups present in the data and yields node values for each group.

    #     Args:
    #         params (List[str]): List of parameter names.

    #     Yields:
    #         dict: A dictionary containing node values and the time stamp.
    #     """
    #     for group in self.data.groups:
    #         data = {'data': self._calculate_node_values(group, params)}
    #         data['time'] = group[-1]
    #         yield data

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

    # def images(self) -> Iterator[dict]:
    #     """
    #     Return an iterator of image data for the Simulation.

    #     Iterates over the groups present in the dynamics data and yields the last
    #     image from each group along with its corresponding adjusted time.

    #     Yields:
    #         dict: Dictionary containing the image data and its associated time.

    #     Note:
    #         Are self-loops good?
    #     """
    #     for group in self.data.dynamics.groups:
    #         image = self.data.dynamics[group[-1]].copy()
    #         time = group[-1] * self.data.model.params.get("dt", 1)
    #         yield {'image': image, 'time': time}

    def _initialize_group(self, i: int) -> Group:
        group = self.data.dynamics.groups[i]
        return Group([self.data.dynamics[j] for j in group], time=np.array(group) * self.data.model.params.get("dt", 1))

    # def groups(self) -> List[Group]:
    #     return [self.group(i) for i in range(len(self.data.dynamics.groups))]

    # def get_node_values(self, image: Any, params: List[str]) -> dict:
    #     """
    #     Fetch the values for the given nodes based on provided parameters for the Simulation.

    #     Args:
    #         image (Any): The image data or identifier for which node values are fetched.
    #         params (List[str]): List of parameter names.

    #     Returns:
    #         dict: A dictionary containing node values based on provided parameters.
    #     """
    #     return self.data.dynamics[image].gatherer.gather(params)

    # def mean_group_values(self, params: List[str], i: int) -> dict:
    #     """
    #     Fetch the mean values for nodes based on provided parameters for a specific group in the Simulation.

    #     Args:
    #         params (List[str]): List of parameter names.
    #         i (int): Index of the group.

    #     Returns:
    #         dict: A dictionary containing mean node values and the adjusted time stamp for the specified group.
    #     """
    #     group = self.data.dynamics.groups[i]
    #     data = {'data': self._calculate_mean_node_values(group, params)}
    #     data['time'] = group[-1] * self.data.model.params.get("dt", 1)
    #     return data

    # def group_values(self, params: List[str], i: int) -> dict:
    #     """
    #     Fetch the node values based on provided parameters for a specific group in the Simulation.

    #     Args:
    #         params (List[str]): List of parameter names.
    #         i (int): Index of the group.

    #     Returns:
    #         dict: A dictionary containing node values and the adjusted time stamp for the specified group.
    #     """
    #     group = self.data.dynamics.groups[i]
    #     data = {'data': self._calculate_node_values(group, params)}
    #     data['time'] = group[-1] * self.data.model.params.get("dt", 1)
    #     return data

    # def mean_group_values_iterator(self, params: List[str]) -> Iterator[dict]:
    #     """
    #     Fetch the mean values for nodes based on provided parameters for the Simulation.

    #     Iterates over the groups present in the dynamics data, calculates mean values
    #     for each group using the appropriate time scaling, and yields the results.

    #     Args:
    #         params (List[str]): List of parameter names.

    #     Yields:
    #         dict: A dictionary containing mean node values and the adjusted time stamp.
    #     """
    #     for group in self.data.dynamics.groups:
    #         data = {'data': self._calculate_mean_node_values(group, params)}
    #         data['time'] = group[-1] * self.data.model.params.get("dt", 1)
    #         yield data

    # def group_values_iterator(self, params: List[str]) -> Iterator[dict]:
    #     """
    #     Fetch the node values based on provided parameters for the Simulation.

    #     Iterates over the groups present in the dynamics data and yields node values
    #     for each group using the appropriate time scaling.

    #     Args:
    #         params (List[str]): List of parameter names.

    #     Yields:
    #         dict: A dictionary containing node values and the adjusted time stamp.
    #     """
    #     for group in self.data.dynamics.groups:
    #         data = {'data': self._calculate_node_values(group, params)}
    #         data['time'] = group[-1] * self.data.model.params.get("dt", 1)
    #         yield data

    @property
    def available_parameters(self) -> list:
        """
        Retrieves the list of available parameters/methods from the data gatherer.

        Returns:
            list: A list of available parameters or methods.
        """
        return self.data.dynamics[0].gatherer.methods
