from abc import ABC, abstractmethod, abstractproperty
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Type, Union

import numpy as np

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

    def __init__(self, data: Any):
        """
        Initialize the Interface instance.

        Args:
            data: The underlying data object to which the interface applies.

        Raises:
            ValueError: If the data is not an instance of REQUIRED_TYPE.
        """
        if not isinstance(data, self.REQUIRED_TYPE):
            raise ValueError(
                f"data must be an instance of {self.REQUIRED_TYPE}")
        self.data = data

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """Auto-register subclasses in available_classes based on their REQUIRED_TYPE."""
        super().__init_subclass__(**kwargs)
        if cls.REQUIRED_TYPE:
            Interface.available_classes[cls.REQUIRED_TYPE] = cls

    @classmethod
    def create_interface(cls, data: Any) -> 'Interface':
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

    @classmethod
    def info(cls) -> str:
        """Return a string representation of the available classes and their mapping.

        Returns:
            A string detailing the available classes and their mappings.
        """
        mappings = ', '.join(
            [f"{key.__name__} -> {value.__name__}" for key, value in cls.available_classes.items()])
        return f"Available Classes: {mappings}"

    @abstractmethod
    def images(self):
        """Return an iterator of image data."""
        raise NotImplementedError(
            "This method must be implemented in subclasses")

    @abstractmethod
    def groups(self):
        """Abstract method to define the behavior for data grouping."""
        raise NotImplementedError(
            "This method must be implemented in subclasses")

    def _calculate_mean_node_values(self, group: List, params: List[str]) -> dict:
        """
        Calculate the mean node values based on parameters.

        Args:
            group (List): List containing data for each node.
            params (List[str]): List of parameter names.

        Returns:
            dict: A dictionary containing mean node values.
        """
        node_values_accumulator = defaultdict(lambda: defaultdict(list))
        results = defaultdict(list)

        # Process each image in the group
        for image in group:
            node_values = self.get_node_values(image, params)

            # Accumulate the parameter values for each node
            for node, parameters in node_values.items():
                for param, value in parameters.items():
                    node_values_accumulator[node][param].append(value)

        # Calculate results for each node and parameter
        for node, parameters in node_values_accumulator.items():
            results['node'].append(node)
            for param, values in parameters.items():
                if param.startswith("max_"):
                    # Compute max value for parameters starting with 'max_'
                    results[param].append(max(values))
                elif param.startswith("min_"):
                    # Compute min value for parameters starting with 'min_'
                    results[param].append(min(values))
                else:
                    # Compute mean value for other parameters
                    results[param].append(np.mean(values))

        return results

    def _calculate_node_values(self, group: List, params: List[str]) -> dict:
        """
        Calculate node values based on parameters.

        Args:
            group (List): List containing data for each node.
            params (List[str]): List of parameter names.

        Returns:
            dict: A dictionary containing node values.
        """
        node_values_accumulator = defaultdict(lambda: defaultdict(list))
        results = defaultdict(lambda: defaultdict(list))

        # Process each image in the group
        for image in group:
            node_values = self.get_node_values(image, params)

            # print(f'{len(node_values.keys()) = }')
            # print(f'{node_values[0].keys() = }')
            # print(f'{node_values[0]["neighbor_mean_opinion"] = }')

            # Accumulate the parameter values for each node
            for node, parameters in node_values.items():
                for param, value in parameters.items():
                    node_values_accumulator[node][param].append(value)

        # Transfer values from the accumulator to the results
        for node, parameters in node_values_accumulator.items():
            for param, values in parameters.items():
                results[node][param] = values

        return results

    @abstractmethod
    def mean_group_values_iterator(self, params: List[str]) -> Iterator[dict]:
        """
        Abstract method to fetch mean node values based on parameters.

        Args:
            params (List[str]): List of parameter names.

        Returns:
            Iterator[dict]: An iterator containing dictionaries of mean node values.
        """
        raise NotImplementedError(
            "This method must be implemented in subclasses")

    @abstractmethod
    def group_values_iterator(self, params: List[str]) -> Iterator[dict]:
        """
        Abstract method to fetch node values based on parameters.

        Args:
            params (List[str]): List of parameter names.

        Returns:
            Iterator[dict]: An iterator containing dictionaries of node values.
        """
        raise NotImplementedError(
            "This method must be implemented in subclasses")

    @abstractproperty
    def available_parameters(self) -> list:
        raise NotImplementedError(
            "This method must be implemented in subclasses")


class HariGraphInterface(Interface):
    """Interface specifically designed for the HariGraph class."""

    REQUIRED_TYPE = HariGraph

    def images(self) -> Iterator[dict]:
        """
        Return an iterator of image data for the HariGraph.

        Yields:
            dict: The image data for the HariGraph with an assigned time of 0.
        """
        image = self.data.copy()
        time = 0
        yield {'image': image, 'time': time}

    def groups(self) -> Any:
        """
        Define the behavior for data grouping in HariGraph.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError(
            "This method must be implemented in subclasses")

    def get_node_values(self, image: Any, params: List[str]) -> dict:
        """
        Fetch the values for the given nodes based on provided parameters.

        Args:
            image (Any): The image data or identifier for which node values are fetched.
            params (List[str]): List of parameter names.

        Returns:
            dict: A dictionary containing node values based on provided parameters.
        """
        return self.data.gatherer.gather(params)

    def mean_group_values_iterator(self, params: List[str]) -> Iterator[dict]:
        """
        Fetch the mean values for groups based on provided parameters for the HariGraph.

        Args:
            params (List[str]): List of parameter names.

        Yields:
            dict: A dictionary containing mean node values and the time stamp.
        """
        data = {'data': self._calculate_mean_node_values(
            [None])}  # No group for single image
        data['time'] = 0
        yield data

    def group_values_iterator(self, params: List[str]) -> Iterator[dict]:
        """
        Fetch the node values based on provided parameters for the HariGraph.

        Args:
            params (List[str]): List of parameter names.

        Yields:
            dict: A dictionary containing node values and the time stamp.
        """
        data = {'data': self._calculate_node_values(
            [None])}  # No group for single image
        data['time'] = 0
        yield data

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

    def images(self) -> Iterator[dict]:
        """
        Return an iterator of image data for the HariDynamics.

        Iterates over the groups present in the data and yields the last image 
        from each group along with its corresponding time.

        Yields:
            dict: Dictionary containing the image data and its associated time.
        """
        for group in self.data.groups:
            image = self.data[group[-1]].copy()
            time = group[-1]
            yield {'image': image, 'time': time}

    def groups(self) -> Any:
        """
        Define the behavior for data grouping in HariDynamics.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError(
            "This method must be implemented in subclasses")

    def get_node_values(self, image: Any, params: List[str]) -> dict:
        """
        Fetch the values for the given nodes based on provided parameters.

        Args:
            image (Any): The image data or identifier for which node values are fetched.
            params (List[str]): List of parameter names.

        Returns:
            dict: A dictionary containing node values based on provided parameters.
        """
        return self.data[image].gatherer.gather(params)

    def mean_group_values_iterator(self, params: List[str]) -> Iterator[dict]:
        """
        Fetch the mean values for nodes based on provided parameters for the HariDynamics.

        Iterates over the groups present in the data, calculates mean values for each 
        group and yields the results.

        Args:
            params (List[str]): List of parameter names.

        Yields:
            dict: A dictionary containing mean node values and the time stamp.
        """
        for group in self.data.groups:
            data = {'data': self._calculate_mean_node_values(group, params)}
            data['time'] = group[-1]
            yield data

    def group_values_iterator(self, params: List[str]) -> Iterator[dict]:
        """
        Fetch the node values based on provided parameters for the HariDynamics.

        Iterates over the groups present in the data and yields node values for each group.

        Args:
            params (List[str]): List of parameter names.

        Yields:
            dict: A dictionary containing node values and the time stamp.
        """
        for group in self.data.groups:
            data = {'data': self._calculate_node_values(group, params)}
            data['time'] = group[-1]
            yield data

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

    def images(self) -> Iterator[dict]:
        """
        Return an iterator of image data for the Simulation.

        Iterates over the groups present in the dynamics data and yields the last 
        image from each group along with its corresponding adjusted time.

        Yields:
            dict: Dictionary containing the image data and its associated time.
        """
        for group in self.data.dynamics.groups:
            image = self.data.dynamics[group[-1]].copy()
            time = group[-1] * self.data.model.params.get("dt", 1)
            yield {'image': image, 'time': time}

    def groups(self) -> Any:
        """
        Define the behavior for data grouping in Simulation.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError(
            "This method must be implemented in subclasses")

    def get_node_values(self, image: Any, params: List[str]) -> dict:
        """
        Fetch the values for the given nodes based on provided parameters for the Simulation.

        Args:
            image (Any): The image data or identifier for which node values are fetched.
            params (List[str]): List of parameter names.

        Returns:
            dict: A dictionary containing node values based on provided parameters.
        """
        return self.data.dynamics[image].gatherer.gather(params)

    def mean_group_values_iterator(self, params: List[str]) -> Iterator[dict]:
        """
        Fetch the mean values for nodes based on provided parameters for the Simulation.

        Iterates over the groups present in the dynamics data, calculates mean values 
        for each group using the appropriate time scaling, and yields the results.

        Args:
            params (List[str]): List of parameter names.

        Yields:
            dict: A dictionary containing mean node values and the adjusted time stamp.
        """
        for group in self.data.dynamics.groups:
            data = {'data': self._calculate_mean_node_values(group, params)}
            data['time'] = group[-1] * self.data.model.params.get("dt", 1)
            yield data

    def group_values_iterator(self, params: List[str]) -> Iterator[dict]:
        """
        Fetch the node values based on provided parameters for the Simulation.

        Iterates over the groups present in the dynamics data and yields node values 
        for each group using the appropriate time scaling.

        Args:
            params (List[str]): List of parameter names.

        Yields:
            dict: A dictionary containing node values and the adjusted time stamp.
        """
        for group in self.data.dynamics.groups:
            data = {'data': self._calculate_node_values(group, params)}
            data['time'] = group[-1] * self.data.model.params.get("dt", 1)
            yield data

    @property
    def available_parameters(self) -> list:
        """
        Retrieves the list of available parameters/methods from the data gatherer.

        Returns:
            list: A list of available parameters or methods.
        """
        return self.data.dynamics[0].gatherer.methods
