from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Type, Union

import numpy as np
import toml

from .hari_graph import HariGraph


class Model(ABC):
    """
    Base abstract class for different types of models.
    """
    _registry: Dict[str, Type["Model"]] = {}

    def __new__(cls, model_type: str, params: Dict[str, Any]):
        """
        Create a new instance of the Model or its registered subclass based on the model_type.

        Parameters:
            model_type: Type of the model to be instantiated.
            params: Parameters for initializing the model.

        Returns:
            Instance of the Model or its registered subclass.
        """
        if cls == Model:  # Checking if the class being instantiated is the base class
            if model_type in cls._registry:
                return super(Model, cls._registry[model_type]).__new__(cls._registry[model_type])
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        return super(Model, cls).__new__(cls)

    def __init__(self, model_type: str, params: Dict[str, Any]):
        """
        Initialize a Model instance.

        Parameters:
            model_type: Type of the model being instantiated.
            params: Parameters for initializing the model.
        """
        self.model_type = model_type
        self.params = params

    @classmethod
    def register(cls, model_type: str) -> Callable[[Type], Type]:
        """
        Class method to register subclasses for different model types.

        Parameters:
            model_type: The type of the model that the subclass represents.

        Returns:
            Decorator for registering the subclass.
        """
        def decorator(subclass: Type) -> Type:
            cls._registry[model_type] = subclass
            return subclass
        return decorator

    @classmethod
    def from_toml(cls, filename: str) -> "Model":
        """
        Load model parameters from a TOML file and instantiate the corresponding Model subclass.

        Parameters:
            filename: Path to the TOML file containing model configuration.

        Returns:
            Instance of the appropriate Model subclass based on the TOML file.
        """
        data = toml.load(filename)
        model_type = data.get("simulation", {}).get("model")
        params = data.get(model_type, {})

        if model_type not in cls._registry:
            raise ValueError(f"Unknown model type: {model_type}")

        return cls._registry[model_type](model_type, params)

    @abstractmethod
    def get_tension(self) -> float:
        """
        Abstract method to compute and return the tension value.

        Returns:
            The computed tension value.
        """
        raise NotImplementedError(
            "The get_tension method must be implemented in subclasses")

    @abstractmethod
    def get_influence(self) -> np.ndarray:
        """
        Abstract method to compute and return the influence values.

        Returns:
            Array containing the computed influence values.
        """
        raise NotImplementedError(
            "The get_influence method must be implemented in subclasses")

    def __repr__(self) -> str:
        """
        Return a string representation of the Model instance.

        Returns:
            String representation of the Model.
        """
        return f"Model(model_type={self.model_type}, params={self.params})"


@Model.register("ActivityDriven")
class ActivityDrivenModel(Model):
    """
    Model representing the ActivityDriven dynamics.
    """

    def get_tension(self, G: HariGraph, norm_type: str = 'squared') -> float:
        """
        Compute and return the tension for the ActivityDriven model.

        Parameters:
            G: The graph on which the tension is to be computed.
            norm_type: The type of norm used to compute the tension. Can be either 'abs' or 'squared'.

        Returns:
            The computed tension value.
        """

        alpha = self.params['alpha']
        K = self.params['K']

        delta_opinions = []
        for node in G:
            sum_influence = sum(G[node][neighbor]['influence'] * np.tanh(alpha * G.nodes[neighbor]['opinion'])
                                for neighbor in G.successors(node))
            adjusted_opinion = -G.nodes[node]['opinion'] + K * sum_influence
            delta_opinions.append(adjusted_opinion - G.nodes[node]['opinion'])

        # Calculate the tension based on the specified norm type
        if norm_type == 'abs':
            tension = np.sum(np.abs(delta_opinions))
        elif norm_type == 'squared':
            tension = np.sqrt(np.sum(np.square(delta_opinions)))
        else:
            raise ValueError(
                f"Invalid norm_type: {norm_type}. Choose either 'abs' or 'squared'.")

        return tension

    def get_influence(self, G: HariGraph) -> np.ndarray:
        """
        Compute and return the influence values for the ActivityDriven model.

        Parameters:
            G: The graph on which the influence values are to be computed.

        Returns:
            Array containing the computed influence values.
        """
        alpha = self.params['alpha']
        K = self.params['K']

        influences = np.zeros(len(G))
        for node in G:
            total_influence = 0
            sum_influence = sum(G[node][neighbor]['influence'] * np.tanh(alpha * G.nodes[neighbor]['opinion'])
                                for neighbor in G.successors(node))
            adjusted_opinion = -G.nodes[node]['opinion'] + K * sum_influence
            for neighbor in G.successors(node):
                delta_opinion = adjusted_opinion - G.nodes[neighbor]['opinion']
                total_influence += abs(delta_opinion)

            influences[node] = total_influence

        return influences


@Model.register("DeGroot")
class DeGrootModel(Model):
    """
    Model representing the DeGroot dynamics.
    """

    def get_tension(self, G: HariGraph, norm_type: str = 'squared') -> float:
        """
        Compute and return the tension for the DeGroot model.

        Parameters:
            G: The graph on which the tension is to be computed.
            norm_type: The type of norm used to compute the tension. Can be either 'abs' or 'squared'.

        Returns:
            The computed tension value.
        """
        delta_opinions = []
        for node in G:
            weighted_sum = sum(G[neighbor][node]['influence'] * G.nodes[neighbor]
                               ['opinion'] for neighbor in G.predecessors(node))
            delta_opinion = G.nodes[node]['opinion'] - weighted_sum
            delta_opinions.append(delta_opinion)

        # Calculate the tension based on the specified norm type
        if norm_type == 'abs':
            tension = np.sum(np.abs(delta_opinions))
        elif norm_type == 'squared':
            tension = np.sqrt(np.sum(np.square(delta_opinions)))
        else:
            raise ValueError(
                f"Invalid norm_type: {norm_type}. Choose either 'abs' or 'squared'.")

        return tension

    def get_influence(self, G: HariGraph) -> np.ndarray:
        """
        Compute and return the influence values for the DeGroot model.

        Parameters:
            G: The graph on which the influence values are to be computed.

        Returns:
            Array containing the computed influence values.
        """
        influences = np.zeros(len(G))

        # For each node, calculate its influence on its neighbors
        for node in G:
            total_influence = 0
            for neighbor in G.successors(node):
                weighted_opinion = G[node][neighbor]['influence'] * \
                    G.nodes[node]['opinion']
                delta_opinion = G.nodes[neighbor]['opinion'] - weighted_opinion
                total_influence += abs(delta_opinion)

            influences[node] = total_influence

        return influences
