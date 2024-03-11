from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Type, Union

import numpy as np
import toml

from .graph import Graph
from .node_gatherer import ActivityDrivenNodeEdgeGatherer


class ModelFactory:
    _registry: Dict[str, Type['Model']] = {}

    @classmethod
    def register(cls, model_type: str):
        """
        Decorator for registering model classes with a specific model type.

        Parameters:
            model_type: The type of the model to register.

        Returns:
            A decorator function that registers the given class.
        """
        def decorator(model_class: Type['Model']) -> Type['Model']:
            cls._registry[model_type] = model_class
            return model_class
        return decorator

    @classmethod
    def create_model(cls, model_type: str, params: Dict[str, Any]) -> 'Model':
        """
        Create a model instance based on the model type and parameters.

        Parameters:
            model_type: The type of the model to create.
            params: Parameters for the model.

        Returns:
            An instance of the requested model type.
        """
        if model_type not in cls._registry:
            raise ValueError(f"Unknown model type: {model_type}")
        model_class = cls._registry[model_type]
        return model_class(params)

    @classmethod
    def from_toml(cls, filename: str) -> 'Model':
        """
        Create a model instance from a TOML configuration file.

        Parameters:
            filename: Path to the TOML file.

        Returns:
            An instance of the model defined in the TOML file.
        """
        data = toml.load(filename)
        model_type = data.get("simulation", {}).get("model")
        params = data.get(model_type, {})
        return cls.create_model(model_type, params)


class Model(ABC):
    """
    Base abstract class for different types of models.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Initialize a Model instance.

        Parameters:
            params: Parameters for initializing the model.
        """
        self.params = params

    @property
    def dt(self):
        return self.params.get("dt", 1)

    @abstractmethod
    def get_tension(self) -> float:
        """
        Abstract method to compute and return the tension value.
        """
        pass

    @abstractmethod
    def get_influence(self) -> np.ndarray:
        """
        Abstract method to compute and return the influence values.
        """
        pass

    @property
    def load_request(self) -> Dict[str, Any]:
        '''
        Results will be sent to Graph.from_network on setup
        '''
        return {}

    def __repr__(self) -> str:
        """
        Return a string representation of the Model instance.
        """
        return f"{self.__class__.__name__}(params={self.params})"


@ModelFactory.register("ActivityDriven")
class ActivityDrivenModel(Model):
    """
    Model representing the ActivityDriven dynamics.
    """

    def get_tension(self, G: Graph, norm_type: str = 'squared') -> float:
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

    def get_influence(self, G: Graph) -> np.ndarray:
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

    @property
    def load_request(self) -> Dict[str, Any]:
        '''
        Results will be sent to Graph.from_network on setup
        '''
        return {'gatherer': ActivityDrivenNodeEdgeGatherer, 'number_of_bots': self.params.get('n_bots', 0)}


@ModelFactory.register("DeGroot")
class DeGrootModel(Model):
    """
    Model representing the DeGroot dynamics.
    """

    def get_tension(self, G: Graph, norm_type: str = 'squared') -> float:
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

    def get_influence(self, G: Graph) -> np.ndarray:
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
