from abc import ABC, abstractmethod

import numpy as np
import toml

from .hari_graph import HariGraph


class Model(ABC):
    _registry = {}

    def __new__(cls, model_type, params):
        if cls == Model:  # Checking if the class being instantiated is the base class
            if model_type in cls._registry:
                return super(Model, cls._registry[model_type]).__new__(cls._registry[model_type])
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        return super(Model, cls).__new__(cls)

    def __init__(self, model_type, params):
        self.model_type = model_type
        self.params = params

    @classmethod
    def register(cls, model_type):
        def decorator(subclass):
            cls._registry[model_type] = subclass
            return subclass
        return decorator

    @classmethod
    def from_toml(cls, filename):
        data = toml.load(filename)
        model_type = data.get("simulation", {}).get("model")
        params = data.get(model_type, {})

        if model_type not in cls._registry:
            raise ValueError(f"Unknown model type: {model_type}")

        return cls._registry[model_type](model_type, params)

    @abstractmethod
    def get_tension(self):
        raise NotImplementedError(
            "The tension property must be implemented in subclasses")

    @abstractmethod
    def get_influence(self):
        raise NotImplementedError(
            "The tension property must be implemented in subclasses")

    def __repr__(self):
        return f"Model(model_type={self.model_type}, params={self.params})"


@Model.register("ActivityDriven")
class ActivityDrivenModel(Model):

    def get_tension(self, G: HariGraph, norm_type='squared') -> float:
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

    def get_tension(self, G: HariGraph, norm_type='squared') -> float:
        # Calculate the opinion change for each node
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
