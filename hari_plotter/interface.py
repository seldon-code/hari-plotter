from abc import ABC, abstractmethod, abstractproperty

from .hari_dynamics import HariDynamics
from .hari_graph import HariGraph
from .simulation import Simulation

from typing import Iterator


class Interface(ABC):
    REQUIRED_TYPE = None
    available_classes = {}

    def __init__(self, data):
        if not isinstance(data, self.REQUIRED_TYPE):
            raise ValueError(
                f"data must be an instance of {self.REQUIRED_TYPE}")
        self.data = data

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.REQUIRED_TYPE:
            Interface.available_classes[cls.REQUIRED_TYPE] = cls

    @classmethod
    def create_interface(cls, data):
        interface_class = cls.available_classes.get(type(data))
        if interface_class:
            return interface_class(data)
        else:
            raise ValueError("Invalid data type, no matching interface found.")

    @classmethod
    def info(cls):
        mappings = ', '.join(
            [f"{key.__name__} -> {value.__name__}" for key, value in cls.available_classes.items()])
        return f"Available Classes: {mappings}"

    @abstractproperty
    def images(self):
        raise NotImplementedError(
            "This property must be implemented in subclasses")

    @abstractproperty
    def groups(self):
        raise NotImplementedError(
            "This property must be implemented in subclasses")


class HariGraphInterface(Interface):
    REQUIRED_TYPE = HariGraph

    @property
    def images(self) -> Iterator[HariGraph]:
        image = self.data.copy()
        image.time = 0
        yield image

    @property
    def groups(self):
        raise NotImplementedError(
            "This property must be implemented in subclasses")


class HariDynamicsInterface(Interface):
    REQUIRED_TYPE = HariDynamics

    @property
    def images(self) -> Iterator[HariGraph]:
        for group in self.data.groups:
            image = self.data[group[0]].copy()
            image.time = group[0]
            yield image

    @property
    def groups(self):
        raise NotImplementedError(
            "This property must be implemented in subclasses")


class SimulationInterface(Interface):
    REQUIRED_TYPE = Simulation

    @property
    def images(self) -> Iterator[HariGraph]:
        for group in self.data.dynamics.groups:
            image = self.data.dynamics[group[0]].copy()
            image.time = group[0]*self.data.model.params.get("dt", 1)
            yield image

    @property
    def groups(self):
        raise NotImplementedError(
            "This property must be implemented in subclasses")
