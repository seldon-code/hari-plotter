import networkx as nx
import numpy as np
import pytest

from hari_plotter import Graph, Model, ModelFactory, Simulation


class TestSimulation:

    @classmethod
    def setup_class(cls):
        cls.degroot = Simulation.from_toml('tests/degroot.toml')
        cls.activity = Simulation.from_toml('tests/activity.toml')

    def test_from_toml(self):
        assert isinstance(self.degroot, Simulation)
        assert isinstance(self.activity, Simulation)
        assert isinstance(self.degroot.model, Model)
        assert isinstance(self.activity.model, Model)
