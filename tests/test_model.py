import networkx as nx
import numpy as np
import pytest

from hari_plotter import Graph, Model, ModelFactory, Simulation


class TestModel:

    @classmethod
    def setup_class(cls):
        cls.degroot: Model = ModelFactory.from_toml('tests/degroot.toml')
        cls.activity: Model = ModelFactory.from_toml('tests/activity.toml')
        cls.activity_bots: Model = ModelFactory.from_toml(
            'tests/activity_bots.toml')

    def test_from_toml(self):
        assert isinstance(self.degroot, Model)
        assert isinstance(self.activity, Model)
        assert isinstance(self.activity_bots, Model)

    def test_init(self):
        model = ModelFactory.create_model("DeGroot", {'gamma': 1})
        assert isinstance(model, Model)

    def test_bots(self):
        assert self.activity_bots.params['n_bots'] == 2, 'Number of bots is incorrect'
