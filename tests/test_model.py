import networkx as nx
import numpy as np
import pytest

from hari_plotter import Graph, Model, ModelFactory, Simulation
from hari_plotter.model import ActivityDrivenModel, DeGrootModel


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

    def test_get_model_name(self):
        model_name = ModelFactory.get_model_name(type(self.degroot))
        assert model_name == "DeGroot", 'Model name is incorrect'
        model_name = ModelFactory.get_model_name(type(self.activity))
        assert model_name == "ActivityDriven", 'Model name is incorrect'

    def test_get_model_class(self):
        model_class = ModelFactory.get_model_class("DeGroot")
        assert model_class == DeGrootModel, 'Model class is incorrect'
        model_class = ModelFactory.get_model_class("ActivityDriven")
        assert model_class == ActivityDrivenModel, 'Model class is incorrect'
