import networkx as nx
import numpy as np
import pytest

from hari_plotter import HariGraph, Interface, Plotter


class TestPlotter:

    @classmethod
    def setup_class(cls):
        H = HariGraph.read_network(
            'tests/network.txt', 'tests/opinions_0.txt')
        cls.H_plotter = Plotter.create_plotter(H)

    def test_from_H(self):
        assert isinstance(self.H_plotter, Plotter)
        assert isinstance(self.H_plotter.interface, Interface)

    def test_draw(self):
        self.H_plotter.draw()

        # plotter.draw(x_parameter='opinion', y_parameter='neighbor_mean_opinion',mode = ['show', 'save', 'gif'],save_dir='test_pics', gif_path='test_pics/gif.gif', show_time=True, scale='tanh')
