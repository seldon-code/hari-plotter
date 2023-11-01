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
        self.H_plotter.draw(mode=[])

    def test_1D_distribution(self):
        self.H_plotter.plot_1D_distribution(mode=[], x_parameter='opinion')
        self.H_plotter.plot_1D_distribution(
            mode=[], x_parameter='opinion', scale='tanh')

    def test_plot_2D_distribution(self):
        self.H_plotter.plot_2D_distribution(
            mode=[], x_parameter='opinion', y_parameter='neighbor_mean_opinion')
        self.H_plotter.plot_2D_distribution(
            mode=[], x_parameter='opinion', y_parameter='neighbor_mean_opinion', scale='tanh')

    def test_plot_2D_distribution_extended(self):
        self.H_plotter.plot_2D_distribution_extended(
            mode=[], x_parameter='opinion', y_parameter='neighbor_mean_opinion')
        self.H_plotter.plot_2D_distribution_extended(
            mode=[], x_parameter='opinion', y_parameter='neighbor_mean_opinion', scale='tanh')
