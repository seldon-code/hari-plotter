import networkx as nx
import numpy as np
import pytest

from hari_plotter import HariGraph, Interface, Plotter, Simulation
from hari_plotter.lazy_hari_graph import LazyHariGraph


class TestPlotter:
    def test_main_plot(self):
        S = Simulation.from_dir("tests/big_test")
        plotter = Plotter.create_plotter(S)
        plotter.interface.regroup(num_intervals=3, interval_size=1)
        cl = {
            "clustering_method": "Interval Clustering",
            "clustering_parameters": ("Opinion", ),
            "scale": ("Linear", ),
            "parameter_boundaries": ((0.0, ), ),
        }
        plotter.add_plot(
            "Scatter",
            {
                "parameters": ["Opinion", "Neighbor mean opinion"],
                # "marker": "x",
                "scale": ["Tanh", "Tanh"],
                # "x_lim": [0, 5],
                # "y_lim": [0, 2],
            },
            row=0,
            col=0,
        )

        plotter.add_plot(
            "Static: Time line",
            {
                "parameters": ["Time", "Opinion"],
            },
            row=0,
            col=1,
        )

        # plotter.add_plot(
        #     "Static: Opinions",
        #     {
        #         "scale": ["Linear", "Tanh"],
        #         "clustering_settings": cl,
        #     },
        #     row=0,
        #     col=1,
        # )
        plotter.size_ratios = [[3, 3], [3]]

        plotter.plot_dynamics(
            mode=[],
            save_dir="test_pics",
            gif_path="test_pics/gif.gif",
        )
