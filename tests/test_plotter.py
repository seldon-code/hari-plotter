import filecmp
import os

import networkx as nx
import numpy as np
import pytest

from hari_plotter import ColorScheme, Graph, Interface, Plotter, Simulation
from hari_plotter.color_scheme import initialize_colormap
from hari_plotter.lazy_graph import LazyGraph


def compare_dirs(dir1, dir2):
    comparison = filecmp.dircmp(dir1, dir2)
    if comparison.left_only or comparison.right_only or comparison.diff_files:
        return False
    for subdir in comparison.common_dirs:
        if not compare_dirs(os.path.join(dir1, subdir), os.path.join(dir2, subdir)):
            return False
    return True


cl = {
    "clustering_method": "K-Means Clustering",
    "clustering_parameters": ["Opinion", "Neighbor mean opinion"],
    "scale": ["Tanh", "Tanh"],
    "n_clusters": 2,
}

plot_list = [
            ["Scatter",
             {
                 "parameters": ["Opinion", "Neighbor mean opinion"],
                 "scale": ["Tanh", "Tanh"],
             },],
            ["Static: Time line",
             {
                 "parameters": ["Time", "Opinion"],
             },],
            ["Histogram",
             {
                 "parameter": "Opinion",
                 "show_x_label": False,
                 "scale": ["Tanh", "Linear"],
             },],
            ["Histogram",
             {
                 "parameter": "Neighbor mean opinion",
                 "rotated": True,
                 "show_y_label": False,
                 "scale": ["Linear", "Tanh"],
             }],
            ["Hexbin",
             {
                 "parameters": ["Opinion", "Neighbor mean opinion"],
                 "scale": ["Tanh", "Tanh"]
             },],
            ["Scatter",
             {
                 "parameters": ["Opinion", "Neighbor mean opinion"],
                 "scale": ["Tanh", "Tanh"],
                 "color": {
                     "mode": "Parameter Colormap",
                     "settings": {"parameter": "Opinion density", "scale": ("Linear",)},
                 },
                 "marker": {"mode": "Cluster Marker", "settings": {"clustering_settings": cl}},
             },],
            ["Scatter",
             {
                 "parameters": ["Opinion", "Neighbor mean opinion"],
                 "scale": ["Tanh", "Tanh"],
                 "color": {
                     "mode": "Cluster Parameter Color",
                     "settings": {"clustering_settings": cl, "parameter": "Opinion"},
                 },
                 "marker": {"mode": "Cluster Marker", "settings": {"clustering_settings": cl}},
             },],
            ["Clustering: Centroids",
             {
                 "clustering_settings": cl,
                 "parameters": ["Opinion", "Neighbor mean opinion"],
                 "scale": ["Tanh", "Tanh"],
                 "color": {
                     "mode": "Cluster Color",
                     "settings": {"clustering_settings": cl},
                 },
                 "marker": {"mode": "Cluster Marker", "settings": {"clustering_settings": cl}},
             },],
            ["Clustering: Centroids",
             {
                 "parameters": ["Opinion", "Neighbor mean opinion"],
                 "clustering_settings": cl,
                 "scale": ["Tanh", "Tanh"],
                 "color": {
                     "mode": "Cluster Parameter Color",
                     "settings": {"clustering_settings": cl, "parameter": "Opinion"},
                 },
                 "marker": {"mode": "Cluster Marker", "settings": {"clustering_settings": cl}},
             },],
            ["Clustering: Fill",
             {
                 "parameters": ["Opinion", "Neighbor mean opinion"],
                 "clustering_settings": cl,
                 "scale": ["Tanh", "Tanh"],
                 "fill_color": {
                     "mode": "Cluster Parameter Color",
                     "settings": {"clustering_settings": cl, "parameter": "Opinion"},
                 },
             },],
            ["Clustering: Fill",
             {
                 "parameters": ["Opinion", "Neighbor mean opinion"],
                 "clustering_settings": cl,
                 "scale": ["Tanh", "Tanh"],
                 "fill_color": {
                     "mode": "Cluster Parameter Color",
                     "settings": {"clustering_settings": cl, "parameter": "Opinion"},
                 },
             },],
            ["Clustering: Degree of Membership",
             {
                 "parameters": ["Opinion", "Neighbor mean opinion"],
                 "clustering_settings": cl,
                 "scale": ["Tanh", "Tanh"],
             },],
            ["Clustering: Density Plot",
             {
                 "parameters": ["Opinion", "Neighbor mean opinion"],
                 "clustering_settings": cl,
                 "scale": ["Tanh", "Tanh"],
                 "fill_color": {
                     "mode": "Cluster Parameter Color",
                     "settings": {"clustering_settings": cl, "parameter": "Opinion"},
                 },
             },],
            ["Scatter",
             {
                 "parameters": ["Opinion", "Neighbor mean opinion"],
                 "scale": ["Tanh", "Tanh"],
                 "color": {
                     "mode": "Cluster Parameter Color",
                     "settings": {
                         "clustering_settings": cl,
                         "parameter": "Opinion"
                     },
                 },
                 "marker": {
                     "mode": "Cluster Marker",
                     "settings": {
                         "clustering_settings": cl
                     }
                 },
             },],
            ["Scatter",
             {
                 "parameters": ["Opinion", "Activity"],
                 "scale": ["Tanh", "Linear"],
                 "color": {
                     "mode": "Parameter Colormap",
                     "settings": {"parameter": "Opinion density", "scale": ("Linear",)},
                 },
             },],
            ["Static: Time line",
             {
                 "parameters": ["Time", "Opinion"],
             },],
            ["Static: Node lines",
             {
                 "parameters": ["Time", "Opinion"],
                 "scale": ["Linear", "Tanh"],

             },],
            ["Static: Node lines",
             {
                 "parameters": ["Time", "Opinion"],
                 "scale": ["Linear", "Tanh"],
                 "color": {
                     "mode": "Cluster Parameter Color",
                     "settings": {
                         "clustering_settings": cl,
                         "parameter": "Opinion",
                         "group_number": -1,
                     },
                 },
             }],
            ["Static: Node lines",
             {
                 "parameters": ["Time", "Opinion"],
                 "scale": ["Linear", "Tanh"],
                 "color": {
                     "mode": "Cluster Parameter Color",
                     "settings": {
                         "clustering_settings": cl,
                         "parameter": "Opinion",
                         "group_number": -1,
                         "None Color": "",
                     },
                 },
             },],
            [
                "Static: Graph line",
                {
                    "parameters": ["Time", "Opinion"],

                },
            ],
    ["Static: Graph line",
                {
                    "parameters": ["Time", "Opinion"],
                    "color": {
                        "mode": "Graph Parameter",
                        "settings": {
                            "parameter": "Opinion",
                        },
                    },
                },],
    ["Static: Graph Range",
                {
                    "parameters": ["Time", "Opinion"],
                    "color": {
                        "mode": "Graph Parameter",
                        "settings": {
                            "parameter": "Opinion",
                        },
                    },
                },],
    ["Static: Clustering Line",
                {
                    "parameter": "Opinion",
                    "clustering_settings": cl,
                },],
    ["Static: Clustering Line",
                {
                    "parameter": "Opinion",
                    "clustering_settings": cl,
                    "color": {
                        "mode": "Cluster Parameter Color",
                        "settings": {
                            "clustering_settings": cl,
                            "parameter": "Opinion",
                            "group_number": -1,
                            "None Color": "",
                        },
                    },
                },],
    ["Static: Clustering Range",
                {
                    "parameter": "Opinion",
                    "range_parameter": "Opinion Standard Deviation",
                    "clustering_settings": cl,
                    "color": {
                        "mode": "Cluster Parameter Color",
                        "settings": {
                            "clustering_settings": cl,
                            "parameter": "Opinion",
                            "group_number": -1,
                            "None Color": "",
                        },
                    },
                },],
    ["Static: Clustering Range",
                {
                    "parameter": "Opinion",
                    "range_parameter": "Opinion Standard Deviation",
                    "clustering_settings": cl,
                },],

]


class TestPlotter:
    @pytest.mark.parametrize("plot_index, plot", [(i, plot) for i, plot in enumerate(plot_list)])
    def test_plot_individual(self, plot_index, plot, setup_simulation, save_dir, tmpdir):
        plotter = Plotter.create_plotter(setup_simulation)
        plotter.regroup(num_intervals=3, interval_size=1)
        plotter.add_plot(
            *plot,
            row=0,
            col=0,
        )
        temp_dir = os.path.join(save_dir, f"plot_{plot_index}")
        os.makedirs(temp_dir, exist_ok=True)
        plotter.plot_dynamics(
            mode=['save'],
            save_dir=temp_dir,
            animation_path=os.path.join(temp_dir, f"gif_{plot_index}.gif"),
        )
        if save_dir != tmpdir:
            baseline_dir = os.path.join("tests/baseline", f"plot_{plot_index}")
            assert compare_dirs(
                temp_dir, baseline_dir), f"Content mismatch in plot {plot_index}"

    def test_plot_2x1(self, setup_simulation, save_dir, tmpdir):
        plotter = Plotter.create_plotter(setup_simulation)
        plotter.regroup(num_intervals=3, interval_size=1)
        plot = ["Scatter",
                {
                    "parameters": ["Opinion", "Neighbor mean opinion"],
                    "scale": ["Tanh", "Tanh"],
                },]

        plotter.add_plot(
            *plot,
            row=0,
            col=0,
        )
        plotter.add_plot(
            *plot,
            row=0,
            col=1,
        )

        temp_dir = os.path.join(save_dir, "plot_2x1")
        os.makedirs(temp_dir, exist_ok=True)
        plotter.plot_dynamics(
            mode=['save'],
            save_dir=str(temp_dir),
            animation_path=os.path.join(temp_dir, "gif.gif"),
        )
        if save_dir != tmpdir:
            baseline_dir = os.path.join("tests/baseline", "plot_2x1")
            assert compare_dirs(
                str(temp_dir), baseline_dir), "Content mismatch in plot_2x1"

    def test_plot_1x2(self, setup_simulation, save_dir, tmpdir):
        plotter = Plotter.create_plotter(setup_simulation)
        plotter.regroup(num_intervals=3, interval_size=1)
        plot = ["Scatter",
                {
                    "parameters": ["Opinion", "Neighbor mean opinion"],
                    "scale": ["Tanh", "Tanh"],
                },]
        plotter.add_plot(
            *plot,
            row=0,
            col=0,
        )
        plotter.add_plot(
            *plot,
            row=1,
            col=0,
        )

        temp_dir = os.path.join(save_dir, "plot_1x2")
        os.makedirs(temp_dir, exist_ok=True)
        plotter.plot_dynamics(
            mode=['save'],
            save_dir=str(temp_dir),
            animation_path=os.path.join(temp_dir, "gif.gif"),
        )
        if save_dir != tmpdir:
            baseline_dir = os.path.join("tests/baseline", "plot_1x2")
            assert compare_dirs(
                str(temp_dir), baseline_dir), "Content mismatch in plot_1x2"

    def test_plot_2x2(self, setup_simulation, save_dir, tmpdir):
        plotter = Plotter.create_plotter(setup_simulation)
        plotter.regroup(num_intervals=3, interval_size=1)
        plot = ["Scatter",
                {
                    "parameters": ["Opinion", "Neighbor mean opinion"],
                    "scale": ["Tanh", "Tanh"],
                },]
        plotter.add_plot(
            *plot,
            row=0,
            col=0,
        )
        plotter.add_plot(
            *plot,
            row=1,
            col=0,
        )
        plotter.add_plot(
            *plot,
            row=1,
            col=1,
        )
        plotter.add_plot(
            *plot,
            row=1,
            col=1,
        )

        temp_dir = os.path.join(save_dir, "plot_2x2")
        os.makedirs(temp_dir, exist_ok=True)
        plotter.plot_dynamics(
            mode=['save'],
            save_dir=str(temp_dir),
            animation_path=os.path.join(temp_dir, "gif.gif"),
        )
        if save_dir != tmpdir:
            baseline_dir = os.path.join("tests/baseline", "plot_2x2")
            assert compare_dirs(
                str(temp_dir), baseline_dir), "Content mismatch in plot_2x2"

    def test_plotter_info(self, setup_simulation):
        plotter = Plotter.create_plotter(setup_simulation)
        plotter.regroup(num_intervals=3, interval_size=1)
        plotter.add_plot(
            "Scatter",
            {
                "parameters": ["Opinion", "Neighbor mean opinion"],
                "scale": ["Tanh", "Tanh"],
            },
            row=0,
            col=0,
        )
        plotter.info()

    def test_multiple_runs_plot(self, setup_simulation, save_dir, tmpdir):
        I1 = Interface.create_interface(setup_simulation)
        I2 = Interface.create_interface(setup_simulation)
        plotter = Plotter([I1, I2])
        plotter.regroup(num_intervals=3, interval_size=1)
        plotter.add_plot(
            "Scatter",
            {
                "parameters": ["Opinion", "Neighbor mean opinion"],
                "scale": ["Tanh", "Tanh"],
            },
            row=0,
            col=0,
        )
        plotter.info()
        temp_dir = os.path.join(save_dir, "multiple_runs_plot")
        os.makedirs(temp_dir, exist_ok=True)
        plotter.plot_dynamics(
            mode=['save'],
            save_dir=str(temp_dir),
            animation_path=os.path.join(temp_dir, "gif.gif"),
        )
        if save_dir != tmpdir:
            baseline_dir = os.path.join("tests/baseline", "multiple_runs_plot")
            assert compare_dirs(
                str(temp_dir), baseline_dir), "Content mismatch in multiple_runs_plot"
