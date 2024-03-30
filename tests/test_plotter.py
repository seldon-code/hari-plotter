import networkx as nx
import numpy as np
import pytest

from hari_plotter import Graph, Interface, Plotter, Simulation, ColorScheme
from hari_plotter.color_scheme import initialize_colormap
from hari_plotter.lazy_graph import LazyGraph

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
    @classmethod
    def setup_class(cls):
        ColorScheme.default_colormap_name, ColorScheme.default_colormap = initialize_colormap(
            {
                "Name": "custom1",
                "Colors": [
                    # (223 / 255, 294 / 255, 255 / 255),
                    (255 / 255, 79 / 255, 20 / 255),
                    (1 / 255, 180 / 255, 155 / 255),
                    (71 / 255, 71 / 255, 240 / 255),
                    # (250 / 255, 200 / 255, 180 / 255),
                ],
            }
        )
        cls.S = Simulation.from_dir("tests/big_test")

    @pytest.mark.parametrize("plot_index, plot", [(i, plot) for i, plot in enumerate(plot_list)])
    def test_plot_individual(self, plot_index, plot, tmpdir):
        plotter = Plotter.create_plotter(self.S)
        plotter.regroup(num_intervals=3, interval_size=1)
        plotter.add_plot(
            *plot,
            row=0,
            col=0,
        )
        # Create a unique directory for each plot
        temp_dir = tmpdir.mkdir(f"plot_{plot_index}")
        plotter.plot_dynamics(
            mode=[],
            save_dir=str(temp_dir),  # Use the pathlib.Path compatible temp_dir
            animation_path=temp_dir.join(
                f"gif_{plot_index}.gif"),  # Construct the file path
        )

    def test_plot_2x1(self, tmpdir):
        plotter = Plotter.create_plotter(self.S)
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

        plotter.plot_dynamics(
            mode=[],
            save_dir=f"{tmpdir}",
            animation_path=f"{tmpdir}/gif.gif",
        )


    def test_plot_1x2(self, tmpdir):
        plotter = Plotter.create_plotter(self.S)
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

        plotter.plot_dynamics(
            mode=[],
            save_dir=f"{tmpdir}",
            animation_path=f"{tmpdir}/gif.gif",
        )

    def test_plot_2x2(self, tmpdir):
        plotter = Plotter.create_plotter(self.S)
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

        plotter.plot_dynamics(
            mode=[],
            save_dir=f"{tmpdir}",
            animation_path=f"{tmpdir}/gif.gif",
        )

    def test_plotter_info(self):
        plotter = Plotter.create_plotter(self.S)
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

    def test_multiple_runs_plot(self, tmpdir):
        I1 = Interface.create_interface(self.S)
        I2 = Interface.create_interface(self.S)
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
        plotter.plot_dynamics(
            mode=[],
            save_dir=f"{tmpdir}",
            animation_path=f"{tmpdir}/gif.gif",
        )
