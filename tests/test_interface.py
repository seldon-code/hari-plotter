import networkx as nx
import numpy as np
import pytest

from hari_plotter import HariGraph, Interface


class TestInterface:

    @classmethod
    def setup_class(cls):
        H = HariGraph.read_network(
            'tests/network.txt', 'tests/opinions_0.txt')
        cls.H_interface = Interface.create_interface(H)
        cls.cl = {
            "clustering_method": "Interval Clustering",
            "clustering_parameters": ("Opinion",),
            "scale": ("Linear",),
            "parameter_boundaries": ((0.0,),),
        }

    def test_from_H(self):
        assert isinstance(self.H_interface, Interface)

    def test_cluster_graph(self):
        self.H_interface.cluster_tracker.cluster_graph(self.cl)

    def test_get_unique_clusters(self):
        self.H_interface.cluster_tracker.get_unique_clusters(self.cl)

    def test_get_cluster_presence(self):
        self.H_interface.cluster_tracker.get_cluster_presence(self.cl)

    def test_get_final_value(self):
        self.H_interface.cluster_tracker.get_final_value(self.cl, 'Opinion')

    def test_available_parameters(self):
        self.H_interface.available_parameters
