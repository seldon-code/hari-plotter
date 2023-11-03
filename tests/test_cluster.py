import networkx as nx
import numpy as np
import pytest

from hari_plotter import Cluster, HariGraph, Interface


class TestCluster:

    @classmethod
    def setup_class(cls):
        H = HariGraph.read_network(
            'tests/network.txt', 'tests/opinions_0.txt')
        cls.H_interface = Interface.create_interface(H)
        for data in cls.H_interface.group_values_iterator(['opinion', 'neighbor_mean_opinion']):
            cls.H_cluster = Cluster.create_cluster('KMeansCluster', data)

    def test_from_H(self):
        assert isinstance(self.H_cluster, Cluster)
