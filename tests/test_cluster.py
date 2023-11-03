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

    def test_KMeansCluster_from_H(self):
        for data in self.H_interface.group_values_iterator(['opinion', 'neighbor_mean_opinion']):
            H_cluster = Cluster.create_cluster('KMeansCluster', data)
            assert isinstance(H_cluster, Cluster)

        for data in self.H_interface.group_values_iterator(['opinion', 'neighbor_mean_opinion']):
            H_cluster = Cluster.create_cluster('KMeansCluster', data, scale={
                                               'opinion': 'tanh', 'neighbor_mean_opinion': 'tanh'})
            assert isinstance(H_cluster, Cluster)

    def test_FuzzyCMeanCluster_from_H(self):
        for data in self.H_interface.group_values_iterator(['opinion', 'neighbor_mean_opinion']):
            H_cluster = Cluster.create_cluster('FuzzyCMeanCluster', data)
            assert isinstance(H_cluster, Cluster)

        for data in self.H_interface.group_values_iterator(['opinion', 'neighbor_mean_opinion']):
            H_cluster = Cluster.create_cluster('FuzzyCMeanCluster', data, scale={
                                               'opinion': 'tanh', 'neighbor_mean_opinion': 'tanh'})
            assert isinstance(H_cluster, Cluster)
