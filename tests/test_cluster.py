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

    def test_from_H(self):
        for clusterization_method in Cluster.clusterization_methods.keys():
            for data in self.H_interface.group_values_iterator(['opinion', 'neighbor_mean_opinion']):
                H_cluster = Cluster.create_cluster(clusterization_method, data)
                assert isinstance(H_cluster, Cluster)
                H_cluster = Cluster.create_cluster(clusterization_method, data, scale={
                    'opinion': 'tanh', 'neighbor_mean_opinion': 'tanh'})
                assert isinstance(H_cluster, Cluster)

    def test_get_number_of_clusters(self):
        for clusterization_method in Cluster.clusterization_methods.keys():
            for data in self.H_interface.group_values_iterator(['opinion', 'neighbor_mean_opinion']):
                H_cluster = Cluster.create_cluster(clusterization_method, data)
                H_cluster.get_number_of_clusters()
                H_cluster = Cluster.create_cluster(clusterization_method, data, scale={
                    'opinion': 'tanh', 'neighbor_mean_opinion': 'tanh'})
                H_cluster.get_number_of_clusters()

    def test_predict_cluster(self):
        for clusterization_method in Cluster.clusterization_methods.keys():
            for data in self.H_interface.group_values_iterator(['opinion', 'neighbor_mean_opinion']):
                H_cluster = Cluster.create_cluster(clusterization_method, data)
                H_cluster.predict_cluster([0., 0.])
                H_cluster = Cluster.create_cluster(clusterization_method, data, scale={
                    'opinion': 'tanh', 'neighbor_mean_opinion': 'tanh'})
                H_cluster.predict_cluster([0., 0.])

    def test_degree_of_membership(self):
        for clusterization_method in Cluster.clusterization_methods.keys():
            for data in self.H_interface.group_values_iterator(['opinion', 'neighbor_mean_opinion']):
                H_cluster = Cluster.create_cluster(clusterization_method, data)
                H_cluster.degree_of_membership([0., 0.])
                H_cluster = Cluster.create_cluster(clusterization_method, data, scale={
                    'opinion': 'tanh', 'neighbor_mean_opinion': 'tanh'})
                H_cluster.degree_of_membership([0., 0.])
