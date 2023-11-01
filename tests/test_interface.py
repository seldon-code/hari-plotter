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

    def test_from_H(self):
        assert isinstance(self.H_interface, Interface)
