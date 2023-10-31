import numpy as np

from hari_plotter import HariDynamics


class TestHariDynamics:

    @classmethod
    def setup_class(cls):
        cls.HD = HariDynamics.read_network(
            'tests/5_ring/network.txt', [f'tests/5_ring/opinions_{i}.txt' for i in range(3)])

    def test_initialize(self):
        self.HD.mean_opinion
        assert all(H.is_initialized for H in self.HD)

    def test_getitem(self):
        self.HD[-1].mean_opinion
        assert self.HD[-1].is_initialized

    def test_group(self):
        self.HD.group(2, 2)

    def test_mapping(self):
        self.HD[0].merge_by_intervals([0.])
        self.HD.merge_nodes_by_index(0)
