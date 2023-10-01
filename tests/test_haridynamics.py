from hari_plotter import HariDynamics
import numpy as np


class TestHariDynamics:

    @classmethod
    def setup_class(cls):
        cls.graph = HariDynamics.read_network(
            'tests/5_ring/network.txt', [f'tests/5_ring/opinions_{i}.txt' for i in range(3)])

    def test_mean_value(self):
        mean_value = np.array(self.graph.weighted_mean_value)
        assert np.all(mean_value <= 1) and np.all(mean_value >= 0)
