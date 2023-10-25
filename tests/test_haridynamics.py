from hari_plotter import HariDynamics
import numpy as np


class TestHariDynamics:

    @classmethod
    def setup_class(cls):
        cls.HD = HariDynamics.read_network(
            'tests/5_ring/network.txt', [f'tests/5_ring/opinions_{i}.txt' for i in range(3)])

    def test_initialize(self):
        self.HD.mean_opinion
        assert all(H.is_initialized() for H in self.HD)

    def test_draw_dynamic_graphs(self):
        self.HD.draw_dynamic_graphs(show=False, show_timestamp=True)

    def test_plot_opinions(self):
        self.HD.plot_opinions(reference_index=0, show=False)

    def test_plot_neighbor_mean_opinion(self):
        self.HD.plot_neighbor_mean_opinion(show=False, show_timestamp=True)
