from hari_plotter import HariDynamics


class TestHariDynamics:

    @classmethod
    def setup_class(cls):
        cls.graph = HariDynamics.read_network('5_ring/network.txt', [f'5_ring/opinions_{i}.txt' for i in range(3)])