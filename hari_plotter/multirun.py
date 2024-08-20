from .simulation import Simulation


class Multirun:
    def __init__(self, simulations: list[Simulation]):
        self.simulations = simulations

    @classmethod
    def from_dirs(cls, dirs: list[str]):
        return cls([Simulation.from_dir(d) for d in dirs])

    def join(self, other: 'Multirun'):
        assert len(self) == 0 or len(self.simulations) == len(
            other.simulations), "Multiruns must have the same length"
        self.simulations.extend(other.simulations)

    def append(self, simulation: Simulation):
        assert isinstance(
            simulation, Simulation), "Can only append Simulation objects"
        assert len(self) == 0 or len(self.simulations[0]) == len(
            simulation), "Simulations must have the same length"
        self.simulations.append(simulation)

    def merge(self) -> Simulation:
        return Simulation.merge(self.simulations)

    @property
    def available_parameters(self) -> list[str]:
        if len(self.simulations) == 0:
            return []
        return self.simulations[0].dynamics[0].gatherer.node_parameters

    def __len__(self):
        if len(self.simulations) == 0:
            return 0
        return len(self.simulations[0])
