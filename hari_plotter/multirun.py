from .simulation import Simulation


class Multirun:
    def __init__(self, simulations: list[Simulation]):
        self.simulations = simulations

    @classmethod
    def from_dirs(cls, dirs: list[str]):
        return cls([Simulation.from_dir(d) for d in dirs])

    def join(self, other: 'Multirun'):
        self.simulations.extend(other.simulations)

    def append(self, simulation: Simulation):
        self.simulations.append(simulation)

    def merge(self) -> Simulation:
        return Simulation.merge(self.simulations)
