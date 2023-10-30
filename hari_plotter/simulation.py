import toml

from .hari_dynamics import HariDynamics
from .model import Model

import pathlib
import re
import os


class Simulation:
    def __init__(self, model, network, max_iterations=None,
                 dynamics: HariDynamics = None, rng_seed=None):
        self.dynamics = dynamics
        self.model = model
        self.rng_seed = rng_seed
        self.max_iterations = max_iterations
        self.network = network

    @classmethod
    def from_toml(cls, filename):
        data = toml.load(filename)

        model_type = data.get("simulation", {}).get("model")
        model_params = data.get(model_type, {})
        model = Model(model_type, model_params)
        rng_seed = data.get("simulation", {}).get("rng_seed", None)
        max_iterations = data.get("model", {}).get("max_iterations", None)
        network = data.get("network", {})

        # Checking if the required keys are present in the data
        if not model:
            raise ValueError(
                "Invalid TOML format for Simulation initialization.")

        return cls(model=model, rng_seed=rng_seed,
                   max_iterations=max_iterations, network=network, dynamics=None)

    @classmethod
    def from_dir(cls, datadir):
        datadir = pathlib.Path(datadir)
        data = toml.load(str(datadir / 'conf.toml'))

        model_type = data.get("simulation", {}).get("model")
        model_params = data.get(model_type, {})
        model = Model(model_type, model_params)
        rng_seed = data.get("simulation", {}).get("rng_seed", None)
        max_iterations = data.get("model", {}).get("max_iterations", None)
        network = data.get("network", {})

        # Checking if the required keys are present in the data
        if not model:
            raise ValueError(
                "Invalid TOML format for Simulation initialization.")

        n_max = max([int(re.search(r'opinions_(\d+).txt', f).group(1))
                    for f in os.listdir(datadir) if re.search(r'opinions_\d+.txt', f)])

        opinion = [str(datadir / f'opinions_{i}.txt')
                   for i in range(n_max + 1)]

        single_network_file = datadir / 'network.txt'

        if single_network_file.exists():
            # If the single 'network.txt' file exists, use it.
            network = [str(single_network_file)]
        else:
            network = [str(datadir / f'network_{i}.txt')
                       for i in range(n_max + 1)]
        HD = HariDynamics.read_network(network, opinion)

        return cls(model=model, rng_seed=rng_seed,
                   max_iterations=max_iterations, network=network, dynamics=HD)

    def to_toml(self, filename):
        data = {
            "simulation": {
                "model": self.model.model_type,
                "rng_seed": self.rng_seed
            },
            "model": {
                "max_iterations": self.max_iterations
            },
            self.model.model_type: self.model.params,
            "network": self.network
        }
        with open(filename, 'w') as f:
            toml.dump(data, f)

    def __repr__(self):
        return f"Simulation(model={self.model}, rng_seed={self.rng_seed}, max_iterations={self.max_iterations}, network={self.network})"
