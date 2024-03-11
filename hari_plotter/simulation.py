import __future__

import os
import pathlib
import re
from typing import Any, Dict, Optional, Union

import toml

from .dynamics import Dynamics
from .lazy_graph import LazyGraph
from .model import Model, ModelFactory


class Simulation:
    """
    A class representing a simulation. 
    Provides methods for initializing from TOML configurations and directories.
    """

    def __init__(self,
                 model: Model,
                 network: Dict[str, Any],
                 max_iterations: Optional[int] = None,
                 dynamics: Optional[Dynamics] = None,
                 rng_seed: Optional[int] = None):
        """
        Initialize a Simulation instance.

        Parameters:
            model: A Model instance used in the simulation.
            network: Network configuration for the simulation.
            max_iterations: Maximum number of iterations for the simulation.
            dynamics: HariDynamics instance used for the simulation. Default is None.
            rng_seed: Seed for random number generation. Default is None.
        """
        self.dynamics: Dynamics = dynamics
        self.model: Model = model
        self.rng_seed: int = rng_seed
        self.max_iterations: int = max_iterations
        self.network: dict = network

    @classmethod
    def from_toml(cls, filename: str) -> 'Simulation':
        """
        Load simulation parameters from a TOML file and instantiate a Simulation instance.

        Parameters:
            filename: Path to the TOML file containing simulation configuration.

        Returns:
            Instance of Simulation based on the TOML file.
        """
        data = toml.load(filename)
        model_type = data.get("simulation", {}).get("model")
        model_params = data.get(model_type, {})
        model = ModelFactory.create_model(model_type, model_params)
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
    def from_dir(cls, datadir: Union[str, pathlib.Path]) -> 'Simulation':
        """
        Load simulation parameters from a directory containing configuration and data.

        Parameters:
            datadir: Path to the directory containing simulation data and configuration.

        Returns:
            Instance of Simulation based on the data in the directory.
        """
        datadir = pathlib.Path(datadir)
        data = toml.load(str(datadir / 'conf.toml'))

        model_type = data.get("simulation", {}).get("model")
        model_params = data.get(model_type, {})
        model = ModelFactory.create_model(model_type, model_params)
        rng_seed = data.get("simulation", {}).get("rng_seed", None)
        max_iterations = data.get("model", {}).get("max_iterations", None)
        network = data.get("network", {})

        load_request = model.load_request

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
        HD = Dynamics.read_network(network, opinion, load_request=load_request)

        return cls(model=model, rng_seed=rng_seed,
                   max_iterations=max_iterations, network=network, dynamics=HD)

    def to_toml(self, filename: str) -> None:
        """
        Serialize the Simulation instance to a TOML file.

        Parameters:
            filename: Path to the TOML file where the simulation configuration will be saved.
        """
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

    def group(self, *args, **kwargs):
        self.dynamics.group(*args, **kwargs)

    @property
    def dt(self):
        return self.model.dt

    def __len__(self) -> int:
        return len(self.dynamics)

    def __getitem__(self, index) -> LazyGraph:
        return self.dynamics[index]

    def __repr__(self) -> str:
        """
        Return a string representation of the Simulation instance.

        Returns:
            String representation of the Simulation.
        """
        return f"Simulation(model={self.model}, rng_seed={self.rng_seed}, max_iterations={self.max_iterations}, network={self.network})"
