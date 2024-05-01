import __future__

import os
import pathlib
import re
import subprocess
from typing import Any, Dict, Optional, Union

import pathlib

import toml

from .dynamics import Dynamics
from .lazy_graph import LazyGraph
from .model import Model, ModelFactory

import copy


class Simulation:
    """
    A class representing a simulation. 
    Provides methods for initializing from TOML configurations and directories.
    """

    def __init__(self,
                 model: Model,
                 parameters: Dict[str, Any],
                 dynamics: Optional[Dynamics | None] = None,):
        """
        Initialize a Simulation instance.

        Parameters:
            model: A Model instance used in the simulation.
            network: Network configuration for the simulation.
            max_iterations: Maximum number of iterations for the simulation.
            dynamics: HariDynamics instance used for the simulation. Default is None.
            rng_seed: Seed for random number generation. Default is None.
        """
        self.model: Model = model
        self.parameters: Dict[str, Any] = parameters
        self.dynamics: Dynamics | None = dynamics

    def run(self, path_to_seldon, directory, n_output_network: int | None = None,  start_numbering_from: int | None = None) -> bool:
        """
        Run the simulation.
        """
        updates = {}
        if n_output_network is not None:
            if 'io' not in updates:
                updates['io'] = {}
            updates['io']['n_output_network'] = n_output_network
        if start_numbering_from is not None:
            if 'io' not in updates:
                updates['io'] = {}
            updates['io']['start_numbering_from'] = start_numbering_from

        try:
            path_to_executable = pathlib.Path(path_to_seldon) / 'build/seldon'
            directory = pathlib.Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            self.to_toml(str(directory / "conf.toml"),
                         updates=updates)

            result = subprocess.run([str(path_to_executable),
                                    str(directory /
                                        "conf.toml"), '-o', str(directory)
                                     ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

            # Output handling
            stdout_output = result.stdout.decode()
            stderr_output = result.stderr.decode()

            if stdout_output:
                print(f'Standard Output:\n{stdout_output}')
            if stderr_output:
                print(f'Standard Error:\n{stderr_output}')
                return False  # Assuming any stderr output indicates a failure

            return True  # Success if no stderr output

        except subprocess.CalledProcessError as e:
            # If subprocess.run fails, print the error and return False
            print(f'An error occurred while running the simulation: {e}')
            print(f'Standard Output:\n{e.stdout.decode()}')
            print(f'Standard Error:\n{e.stderr.decode()}')
            return False

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
        if not model_type:
            raise ValueError(
                "Invalid TOML format for Simulation initialization.")
        model_params = data.get(model_type, {})
        model = ModelFactory.create_model(model_type, model_params)

        # Checking if the required keys are present in the data
        if not model:
            raise ValueError(
                "Invalid TOML format for Simulation initialization.")

        data.pop(model_type)
        data["simulation"].pop("model")

        return cls(model=model, parameters=data, dynamics=None)

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
        if not model_type:
            raise ValueError(
                "Invalid TOML format for Simulation initialization.")
        model_params = data.get(model_type, {})
        model = ModelFactory.create_model(model_type, model_params)

        # Checking if the required keys are present in the data
        if not model:
            raise ValueError(
                "Invalid TOML format for Simulation initialization.")

        data.pop(model_type)
        data["simulation"].pop("model")

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

        return cls(model=model, parameters=data, dynamics=HD)

    def parameters_with_model(self, updates: dict | None = None):
        """
        Return a dictionary containing the model parameters and simulation parameters.
        """
        # Make a deep copy of self.parameters
        data = copy.deepcopy(self.parameters)
        model_type = self.model.model_type

        # Ensure the 'simulation' key exists in the dictionary
        if 'simulation' not in data:
            data['simulation'] = {}

        data["simulation"]['model'] = model_type
        data[model_type] = self.model.params

        data = Simulation.update_nested_dict(data, updates)

        return data

    def to_toml(self, filename: str, updates: dict | None = None) -> None:
        """
        Serialize the Simulation instance to a TOML file.

        Parameters:
            filename: Path to the TOML file where the simulation configuration will be saved.
        """
        data = self.parameters_with_model(updates=updates)

        with open(filename, 'w') as f:
            toml.dump(data, f)

    def group(self, num_intervals: int, interval_size: int = 1, offset: int = 0):
        self.dynamics.group(num_intervals, interval_size, offset)

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
        return f"Simulation(model={self.model}, parameters={self.parameters}, dynamics={self.dynamics})"

    def __str__(self) -> str:
        """
        Return a fancy string representation of the Simulation instance.

        Returns:
            Fancy string representation of the Simulation.
        """

        data = self.parameters_with_model()

        model_str = f"Model:\n    Type: {self.model.model_type} ({type(self.model).__name__})\n    Parameters: {self.model.params}"
        # Create a new dictionary that merges self.parameters and {'model': self.model.model_type}

        parameters_str = "Parameters:\n" + \
            "\n".join([f"    {k}: {v}" for k,
                       v in data.items()])
        dynamics_str = f"Dynamics: {self.dynamics if self.dynamics is not None else 'Not specified'}"

        return f"Simulation Instance\n-------------------\n{model_str}\n\n{parameters_str}\n\n{dynamics_str}"

    @staticmethod
    def update_nested_dict(original: dict, updates: dict | None):
        """
        Recursively update the dictionary 'original' with values from 'updates'.
        If a key in 'updates' refers to a dictionary, dive deeper.
        Otherwise, simply update the value.
        """
        if updates:
            for key, value in updates.items():
                if isinstance(value, dict) and key in original:
                    original[key] = Simulation.update_nested_dict(
                        original.get(key, {}), value)
                else:
                    original[key] = value
        return original
