import toml
from .model import Model


class Simulation:
    def __init__(self, model, max_iterations, network, rng_seed=None):
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
        max_iterations = data.get("model", {}).get("max_iterations")
        network = data.get("network", {})

        # Checking if the required keys are present in the data
        if not all([model, max_iterations]):
            raise ValueError(
                "Invalid TOML format for Simulation initialization.")

        return cls(model, rng_seed, max_iterations, network)

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
