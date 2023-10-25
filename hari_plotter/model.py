import toml


class Model:
    def __init__(self, model_type, params):
        self.model_type = model_type
        self.params = params

    @classmethod
    def from_dict(cls, dictionary):
        model_type = dictionary.get('model_type')
        params = dictionary.get('params')
        if model_type is None or params is None:
            raise ValueError(
                "Dictionary must contain 'model_type' and 'params' keys.")
        return cls(model_type, params)

    @classmethod
    def read(cls, filename):
        # Read the toml file
        data = toml.load(filename)

        # Extract model type and parameters
        model_type = data.get("simulation", {}).get("model")
        # Assuming the model's parameters are in a table named after the model
        params = data.get(model_type, {})

        if not model_type or not params:
            raise ValueError("Invalid TOML format for Model initialization.")

        return cls(model_type, params)

    def __repr__(self):
        return f"Model(model_type={self.model_type}, params={self.params})"
