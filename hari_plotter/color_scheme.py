import matplotlib.pyplot as plt
from .interface import Interface


class ColorScheme:
    _methods = {}

    def __init__(self, interface: Interface = None) -> None:
        self.interface = interface
        self.scatter_default_marker = 'o'
        self.scatter_default_color = 'blue'
        self.color_palette = plt.cm.coolwarm

    @classmethod
    def method(cls, property_name):
        """
        Class method decorator to register methods that provide various properties of nodes.

        Parameters:
        -----------
        property_name : str
            The name of the property that the method provides.

        Raises:
        -------
        ValueError: 
            If the property name is already defined.
        """
        def decorator(method):
            if property_name in cls._methods:
                raise ValueError(
                    f"Property {property_name} is already defined.")
            cls._methods[property_name] = method
            return method
        return decorator

    @property
    def methods(self) -> list:
        """Returns a list of property names that have been registered."""
        return list(self._methods.keys())

    def scatter_marker_nodes(self, parameter=None, scale='Linear'):
        return self.scatter_default_marker

    def scatter_colors_nodes(self, parameter=None, scale='Linear'):
        return {node: self.scatter_default_color for node in self.interface.nodes}

    def copy(self):
        return ColorScheme(self.interface)

    def apply_changes(self, changes):
        pass

    def variation(self, changes):
        copy = self.copy()
        copy.apply_changes(changes)
        return copy
