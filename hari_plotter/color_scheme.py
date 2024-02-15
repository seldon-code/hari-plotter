import matplotlib.pyplot as plt
from .interface import Interface


class ColorScheme:
    def __init__(self, interface: Interface = None) -> None:
        self.interface = interface
        self.scatter_default_marker = 'o'
        self.scatter_default_color = 'blue'
        self.color_palette = plt.cm.coolwarm

    def copy(self):
        return ColorScheme(self.interface)

    def apply_changes(self, changes):
        pass

    def variation(self, changes):
        copy = self.copy()
        copy.apply_changes(changes)
        return copy
