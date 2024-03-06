import matplotlib.pyplot as plt

from hari_plotter import Plotter
from hari_plotter.interface import Interface


class QtPlotter(Plotter):
    def __init__(self, interface: Interface, figsize=None, figure=None):
        super().__init__(interface, figsize)
        self.figure = figure
        # self.current_group = current_group

    def create_fig_and_axs(self):
        if not self.figure:
            self.figure = plt.figure(figsize=self.figsize)
        fig = self.figure
        fig.clf()  # Clear the figure to prevent overlaying of plots
        axs = fig.subplots(self
                           .num_rows, self.num_cols, gridspec_kw={
                               'width_ratios': self.size_ratios[0], 'height_ratios': self.size_ratios[1]})
        # Ensure axs is a 2D array for consistency
        if self.num_rows == 1 and self.num_cols == 1:
            axs = [[axs]]  # Single plot
        elif self.num_rows == 1:
            axs = [axs]  # Single row, multiple columns
        elif self.num_cols == 1:
            axs = [[ax] for ax in axs]  # Multiple rows, single column

        return fig, axs

    # def update_plot(self):
    #     self.plot(self.current_group)
