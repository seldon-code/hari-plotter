import math
import os
import shutil
import tempfile
from typing import List, Optional, Union

import imageio
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .interface import Interface


class PlotSaver:
    def __init__(self, mode='show', save_path=None, gif_path=None):
        """
        :param mode: a list or a single string, e.g. ['show', 'save'] or 'gif'
        :param save_path: path to save individual plots (used if 'save' is in mode)
        :param gif_path: path to save gif (used if 'gif' is in mode)
        """
        # Ensure mode is a list even if a single mode string is provided
        self.mode = mode if isinstance(mode, list) else [mode]
        self.save_path = save_path
        self.gif_path = gif_path
        self.saved_images = []
        self.temp_dir = None

    @staticmethod
    def is_inside_jupyter():
        try:
            get_ipython
            return True
        except NameError:
            return False

    def __enter__(self):
        return self

    def save(self, fig):
        if 'save' in self.mode and self.save_path:
            path = self.save_path.format(len(self.saved_images))
            fig.savefig(path)
            self.saved_images.append(path)
        elif 'gif' in self.mode and not self.save_path:  # Only gif mode selected
            if not self.temp_dir:
                self.temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(
                self.temp_dir, "tmp_plot_{}.png".format(len(self.saved_images)))
            fig.savefig(temp_path)
            self.saved_images.append(temp_path)

        if 'show' in self.mode:
            if self.is_inside_jupyter():
                # In Jupyter, just let the figure be displayed automatically
                display(fig)
            else:
                # Outside Jupyter, use fig.show() to display the figure
                fig.show()

        # Close the figure after processing
        plt.close(fig)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'gif' in self.mode and self.gif_path and self.saved_images:
            with imageio.get_writer(self.gif_path, mode='I') as writer:
                for img_path in self.saved_images:
                    image = imageio.imread(img_path)
                    writer.append_data(image)

        # Cleanup temporary directory if it was used
        if self.temp_dir:
            shutil.rmtree(self.temp_dir)


def draw(interface: Interface, mode='show', save_dir: str = None, gif_path: str = None,
         pos=None, node_info_mode='none', use_node_color=True,
         use_edge_thickness=True, show_edge_influences=False,
         node_size_multiplier=200,
         arrowhead_length=0.2, arrowhead_width=0.2,
         min_line_width=0.1, max_line_width=3.0,
         seed=None,
         show_time: bool = False,
         fig=None, ax=None):
    """
    Draws a graphical representation of the first graph in each group of the given interface. The positions 
    of the nodes are determined by the graph at the reference_index.

    Parameters:
    -----------
    interface : Interface
        The interface containing the graphs to be drawn.

    mode : str, optional (default='show')
        The mode of displaying the graph. If 'show', the graphs are displayed. Other modes may be supported
        depending on the context.

    save_dir : str, optional
        The directory where the drawn graphs will be saved. If specified, each graph is saved with a filename '{i}.png'.

    gif_path : str, optional
        Path to save the animation as a gif, if desired.

    pos : dict, optional
        Position of nodes as a dictionary of coordinates. If not provided, the spring layout is used to position nodes.

    node_info_mode : str, optional (default='none')
        Mode to determine which node information to display. Options include:
        - 'opinions': Display node opinions.
        - 'ids': Display node IDs.
        - 'labels': Display node labels.
        - 'size': Display size of the node based on labels.

    use_node_color : bool, optional (default=True)
        Whether to use colors for nodes based on their attributes.

    use_edge_thickness : bool, optional (default=True)
        Whether to use edge thickness based on their influence attributes.

    show_edge_influences : bool, optional (default=False)
        If True, displays the influence values on the edges.

    node_size_multiplier : int, optional (default=200)
        Multiplier for node sizes to enhance visibility.

    arrowhead_length : float, optional (default=0.2)
        Length of the arrowhead.

    arrowhead_width : float, optional (default=0.2)
        Width of the arrowhead.

    min_line_width : float, optional (default=0.1)
        Minimum line width for edges.

    max_line_width : float, optional (default=3.0)
        Maximum line width for edges.

    seed : int or None, optional
        Seed for reproducibility in node positioning.

    show_time : bool, optional (default=False)
        If True, the timestamp of each graph is displayed in the bottom right corner of the plot.

    fig : Matplotlib figure object, optional
        If provided, the graph is drawn on this figure.

    ax : Matplotlib axis object, optional
        If provided, the graph is drawn on this axis.

    Returns:
    --------
    None

    """

    with PlotSaver(mode=mode, save_path=f"{save_dir}/"+"plot_{}.png", gif_path=gif_path) as saver:
        for image in interface.images:
            if fig is None or ax is None:
                fig, ax = plt.subplots(figsize=(10, 7))

            if pos is None:
                pos = image.position_nodes(seed=seed)

            node_attributes = nx.get_node_attributes(image, 'opinion')
            edge_attributes = nx.get_edge_attributes(image, 'influence')

            # Prepare Node Labels
            node_labels = {}
            match node_info_mode:
                case 'opinions':
                    node_labels = {node: f"{opinion:.2f}" for node,
                                   opinion in node_attributes.items()}
                case 'ids':
                    node_labels = {node: f"{node}" for node in image.nodes}
                case 'labels':
                    for node in image.nodes:
                        label = image.nodes[node].get('label', None)
                        if label is not None:
                            node_labels[node] = ','.join(map(str, label))
                        else:  # If label is not defined, show id instead
                            node_labels[node] = str(node)
                case 'size':
                    for node in image.nodes:
                        label_len = len(image.nodes[node].get('label', [node]))
                        node_labels[node] = str(label_len)

            # Prepare Node Colors
            if use_node_color:
                node_colors = [cm.bwr(opinion)
                               for opinion in node_attributes.values()]
            else:
                node_colors = 'lightblue'

            # Prepare Edge Widths
            if use_edge_thickness:

                # Gather edge weights
                edge_weights = list(edge_attributes.values())

                # Scale edge weights non-linearly
                # or np.log1p(edge_weights) for logarithmic scaling
                scaled_weights = np.sqrt(edge_weights)

                # Normalize scaled weights to range [min_line_width,
                # max_line_width]
                max_scaled_weight = max(scaled_weights)
                min_scaled_weight = min(scaled_weights)

                edge_widths = [
                    min_line_width + (max_line_width - min_line_width) * (weight -
                                                                          min_scaled_weight) / (max_scaled_weight - min_scaled_weight)
                    for weight in scaled_weights
                ]

            else:
                # Default line width applied to all edges
                edge_widths = [1.0] * image.number_of_edges()

            # Prepare Edge Labels
            edge_labels = None
            if show_edge_influences:
                edge_labels = {(u, v): f"{influence:.2f}" for (u, v),
                               influence in edge_attributes.items()}

            # Calculate Node Sizes
            node_sizes = []
            for node in image.nodes:
                label_len = len(image.nodes[node].get('label', [node]))
                size = node_size_multiplier * \
                    math.sqrt(label_len)  # Nonlinear scaling
                node_sizes.append(size)

            # Draw Nodes and Edges
            nx.draw_networkx_nodes(
                image, pos, node_color=node_colors, node_size=node_sizes, ax=ax)

            for (u, v), width in zip(image.edges(), edge_widths):
                # Here, node_v_size and node_u_size represent the sizes (or the
                # "radii") of the nodes.
                node_v_size = node_sizes[list(image.nodes).index(v)]
                node_u_size = node_sizes[list(image.nodes).index(u)]

                # Adjust the margins based on node sizes to avoid collision with
                # arrowheads and to avoid unnecessary gaps.
                target_margin = 5 * node_v_size / node_size_multiplier
                source_margin = 5 * node_u_size / node_size_multiplier

                if image.has_edge(v, u):
                    nx.draw_networkx_edges(image, pos, edgelist=[(u, v)], width=width, connectionstyle='arc3,rad=0.3',
                                           arrowstyle=f'->,head_length={arrowhead_length},head_width={arrowhead_width}', min_target_margin=target_margin, min_source_margin=source_margin)
                else:
                    nx.draw_networkx_edges(image, pos, edgelist=[(
                        u, v)], width=width, arrowstyle=f'-|>,head_length={arrowhead_length},head_width={arrowhead_width}', min_target_margin=target_margin, min_source_margin=source_margin)

            # Draw Labels
            nx.draw_networkx_labels(image, pos, labels=node_labels)
            if edge_labels:
                nx.draw_networkx_edge_labels(
                    image, pos, edge_labels=edge_labels)
            if show_time:
                ax.set_title(f't = {image.time}')

            saver.save(fig)
