from .hari_graph import HariGraph
from .lazy_hari_graph import LazyHariGraph


class HariDynamics:
    def __init__(self):
        self.lazy_hari_graphs = []  # Initialize an empty list to hold LazyHariGraph objects

    @classmethod
    def read_network(cls, network_file, opinion_files):
        """
        Reads the network file and a list of opinion files to create LazyHariGraph objects
        and appends them to the lazy_hari_graphs list of a HariDynamics instance.

        Parameters:
            network_file (str): The path to the network file.
            opinion_files (List[str]): A list of paths to the opinion files.

        Returns:
            HariDynamics: An instance of HariDynamics with lazy_hari_graphs populated.
        """
        # Create an instance of HariDynamics
        dynamics_instance = cls()

        for opinion_file in opinion_files:
            # Append LazyHariGraph objects to the list, with the class method and parameters needed
            # to create the actual HariGraph instances when they are required.
            dynamics_instance.lazy_hari_graphs.append(
                LazyHariGraph(HariGraph.read_network,
                              network_file, opinion_file)
            )

        return dynamics_instance
    
    def __getattr__(self, name):
        # Try to get the attribute from the first LazyHariGraph object in the list.
        # If it exists, assume it exists on all HariGraph instances in the list.
        if self.lazy_hari_graphs:
            try:
                attr = getattr(self.lazy_hari_graphs[0], name)
            except AttributeError:
                pass  # Handle below
            else:
                if callable(attr):
                    # If the attribute is callable, return a function that calls it on all HariGraph instances.
                    def forwarded(*args, **kwargs):
                        return [getattr(lazy_graph, name)(*args, **kwargs) for lazy_graph in self.lazy_hari_graphs]
                    return forwarded
                else:
                    # If the attribute is not callable, return a list of its values from all HariGraph instances.
                    return [getattr(lazy_graph, name) for lazy_graph in self.lazy_hari_graphs]
        
        # If the attribute does not exist on HariGraph instances, raise an AttributeError.
        raise AttributeError(f"'HariDynamics' object and its 'HariGraph' instances have no attribute '{name}'")
