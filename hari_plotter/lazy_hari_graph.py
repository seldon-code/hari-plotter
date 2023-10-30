import warnings

from .hari_graph import HariGraph


class LazyHariGraph:
    def __init__(self, classmethod, *args, **kwargs):
        # The actual HariGraph instance (initialized later)
        self._hari_graph: HariGraph = None
        # The class method used to create the HariGraph instance
        self._classmethod = classmethod
        self._args = args  # Positional arguments to pass to the class method
        self._kwargs = kwargs  # Keyword arguments to pass to the class method
        self._mapping = None

    def _initialize(self):
        if self._hari_graph is None:
            self._hari_graph = self._classmethod(*self._args, **self._kwargs)
            if self._mapping and not (self._is_trivial_mapping(self._mapping)):
                self._hari_graph.merge_clusters(self._mapping)

    def uninitialize(self):
        self._hari_graph = None

    def reinitialize(self):
        self._hari_graph = self._classmethod(*self._args, **self._kwargs)
        if self._mapping and not (self._is_trivial_mapping(self._mapping)):
            self._hari_graph.merge_clusters(self._mapping)

    @property
    def is_initialized(self):
        return self._hari_graph is not None

    @property
    def mapping(self):
        return self._mapping

    @mapping.setter
    def mapping(self, new_mapping):
        """
        To be replaced with something better handling reinitialize
        """
        self._mapping = new_mapping
        if self.is_initialized:
            self.reinitialize()

    def _is_trivial_mapping(self, mapping):
        return all(k == v for k, v in mapping.items())

    @property
    def is_correctly_mapped(self):
        real_mapping = self._hari_graph.get_cluster_mapping()
        return self._mapping != real_mapping

    def get_graph(self):
        self._initialize()
        return self._hari_graph

    def __getattr__(self, name):
        self._initialize()
        return getattr(self._hari_graph, name)

    def __getitem__(self, n):
        self._initialize()
        return self._hari_graph.__getitem__(n)

    def __str__(self):
        if self._hari_graph is not None:
            return f"<LazyHariGraph with {self._hari_graph.number_of_nodes()} nodes and {self._hari_graph.number_of_edges()} edges>"
        else:
            return f"<Uninitialized LazyHariGraph object at {id(self)}>"
