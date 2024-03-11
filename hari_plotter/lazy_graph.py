import warnings
from typing import Any, Callable, Dict, List, Tuple, Union

from .graph import Graph


class LazyGraph:
    """
    The `LazyHariGraph` class is a wrapper that lazily initializes and provides access to a `HariGraph`
    instance, allowing for delayed initialization and dynamic mapping updates.
    """

    def __init__(self, classmethod: Callable[..., Graph], *args: Any, **kwargs: Any) -> None:
        """
        Initialize a new LazyHariGraph instance.

        Parameters:
            classmethod: The class method used to create the HariGraph instance.
            *args: Positional arguments to pass to the class method.
            **kwargs: Keyword arguments to pass to the class method.
        """
        # The actual HariGraph instance (initialized later)
        self._hari_graph: Graph = None
        # The class method used to create the HariGraph instance
        self._classmethod = classmethod
        self._args = args  # Positional arguments to pass to the class method
        self._kwargs = kwargs  # Keyword arguments to pass to the class method
        self._mapping = None

    def _initialize(self) -> None:
        """
        Initializes the internal HariGraph instance if not already initialized.
        """
        if self._hari_graph is None:
            self._hari_graph = self._classmethod(*self._args, **self._kwargs)
            if self._mapping and not (self._is_trivial_mapping(self._mapping)):
                self._hari_graph.merge_clusters(self._mapping)

    def uninitialize(self) -> None:
        """
        Resets the internal HariGraph instance to its uninitialized state.
        """
        self._hari_graph = None

    def reinitialize(self) -> None:
        """
        Reinitializes the internal HariGraph instance, preserving the current mapping.
        """
        self._hari_graph = self._classmethod(*self._args, **self._kwargs)
        if self._mapping and not (self._is_trivial_mapping(self._mapping)):
            self._hari_graph.merge_clusters(self._mapping)

    @property
    def is_initialized(self) -> bool:
        """
        Checks whether the internal HariGraph instance has been initialized.

        Returns:
            bool: True if the internal HariGraph is initialized, otherwise False.
        """
        return self._hari_graph is not None

    @property
    def mapping(self) -> List[List[Tuple[int]]]:
        """
        Get the current mapping of the LazyHariGraph.

        Returns:
            List[List[Tuple[int]]]: The current mapping.
        """
        return self._mapping

    @mapping.setter
    def mapping(self, new_mapping: List[List[Tuple[int]]]) -> None:
        """
        Sets a new mapping and reinitializes the internal HariGraph instance with this mapping.

        Parameters:
            new_mapping: The new mapping to set.
        """
        self._mapping = new_mapping
        if self.is_initialized:
            self.reinitialize()

    def _is_trivial_mapping(self, mapping: List[List[Tuple[int]]]) -> bool:
        """
        Checks whether a given mapping is trivial.

        Parameters:
            mapping: The mapping to check.

        Returns:
            bool: True if the mapping is trivial, otherwise False.
        """
        return all(len(i) == 1 for i in mapping)

    @property
    def is_correctly_mapped(self) -> bool:
        """
        Checks whether the graph's mapping matches the real cluster mapping of the internal HariGraph.

        Mappings are expected to have sorted elements of sorted elements!

        Returns:
            bool: True if the mappings match, otherwise False.
        """

        return self._mapping == self._hari_graph.get_cluster_mapping()

    def get_graph(self) -> Graph:
        """
        Get the initialized HariGraph instance.

        Returns:
            HariGraph: The initialized HariGraph.
        """
        self._initialize()
        return self._hari_graph

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the internal HariGraph instance.

        Parameters:
            name: The attribute name.

        Returns:
            Any: The value of the requested attribute from the internal HariGraph.
        """
        self._initialize()
        return getattr(self._hari_graph, name)

    def __getitem__(self, n: int) -> Any:
        """
        Delegate item access to the internal HariGraph instance.

        Parameters:
            n: The index or key.

        Returns:
            Any: The item from the internal HariGraph at the specified index or key.
        """
        self._initialize()
        return self._hari_graph.__getitem__(n)

    def __str__(self) -> str:
        """
        Return a string representation of the LazyHariGraph.

        Returns:
            str: A string representation of the LazyHariGraph.
        """
        if self._hari_graph is not None:
            return f"<LazyHariGraph with {self._hari_graph.number_of_nodes()} nodes and {self._hari_graph.number_of_edges()} edges>"
        else:
            return f"<Uninitialized LazyHariGraph object at {id(self)}>"
