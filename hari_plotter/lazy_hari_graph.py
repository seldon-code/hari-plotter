class LazyHariGraph:
    def __init__(self, classmethod, *args, **kwargs):
        # The actual HariGraph instance (initialized later)
        self._hari_graph = None
        # The class method used to create the HariGraph instance
        self._classmethod = classmethod
        self._args = args  # Positional arguments to pass to the class method
        self._kwargs = kwargs  # Keyword arguments to pass to the class method

    def _initialize(self):
        if self._hari_graph is None:
            self._hari_graph = self._classmethod(*self._args, **self._kwargs)

    def __getattr__(self, name):
        self._initialize()  # Initialize the HariGraph instance if not already done
        return getattr(self._hari_graph, name)
