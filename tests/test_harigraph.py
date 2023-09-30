import networkx as nx
import pytest

from hari_plotter import HariGraph


class TestHariGraph:

    @classmethod
    def setup_class(cls):
        cls.graph = HariGraph.read_json('tests/test.json')
        cls.graph_from_files = HariGraph.read_network(
            'tests/network.txt', 'tests/opinions_0.txt')

    def test_read_json(self):
        assert isinstance(
            self.graph, HariGraph), "Read JSON should return an instance of HariGraph."

    def test_generate_labels(self):
        self.graph.generate_labels()
        for node in self.graph.nodes:
            assert 'label' in self.graph.nodes[
                node], f"Node {node} should have a label after calling generate_labels."

    def test_weighted_mean_value(self):
        assert self.graph.weighted_mean_value == pytest.approx(
            0.79070086941968), "Weighted mean value is not as expected."

    def test_default_similarity_function(self):
        similarity = self.graph.compute_similarity(56, 75)
        assert similarity == pytest.approx(
            7.061116043349517), "Default similarity function is not returning expected values."

    def test_compute_similarity(self):
        similarity = self.graph.compute_similarity(56, 75)
        assert similarity == pytest.approx(
            7.061116043349517), "Compute similarity is not returning expected values."

    def test_merge_nodes(self):
        i, j = 56, 75
        old_number_of_nodes = self.graph.number_of_nodes()
        self.graph.merge_nodes(i, j)
        new_number_of_nodes = self.graph.number_of_nodes()
        assert new_number_of_nodes == old_number_of_nodes - \
            1, "Number of nodes should decrease by one after merging."

    def test_draw(self):
        # This method can be tested by visually inspecting the drawn graph or by checking if it raises any exceptions during execution.
        try:
            self.graph.draw(show=False)
        except Exception as e:
            pytest.fail(
                f"Draw method should not raise any exceptions. Raised: {str(e)}")

    def test_str(self):
        # Test the __str__ method
        assert str(
            self.graph) == f"<HariGraph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges>"

    def test_repr(self):
        # Test the __repr__ method
        assert repr(
            self.graph) == f"<HariGraph object at {id(self.graph)}: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges>"

    def test_load_from_network(self):
        assert self.graph_from_files.weighted_mean_value == pytest.approx(
            0.6279692643403699), "Weighted mean value from network and opinions files is not as expected."

    def test_strongly_connected_components(self):
        n1, n2 = 5, 4  # Number of nodes in the two components
        connect_nodes = 1  # Number of nodes to connect between components
        graph = HariGraph.strongly_connected_components(n1, n2, connect_nodes)

        assert isinstance(
            graph, HariGraph), "Method should return an instance of HariGraph."

        # Check if the generated graph has two strongly connected components
        strongly_connected_components = [
            c for c in nx.strongly_connected_components(graph)]
        assert len(
            strongly_connected_components) == 2, "Graph should have two strongly connected components."

        # Check the number of nodes in each component
        assert len(strongly_connected_components[0]) == n1 or len(
            strongly_connected_components[0]) == n2, "Number of nodes in the first component is incorrect."
        assert len(strongly_connected_components[1]) == n1 or len(
            strongly_connected_components[1]) == n2, "Number of nodes in the second component is incorrect."

        # Check if the graph has weak connectivity between the two components
        # Convert to undirected graph for weak connectivity check
        weakly_connected_subgraph = nx.Graph(graph)
        assert nx.is_connected(
            weakly_connected_subgraph), "The graph should be weakly connected."

        # Check the values of the nodes in each strongly connected component
        component_1_values = [graph.nodes[node]['value']
                              for node in strongly_connected_components[0]]
        component_2_values = [graph.nodes[node]['value']
                              for node in strongly_connected_components[1]]

        assert all(0.9 <= value <= 1.0 for value in component_1_values) or all(
            0.0 <= value <= 0.1 for value in component_1_values), "Values in the first component are incorrect."
        assert all(0.9 <= value <= 1.0 for value in component_2_values) or all(
            0.0 <= value <= 0.1 for value in component_2_values), "Values in the second component are incorrect."