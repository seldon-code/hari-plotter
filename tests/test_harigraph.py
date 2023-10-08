import networkx as nx
import numpy as np
import pytest

from hari_plotter import HariGraph


class TestHariGraph:

    @classmethod
    def setup_class(cls):
        cls.graph = HariGraph.read_json('tests/test.json')
        cls.graph_from_files = HariGraph.read_network(
            'tests/network.txt', 'tests/opinions_0.txt')
        cls.strongly_connected_graph = HariGraph.strongly_connected_components(
            15, 25, 6)

    def test_read_json(self):
        assert isinstance(
            self.graph, HariGraph), "Read JSON should return an instance of HariGraph."

    def test_generate_labels(self):
        self.graph.generate_labels()
        for node in self.graph.nodes:
            assert 'label' in self.graph.nodes[
                node], f"Node {node} should have a label after calling generate_labels."

    def test_mean_opinion(self):
        assert self.graph.mean_opinion == pytest.approx(
            0.79070086941968), "Mean opinion is not as expected."

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
        assert self.graph_from_files.mean_opinion == pytest.approx(
            0.6279692643403699), "Mean opinion from network and opinions files is not as expected."

    def test_strongly_connected_components(self):
        n1, n2 = 8, 9  # Number of nodes in the two components
        connect_nodes = 3  # Number of nodes to connect between components
        graph = HariGraph.strongly_connected_components(n1, n2, connect_nodes)

        assert isinstance(
            graph, HariGraph), "Method should return an instance of HariGraph."

        # Assert that the number of nodes is correct
        assert graph.number_of_nodes() == n1 + \
            n2, f"Graph should have {n1 + n2} nodes."

        # Assert that the graph has the 'opinion' attribute for every node and edge
        for _, data in graph.nodes(data=True):
            assert 'opinion' in data, "Every node should have a 'opinion' attribute."

        for _, _, data in graph.edges(data=True):
            assert 'influence' in data, "Every edge should have a 'influence' attribute."

        assert graph.check_all_paths_exist(), "All paths should exist in the graph."

    def test_cluster_size(self):
        cluster_size = self.graph.cluster_size
        assert isinstance(
            cluster_size, dict), "cluster_size should return a dictionary."

        for node in self.graph.nodes:
            label = self.graph.nodes[node].get('label', [node])
            assert cluster_size[node] == len(
                label), f"Size of node {node} is incorrect."

    def test_importance(self):
        importance = self.graph.importance
        cluster_size = self.graph.cluster_size
        assert isinstance(
            importance, dict), "importance should return a dictionary."

        for node in self.graph.nodes:
            influences_sum = sum(
                data['influence'] for _, _, data in self.graph.edges(node, data=True))
            calculated_importance = influences_sum / \
                cluster_size[node] if cluster_size[node] != 0 else 0
            assert importance[node] == pytest.approx(
                calculated_importance), f"Importance of node {node} is incorrect."

    def test_find_clusters(self):
        G = HariGraph()

        # Add nodes and edges
        G.add_node(1, opinion=0.1)
        G.add_node(2, opinion=0.2)
        G.add_node(3, opinion=0.9)
        G.add_node(4, opinion=0.95)
        G.add_edge(1, 2, influence=0.15)
        G.add_edge(3, 4, influence=0.15)

        clusters = G.find_clusters(
            max_opinion_difference=0.1, min_influence=0.1)

        # Validate clusters
        assert len(
            clusters) == 2, f"Expected 2 clusters, but got {len(clusters)}"
        assert set(clusters[0]) == {1, 2}, "Unexpected nodes in first cluster"
        assert set(clusters[1]) == {3, 4}, "Unexpected nodes in second cluster"

    def test_merge_clusters(self):
        G = HariGraph()

        # Add nodes and edges
        G.add_node(1, opinion=0.1, label=[1])
        G.add_node(2, opinion=0.2, label=[2])
        G.add_node(3, opinion=0.9, label=[3])
        G.add_node(4, opinion=0.95, label=[4])
        G.add_edge(1, 2, influence=0.15)
        G.add_edge(3, 4, influence=0.05)

        clusters = [{1, 2}, {3, 4}]
        G.merge_clusters(clusters)

        # Validate merge
        assert len(G.nodes) == 2, f"Expected 2 nodes, but got {len(G.nodes)}"
        assert len(G.edges) == 0, f"Expected 0 edges, but got {len(G.edges)}"

        # Validate new nodes' opinions, labels, and importances
        for node in G.nodes:
            assert G.nodes[node]['opinion'] == pytest.approx(
                0.15) or G.nodes[node]['opinion'] == pytest.approx(0.925), f"Unexpected opinion in merged node {node}. {G.nodes[node]['opinion'] = }"
            assert len(G.nodes[node]['label']
                       ) == 2, "Unexpected label length in merged node"

    def test_min_max_opinions(self):
        self.graph.add_parameters_to_nodes()
        opinions = self.graph.node_values
        min_opinions = opinions['min_opinion']
        max_opinions = opinions['max_opinion']
        assert isinstance(
            min_opinions, dict), "min_opinions should return a dictionary."

        assert isinstance(
            max_opinions, dict), "max_opinions should return a dictionary."
        assert np.all(np.array(list(min_opinions.values())) >= 0) and np.all(
            np.array(list(min_opinions.values())) <= 1), "min_opinions are not in range."
        assert np.all(np.array(list(max_opinions.values())) >= 0) and np.all(
            np.array(list(max_opinions.values())) <= 1), "max_opinions are not in range."

    def test_merge_by_intervals(self):
        self.graph.merge_by_intervals([0.25, 0.75])
        assert self.graph.number_of_nodes(
        ) <= 3, f"Expected maximum 3 nodes, but got {self.graph.number_of_nodes()}"

    def test_merge_by_intervals_2(self):
        expected_clusters = {
            40: set(range(15, 40)),
            41: set(range(15))
        }

        assert len(self.strongly_connected_graph.nodes) == 40
        assert len(self.strongly_connected_graph.edges) == 816

        self.strongly_connected_graph.merge_by_intervals([0.5])

        # Check the number of nodes and edges after merging
        assert len(self.strongly_connected_graph.nodes) == 2
        assert len(self.strongly_connected_graph.edges) == 2

        # Check the cluster mapping
        cluster_mapping = self.strongly_connected_graph.get_cluster_mapping()
        assert cluster_mapping == expected_clusters

        # Check each node in the cluster mapping
        for new_id, cluster in cluster_mapping.items():
            for old_id in cluster:
                assert old_id in self.strongly_connected_graph.nodes[new_id][
                    'label'], f"Node {old_id} should be in the label of the new node {new_id}"
