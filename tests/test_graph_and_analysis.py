"""Unit tests for graph_builder and analysis modules."""

import pytest
import networkx as nx
import pandas as pd
from src.graph_builder import build_graph, clean_graph, get_graph_summary, graph_to_dataframe
from src.analysis import (
    compute_centrality,
    detect_communities,
    label_communities,
    semantic_clustering,
    find_bridge_topics,
    find_shortest_path,
    compute_surprise_score,
    compare_communities_clusters,
    get_community_sizes,
)


class MockPage:
    """Mock page object to simulate crawler output."""
    def __init__(self, title, summary, links, depth=0):
        self.title = title
        self.summary = summary
        self.links = links
        self.depth = depth
        self.exists = True
        self.is_redirect = False


@pytest.fixture
def mock_pages():
    return {
        "Physics": MockPage("Physics", "Physics is the natural science of matter, energy, and their interactions.", ["Quantum mechanics", "Thermodynamics", "Mathematics"], 0),
        "Quantum mechanics": MockPage("Quantum mechanics", "Quantum mechanics is a fundamental theory describing nature at atomic scales.", ["Physics", "Wave function", "Photon"], 1),
        "Thermodynamics": MockPage("Thermodynamics", "Thermodynamics is the study of heat, work, temperature, and energy.", ["Physics", "Entropy", "Temperature"], 1),
        "Mathematics": MockPage("Mathematics", "Mathematics studies numbers, shapes, patterns, and abstract structures.", ["Physics", "Algebra", "Geometry"], 1),
        "Wave function": MockPage("Wave function", "A wave function describes the quantum state of a particle or system.", ["Quantum mechanics", "Physics"], 2),
        "Photon": MockPage("Photon", "A photon is an elementary particle and quantum of electromagnetic radiation.", ["Quantum mechanics", "Electromagnetic radiation"], 2),
        "Entropy": MockPage("Entropy", "Entropy measures the number of microscopic configurations in thermodynamics.", ["Thermodynamics"], 2),
        "Temperature": MockPage("Temperature", "Temperature is a physical quantity expressing thermal energy.", ["Thermodynamics", "Physics"], 2),
        "Algebra": MockPage("Algebra", "Algebra is a branch of mathematics with symbols and rules for manipulating them.", ["Mathematics"], 2),
        "Geometry": MockPage("Geometry", "Geometry studies shapes, sizes, positions, and properties of space.", ["Mathematics", "Algebra"], 2),
    }


@pytest.fixture
def sample_graph(mock_pages):
    return build_graph(mock_pages, "Physics")


@pytest.fixture
def cleaned_graph(sample_graph):
    return clean_graph(sample_graph, keep_largest_wcc=True)


def test_build_graph_node_count(sample_graph):
    assert sample_graph.number_of_nodes() == 10, "Should have 10 nodes"


def test_build_graph_has_edges(sample_graph):
    assert sample_graph.has_edge("Physics", "Quantum mechanics")
    assert sample_graph.has_edge("Physics", "Thermodynamics")
    assert sample_graph.has_edge("Physics", "Mathematics")


def test_build_graph_no_external_edges(sample_graph):
    # "Electromagnetic radiation" is referenced but not in mock_pages
    assert "Electromagnetic radiation" not in sample_graph.nodes()
    assert not sample_graph.has_edge("Photon", "Electromagnetic radiation")


def test_clean_graph_no_self_loops(mock_pages):
    # Add a self-loop manually
    G = build_graph(mock_pages, "Physics")
    G.add_edge("Physics", "Physics")
    cleaned = clean_graph(G)
    assert not list(nx.selfloop_edges(cleaned)), "No self-loops after cleaning"


def test_clean_graph_no_isolates(mock_pages):
    G = build_graph(mock_pages, "Physics")
    G.add_node("IsolatedNode")
    cleaned = clean_graph(G)
    assert "IsolatedNode" not in cleaned.nodes()


def test_graph_summary(cleaned_graph):
    summary = get_graph_summary(cleaned_graph)
    assert "num_nodes" in summary
    assert "num_edges" in summary
    assert "density" in summary
    assert summary["num_nodes"] > 0


def test_graph_to_dataframe(cleaned_graph):
    df = graph_to_dataframe(cleaned_graph)
    assert "title" in df.columns
    assert "summary" in df.columns
    assert "in_degree" in df.columns
    assert len(df) == cleaned_graph.number_of_nodes()


def test_compute_centrality(cleaned_graph):
    df = compute_centrality(cleaned_graph)
    assert "pagerank" in df.columns
    assert "betweenness" in df.columns
    assert "in_degree" in df.columns
    assert len(df) == cleaned_graph.number_of_nodes()
    assert all(df["pagerank"] >= 0)


def test_detect_communities(cleaned_graph):
    comms = detect_communities(cleaned_graph)
    assert len(comms) == cleaned_graph.number_of_nodes()
    for node in cleaned_graph.nodes():
        assert node in comms


def test_label_communities(cleaned_graph):
    comms = detect_communities(cleaned_graph)
    summaries = {n: cleaned_graph.nodes[n].get("summary", "") for n in cleaned_graph.nodes()}
    labels = label_communities(comms, summaries)
    assert len(labels) > 0
    for cid, label in labels.items():
        assert isinstance(label, str) and len(label) > 0


def test_find_shortest_path(cleaned_graph):
    path = find_shortest_path(cleaned_graph, "Physics", "Geometry")
    assert path is not None
    assert path[0] == "Physics"
    assert path[-1] == "Geometry"


def test_find_shortest_path_no_path():
    G = nx.DiGraph()
    G.add_node("A")
    G.add_node("B")
    assert find_shortest_path(G, "A", "B") is None


def test_compute_surprise_score():
    summaries = {
        "Physics": "Physics is the study of matter and energy.",
        "Cooking": "Cooking is the art of preparing food by applying heat.",
    }
    score = compute_surprise_score("Physics", "Cooking", summaries)
    assert 0.0 <= score <= 1.0


def test_semantic_clustering(cleaned_graph):
    summaries = {n: cleaned_graph.nodes[n].get("summary", "") for n in cleaned_graph.nodes()}
    clusters = semantic_clustering(summaries, n_clusters=3)
    assert len(clusters) == len(summaries)


def test_community_sizes(cleaned_graph):
    comms = detect_communities(cleaned_graph)
    sizes = get_community_sizes(comms)
    assert "community_id" in sizes.columns
    assert "size" in sizes.columns
    assert sizes["size"].sum() == cleaned_graph.number_of_nodes()


def test_find_bridge_topics(cleaned_graph):
    comms = detect_communities(cleaned_graph)
    bridges = find_bridge_topics(cleaned_graph, comms)
    assert "title" in bridges.columns
    assert "betweenness" in bridges.columns
    assert "num_neighbor_communities" in bridges.columns
