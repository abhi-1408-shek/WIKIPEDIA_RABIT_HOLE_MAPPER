"""Build and clean the NetworkX directed graph from crawl results."""

import logging
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def build_graph(pages: dict[str, Any], start_topic: str) -> nx.DiGraph:
    """
    Build a directed graph from crawled page data.

    Each node has attributes: title, summary, depth, in_degree_val, out_degree_val.
    Edge A -> B exists if page A links to page B and B is in our crawled set.
    """
    G = nx.DiGraph()

    for title, info in pages.items():
        G.add_node(
            title,
            title=title,
            summary=getattr(info, "summary", ""),
            depth=getattr(info, "depth", 0),
        )

    for title, info in pages.items():
        for link in getattr(info, "links", []):
            if link in pages and link != title:
                G.add_edge(title, link)

    for node in G.nodes():
        G.nodes[node]["in_degree_val"] = G.in_degree(node)
        G.nodes[node]["out_degree_val"] = G.out_degree(node)

    logger.info("Built graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def clean_graph(G: nx.DiGraph, keep_largest_wcc: bool = True) -> nx.DiGraph:
    """Remove self-loops, isolated nodes, optionally keep largest WCC."""
    self_loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loops)
    if self_loops:
        logger.info("Removed %d self-loops", len(self_loops))

    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)
    if isolated:
        logger.info("Removed %d isolated nodes", len(isolated))

    if keep_largest_wcc and len(G) > 0:
        components = list(nx.weakly_connected_components(G))
        if len(components) > 1:
            largest = max(components, key=len)
            removed_count = len(G) - len(largest)
            G = G.subgraph(largest).copy()
            logger.info("Kept largest WCC (%d nodes), removed %d", len(G), removed_count)

    for node in G.nodes():
        G.nodes[node]["in_degree_val"] = G.in_degree(node)
        G.nodes[node]["out_degree_val"] = G.out_degree(node)

    return G


def get_graph_summary(G: nx.DiGraph) -> dict[str, Any]:
    """Return basic graph statistics."""
    n = G.number_of_nodes()
    return {
        "num_nodes": n,
        "num_edges": G.number_of_edges(),
        "density": round(nx.density(G), 6),
        "avg_in_degree": round(sum(d for _, d in G.in_degree()) / max(n, 1), 2),
        "avg_out_degree": round(sum(d for _, d in G.out_degree()) / max(n, 1), 2),
        "num_weakly_connected_components": nx.number_weakly_connected_components(G),
        "is_dag": nx.is_directed_acyclic_graph(G),
    }


def export_graphml(G: nx.DiGraph, filepath: str) -> None:
    """Export the graph to GraphML format."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, filepath)
    logger.info("Graph exported to %s", filepath)


def graph_to_dataframe(G: nx.DiGraph) -> pd.DataFrame:
    """Convert graph nodes to a pandas DataFrame."""
    records = []
    for node, attrs in G.nodes(data=True):
        records.append({
            "title": attrs.get("title", node),
            "summary": attrs.get("summary", ""),
            "depth": attrs.get("depth", -1),
            "in_degree": attrs.get("in_degree_val", 0),
            "out_degree": attrs.get("out_degree_val", 0),
        })
    return pd.DataFrame(records)
