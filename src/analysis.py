"""Graph analysis: centrality, communities, semantic clustering, path finding."""

import logging
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

logger = logging.getLogger(__name__)

# Try Louvain; fall back to NetworkX greedy modularity
try:
    import community.community_louvain as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    logger.info("python-louvain not found; using greedy_modularity_communities")


def compute_centrality(G: nx.DiGraph) -> pd.DataFrame:
    """Compute PageRank, betweenness, and in-degree centrality."""
    if len(G) == 0:
        return pd.DataFrame(columns=["title", "pagerank", "betweenness", "in_degree"])

    pr = nx.pagerank(G, alpha=0.85)
    bc = nx.betweenness_centrality(G)
    records = []
    for node in G.nodes():
        records.append({
            "title": node,
            "pagerank": round(pr.get(node, 0.0), 6),
            "betweenness": round(bc.get(node, 0.0), 6),
            "in_degree": G.in_degree(node),
        })
    df = pd.DataFrame(records).sort_values("pagerank", ascending=False).reset_index(drop=True)
    return df


def detect_communities(G: nx.DiGraph) -> dict[str, int]:
    """Detect communities using Louvain (or greedy modularity fallback)."""
    if len(G) == 0:
        return {}

    undirected = G.to_undirected()

    if HAS_LOUVAIN:
        partition = community_louvain.best_partition(undirected)
        return partition
    else:
        communities_gen = nx.community.greedy_modularity_communities(undirected)
        partition = {}
        for i, comm in enumerate(communities_gen):
            for node in comm:
                partition[node] = i
        return partition


def label_communities(
    communities: dict[str, int],
    summaries: dict[str, str],
    top_n: int = 3,
) -> dict[int, str]:
    """Label each community by its top TF-IDF keywords."""
    comm_to_nodes: dict[int, list[str]] = {}
    for node, cid in communities.items():
        comm_to_nodes.setdefault(cid, []).append(node)

    labels: dict[int, str] = {}
    for cid, nodes in comm_to_nodes.items():
        texts = [summaries.get(n, "") for n in nodes]
        texts = [t for t in texts if t.strip()]
        if not texts:
            labels[cid] = f"Community {cid}"
            continue
        try:
            vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
            tfidf = vectorizer.fit_transform(texts)
            mean_tfidf = np.asarray(tfidf.mean(axis=0)).flatten()
            feature_names = vectorizer.get_feature_names_out()
            top_indices = mean_tfidf.argsort()[-top_n:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            labels[cid] = ", ".join(keywords)
        except Exception:
            labels[cid] = f"Community {cid}"
    return labels


def semantic_clustering(
    summaries: dict[str, str],
    n_clusters: int = 5,
) -> dict[str, int]:
    """Cluster articles by TF-IDF of their summaries using KMeans."""
    if not summaries:
        return {}

    titles = list(summaries.keys())
    texts = [summaries.get(t, "") for t in titles]
    non_empty = [(t, txt) for t, txt in zip(titles, texts) if txt.strip()]

    if len(non_empty) < 2:
        return {t: 0 for t in titles}

    filtered_titles, filtered_texts = zip(*non_empty)
    actual_k = min(n_clusters, len(filtered_titles))

    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    tfidf_matrix = vectorizer.fit_transform(filtered_texts)

    km = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
    labels = km.fit_predict(tfidf_matrix)

    result = {t: int(labels[i]) for i, t in enumerate(filtered_titles)}
    for t in titles:
        if t not in result:
            result[t] = 0
    return result


def compare_communities_clusters(
    communities: dict[str, int],
    clusters: dict[str, int],
) -> float:
    """Compute adjusted Rand index between communities and clusters."""
    common = sorted(set(communities.keys()) & set(clusters.keys()))
    if len(common) < 2:
        return 0.0
    labels_a = [communities[n] for n in common]
    labels_b = [clusters[n] for n in common]
    return float(adjusted_rand_score(labels_a, labels_b))


def find_bridge_topics(
    G: nx.DiGraph,
    communities: dict[str, int],
    top_n: int = 10,
) -> pd.DataFrame:
    """Find nodes connecting different communities (high betweenness, multi-community neighbors)."""
    if len(G) == 0:
        return pd.DataFrame(columns=["title", "betweenness", "community", "num_neighbor_communities"])

    bc = nx.betweenness_centrality(G)
    records = []
    undirected = G.to_undirected()
    for node in G.nodes():
        neighbor_comms = set()
        for neighbor in undirected.neighbors(node):
            if neighbor in communities:
                neighbor_comms.add(communities[neighbor])
        records.append({
            "title": node,
            "betweenness": round(bc.get(node, 0.0), 6),
            "community": communities.get(node, -1),
            "num_neighbor_communities": len(neighbor_comms),
        })

    df = pd.DataFrame(records)
    df = df[df["num_neighbor_communities"] > 1].sort_values("betweenness", ascending=False)
    return df.head(top_n).reset_index(drop=True)


def find_shortest_path(G: nx.DiGraph, source: str, target: str) -> Optional[list[str]]:
    """Find shortest path in the undirected version."""
    if source not in G or target not in G:
        return None
    try:
        undirected = G.to_undirected()
        return nx.shortest_path(undirected, source, target)
    except nx.NetworkXNoPath:
        return None


def compute_surprise_score(
    source: str,
    target: str,
    summaries: dict[str, str],
) -> float:
    """Compute 1 - cosine_similarity between TF-IDF vectors (higher = more surprising)."""
    s1 = summaries.get(source, "")
    s2 = summaries.get(target, "")
    if not s1.strip() or not s2.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform([s1, s2])
        sim = sklearn_cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return round(float(1.0 - sim), 4)
    except Exception:
        return 0.0


def get_community_sizes(communities: dict[str, int]) -> pd.DataFrame:
    """Return DataFrame of community sizes."""
    if not communities:
        return pd.DataFrame(columns=["community_id", "size"])
    counts: dict[int, int] = {}
    for cid in communities.values():
        counts[cid] = counts.get(cid, 0) + 1
    records = [{"community_id": cid, "size": sz} for cid, sz in sorted(counts.items())]
    return pd.DataFrame(records).sort_values("size", ascending=False).reset_index(drop=True)
