"""Wikipedia Rabbit Hole Mapper — interactive Streamlit app.

Crawl Wikipedia from a starting topic, build a link graph, detect
communities, and explore how ideas connect.
"""

import io
import tempfile
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.analysis import (
    compare_communities_clusters,
    compute_centrality,
    compute_surprise_score,
    detect_communities,
    find_bridge_topics,
    find_shortest_path,
    get_community_sizes,
    label_communities,
    semantic_clustering,
)
from src.crawler import PageInfo, crawl
from src.graph_builder import build_graph, clean_graph, get_graph_summary, graph_to_dataframe
from src.visualize import (
    create_centrality_bar_chart,
    create_community_chart,
    create_community_legend,
    create_degree_distribution,
    create_network_html,
)

st.set_page_config(
    page_title="Wikipedia Rabbit Hole Mapper",
    page_icon="🕳️",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def cached_crawl(topic: str, depth: int, max_links: int, node_cap: int) -> dict[str, PageInfo]:
    """Crawl Wikipedia, caching results per parameter combination.

    The progress bar is created inside the cached function so Streamlit can
    replay its updates on cache hits (calling elements on outside blocks
    raises CacheReplayClosureError).
    """
    bar = st.progress(0.0, text="Starting crawl…")

    def on_progress(done: int, message: str) -> None:
        bar.progress(min(done / max(node_cap, 1), 0.99), text=message)

    try:
        pages = crawl(
            topic,
            max_depth=depth,
            max_links_per_page=max_links,
            node_cap=node_cap,
            progress_callback=on_progress,
        )
    except ValueError:
        bar.empty()
        raise
    bar.progress(1.0, text=f"Crawled {len(pages)} pages.")
    return pages


def analyze_graph(G: nx.DiGraph) -> dict[str, Any]:
    """Run the full analysis pipeline on a cleaned graph."""
    summaries = {n: G.nodes[n].get("summary", "") for n in G.nodes()}
    centrality = compute_centrality(G)
    communities = detect_communities(G)
    labels = label_communities(communities, summaries)
    sizes = get_community_sizes(communities)
    clusters = semantic_clustering(summaries, n_clusters=max(len(labels), 2))
    ari = compare_communities_clusters(communities, clusters)
    bridges = find_bridge_topics(G, communities)
    return {
        "summaries": summaries,
        "centrality": centrality,
        "communities": communities,
        "labels": labels,
        "sizes": sizes,
        "clusters": clusters,
        "ari": ari,
        "bridges": bridges,
    }


def render_network_html(
    G: nx.DiGraph,
    communities: dict[str, int],
    labels: dict[int, str],
) -> str:
    """Generate the pyvis network and return its HTML for embedding."""
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "network.html")
        create_network_html(G, communities, labels, output_path=path, height="720px")
        return Path(path).read_text(encoding="utf-8")


def graphml_bytes(G: nx.DiGraph) -> bytes:
    """Serialize the graph to GraphML bytes for download."""
    buffer = io.BytesIO()
    nx.write_graphml(G, buffer)
    return buffer.getvalue()


def wiki_link(title: str) -> str:
    """Return a Markdown link to a Wikipedia article."""
    url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
    return f"[{title}]({url})"


def sidebar() -> Optional[dict[str, Any]]:
    """Render sidebar controls; return params when the user builds a map."""
    with st.sidebar:
        st.title("🕳️ Rabbit Hole Mapper")
        st.caption("Follow the links. Find the hidden connections.")
        with st.form("crawl_form"):
            topic = st.text_input("Starting topic", value="Black hole")
            depth = st.slider("Crawl depth", 1, 3, 1)
            max_links = st.slider("Max links per page", 5, 50, 25)
            node_cap = st.slider("Max total nodes", 25, 500, 300, step=25)
            submitted = st.form_submit_button("🔭 Build map", use_container_width=True)
        st.markdown("**Try:** Black hole · Jazz · Cryptography")
        st.markdown(
            "Depth 1 is fast; depth 2–3 can take a few minutes on a cold cache. "
            "Repeat runs are instant thanks to disk caching."
        )
        if submitted:
            return {
                "topic": topic.strip(),
                "depth": depth,
                "max_links": max_links,
                "node_cap": node_cap,
            }
    return None


def tab_network_map(G: nx.DiGraph, results: dict[str, Any], stats: dict[str, Any]) -> None:
    """Network Map tab: interactive pyvis graph plus downloads."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Topics", stats["num_nodes"])
    c2.metric("Links", stats["num_edges"])
    c3.metric("Communities", len(results["labels"]))
    c4.metric("Density", f"{stats['density']:.3f}")

    html = render_network_html(G, results["communities"], results["labels"])
    components.html(html, height=740, scrolling=True)
    st.caption(
        "Node size = PageRank · color = community · hover for a summary · "
        "click a node to open its Wikipedia article."
    )

    st.markdown(create_community_legend(results["labels"], results["sizes"]), unsafe_allow_html=True)

    st.subheader("Downloads")
    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇️ Graph (GraphML)",
        data=graphml_bytes(G),
        file_name="rabbit_hole.graphml",
        mime="application/xml",
    )
    node_df = graph_to_dataframe(G)
    merged = node_df.merge(results["centrality"], on="title", how="left")
    merged["community"] = merged["title"].map(results["communities"])
    merged["community_label"] = merged["community"].map(results["labels"])
    d2.download_button(
        "⬇️ Node table (CSV)",
        data=merged.to_csv(index=False).encode("utf-8"),
        file_name="rabbit_hole_nodes.csv",
        mime="text/csv",
    )

    st.plotly_chart(create_degree_distribution(G), use_container_width=True)


def tab_communities(G: nx.DiGraph, results: dict[str, Any]) -> None:
    """Communities tab: sizes, labels, semantic comparison, bridges."""
    st.plotly_chart(create_community_chart(results["sizes"]), use_container_width=True)
    st.markdown(create_community_legend(results["labels"], results["sizes"]), unsafe_allow_html=True)

    st.subheader("Link communities vs. text clusters")
    st.markdown(
        "Communities come from the **link structure** (Louvain modularity). "
        "Clusters come from **summary text** (TF-IDF + KMeans). The adjusted Rand "
        "index (ARI) measures how much they agree: 1 = identical, 0 = chance."
    )
    st.metric("Adjusted Rand index", f"{results['ari']:.3f}")
    comparison = pd.DataFrame(
        {
            "topic": list(results["communities"].keys()),
            "link_community": [results["communities"][t] for t in results["communities"]],
            "text_cluster": [results["clusters"].get(t, -1) for t in results["communities"]],
        }
    ).sort_values(["link_community", "text_cluster"])
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.subheader("Bridge topics")
    st.markdown(
        "High-betweenness topics whose neighbors span multiple communities — "
        "the connectors between different worlds of ideas."
    )
    if results["bridges"].empty:
        st.info("No bridge topics found (the graph may be a single tight community).")
    else:
        bridges = results["bridges"].copy()
        bridges["community_label"] = bridges["community"].map(results["labels"])
        st.dataframe(bridges, use_container_width=True, hide_index=True)


def tab_influential(results: dict[str, Any]) -> None:
    """Influential Topics tab: centrality chart and table."""
    top_n = st.slider("Show top N", 5, 30, 15)
    st.plotly_chart(
        create_centrality_bar_chart(results["centrality"], top_n=top_n),
        use_container_width=True,
    )
    table = results["centrality"].head(top_n).copy()
    table["summary"] = table["title"].map(results["summaries"])
    table["community"] = table["title"].map(results["communities"]).map(results["labels"])
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(
        "PageRank = influence via incoming links · betweenness = how often a topic "
        "lies on shortest paths · in-degree = raw number of linking topics."
    )


def tab_path_finder(G: nx.DiGraph, results: dict[str, Any]) -> None:
    """Path Finder tab: shortest chain between two topics plus surprise score."""
    titles = sorted(G.nodes())
    c1, c2 = st.columns(2)
    source = c1.selectbox("From topic", titles, index=0)
    default_target = 1 if len(titles) > 1 else 0
    target = c2.selectbox("To topic", titles, index=min(len(titles) - 1, max(default_target, len(titles) // 2)))

    if source == target:
        st.info("Pick two different topics.")
        return

    path = find_shortest_path(G, source, target)
    if path is None:
        st.warning("No path between these topics in the crawled graph. Try a deeper crawl.")
        return

    st.subheader("The rabbit hole path")
    st.markdown(" → ".join(wiki_link(t) for t in path))
    st.caption(f"{len(path) - 1} hop(s)")

    with st.expander("What each stop is about", expanded=False):
        for t in path:
            st.markdown(f"**{wiki_link(t)}** — {results['summaries'].get(t, 'No summary.')}")

    surprise = compute_surprise_score(source, target, results["summaries"])
    st.metric("Surprise score", f"{surprise:.2f}")
    if surprise > 0.8:
        st.markdown("🤯 **Wild jump** — these topics barely share any vocabulary.")
    elif surprise > 0.5:
        st.markdown("😮 **Surprising connection** — quite different subjects.")
    else:
        st.markdown("🙂 **Close cousins** — semantically similar topics.")
    st.caption("Surprise = 1 − cosine similarity of the two summaries' TF-IDF vectors.")


def tab_about() -> None:
    """About tab: methodology and limitations."""
    st.subheader("How it works")
    st.markdown(
        """
1. **Crawl** — the MediaWiki API (`prop=links`, `plnamespace=0`) fetches outgoing
   article links breadth-first from your topic, with pagination, redirects
   resolved, and non-article pages (files, categories, disambiguations, year
   pages) skipped. Intro extracts come from `prop=extracts`, batched 20 titles
   per request. Every response is cached to disk; requests are rate-limited
   and retried with backoff.
2. **Graph** — a directed NetworkX graph where A → B means article A links to
   article B. Self-loops and isolated nodes are removed; only the largest
   weakly connected component is kept.
3. **Communities** — Louvain modularity optimization on the undirected graph
   (greedy modularity as fallback). Each community is auto-named by its top
   TF-IDF keywords.
4. **Semantic clustering** — TF-IDF on summaries + KMeans, compared against
   link communities via the adjusted Rand index.
5. **Centrality & paths** — PageRank, betweenness, and in-degree rank the most
   influential topics; bridge topics connect communities; the path finder
   shows the shortest chain between any two topics with a semantic
   "surprise score".
        """
    )
    st.subheader("Limitations")
    st.markdown(
        """
- Only the first *N* links per page are kept (API returns them alphabetically),
  so the map is a sample of the neighborhood, not the full link set.
- Summaries are truncated intros; TF-IDF labels are keywords, not human titles.
- Louvain is stochastic — community boundaries can shift slightly between runs.
- English Wikipedia only; link structure reflects editorial choices, not
  ground-truth semantic relatedness.
        """
    )
    st.subheader("API etiquette")
    st.markdown(
        "Custom User-Agent, 0.1–0.2 s between requests, batched extracts, "
        "exponential backoff honoring `Retry-After`, and a persistent disk "
        "cache so repeated runs never re-hit the API."
    )


def main() -> None:
    """App entry point."""
    params = sidebar()
    if params is not None:
        if not params["topic"]:
            st.sidebar.error("Enter a topic first.")
        else:
            st.session_state["params"] = params

    if "params" not in st.session_state:
        st.title("🕳️ Wikipedia Rabbit Hole Mapper")
        st.markdown(
            "Enter a topic in the sidebar and press **Build map** to crawl "
            "Wikipedia, map how ideas connect, and discover hidden communities. "
            "Good starting points: **Black hole**, **Jazz**, **Cryptography**."
        )
        return

    p = st.session_state["params"]
    st.title(f"🕳️ Rabbit holes from “{p['topic']}”")

    try:
        pages = cached_crawl(p["topic"], p["depth"], p["max_links"], p["node_cap"])
    except ValueError as e:
        st.error(str(e))
        st.stop()

    G = clean_graph(build_graph(pages, p["topic"]))
    if G.number_of_nodes() < 2:
        st.warning("This topic produced too few connected articles to map. Try another topic or more links per page.")
        st.stop()

    results = analyze_graph(G)
    stats = get_graph_summary(G)

    tabs = st.tabs(["🕸️ Network Map", "🧩 Communities", "👑 Influential Topics", "🧭 Path Finder", "ℹ️ About"])
    with tabs[0]:
        tab_network_map(G, results, stats)
    with tabs[1]:
        tab_communities(G, results)
    with tabs[2]:
        tab_influential(results)
    with tabs[3]:
        tab_path_finder(G, results)
    with tabs[4]:
        tab_about()


if __name__ == "__main__":
    main()