"""Visualization: pyvis interactive network, plotly charts."""

import logging
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyvis.network import Network

logger = logging.getLogger(__name__)

COMMUNITY_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#000000",
]


def _get_color(community_id: int) -> str:
    """Return a color for a community, cycling through the palette."""
    return COMMUNITY_COLORS[community_id % len(COMMUNITY_COLORS)]


def create_network_html(
    G: nx.DiGraph,
    communities: dict[str, int],
    community_labels: dict[int, str],
    output_path: str = "network.html",
    height: str = "700px",
) -> str:
    """Create an interactive pyvis network HTML file."""
    net = Network(
        directed=True,
        notebook=False,
        height=height,
        width="100%",
        cdn_resources="remote",
    )

    # Compute PageRank for sizing
    try:
        pr = nx.pagerank(G, alpha=0.85)
    except Exception:
        pr = {n: 1.0 / max(len(G), 1) for n in G.nodes()}

    pr_values = list(pr.values())
    pr_min = min(pr_values) if pr_values else 0
    pr_max = max(pr_values) if pr_values else 1
    pr_range = pr_max - pr_min if pr_max > pr_min else 1.0

    for node in G.nodes():
        node_data = G.nodes[node]
        summary_snippet = node_data.get("summary", "")[:150]
        comm_id = communities.get(node, 0)
        comm_label = community_labels.get(comm_id, f"Community {comm_id}")
        pr_val = pr.get(node, 0)

        size = 10 + 40 * ((pr_val - pr_min) / pr_range)
        color = _get_color(comm_id)
        wiki_url = "https://en.wikipedia.org/wiki/" + node.replace(" ", "_")

        tooltip = f"<b>{node}</b><br><br>{summary_snippet}<br><br>Community: {comm_label}<br>PageRank: {pr_val:.6f}"

        net.add_node(
            node,
            label=node,
            title=tooltip,
            size=size,
            color=color,
            url=wiki_url,
        )

    for src, dst in G.edges():
        net.add_edge(src, dst)

    net.set_options("""
    {
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -3000,
                "centralGravity": 0.3,
                "springLength": 100,
                "springConstant": 0.04,
                "damping": 0.09
            },
            "minVelocity": 0.75
        },
        "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
            "color": {"opacity": 0.4},
            "smooth": {"type": "continuous"}
        },
        "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
        }
    }
    """)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(output_path)

    # Open Wikipedia in a new tab on node click (default navigates the iframe).
    html_path = Path(output_path)
    html = html_path.read_text(encoding="utf-8")
    click_handler = """
    <script type="text/javascript">
        if (typeof network !== "undefined" && typeof nodes !== "undefined") {
            network.on("click", function (params) {
                if (params.nodes.length > 0) {
                    var node = nodes.get(params.nodes[0]);
                    if (node && node.url) { window.open(node.url, "_blank"); }
                }
            });
        }
    </script>
    </body>"""
    html = html.replace("</body>", click_handler)
    html_path.write_text(html, encoding="utf-8")

    logger.info("Network HTML saved to %s", output_path)
    return output_path


def create_centrality_bar_chart(
    centrality_df: pd.DataFrame,
    top_n: int = 15,
) -> go.Figure:
    """Horizontal bar chart of top nodes by PageRank."""
    df = centrality_df.head(top_n).sort_values("pagerank", ascending=True)
    fig = px.bar(
        df, x="pagerank", y="title", orientation="h",
        color="pagerank", color_continuous_scale="Viridis",
        title=f"Top {top_n} Most Influential Topics (by PageRank)",
        labels={"pagerank": "PageRank Score", "title": "Topic"},
    )
    fig.update_layout(
        height=max(400, top_n * 30),
        showlegend=False,
        yaxis_title="",
        xaxis_title="PageRank Score",
    )
    return fig


def create_community_chart(community_sizes: pd.DataFrame) -> go.Figure:
    """Bar chart of community sizes."""
    df = community_sizes.copy()
    df["community_id"] = df["community_id"].astype(str)
    colors = [_get_color(int(c)) for c in df["community_id"]]
    fig = go.Figure(data=[
        go.Bar(
            x=df["community_id"],
            y=df["size"],
            marker_color=colors,
            text=df["size"],
            textposition="auto",
        )
    ])
    fig.update_layout(
        title="Community Sizes",
        xaxis_title="Community",
        yaxis_title="Number of Topics",
        height=400,
    )
    return fig


def create_degree_distribution(G: nx.DiGraph) -> go.Figure:
    """Histogram of in-degree distribution with log-scale y-axis."""
    in_degrees = [d for _, d in G.in_degree()]
    fig = px.histogram(
        x=in_degrees, nbins=30,
        title="In-Degree Distribution",
        labels={"x": "In-Degree", "y": "Count"},
    )
    fig.update_layout(
        yaxis_type="log",
        yaxis_title="Count (log scale)",
        xaxis_title="In-Degree",
        height=400,
    )
    return fig


def create_community_legend(
    community_labels: dict[int, str],
    community_sizes: pd.DataFrame,
) -> str:
    """Return styled HTML legend for communities."""
    rows = []
    for _, row in community_sizes.iterrows():
        cid = int(row["community_id"])
        color = _get_color(cid)
        label = community_labels.get(cid, f"Community {cid}")
        size = int(row["size"])
        rows.append(
            f'<div style="display:flex;align-items:center;margin:4px 0;">'
            f'<span style="display:inline-block;width:16px;height:16px;'
            f'background:{color};border-radius:3px;margin-right:8px;"></span>'
            f'<b>Community {cid}</b>: {label} ({size} topics)</div>'
        )
    return '<div style="font-family:sans-serif;font-size:14px;">' + "".join(rows) + "</div>"
