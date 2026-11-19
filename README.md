# 🕳️ Wikipedia Rabbit Hole Mapper

Given a starting Wikipedia topic, crawl the articles it links to, build a graph of how ideas connect, detect hidden communities of related topics, and explore everything in an interactive Streamlit app.

Ever clicked from **Black hole** to **Accretion disk** to **Quasar** and wondered how far one idea can take you? This project maps that journey — and finds the surprising bridges between worlds of knowledge.

## Screenshots

| Network Map | Communities | Path Finder |
|---|---|---|
| ![Network map](docs/screenshot_network.png) | ![Communities](docs/screenshot_communities.png) | ![Path finder](docs/screenshot_path.png) |

*(Add screenshots to `docs/` after running the app.)*

## Features

- **Polite MediaWiki crawler** — uses the official API (`prop=links`, `prop=extracts`), never HTML scraping. Handles pagination (`plcontinue`), redirects, missing pages, and skips non-article pages (File:, Category:, disambiguations, year/list pages). Custom User-Agent, 0.1–0.2 s rate limiting, exponential backoff honoring `Retry-After`, batched extracts (20 titles/request), and a persistent JSON disk cache.
- **Directed link graph** — NetworkX DiGraph with title, summary, depth, and degree attributes; self-loops and isolates removed; largest weakly connected component kept.
- **Community detection** — Louvain modularity (greedy-modularity fallback), with each community auto-named by its top 3 TF-IDF keywords.
- **Semantic clustering** — TF-IDF + KMeans on article summaries, compared to link communities via the adjusted Rand index.
- **Centrality analysis** — PageRank, betweenness, and in-degree; "most influential topics" table and charts.
- **Bridge topics** — high-betweenness nodes connecting different communities.
- **Rabbit hole path finder** — shortest chain between any two crawled topics, with a "surprise score" (1 − cosine similarity of the endpoints' summaries).
- **Interactive visualization** — pyvis network (size = PageRank, color = community, hover = summary, click = open Wikipedia), Plotly centrality bars, community-size chart, log-scale degree histogram, and a keyword legend.
- **Streamlit UI** — sidebar controls (topic, depth 1–3, links/page, node cap), live crawl progress, five tabs, GraphML + CSV downloads, friendly errors with search-API suggestions for unknown topics, and `st.cache_data` so repeat runs never re-crawl.

## Installation

Requires Python 3.10+.

```bash
git clone <your-repo-url>
cd wikipedia-rabbit-hole-mapper
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open http://localhost:8501, type a topic, and press **Build map**.

## 🚀 Deploy to Render

### Method 1: Web Service (Direct Git)
1. Push this repository to GitHub or GitLab.
2. In the Render Dashboard, click **New +** → **Web Service** and connect your repository.
3. Configure the service:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Under **Advanced** → **Environment Variables**, add:
   - `PYTHON_VERSION`: `3.10.12`
5. Click **Deploy Web Service**.

### Method 2: Render Blueprint (Infrastructure-as-Code)
This repo contains a pre-configured `render.yaml`. Click **New +** → **Blueprint** on Render, select this repository, and it will auto-configure everything.

### Method 3: Docker on Render
Render automatically recognizes the included `Dockerfile`:
- Select **Environment**: `Docker` when creating the Web Service.

### Example topics to try

| Topic | What you'll find |
|---|---|
| **Black hole** | Astrophysics core (accretion, quasars) vs. relativity-theory community |
| **Jazz** | Genres, instruments, and era-based communities |
| **Cryptography** | Math vs. computer-security clusters, with number theory as a bridge |

Depth 1 is fast (~30 s cold). Depth 2–3 can take several minutes on a cold cache; repeat runs are instant thanks to disk caching.

### Tests

```bash
pytest tests/ -q
```

### Notebook

`notebooks/exploration.ipynb` runs the full pipeline on one example topic with a markdown walkthrough of the findings:

```bash
jupyter notebook notebooks/exploration.ipynb
```

## How the analysis works

1. **Crawl (BFS).** From the resolved start title, fetch up to *N* outgoing article links per page, level by level, until the depth limit or node cap is hit. Intro extracts are fetched in batches of 20 titles. Every API response is cached under `data/cache/` keyed by a hash of the request.
2. **Graph.** Edge A → B exists when A links to B *and* B was crawled. Cleaning removes self-loops and isolated nodes, then keeps the largest weakly connected component so the map is one coherent territory.
3. **Communities.** Louvain maximizes modularity on the undirected graph — groups of articles that link to each other more than to the rest. Each group is labeled by the highest mean-TF-IDF terms across its members' summaries.
4. **Text vs. links.** KMeans on TF-IDF vectors clusters articles by *what they say*; Louvain clusters by *what they link to*. The adjusted Rand index quantifies the agreement (1 = identical partitions, 0 = chance).
5. **Influence & bridges.** PageRank measures influence through incoming links; betweenness measures how often a node sits on shortest paths. Nodes with high betweenness *and* neighbors in several communities are "bridges".
6. **Paths & surprise.** Shortest paths are computed on the undirected graph (links aren't always reciprocal). The surprise score is 1 − cosine similarity between the endpoints' TF-IDF vectors: near 1 means the topics share almost no vocabulary.

## Project structure

```
wikipedia-rabbit-hole-mapper/
├── app.py                  # Streamlit app
├── src/
│   ├── crawler.py          # MediaWiki API: links, extracts, BFS crawl
│   ├── graph_builder.py    # NetworkX graph construction and cleaning
│   ├── analysis.py         # centrality, communities, clustering, paths
│   ├── visualize.py        # pyvis network + Plotly charts
│   └── utils.py            # caching, rate limiting, retries, filtering
├── tests/                  # pytest unit tests (mock graph)
├── notebooks/exploration.ipynb
├── data/cache/             # cached API responses (gitignored)
├── requirements.txt
└── README.md
```

## Limitations

- The API returns links alphabetically; keeping the first *N* per page samples the neighborhood rather than capturing the most relevant links.
- Summaries are truncated intros (~300 chars), so TF-IDF labels are keyword hints, not polished names.
- Louvain is stochastic — community boundaries can shift slightly between runs.
- English Wikipedia only; the link graph reflects editorial structure, not ground-truth semantics.
- Very deep crawls are bounded by the node cap (default 300) to stay fast and polite.

## Future improvements

- Rank candidate links by semantic similarity to the start topic before crawling (guided rabbit holes).
- Weighted edges from link position/section; multilingual support.
- Persist full crawls (not just API responses) and diff maps over time.
- Community labeling with keyphrase extraction or an LLM; hierarchical Louvain levels.
- Export to Gephi/Cytoscape formats and shareable map permalinks.

## API etiquette

This project follows the [Wikimedia API etiquette guidelines](https://www.mediawiki.org/wiki/API:Etiquette): serial requests with a descriptive User-Agent, small delays, batched queries where possible, retry with backoff on 429/503, and aggressive disk caching so repeated runs make zero network calls.