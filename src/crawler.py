"""Wikipedia crawler using the MediaWiki API."""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import requests

from src.utils import (
    RateLimiter,
    cache_key,
    clean_summary,
    load_from_cache,
    retry_with_backoff,
    save_to_cache,
    should_skip_title,
)

logger = logging.getLogger(__name__)

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "WikiRabbitHoleMapper/1.0 (educational project; Python/requests)"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})

_rate_limiter = RateLimiter()


@dataclass
class PageInfo:
    """Information about a Wikipedia page."""

    title: str
    summary: str = ""
    links: list[str] = field(default_factory=list)
    exists: bool = True
    is_redirect: bool = False
    redirect_target: str = ""
    depth: int = 0


def _api_request(params: dict[str, Any]) -> Optional[dict]:
    """Make a cached, rate-limited API request with retry logic."""
    key = cache_key(API_URL, params)
    cached = load_from_cache(key)
    if cached is not None:
        return cached

    _rate_limiter.wait()

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def _do_request() -> dict:
        resp = SESSION.get(API_URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    try:
        data = _do_request()
        save_to_cache(key, data)
        return data
    except Exception as e:
        logger.error("API request failed: %s", e)
        return None


def search_titles(query: str, limit: int = 5) -> list[str]:
    """Search for Wikipedia page titles matching a query."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
    }
    data = _api_request(params)
    if not data:
        return []
    return [r["title"] for r in data.get("query", {}).get("search", [])]


def fetch_page_links(title: str, max_links: int = 25) -> list[str]:
    """Fetch outgoing links from a Wikipedia page with pagination."""
    all_links: list[str] = []
    params: dict[str, Any] = {
        "action": "query",
        "titles": title,
        "prop": "links",
        "plnamespace": 0,
        "pllimit": "max",
        "format": "json",
        "redirects": 1,
    }
    while True:
        data = _api_request(params)
        if not data:
            break
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id == "-1":
                return []
            for link in page_data.get("links", []):
                link_title = link.get("title", "")
                if link_title and not should_skip_title(link_title):
                    all_links.append(link_title)
                    if len(all_links) >= max_links:
                        return all_links
        if "continue" in data:
            params["plcontinue"] = data["continue"]["plcontinue"]
        else:
            break
    return all_links


def fetch_page_summary(title: str) -> str:
    """Fetch the introductory extract of a page."""
    return fetch_summaries_batch([title]).get(title, "")


def fetch_summaries_batch(titles: list[str], batch_size: int = 20) -> dict[str, str]:
    """Fetch intro extracts for many titles in batched API requests.

    The extracts prop accepts up to 20 titles per request, which is far
    more polite (and faster) than one request per page.
    """
    summaries: dict[str, str] = {}
    for i in range(0, len(titles), batch_size):
        chunk = titles[i : i + batch_size]
        params: dict[str, Any] = {
            "action": "query",
            "titles": "|".join(chunk),
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "exsectionformat": "plain",
            "format": "json",
            "redirects": 1,
        }
        data = _api_request(params)
        if not data:
            continue
        query = data.get("query", {})
        by_title = {
            p.get("title", ""): clean_summary(p.get("extract", ""))
            for p in query.get("pages", {}).values()
        }
        renames: dict[str, str] = {}
        for item in query.get("normalized", []) + query.get("redirects", []):
            renames[item["from"]] = item["to"]
        for title in chunk:
            resolved = title
            seen: set[str] = set()
            while resolved in renames and resolved not in seen:
                seen.add(resolved)
                resolved = renames[resolved]
            summaries[title] = by_title.get(resolved, by_title.get(title, ""))
    return summaries


def check_page_exists(title: str) -> tuple[bool, str]:
    """Check if a page exists and resolve redirects."""
    params: dict[str, Any] = {
        "action": "query",
        "titles": title,
        "redirects": 1,
        "format": "json",
    }
    data = _api_request(params)
    if not data:
        return False, title
    redirects = data.get("query", {}).get("redirects", [])
    resolved = redirects[-1]["to"] if redirects else title
    pages = data.get("query", {}).get("pages", {})
    for page_id in pages:
        if page_id == "-1":
            return False, resolved
        return True, resolved
    return False, title


def crawl(
    start_topic: str,
    max_depth: int = 2,
    max_links_per_page: int = 25,
    node_cap: int = 300,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> dict[str, PageInfo]:
    """BFS crawl from start_topic up to max_depth."""
    exists, resolved_title = check_page_exists(start_topic)
    if not exists:
        suggestions = search_titles(start_topic)
        raise ValueError(
            f"Page '{start_topic}' not found. "
            f"Did you mean: {', '.join(suggestions)}"
            if suggestions
            else f"Page '{start_topic}' not found."
        )

    pages: dict[str, PageInfo] = {}
    visited: set[str] = {resolved_title}
    current_level: list[str] = [resolved_title]
    depth = 0

    while current_level and len(pages) < node_cap:
        level_links: list[tuple[str, list[str]]] = []
        for title in current_level:
            if len(pages) + len(level_links) >= node_cap:
                break
            if progress_callback:
                progress_callback(
                    len(pages),
                    f"Crawling: {title} (depth {depth}, {len(pages)} nodes so far)",
                )
            links = fetch_page_links(title, max_links=max_links_per_page)
            level_links.append((title, links))

        summaries = fetch_summaries_batch([t for t, _ in level_links])

        next_level: list[str] = []
        for title, links in level_links:
            pages[title] = PageInfo(
                title=title,
                summary=summaries.get(title, ""),
                links=links,
                exists=True,
                depth=depth,
            )
            if depth < max_depth:
                for link_title in links:
                    if link_title not in visited and len(visited) < node_cap:
                        visited.add(link_title)
                        next_level.append(link_title)

        current_level = next_level
        depth += 1

    if progress_callback:
        progress_callback(len(pages), f"Crawling complete: {len(pages)} pages")

    return pages
