"""Utility functions: caching, rate limiting, and helpers."""

import hashlib
import json
import re
import time
import random
import logging
from pathlib import Path
from typing import Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)

# Default cache directory
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"


def get_cache_dir() -> Path:
    """Return the cache directory, creating it if needed."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def cache_key(url: str, params: dict[str, Any]) -> str:
    """Generate a deterministic cache key from URL and parameters."""
    raw = url + json.dumps(params, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def load_from_cache(key: str) -> Optional[dict]:
    """Load a cached API response from disk."""
    path = get_cache_dir() / f"{key}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Cache read error for %s: %s", key, e)
    return None


def save_to_cache(key: str, data: dict) -> None:
    """Save an API response to disk cache."""
    path = get_cache_dir() / f"{key}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.warning("Cache write error for %s: %s", key, e)


class RateLimiter:
    """Simple rate limiter with configurable delay between calls."""

    def __init__(self, min_delay: float = 0.1, max_delay: float = 0.2) -> None:
        self.min_delay = min_delay
        self.max_delay = max_delay
        self._last_call: float = 0.0

    def wait(self) -> None:
        """Wait the appropriate amount of time before the next API call."""
        now = time.time()
        elapsed = now - self._last_call
        delay = random.uniform(self.min_delay, self.max_delay)
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_call = time.time()


def retry_with_backoff(
    func: Any = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> Any:
    """Decorator: retry a function call with exponential backoff."""
    def decorator(f: Any) -> Any:
        @wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None
            for attempt in range(max_retries + 1):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (backoff_factor ** attempt)
                        response = getattr(e, "response", None)
                        if response is not None and response.status_code in (429, 503):
                            retry_after = response.headers.get("Retry-After")
                            try:
                                delay = max(delay, float(retry_after))
                            except (TypeError, ValueError):
                                delay = max(delay, 5.0)
                        logger.warning(
                            "Attempt %d/%d failed: %s. Retrying in %.1fs...",
                            attempt + 1, max_retries + 1, e, delay,
                        )
                        time.sleep(delay)
            raise last_exception  # type: ignore[misc]
        return wrapper
    if func is not None:
        return decorator(func)
    return decorator


# Pages to skip
SKIP_PREFIXES = ("File:", "Category:", "Help:", "Template:", "Wikipedia:",
                 "Portal:", "Draft:", "Module:", "MediaWiki:", "Talk:",
                 "User:", "Special:", "Book:")

SKIP_PATTERNS = ("disambiguation", "List of", "Index of", "Outline of",
                 "Timeline of")

_YEAR_RE = re.compile(r"^\d{1,4}(\s*(BC|AD|CE|BCE))?$")


def should_skip_title(title: str) -> bool:
    """Check if a Wikipedia title should be skipped."""
    if any(title.startswith(prefix) for prefix in SKIP_PREFIXES):
        return True
    if any(pattern.lower() in title.lower() for pattern in SKIP_PATTERNS):
        return True
    if _YEAR_RE.match(title.strip()):
        return True
    return False


def clean_summary(text: str, max_length: int = 300) -> str:
    """Truncate a summary to max_length at a sentence boundary."""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    last_period = truncated.rfind(".")
    if last_period > max_length // 2:
        return truncated[: last_period + 1]
    return truncated + "..."
