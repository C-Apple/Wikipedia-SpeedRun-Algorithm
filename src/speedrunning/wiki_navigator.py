"""Wikipedia hyperlink navigation helpers.

The navigator fetches outgoing links for a page, scores each candidate against a
target title, and repeats until the target is reached or a stopping condition is
met.  It is intentionally small and testable: network access is isolated behind
``WikipediaClient`` and scoring can use either trained title embeddings or a
string-similarity fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from difflib import SequenceMatcher
import json
import logging
import math
import time
from typing import Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from torch import nn

LOGGER = logging.getLogger(__name__)
from src.config import LINK_LIMIT


@dataclass(frozen=True)
class PageVisitLog:
    """One navigation step with timing and scoring details."""

    page: str
    elapsed_seconds: float
    links_found: int
    selected_link: str | None
    selected_score: float | None


@dataclass
class NavigationResult:
    """Path and per-page logs returned by ``navigate``."""

    path: list[str]
    logs: list[PageVisitLog] = field(default_factory=list)
    reached_target: bool = False
    reason: str = ""


class WikipediaClient:
    """Small client for the MediaWiki API link list endpoint.

    The frontend can request the same page repeatedly during experiments. This
    client keeps a small in-memory cache and backs off on HTTP 429 responses so
    a good run is reusable instead of immediately causing repeated rate-limit
    failures.
    """

    def __init__(
        self,
        api_url: str = "https://en.wikipedia.org/w/api.php",
        timeout: float = 10.0,
        *,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
    ):
        self.api_url = api_url
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.backoff_seconds = max(0.0, backoff_seconds)
        self._cache: dict[tuple[str, int], list[str]] = {}

    def normalize_title(self, title: str) -> str:
        """Normalize a Wikipedia title for scoring/training."""

        return " ".join(title.replace("_", " ").casefold().split())

    def get_hyperlinks(self, page_title: str, limit: int = 500) -> list[str]:
        """Return article-title hyperlinks from ``page_title``.

        The MediaWiki ``links`` property can be paginated. This method follows
        continuation tokens, excludes non-article namespaces, and de-duplicates
        titles while preserving first-seen order.
        """

        if not page_title or not page_title.strip():
            raise ValueError("page_title must be a non-empty string")

        cache_key = ((self.normalize_title(page_title)), max(1, limit))
        if cache_key in self._cache:
            return list(self._cache[cache_key])

        remaining = max(1, limit)
        params = {
            "action": "query",
            "format": "json",
            "prop": "links",
            "titles": page_title,
            "plnamespace": 0,
            "pllimit": min(remaining, 500),
            "redirects": 1,
        }
        links: list[str] = []
        seen: set[str] = set()

        while remaining > 0:
            payload = self._get_json(params)
            pages = payload.get("query", {}).get("pages", {})
            for page in pages.values():
                for link in page.get("links", []):
                    title = link.get("title")
                    if title and title not in seen:
                        links.append(title)
                        seen.add(title)
                        remaining -= 1
                        if remaining == 0:
                            break
                if remaining == 0:
                    break

            continuation = payload.get("continue")
            if not continuation:
                break
            params.update(continuation)
            params["pllimit"] = min(remaining, 500)

        self._cache[cache_key] = list(links)
        return links

    def _get_json(self, params: dict[str, object]) -> dict[str, object]:
        url = f"{self.api_url}?{urlencode(params)}"
        request = Request(url, headers={"User-Agent": "Wikipedia-SpeedRun-Algorithm/0.1 (educational speedrun test bench)"})
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                if exc.code != 429 or attempt >= self.max_retries:
                    raise ConnectionError(f"Unable to fetch Wikipedia links: {exc}") from exc
                time.sleep(self._retry_delay(exc, attempt))
            except (URLError, TimeoutError) as exc:
                if attempt >= self.max_retries:
                    raise ConnectionError(f"Unable to fetch Wikipedia links: {exc}") from exc
                time.sleep(self.backoff_seconds * (2**attempt))
        raise ConnectionError("Unable to fetch Wikipedia links after retries")

    def _retry_delay(self, exc: HTTPError, attempt: int) -> float:
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    return max(0.0, retry_at.timestamp() - time.time())
                except (TypeError, ValueError):
                    pass
        return self.backoff_seconds * (2**attempt)