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

LOGGER = logging.getLogger(__name__)


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

    def get_hyperlinks(self, page_title: str, limit: int = 500) -> list[str]:
        """Return article-title hyperlinks from ``page_title``.

        The MediaWiki ``links`` property can be paginated. This method follows
        continuation tokens, excludes non-article namespaces, and de-duplicates
        titles while preserving first-seen order.
        """

        if not page_title or not page_title.strip():
            raise ValueError("page_title must be a non-empty string")

        cache_key = (normalize_title(page_title), max(1, limit))
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


def normalize_title(title: str) -> str:
    """Normalize a Wikipedia title for scoring/training."""

    return " ".join(title.replace("_", " ").casefold().split())


def tokenize_title(title: str) -> list[str]:
    """Tokenize multi-word page titles such as 'The White House'."""

    return normalize_title(title).split()


def title_embedding(title: str, word_to_id: dict[str, int], embedding_matrix: np.ndarray) -> np.ndarray | None:
    """Average known token embeddings for a page title.

    Averaging title tokens lets the existing SGNS word vectors represent page
    names containing multiple words, and can be trained further using
    title/link co-occurrence pairs.
    """

    vectors = []
    for token in tokenize_title(title):
        idx = word_to_id.get(token)
        if idx is not None and idx < len(embedding_matrix):
            vectors.append(embedding_matrix[idx])
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def string_similarity(candidate: str, target: str) -> float:
    """Deterministic fallback when trained embeddings are unavailable."""

    return SequenceMatcher(None, normalize_title(candidate), normalize_title(target)).ratio()


def build_title_scorer(
    target_title: str,
    word_to_id: dict[str, int] | None = None,
    embedding_matrix: np.ndarray | None = None,
) -> Callable[[str], float]:
    """Create a candidate scorer using embeddings when possible."""

    target_vec = None
    if word_to_id is not None and embedding_matrix is not None:
        target_vec = title_embedding(target_title, word_to_id, embedding_matrix)

    def score(candidate_title: str) -> float:
        if target_vec is not None and word_to_id is not None and embedding_matrix is not None:
            candidate_vec = title_embedding(candidate_title, word_to_id, embedding_matrix)
            if candidate_vec is not None:
                return cosine_similarity(candidate_vec, target_vec)
        return string_similarity(candidate_title, target_title)

    return score


def find_closest_hyperlink(links: Iterable[str], target_title: str, scorer: Callable[[str], float] | None = None) -> tuple[str, float]:
    """Return the highest-scoring outgoing link for ``target_title``."""

    candidates = list(dict.fromkeys(links))
    if not candidates:
        raise ValueError("Cannot choose a hyperlink from an empty link list")
    scorer = scorer or build_title_scorer(target_title)
    scored = [(candidate, scorer(candidate)) for candidate in candidates]
    return max(scored, key=lambda item: (item[1], item[0]))


def navigate(
    start_page: str,
    target_page: str,
    client: WikipediaClient | None = None,
    max_steps: int = 25,
    link_limit: int = 500,
    scorer: Callable[[str], float] | None = None,
) -> NavigationResult:
    """Greedily navigate Wikipedia hyperlinks toward ``target_page``."""

    client = client or WikipediaClient()
    scorer = scorer or build_title_scorer(target_page)
    current = start_page
    path = [current]
    logs: list[PageVisitLog] = []
    visited = {normalize_title(current)}

    for _ in range(max_steps):
        if normalize_title(current) == normalize_title(target_page):
            return NavigationResult(path=path, logs=logs, reached_target=True, reason="target reached")

        started = time.perf_counter()
        links = client.get_hyperlinks(current, limit=link_limit)
        elapsed = time.perf_counter() - started
        if not links:
            logs.append(PageVisitLog(current, elapsed, 0, None, None))
            return NavigationResult(path=path, logs=logs, reason="no outgoing article links")

        if normalize_title(target_page) in {normalize_title(link) for link in links}:
            selected, score = target_page, math.inf
        else:
            unvisited = [link for link in links if normalize_title(link) not in visited]
            if not unvisited:
                logs.append(PageVisitLog(current, elapsed, len(links), None, None))
                return NavigationResult(path=path, logs=logs, reason="all outgoing links already visited")
            selected, score = find_closest_hyperlink(unvisited, target_page, scorer)

        logs.append(PageVisitLog(current, elapsed, len(links), selected, score))
        LOGGER.info("visited=%s links=%s selected=%s score=%s elapsed=%.3fs", current, len(links), selected, score, elapsed)
        current = selected
        path.append(current)
        visited.add(normalize_title(current))

    return NavigationResult(path=path, logs=logs, reached_target=normalize_title(current) == normalize_title(target_page), reason="max steps reached")


def title_cooccurrence_corpus(page_to_links: dict[str, Iterable[str]]) -> list[str]:
    """Build training lines from titles that appear together in link graphs.

    Each line contains a source title and one linked title. Feeding these lines
    into the existing SGNS dataloader trains tokens from related page titles to
    share context, e.g. ``the white house`` with ``barack obama``.
    """

    corpus: list[str] = []
    for page, links in page_to_links.items():
        source = normalize_title(page)
        for link in links:
            linked = normalize_title(link)
            if source and linked:
                corpus.append(f"{source} {linked}")
    return corpus
