import math

import numpy as np
import pytest

from src.speedrunning.wiki_navigator import (
    build_title_scorer,
    cosine_similarity,
    find_closest_hyperlink,
    navigate,
    string_similarity,
    title_cooccurrence_corpus,
    title_embedding,
)


class FakeClient:
    def __init__(self, graph):
        self.graph = graph
        self.calls = []

    def get_hyperlinks(self, page_title, limit=500):
        self.calls.append((page_title, limit))
        return self.graph.get(page_title, [])[:limit]


def test_cosine_similarity_returns_zero_for_zero_vector():
    assert cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 2.0])) == 0.0


def test_string_similarity_is_case_and_underscore_insensitive():
    assert string_similarity("Artificial_Intelligence", "artificial intelligence") == 1.0


def test_title_embedding_ignores_unknown_tokens_and_returns_none_when_empty():
    matrix = np.array([[2.0, 0.0]])
    assert np.allclose(title_embedding("Known Missing", {"known": 0}, matrix), [2.0, 0.0])
    assert title_embedding("Missing", {"known": 0}, matrix) is None


def test_build_title_scorer_falls_back_for_unknown_candidate_tokens():
    scorer = build_title_scorer("Deep Learning", {"deep": 0, "learning": 1}, np.eye(2))

    assert scorer("Deep Learning") == pytest.approx(1.0)


def test_find_closest_hyperlink_deduplicates_and_tie_breaks_by_title():
    link, score = find_closest_hyperlink(["Beta", "Alpha", "Beta"], "Target", scorer=lambda _: 0.5)

    assert link == "Beta"
    assert score == 0.5


def test_find_closest_hyperlink_rejects_empty_links():
    with pytest.raises(ValueError, match="empty link list"):
        find_closest_hyperlink([], "Target")


def test_navigate_stops_immediately_when_start_matches_target():
    result = navigate("Target", "target", client=FakeClient({}), max_steps=3)

    assert result.reached_target is True
    assert result.path == ["Target"]
    assert result.logs == []


def test_navigate_reports_no_outgoing_links():
    result = navigate("Start", "Target", client=FakeClient({"Start": []}), max_steps=3)

    assert result.reached_target is False
    assert result.reason == "no outgoing article links"
    assert result.logs[0].links_found == 0


def test_navigate_respects_link_limit_and_records_score():
    client = FakeClient({"Start": ["Wrong", "Target"]})
    result = navigate("Start", "Target", client=client, max_steps=1, link_limit=1, scorer=lambda title: 0.25 if title == "Wrong" else math.inf)

    assert client.calls == [("Start", 1)]
    assert result.path == ["Start", "Wrong"]
    assert result.logs[0].selected_score == 0.25
    assert result.reason == "max steps reached"


def test_title_cooccurrence_corpus_skips_blank_titles_after_normalization():
    assert title_cooccurrence_corpus({"  ": ["Target"], "Start": [" ", "Target"]}) == ["start target"]
