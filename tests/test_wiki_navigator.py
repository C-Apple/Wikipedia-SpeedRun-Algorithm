import math

import numpy as np

from src.speedrunning.wiki_navigator import (
    WikipediaClient,
    build_title_scorer,
    find_closest_hyperlink,
    navigate,
    normalize_title,
    title_cooccurrence_corpus,
    title_embedding,
    tokenize_title,
)


class FakeClient:
    def __init__(self, graph):
        self.graph = graph

    def get_hyperlinks(self, page_title, limit=500):
        return self.graph.get(page_title, [])[:limit]


def test_tokenize_and_normalize_multi_word_title():
    assert normalize_title("The_White   House") == "the white house"
    assert tokenize_title("Barack Obama") == ["barack", "obama"]


def test_title_embedding_averages_known_title_tokens():
    word_to_id = {"barack": 0, "obama": 1}
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(title_embedding("Barack Obama", word_to_id, matrix), [0.5, 0.5])


def test_find_closest_hyperlink_prefers_semantic_embedding_score():
    word_to_id = {"barack": 0, "obama": 1, "white": 2, "house": 3, "python": 4}
    matrix = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.1],
            [0.9, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ]
    )
    scorer = build_title_scorer("Barack Obama", word_to_id, matrix)

    link, score = find_closest_hyperlink(["Python", "The White House"], "Barack Obama", scorer)

    assert link == "The White House"
    assert score > 0.9


def test_navigate_logs_each_page_and_stops_when_target_link_is_found():
    client = FakeClient(
        {
            "Start": ["A", "Target"],
            "A": ["Target"],
        }
    )

    result = navigate("Start", "Target", client=client, max_steps=3)

    assert result.reached_target is True
    assert result.path == ["Start", "Target"]
    assert len(result.logs) == 1
    assert result.logs[0].page == "Start"
    assert result.logs[0].links_found == 2
    assert result.logs[0].selected_link == "Target"
    assert math.isinf(result.logs[0].selected_score)


def test_navigate_avoids_cycles_and_reports_no_unvisited_links():
    client = FakeClient({"Start": ["A"], "A": ["Start"]})

    result = navigate("Start", "Target", client=client, max_steps=5)

    assert result.reached_target is False
    assert result.path == ["Start", "A"]
    assert result.reason == "all outgoing links already visited"


def test_title_cooccurrence_corpus_pairs_source_and_link_titles():
    corpus = title_cooccurrence_corpus({"The White House": ["Barack Obama", "United States"]})

    assert corpus == ["the white house barack obama", "the white house united states"]


def test_wikipedia_client_caches_hyperlinks_by_title_and_limit():
    class CachedClient(WikipediaClient):
        def __init__(self):
            super().__init__(api_url="https://example.invalid", max_retries=0)
            self.calls = 0

        def _get_json(self, params):
            self.calls += 1
            return {"query": {"pages": {"1": {"links": [{"title": "A"}, {"title": "B"}]}}}}

    client = CachedClient()

    assert client.get_hyperlinks("Start", limit=2) == ["A", "B"]
    assert client.get_hyperlinks("Start", limit=2) == ["A", "B"]
    assert client.calls == 1
