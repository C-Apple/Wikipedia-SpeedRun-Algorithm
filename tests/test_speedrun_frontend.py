import pytest

from src.speedrun_frontend import (
    SpeedrunResult,
    StepSnapshot,
    load_results,
    parse_wikipedia_title,
    ranked_links,
    render_index,
    run_speedrun,
    save_result,
    summarize_results,
)


class FakeClient:
    def __init__(self, graph):
        self.graph = graph

    def get_hyperlinks(self, page_title, limit=500):
        return self.graph.get(page_title, [])[:limit]


def test_parse_wikipedia_title_accepts_titles_and_urls():
    assert parse_wikipedia_title("Python_(programming_language)") == "Python (programming language)"
    assert parse_wikipedia_title("https://en.wikipedia.org/wiki/Artificial_intelligence") == "Artificial intelligence"
    assert parse_wikipedia_title("https://en.wikipedia.org/w/index.php?title=Alan_Turing") == "Alan Turing"


def test_parse_wikipedia_title_rejects_non_wikipedia_urls():
    with pytest.raises(ValueError, match="Only Wikipedia URLs"):
        parse_wikipedia_title("https://example.com/wiki/Python")


def test_ranked_links_returns_top_scores_in_display_order():
    ranked = ranked_links(["A", "B", "A"], "Target", scorer=lambda title: 1.0 if title == "B" else 0.5)

    assert ranked == [{"title": "B", "score": 1.0}, {"title": "A", "score": 0.5}]


def test_run_speedrun_emits_steps_and_reaches_target_without_network():
    snapshots = []
    result = run_speedrun(
        "Start",
        "Target",
        client=FakeClient({"Start": ["Middle"], "Middle": ["Target"]}),
        max_steps=5,
        on_step=snapshots.append,
    )

    assert result.reached_target is True
    assert result.path == ["Start", "Middle", "Target"]
    assert [step.best_link for step in snapshots] == ["Middle", "Target"]
    assert result.links_visited == 2


def test_save_load_and_summarize_results(tmp_path):
    result = SpeedrunResult(
        id="r1",
        started_at="2026-07-08T00:00:00+00:00",
        start_page="Start",
        target_page="Target",
        model_path=None,
        path=["Start", "Target"],
        steps=[StepSnapshot(0, "Start", 1, "Target", float("inf"), 0.1)],
        reached_target=True,
        reason="target reached",
        total_seconds=0.25,
    )
    file_path = tmp_path / "results.jsonl"
    save_result(result, file_path)

    rows = load_results(file_path)
    assert rows[0]["links_visited"] == 1
    assert summarize_results(rows) == {
        "runs": 1,
        "successes": 1,
        "success_rate": 1.0,
        "avg_seconds": 0.25,
        "avg_links_visited": 1.0,
    }

def test_render_index_contains_speedrun_controls():
    html = render_index()

    assert "Wikipedia SpeedRun Test Bench" in html
    assert "Run 1000 random speedruns" in html
    assert "Local model checkpoint" in html
    assert "/api/jobs" in html
    assert "startLiveTimer" in html
    assert "rank-bar" in html
    assert "scoreColor" in html


def test_bench_job_json_includes_live_path_and_top_links():
    from src.speedrun_frontend import BenchJob

    job = BenchJob(id="job-1", mode="single", total_runs=1)
    job.current_path = ["Start", "Middle"]
    job.current_top_links = [{"title": "Middle", "score": 0.9}]

    payload = job.to_json()

    assert payload["current_path"] == ["Start", "Middle"]
    assert payload["current_top_links"] == [{"title": "Middle", "score": 0.9}]
