"""Browser test bench for running Wikipedia speedruns.

Run ``python -m src.speedrun_frontend`` and open the printed localhost URL.
The app lets you choose start/end Wikipedia URLs or titles, optionally point at a
local SGNS checkpoint, watch each greedy navigation step, persist results, and
launch batch benchmark runs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

from src.config import LOAD_MODEL_PATH
from src.speedrunning.vector_visualization import EmbeddingProjector3D
from src.speedrunning.wiki_navigator import WikipediaClient
from src.speedrunning.speedrunner import build_title_scorer, normalize_title, project_title, speedrun_step
from src.training.save_configs import load_checkpoint
from src.training.word_embedding import SkipGramNegSampling

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "runs" / "speedrun_bench"
RESULTS_FILE = RESULTS_DIR / "results.jsonl"
RANDOM_PAIRS = [
    ("Python (programming language)", "Artificial intelligence"),
    ("Northwestern University", "Chicago"),
    ("The White House", "Barack Obama"),
    ("Computer science", "Alan Turing"),
    ("Basketball", "Michael Jordan"),
    ("World War II", "United Nations"),
    ("Machine learning", "Neural network"),
    ("United States", "Mount Rushmore"),
]

@dataclass
@dataclass
class StepSnapshot:
    step: int
    page: str
    links_found: int
    best_link: str | None
    best_score: float | None
    elapsed_seconds: float
    top_links: list[dict[str, float | str]] = field(
        default_factory=list
    )

    vector_path: list[dict[str, Any]] = field(
        default_factory=list
    )

    target_vector: dict[str, Any] | None = None


@dataclass
class SpeedrunResult:
    id: str
    started_at: str
    start_page: str
    target_page: str
    model_path: str | None
    path: list[str]
    steps: list[StepSnapshot]
    reached_target: bool
    reason: str
    total_seconds: float

    @property
    def links_visited(self) -> int:
        return max(0, len(self.path) - 1)

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["links_visited"] = self.links_visited
        return payload


@dataclass
class BenchJob:
    id: str
    mode: str
    total_runs: int
    status: str = "queued"
    current_run: int = 0
    current_page: str = ""
    best_link: str | None = None
    elapsed_seconds: float = 0.0
    current_path: list[str] = field(default_factory=list)
    current_top_links: list[
        dict[str, float | str]
    ] = field(default_factory=list)

    current_vector_path: list[
        dict[str, Any]
    ] = field(default_factory=list)

    target_vector: dict[str, Any] | None = None

    results: list[SpeedrunResult] = field(
        default_factory=list
    )
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mode": self.mode,
            "total_runs": self.total_runs,
            "status": self.status,
            "current_run": self.current_run,
            "current_page": self.current_page,
            "best_link": self.best_link,
            "elapsed_seconds": self.elapsed_seconds,
            "current_path": self.current_path,
            "current_top_links": self.current_top_links,
            "error": self.error,
            "results": [result.to_json() for result in self.results[-25:]],
            "summary": summarize_results(self.results),
        }


def parse_wikipedia_title(value: str) -> str:
    """Extract a page title from a Wikipedia URL or pass through a title."""

    value = value.strip()
    if not value:
        raise ValueError("Wikipedia title or URL is required")
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        if "wikipedia.org" not in parsed.netloc:
            raise ValueError("Only Wikipedia URLs are supported")
        if parsed.path.startswith("/wiki/"):
            return unquote(parsed.path.removeprefix("/wiki/")).replace("_", " ")
        query_title = parse_qs(parsed.query).get("title", [""])[0]
        if query_title:
            return unquote(query_title).replace("_", " ")
        raise ValueError("Wikipedia URL must contain /wiki/<title> or a title= query parameter")
    return value.replace("_", " ")


def load_checkpoint_scorer(model_path: str | None, target_page: str) -> tuple[Callable[[str], float], str | None]:
    """Build a title scorer from a local checkpoint path or fall back to strings."""

    if not model_path:
        return build_title_scorer(target_page), None
    path = Path(model_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")
    from src.training.save_configs import load_checkpoint
    from src.training.word_embedding import SkipGramNegSampling

    model, word_to_id, _id_to_word, _config, _ckpt = load_checkpoint(path, SkipGramNegSampling)
    matrix = model.in_embedding.weight.detach().cpu().numpy()
    return build_title_scorer(target_page, word_to_id, matrix), str(path)


def ranked_links(links: list[str], target_page: str, scorer: Callable[[str], float], limit: int = 10) -> list[dict[str, float | str]]:
    """Return top-ranked candidate links for display."""

    unique = list(dict.fromkeys(links))
    ranked = sorted(((link, scorer(link)) for link in unique), key=lambda item: (item[1], item[0]), reverse=True)
    return [{"title": title, "score": score} for title, score in ranked[:limit]]


def run_speedrun(
    start_page: str,
    target_page: str,
    *,
    max_steps: int = 25,
    link_limit: int = 500,
    client: WikipediaClient | None = None,
    on_step: Callable[[StepSnapshot], None] | None = None,
    model_path: str | None = None,
) -> SpeedrunResult:
    """Run one speedrun using speedrun_step as the navigation source."""

    client = client or WikipediaClient()

    # The centralized speedrun algorithm requires the trained model config.
    checkpoint_path = Path(
        model_path or LOAD_MODEL_PATH
    ).expanduser()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {checkpoint_path}"
        )

    model, word_to_id, id_to_word, model_config, _ckpt = (
        load_checkpoint(
            checkpoint_path,
            SkipGramNegSampling,
        )
    )

    config = {
        "model": model,
        "word_to_id": word_to_id,
        "id_to_word": id_to_word,
        "model_config": model_config,
    }

    # Build one stable 3D projection for the entire speedrun.
    embedding_matrix = (
        model.in_embedding.weight
        .detach()
        .cpu()
        .numpy()
    )

    projector = EmbeddingProjector3D(
        embedding_matrix
    )

    start_position = project_title(
        start_page,
        word_to_id,
        model,
        projector,
    )

    target_position = project_title(
        target_page,
        word_to_id,
        model,
        projector,
    )

    vector_path: list[dict[str, Any]] = []

    if start_position is not None:
        vector_path.append(
            {
                "title": start_page,
                "position": start_position,
            }
        )

    target_vector = None

    if target_position is not None:
        target_vector = {
            "title": target_page,
            "position": target_position,
        }

    run_id = uuid.uuid4().hex
    started_at = datetime.now(
        timezone.utc
    ).isoformat()
    timer_started = time.perf_counter()

    current_page = start_page

    # Active path used by speedrun_step for backtracking.
    page_history = [start_page]

    # Every page ever visited, including dead ends that were backtracked from.
    visited_pages = {
        normalize_title(start_page)
    }

    # Full traversal shown and saved by the frontend.
    traversal_path = [start_page]

    steps: list[StepSnapshot] = []

    reached = (
        normalize_title(start_page)
        == normalize_title(target_page)
    )

    reason = (
        "target reached"
        if reached
        else "max steps reached"
    )

    for step_number in range(max_steps):
        if reached:
            break

        result = speedrun_step(
            current_page=current_page,
            end_page=target_page,
            config=config,
            page_history=page_history,
            visited_pages=visited_pages,
            started_at=timer_started,
            wiki_client=client,
            link_limit=link_limit,
        )

        top_links = [
            {
                "title": title,
                "score": float(score),
            }
            for title, score
            in result.ranked_links[:10]
        ]

        # Project the selected next page into the same 3D space.
        if not result.stopped:
            next_position = project_title(
                result.next_page,
                word_to_id,
                model,
                projector,
            )

            if next_position is not None:
                if (
                    not vector_path
                    or vector_path[-1]["title"]
                    != result.next_page
                ):
                    vector_path.append(
                        {
                            "title": result.next_page,
                            "position": next_position,
                        }
                    )

        snapshot = StepSnapshot(
            step=step_number,
            page=result.current_page,
            links_found=result.links_found,
            best_link=(
                None
                if result.stopped
                else result.next_page
            ),
            best_score=result.selected_score,
            elapsed_seconds=result.elapsed_seconds,
            top_links=top_links,

            # Send a copy so later mutations do not alter old snapshots.
            vector_path=[
                {
                    "title": point["title"],
                    "position": list(
                        point["position"]
                    ),
                }
                for point in vector_path
            ],
            target_vector=(
                {
                    "title": target_vector["title"],
                    "position": list(
                        target_vector["position"]
                    ),
                }
                if target_vector is not None
                else None
            ),
        )

        steps.append(snapshot)

        if on_step is not None:
            print(
                f"[FRONTEND CALLBACK] "
                f"{snapshot.page} -> "
                f"{snapshot.best_link}"
            )

            on_step(snapshot)

        if result.stopped:
            reason = (
                result.reason
                or "speedrun stopped"
            )
            break

        current_page = result.next_page
        traversal_path.append(current_page)

        # speedrun_step already pops page_history when backtracking.
        if not result.backtracked:
            page_history.append(
                current_page
            )
            visited_pages.add(
                normalize_title(current_page)
            )

        if result.page_found:
            reached = True
            reason = "target reached"
            break

    else:
        reached = (
            normalize_title(current_page)
            == normalize_title(target_page)
        )

        if reached:
            reason = "target reached"
        else:
            reason = "max steps reached"

    total_seconds = (
        time.perf_counter()
        - timer_started
    )

    return SpeedrunResult(
        id=run_id,
        started_at=started_at,
        start_page=start_page,
        target_page=target_page,
        model_path=str(checkpoint_path),
        path=traversal_path,
        steps=steps,
        reached_target=reached,
        reason=reason,
        total_seconds=total_seconds,
    )

def save_result(result: SpeedrunResult, results_file: Path = RESULTS_FILE) -> None:
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with results_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.to_json()) + "\n")


def load_results(results_file: Path = RESULTS_FILE) -> list[dict[str, Any]]:
    if not results_file.exists():
        return []
    rows = []
    with results_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_results(results: list[SpeedrunResult] | list[dict[str, Any]]) -> dict[str, float | int]:
    if not results:
        return {"runs": 0, "successes": 0, "success_rate": 0.0, "avg_seconds": 0.0, "avg_links_visited": 0.0}
    def get(item: Any, key: str) -> Any:
        return item.get(key) if isinstance(item, dict) else getattr(item, key)
    runs = len(results)
    successes = sum(1 for result in results if get(result, "reached_target"))
    avg_seconds = sum(float(get(result, "total_seconds")) for result in results) / runs
    avg_links = sum(int(get(result, "links_visited") if isinstance(result, dict) else result.links_visited) for result in results) / runs
    return {"runs": runs, "successes": successes, "success_rate": successes / runs, "avg_seconds": avg_seconds, "avg_links_visited": avg_links}


JOBS: dict[str, BenchJob] = {}
JOBS_LOCK = threading.Lock()


def run_job(
    job: BenchJob,
    config: dict[str, Any],
) -> None:
    print(
    f"[JOB STARTED] {job.id}",
    flush=True,
)
    try:
        with JOBS_LOCK:
            job.status = "running"
            job.error = None

        client = WikipediaClient()

        for index in range(job.total_runs):
            if config.get("randomize"):
                start_page, target_page = random.choice(
                    RANDOM_PAIRS
                )
            else:
                start_page = parse_wikipedia_title(
                    config["start"]
                )
                target_page = parse_wikipedia_title(
                    config["target"]
                )

            with JOBS_LOCK:
                job.current_run = index + 1
                job.current_page = start_page
                job.best_link = None
                job.elapsed_seconds = 0.0
                job.current_path = [start_page]
                job.current_top_links = []

            def update(snapshot: StepSnapshot) -> None:
                """Receive one completed step from run_speedrun."""

                with JOBS_LOCK:
                    job.current_page = snapshot.page
                    job.best_link = snapshot.best_link
                    job.elapsed_seconds = (
                        snapshot.elapsed_seconds
                    )
                    job.current_top_links = list(
                        snapshot.top_links
                    )

                    if snapshot.best_link is not None:
                        job.current_path = [
                            *job.current_path,
                            snapshot.best_link,
                        ]

            result = run_speedrun(
                start_page=start_page,
                target_page=target_page,
                model_path=(
                    config.get("model_path") or None
                ),
                max_steps=int(
                    config.get("max_steps", 25)
                ),
                link_limit=int(
                    config.get("link_limit", 500)
                ),
                client=client,
                on_step=update,
            )

            save_result(result)

            with JOBS_LOCK:
                job.results.append(result)
                job.current_page = (
                    result.path[-1]
                    if result.path
                    else start_page
                )
                job.current_path = list(result.path)
                job.elapsed_seconds = result.total_seconds

        with JOBS_LOCK:
            job.status = "completed"

    except Exception as exc:
        with JOBS_LOCK:
            job.status = "failed"
            job.error = str(exc)


def start_job(
    config: dict[str, Any],
    total_runs: int,
    mode: str,
) -> BenchJob:
    job = BenchJob(
        id=uuid.uuid4().hex,
        mode=mode,
        total_runs=total_runs,
    )

    with JOBS_LOCK:
        JOBS[job.id] = job

    print(
        f"[JOB CREATED] {job.id} mode={mode}",
        flush=True,
    )

    thread = threading.Thread(
        target=run_job,
        args=(job, config),
        daemon=True,
    )

    thread.start()

    return job



class SpeedrunFrontendHandler(BaseHTTPRequestHandler):
    server_version = "WikipediaSpeedRunBench/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(render_index())
        elif parsed.path == "/api/results":
            rows = load_results()
            self._send_json({"results": rows, "summary": summarize_results(rows)})
        elif parsed.path.startswith("/api/jobs/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            job = JOBS.get(job_id)
            if not job:
                self.send_error(HTTPStatus.NOT_FOUND, "Unknown job")
                return
            self._send_json(job.to_json())
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/jobs":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        data = self._read_json()
        runs = int(data.get("runs", 1))
        runs = max(1, min(1000, runs))
        job = start_job(data, runs, str(data.get("mode", "single")))
        self._send_json(job.to_json(), status=HTTPStatus.ACCEPTED)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        return json.loads(raw or "{}")

    def _send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def build_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), SpeedrunFrontendHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Wikipedia speedrun benchmark UI")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    server = build_server(args.host, args.port)
    print(f"Speedrun test bench running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down speedrun test bench")
    finally:
        server.server_close()

def render_index() -> str:
    pair_options = "".join(
        f'<option value="{start} → {target}">'
        for start, target in RANDOM_PAIRS
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta
    name="viewport"
    content="width=device-width, initial-scale=1"
>
<title>Wikipedia SpeedRun Test Bench</title>

<style>
:root {{
    font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    color: #e5eefb;
    background: #07111f;
}}

body {{
    margin: 0;
}}

main {{
    max-width: 1180px;
    margin: auto;
    padding: 28px;
}}

.hero {{
    display: grid;
    gap: 10px;
    margin-bottom: 22px;
}}

h1 {{
    font-size: clamp(2.2rem, 6vw, 4.4rem);
    margin: 0;
}}

p {{
    color: #9fb3ca;
}}

.panel {{
    background: #0d1b2f;
    border: 1px solid #233b5e;
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 24px 80px #0008;
}}

.grid {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(250px, 1fr));
    gap: 16px;
}}

label {{
    display: grid;
    gap: 7px;
    font-weight: 700;
}}

input {{
    border: 1px solid #34506f;
    border-radius: 12px;
    background: #081525;
    color: #f8fafc;
    padding: 12px;
}}

.actions {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    align-items: center;
    margin-top: 16px;
}}

button {{
    border: 0;
    border-radius: 999px;
    padding: 13px 20px;
    font-weight: 900;
    cursor: pointer;
}}

.primary {{
    background: #38bdf8;
    color: #082f49;
}}

.secondary {{
    background: #a78bfa;
    color: #201047;
}}

.ghost {{
    background: #203450;
    color: #e2e8f0;
}}

.metrics {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(170px, 1fr));
    gap: 12px;
    margin: 18px 0;
}}

.metric {{
    background: #081525;
    border-radius: 16px;
    padding: 16px;
}}

.metric b {{
    display: block;
    font-size: 1.8rem;
}}

.status {{
    color: #93c5fd;
    font-weight: 800;
}}

.path,
.top {{
    background: #020617;
    border-radius: 16px;
    padding: 16px;
    min-height: 90px;
    overflow: auto;
}}

canvas {{
    width: 100%;
    min-height: 260px;
    background: #020617;
    border-radius: 16px;
    margin-top: 12px;
}}

.rank-row {{
    display: grid;
    grid-template-columns:
        minmax(120px, 1fr)
        minmax(110px, 36%);
    gap: 12px;
    align-items: center;
    margin: 8px 0;
}}

.rank-bar-track {{
    height: 14px;
    background: #172554;
    border-radius: 999px;
    overflow: hidden;
}}

.rank-bar {{
    height: 100%;
    border-radius: 999px;
    min-width: 3px;
}}

.vector-panel {{
    margin-top: 22px;
}}

.vector-header {{
    display: flex;
    justify-content: space-between;
    gap: 18px;
    align-items: center;
    flex-wrap: wrap;
}}

.vector-legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    color: #cbd5e1;
}}

.legend-item {{
    display: flex;
    align-items: center;
    gap: 7px;
}}

.legend-dot {{
    width: 12px;
    height: 12px;
    border-radius: 50%;
}}

.legend-start {{
    background: #22c55e;
}}

.legend-path {{
    background: #38bdf8;
}}

.legend-current {{
    background: #facc15;
}}

.legend-target {{
    background: #ef4444;
}}

#vectorViz {{
    width: 100%;
    height: 540px;
    background: #020617;
    border: 1px solid #233b5e;
    border-radius: 18px;
    overflow: hidden;
    margin-top: 12px;
}}

.vector-message {{
    color: #94a3b8;
    margin-top: 8px;
}}
</style>
</head>

<body>
<main>

<section class="hero">
    <h1>Wikipedia SpeedRun Test Bench</h1>
    <p>
        Pick start/end pages or random challenges, choose a local
        trained model path, and watch the greedy navigator rank
        links in real time.
    </p>
</section>

<section class="panel">
    <div class="grid">
        <label>
            Start URL or title
            <input
                id="start"
                list="pairs"
                value="Python (programming language)"
            >
        </label>

        <label>
            End URL or title
            <input
                id="target"
                value="Artificial intelligence"
            >
        </label>

        <label>
            Local model checkpoint/folder
            <input
                id="model"
                placeholder="runs/sgns_wiki_v2 or /Users/me/checkpoint.pt"
            >
        </label>

        <label>
            Max steps
            <input
                id="maxSteps"
                type="number"
                min="1"
                value="25"
            >
        </label>

        <label>
            Link limit per page
            <input
                id="linkLimit"
                type="number"
                min="1"
                value="500"
            >
        </label>
    </div>

    <datalist id="pairs">
        {pair_options}
    </datalist>

    <div class="actions">
        <button class="primary" id="runOne">
            Run speedrun
        </button>

        <button class="secondary" id="runBatch">
            Run 1000 random speedruns
        </button>

        <button class="ghost" id="loadResults">
            Reload saved results
        </button>

        <label>
            <input id="randomize" type="checkbox">
            Use random start/end
        </label>

        <span class="status" id="status">
            Idle
        </span>
    </div>
</section>

<section class="metrics">
    <div class="metric">
        Run timer
        <b id="timer">0.00s</b>
    </div>

    <div class="metric">
        Current page
        <b id="page">—</b>
    </div>

    <div class="metric">
        Highest ranked link
        <b id="best">—</b>
    </div>

    <div class="metric">
        Batch avg time
        <b id="avgTime">—</b>
    </div>

    <div class="metric">
        Avg links visited
        <b id="avgLinks">—</b>
    </div>

    <div class="metric">
        Success rate
        <b id="success">—</b>
    </div>
</section>

<section class="grid">
    <div>
        <h2>Current path</h2>
        <div class="path" id="path">
            No run yet.
        </div>
    </div>

    <div>
        <h2>Top ranked hyperlinks now</h2>
        <div class="top" id="top">
            No ranking yet.
        </div>
    </div>
</section>

<section class="vector-panel">
    <div class="vector-header">
        <h2>3D embedding-space path</h2>

        <div class="vector-legend">
            <span class="legend-item">
                <i class="legend-dot legend-start"></i>
                Start
            </span>

            <span class="legend-item">
                <i class="legend-dot legend-path"></i>
                Visited
            </span>

            <span class="legend-item">
                <i class="legend-dot legend-current"></i>
                Current
            </span>

            <span class="legend-item">
                <i class="legend-dot legend-target"></i>
                Target
            </span>
        </div>
    </div>

    <div id="vectorViz"></div>

    <div class="vector-message" id="vectorMessage">
        Run a speedrun to display projected title vectors.
    </div>
</section>

<section>
    <h2>Saved results graph</h2>
    <canvas
        id="chart"
        width="1100"
        height="320"
    ></canvas>
</section>

</main>

<script type="importmap">
{{
    "imports": {{
        "three":
            "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js",
        "three/addons/":
            "https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/"
    }}
}}
</script>

<script type="module">
import * as THREE from "three";
import {{ OrbitControls }} from
    "three/addons/controls/OrbitControls.js";

const container = document.getElementById("vectorViz");
const message = document.getElementById("vectorMessage");

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(
    55,
    container.clientWidth / container.clientHeight,
    0.01,
    100
);

camera.position.set(2.7, 2.2, 3.4);

const renderer = new THREE.WebGLRenderer({{
    antialias: true,
    alpha: true
}});

renderer.setPixelRatio(
    Math.min(window.devicePixelRatio, 2)
);

renderer.setSize(
    container.clientWidth,
    container.clientHeight
);

container.appendChild(renderer.domElement);

const controls = new OrbitControls(
    camera,
    renderer.domElement
);

controls.enableDamping = true;
controls.dampingFactor = 0.08;

scene.add(
    new THREE.AmbientLight(
        0xffffff,
        1.5
    )
);

const directionalLight =
    new THREE.DirectionalLight(
        0xffffff,
        2.2
    );

directionalLight.position.set(3, 4, 5);
scene.add(directionalLight);

const grid = new THREE.GridHelper(
    4,
    12,
    0x334155,
    0x172554
);

grid.position.y = -1.5;
scene.add(grid);

const axes = new THREE.AxesHelper(1.2);
scene.add(axes);

const pathGroup = new THREE.Group();
scene.add(pathGroup);

function clearGroup(group) {{
    while (group.children.length > 0) {{
        const child = group.children.pop();

        if (child.geometry) {{
            child.geometry.dispose();
        }}

        if (child.material) {{
            if (Array.isArray(child.material)) {{
                child.material.forEach(
                    material => material.dispose()
                );
            }} else {{
                child.material.dispose();
            }}
        }}
    }}
}}

function toVector3(position) {{
    return new THREE.Vector3(
        Number(position[0]),
        Number(position[1]),
        Number(position[2])
    );
}}

function addPoint(
    position,
    color,
    radius
) {{
    const geometry =
        new THREE.SphereGeometry(
            radius,
            24,
            16
        );

    const material =
        new THREE.MeshStandardMaterial({{
            color,
            emissive: color,
            emissiveIntensity: 0.25
        }});

    const sphere = new THREE.Mesh(
        geometry,
        material
    );

    sphere.position.copy(
        toVector3(position)
    );

    pathGroup.add(sphere);

    return sphere;
}}

function addArrow(
    fromPosition,
    toPosition
) {{
    const origin = toVector3(fromPosition);
    const destination = toVector3(toPosition);

    const direction =
        new THREE.Vector3().subVectors(
            destination,
            origin
        );

    const length = direction.length();

    if (length < 0.0001) {{
        return;
    }}

    direction.normalize();

    const arrow = new THREE.ArrowHelper(
        direction,
        origin,
        length,
        0xa78bfa,
        Math.min(0.12, length * 0.2),
        Math.min(0.08, length * 0.12)
    );

    pathGroup.add(arrow);
}}

function addTargetConnection(
    currentPosition,
    targetPosition
) {{
    const points = [
        toVector3(currentPosition),
        toVector3(targetPosition)
    ];

    const geometry =
        new THREE.BufferGeometry()
            .setFromPoints(points);

    const material =
        new THREE.LineDashedMaterial({{
            color: 0xef4444,
            dashSize: 0.05,
            gapSize: 0.035,
            transparent: true,
            opacity: 0.65
        }});

    const line = new THREE.Line(
        geometry,
        material
    );

    line.computeLineDistances();
    pathGroup.add(line);
}}

window.renderVectorScene = function(
    vectorPath,
    targetVector
) {{
    clearGroup(pathGroup);

    if (
        !Array.isArray(vectorPath)
        || vectorPath.length === 0
    ) {{
        message.textContent =
            "No projected vectors are available yet.";
        return;
    }}

    message.textContent =
        "Drag to rotate, scroll to zoom, and right-drag to pan.";

    vectorPath.forEach(
        (point, index) => {{
            if (
                !point
                || !Array.isArray(point.position)
                || point.position.length !== 3
            ) {{
                return;
            }}

            const isStart = index === 0;
            const isCurrent =
                index === vectorPath.length - 1;

            let color = 0x38bdf8;
            let radius = 0.05;

            if (isStart) {{
                color = 0x22c55e;
                radius = 0.08;
            }}

            if (isCurrent) {{
                color = 0xfacc15;
                radius = 0.09;
            }}

            addPoint(
                point.position,
                color,
                radius
            );

            if (
                index > 0
                && vectorPath[index - 1]?.position
            ) {{
                addArrow(
                    vectorPath[index - 1].position,
                    point.position
                );
            }}
        }}
    );

    if (
        targetVector
        && Array.isArray(targetVector.position)
        && targetVector.position.length === 3
    ) {{
        addPoint(
            targetVector.position,
            0xef4444,
            0.095
        );

        const current =
            vectorPath[vectorPath.length - 1];

        if (current?.position) {{
            addTargetConnection(
                current.position,
                targetVector.position
            );
        }}
    }}
}};

function animate() {{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}}

animate();

window.addEventListener(
    "resize",
    () => {{
        const width = container.clientWidth;
        const height = container.clientHeight;

        camera.aspect = width / height;
        camera.updateProjectionMatrix();

        renderer.setSize(width, height);
    }}
);
</script>

<script>
let activeJob = null;
let poller = null;
let timerTicker = null;
let timerStartedAt = null;
let timerBaseSeconds = 0;

const $ = id =>
    document.getElementById(id);

function setTimer(seconds) {{
    $("timer").textContent =
        Math.max(0, seconds).toFixed(2) + "s";
}}

function startLiveTimer(baseSeconds = 0) {{
    timerBaseSeconds = baseSeconds;
    timerStartedAt = performance.now();

    clearInterval(timerTicker);

    timerTicker = setInterval(
        () => setTimer(
            timerBaseSeconds
            + (
                performance.now()
                - timerStartedAt
            ) / 1000
        ),
        50
    );
}}

function stopLiveTimer(
    finalSeconds = timerBaseSeconds
) {{
    clearInterval(timerTicker);

    timerTicker = null;
    timerStartedAt = null;

    setTimer(finalSeconds);
}}

function scoreStrength(score) {{
    if (!Number.isFinite(score)) {{
        return 1;
    }}

    return Math.max(
        0,
        Math.min(1, score)
    );
}}

function scoreColor(strength) {{
    const hue =
        120 * Math.max(
            0,
            Math.min(1, strength)
        );

    return `hsl(${{hue}} 85% 52%)`;
}}

function payload() {{
    return {{
        start: $("start").value,
        target: $("target").value,
        model_path: $("model").value,
        max_steps: +$("maxSteps").value,
        link_limit: +$("linkLimit").value,
        randomize: $("randomize").checked
    }};
}}

async function start(mode) {{
    const response = await fetch(
        "/api/jobs",
        {{
            method: "POST",
            headers: {{
                "Content-Type":
                    "application/json"
            }},
            body: JSON.stringify({{
                ...payload(),
                runs: mode === "batch"
                    ? 1000
                    : 1,
                mode
            }})
        }}
    );

    const job = await response.json();

    activeJob = job.id;

    $("status").textContent =
        "Running " + mode;

    clearInterval(poller);

    startLiveTimer(0);

    poller = setInterval(
        poll,
        700
    );

    await poll();
}}

async function poll() {{
    if (!activeJob) {{
        return;
    }}

    try {{
        const response = await fetch(
            "/api/jobs/"
            + activeJob
            + "?t="
            + Date.now(),
            {{
                cache: "no-store"
            }}
        );

        if (!response.ok) {{
            throw new Error(
                "Polling failed: "
                + response.status
            );
        }}

        const job = await response.json();

        drawJob(job);

        if (
            job.status === "completed"
            || job.status === "failed"
        ) {{
            clearInterval(poller);
            poller = null;

            stopLiveTimer(
                job.elapsed_seconds
            );

            await loadResults();
        }}
    }} catch (error) {{
        console.error(error);

        clearInterval(poller);
        poller = null;

        stopLiveTimer();

        $("status").textContent =
            "Frontend error: "
            + error.message;
    }}
}}

function drawJob(job) {{
    $("status").textContent =
        `${{job.status}} `
        + `(${{job.current_run}}/${{job.total_runs}})`
        + (
            job.error
                ? " — " + job.error
                : ""
        );

    if (!timerTicker) {{
        setTimer(job.elapsed_seconds);
    }}

    $("page").textContent =
        job.current_page || "—";

    $("best").textContent =
        job.best_link || "—";

    const last =
        job.results.at(-1);

    const path =
        job.current_path?.length
            ? job.current_path
            : (
                last
                    ? last.path
                    : []
            );

    $("path").textContent =
        path.length
            ? path.join(" → ")
            : "No run yet.";

    const top =
        job.current_top_links?.length
            ? job.current_top_links
            : (
                last?.steps?.at(-1)
                    ?.top_links
                || []
            );

    $("top").innerHTML =
        top.length
            ? top.map(link => {{
                const score =
                    Number(link.score);

                const strength =
                    scoreStrength(score);

                return `
                    <div class="rank-row">
                        <div>
                            ${{link.title}}
                            <small>
                                (${{score.toFixed(3)}})
                            </small>
                        </div>

                        <div class="rank-bar-track">
                            <div
                                class="rank-bar"
                                style="
                                    width:
                                    ${{Math.round(
                                        strength * 100
                                    )}}%;
                                    background:
                                    ${{scoreColor(
                                        strength
                                    )}};
                                "
                            ></div>
                        </div>
                    </div>
                `;
            }}).join("")
            : "No ranking yet.";

    if (window.renderVectorScene) {{
        window.renderVectorScene(
            job.current_vector_path || [],
            job.target_vector || null
        );
    }}

    updateSummary(job.summary);
}}

function updateSummary(summary) {{
    $("avgTime").textContent =
        summary.runs
            ? summary.avg_seconds
                .toFixed(2) + "s"
            : "—";

    $("avgLinks").textContent =
        summary.runs
            ? summary.avg_links_visited
                .toFixed(1)
            : "—";

    $("success").textContent =
        summary.runs
            ? Math.round(
                summary.success_rate * 100
            ) + "%"
            : "—";
}}

async function loadResults() {{
    const response = await fetch(
        "/api/results?t=" + Date.now(),
        {{
            cache: "no-store"
        }}
    );

    const data = await response.json();

    updateSummary(data.summary);

    drawChart(
        data.results.slice(-100)
    );
}}

function drawChart(rows) {{
    const canvas = $("chart");
    const context =
        canvas.getContext("2d");

    context.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    context.fillStyle = "#94a3b8";

    context.fillText(
        "Last "
        + rows.length
        + " saved runs: "
        + "cyan=time seconds, "
        + "purple=links visited",
        20,
        24
    );

    if (!rows.length) {{
        return;
    }}

    const maxTime = Math.max(
        ...rows.map(
            result => result.total_seconds
        ),
        1
    );

    const maxLinks = Math.max(
        ...rows.map(
            result => result.links_visited
        ),
        1
    );

    rows.forEach(
        (result, index) => {{
            const x =
                40
                + index
                * (
                    (canvas.width - 80)
                    / Math.max(
                        rows.length - 1,
                        1
                    )
                );

            const timeHeight =
                (
                    result.total_seconds
                    / maxTime
                )
                * (canvas.height - 70);

            context.fillStyle =
                "#38bdf8";

            context.fillRect(
                x - 3,
                canvas.height
                    - 30
                    - timeHeight,
                6,
                timeHeight
            );

            const linksHeight =
                (
                    result.links_visited
                    / maxLinks
                )
                * (canvas.height - 70);

            context.fillStyle =
                "#a78bfa";

            context.fillRect(
                x + 4,
                canvas.height
                    - 30
                    - linksHeight,
                6,
                linksHeight
            );
        }}
    );
}}

$("runOne").onclick =
    () => start("single");

$("runBatch").onclick =
    () => start("batch");

$("loadResults").onclick =
    loadResults;

loadResults();
</script>

</body>
</html>"""

if __name__ == "__main__":
    main()