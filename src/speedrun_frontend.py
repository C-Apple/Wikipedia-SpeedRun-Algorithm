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

from src.wiki_navigator import WikipediaClient, build_title_scorer, find_closest_hyperlink, normalize_title

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
class StepSnapshot:
    step: int
    page: str
    links_found: int
    best_link: str | None
    best_score: float | None
    elapsed_seconds: float
    top_links: list[dict[str, float | str]] = field(default_factory=list)


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
    current_top_links: list[dict[str, float | str]] = field(default_factory=list)
    results: list[SpeedrunResult] = field(default_factory=list)
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
    from src.save_configs import load_checkpoint
    from src.word_embedding import SkipGramNegSampling

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
    model_path: str | None = None,
    max_steps: int = 25,
    link_limit: int = 500,
    client: WikipediaClient | None = None,
    on_step: Callable[[StepSnapshot], None] | None = None,
) -> SpeedrunResult:
    """Run one greedy speedrun while emitting step snapshots."""

    client = client or WikipediaClient()
    scorer, resolved_model_path = load_checkpoint_scorer(model_path, target_page)
    run_id = uuid.uuid4().hex
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    path = [start_page]
    steps: list[StepSnapshot] = []
    visited = {normalize_title(start_page)}
    current = start_page
    reason = "max steps reached"
    reached = normalize_title(start_page) == normalize_title(target_page)

    for step in range(max_steps):
        if normalize_title(current) == normalize_title(target_page):
            reached = True
            reason = "target reached"
            break
        links = client.get_hyperlinks(current, limit=link_limit)
        top = ranked_links(links, target_page, scorer)
        if not links:
            reason = "no outgoing article links"
            snapshot = StepSnapshot(step, current, 0, None, None, time.perf_counter() - started, [])
            steps.append(snapshot)
            if on_step:
                on_step(snapshot)
            break
        if normalize_title(target_page) in {normalize_title(link) for link in links}:
            selected, score = target_page, math.inf
        else:
            unvisited = [link for link in links if normalize_title(link) not in visited]
            if not unvisited:
                reason = "all outgoing links already visited"
                snapshot = StepSnapshot(step, current, len(links), None, None, time.perf_counter() - started, top)
                steps.append(snapshot)
                if on_step:
                    on_step(snapshot)
                break
            selected, score = find_closest_hyperlink(unvisited, target_page, scorer)
        snapshot = StepSnapshot(step, current, len(links), selected, score, time.perf_counter() - started, top)
        steps.append(snapshot)
        if on_step:
            on_step(snapshot)
        current = selected
        path.append(current)
        visited.add(normalize_title(current))
    else:
        reached = normalize_title(current) == normalize_title(target_page)
        if reached:
            reason = "target reached"

    total_seconds = time.perf_counter() - started
    return SpeedrunResult(run_id, started_at, start_page, target_page, resolved_model_path, path, steps, reached, reason, total_seconds)


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


def run_job(job: BenchJob, config: dict[str, Any]) -> None:
    try:
        job.status = "running"
        client = WikipediaClient()
        for index in range(job.total_runs):
            job.current_run = index + 1
            if config.get("randomize"):
                start_page, target_page = random.choice(RANDOM_PAIRS)
            else:
                start_page = parse_wikipedia_title(config["start"])
                target_page = parse_wikipedia_title(config["target"])

            job.current_path = [start_page]
            job.current_top_links = []

            def update(snapshot: StepSnapshot) -> None:
                job.current_page = snapshot.page
                job.best_link = snapshot.best_link
                job.elapsed_seconds = snapshot.elapsed_seconds
                job.current_top_links = snapshot.top_links
                if snapshot.best_link:
                    job.current_path = [*job.current_path, snapshot.best_link]

            result = run_speedrun(
                start_page,
                target_page,
                model_path=config.get("model_path") or None,
                max_steps=int(config.get("max_steps", 25)),
                link_limit=int(config.get("link_limit", 500)),
                client=client,
                on_step=update,
            )
            save_result(result)
            job.results.append(result)
        job.status = "completed"
    except Exception as exc:  # keep background job errors visible in the UI
        job.status = "failed"
        job.error = str(exc)


def start_job(config: dict[str, Any], total_runs: int, mode: str) -> BenchJob:
    job = BenchJob(id=uuid.uuid4().hex, mode=mode, total_runs=total_runs)
    with JOBS_LOCK:
        JOBS[job.id] = job
    thread = threading.Thread(target=run_job, args=(job, config), daemon=True)
    thread.start()
    return job


def render_index() -> str:
    pair_options = "".join(f'<option value="{start} → {target}">' for start, target in RANDOM_PAIRS)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wikipedia SpeedRun Test Bench</title>
<style>
:root {{ font-family: Inter, ui-sans-serif, system-ui, sans-serif; color: #e5eefb; background: #07111f; }}
body {{ margin: 0; }} main {{ max-width: 1180px; margin: auto; padding: 28px; }}
.hero {{ display:grid; gap:10px; margin-bottom:22px; }} h1 {{ font-size: clamp(2.2rem, 6vw, 4.4rem); margin:0; }} p {{ color:#9fb3ca; }}
.panel {{ background:#0d1b2f; border:1px solid #233b5e; border-radius:22px; padding:22px; box-shadow: 0 24px 80px #0008; }}
.grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap:16px; }}
label {{ display:grid; gap:7px; font-weight:700; }} input {{ border:1px solid #34506f; border-radius:12px; background:#081525; color:#f8fafc; padding:12px; }}
.actions {{ display:flex; gap:12px; flex-wrap:wrap; align-items:center; margin-top:16px; }} button {{ border:0; border-radius:999px; padding:13px 20px; font-weight:900; cursor:pointer; }}
.primary {{ background:#38bdf8; color:#082f49; }} .secondary {{ background:#a78bfa; color:#201047; }} .ghost {{ background:#203450; color:#e2e8f0; }}
.metrics {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap:12px; margin:18px 0; }}
.metric {{ background:#081525; border-radius:16px; padding:16px; }} .metric b {{ display:block; font-size:1.8rem; }}
.status {{ color:#93c5fd; font-weight:800; }} .path, .top {{ background:#020617; border-radius:16px; padding:16px; min-height:90px; overflow:auto; }}
canvas {{ width:100%; min-height:260px; background:#020617; border-radius:16px; margin-top:12px; }}
.rank-row {{ display:grid; grid-template-columns:minmax(120px, 1fr) minmax(110px, 36%); gap:12px; align-items:center; margin:8px 0; }}
.rank-bar-track {{ height:14px; background:#172554; border-radius:999px; overflow:hidden; }}
.rank-bar {{ height:100%; border-radius:999px; min-width:3px; }}
</style></head>
<body><main>
<section class="hero"><h1>Wikipedia SpeedRun Test Bench</h1><p>Pick start/end pages or random challenges, choose a local trained model path, and watch the greedy navigator rank links in real time.</p></section>
<section class="panel"><div class="grid">
<label>Start URL or title<input id="start" list="pairs" value="Python (programming language)"></label>
<label>End URL or title<input id="target" value="Artificial intelligence"></label>
<label>Local model checkpoint/folder<input id="model" placeholder="runs/sgns_wiki_v2 or /Users/me/checkpoint.pt"></label>
<label>Max steps<input id="maxSteps" type="number" min="1" value="25"></label>
<label>Link limit per page<input id="linkLimit" type="number" min="1" value="500"></label>
</div><datalist id="pairs">{pair_options}</datalist>
<div class="actions"><button class="primary" id="runOne">Run speedrun</button><button class="secondary" id="runBatch">Run 1000 random speedruns</button><button class="ghost" id="loadResults">Reload saved results</button><label><input id="randomize" type="checkbox"> Use random start/end</label><span class="status" id="status">Idle</span></div></section>
<section class="metrics"><div class="metric">Run timer<b id="timer">0.00s</b></div><div class="metric">Current page<b id="page">—</b></div><div class="metric">Highest ranked link<b id="best">—</b></div><div class="metric">Batch avg time<b id="avgTime">—</b></div><div class="metric">Avg links visited<b id="avgLinks">—</b></div><div class="metric">Success rate<b id="success">—</b></div></section>
<section class="grid"><div><h2>Current path</h2><div class="path" id="path">No run yet.</div></div><div><h2>Top ranked hyperlinks now</h2><div class="top" id="top">No ranking yet.</div></div></section>
<section><h2>Saved results graph</h2><canvas id="chart" width="1100" height="320"></canvas></section>
</main><script>
let activeJob=null, poller=null, timerTicker=null, timerStartedAt=null, timerBaseSeconds=0;
const $=id=>document.getElementById(id);
function setTimer(seconds) {{ $('timer').textContent=Math.max(0, seconds).toFixed(2)+'s'; }}
function startLiveTimer(baseSeconds=0) {{ timerBaseSeconds=baseSeconds; timerStartedAt=performance.now(); clearInterval(timerTicker); timerTicker=setInterval(()=>setTimer(timerBaseSeconds+(performance.now()-timerStartedAt)/1000), 50); }}
function stopLiveTimer(finalSeconds=timerBaseSeconds) {{ clearInterval(timerTicker); timerTicker=null; timerStartedAt=null; setTimer(finalSeconds); }}
function scoreStrength(score) {{ if(!Number.isFinite(score)) return 1; return Math.max(0, Math.min(1, score)); }}
function scoreColor(strength) {{ const hue=120*Math.max(0, Math.min(1, strength)); return `hsl(${{hue}} 85% 52%)`; }}
function payload() {{ return {{start:$('start').value,target:$('target').value,model_path:$('model').value,max_steps:+$('maxSteps').value,link_limit:+$('linkLimit').value,randomize:$('randomize').checked}}; }}
async function start(mode) {{ const res=await fetch('/api/jobs',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{...payload(),runs: mode==='batch'?1000:1,mode}})}}); activeJob=(await res.json()).id; $('status').textContent='Running '+mode; clearInterval(poller); startLiveTimer(0); poller=setInterval(poll,700); poll(); }}
async function poll() {{ if(!activeJob) return; const job=await (await fetch('/api/jobs/'+activeJob)).json(); drawJob(job); if(['completed','failed'].includes(job.status)) {{ clearInterval(poller); stopLiveTimer(job.elapsed_seconds); await loadResults(); }} }}
function drawJob(job) {{ $('status').textContent=`${{job.status}} (${{job.current_run}}/${{job.total_runs}})`+(job.error?' — '+job.error:''); if(!timerTicker) setTimer(job.elapsed_seconds); $('page').textContent=job.current_page||'—'; $('best').textContent=job.best_link||'—'; const last=job.results.at(-1); const path=job.current_path?.length?job.current_path:(last?last.path:[]); $('path').textContent=path.length?path.join(' → '):'No run yet.'; const top=job.current_top_links?.length?job.current_top_links:(last?.steps?.at(-1)?.top_links||[]); $('top').innerHTML=top.length ? top.map(l=>{{ const score=Number(l.score); const strength=scoreStrength(score); return `<div class="rank-row"><div>${{l.title}} <small>(${{score.toFixed(3)}})</small></div><div class="rank-bar-track"><div class="rank-bar" style="width:${{Math.round(strength*100)}}%;background:${{scoreColor(strength)}}"></div></div></div>`; }}).join('') : 'No ranking yet.'; updateSummary(job.summary); }}
function updateSummary(s) {{ $('avgTime').textContent=s.runs?s.avg_seconds.toFixed(2)+'s':'—'; $('avgLinks').textContent=s.runs?s.avg_links_visited.toFixed(1):'—'; $('success').textContent=s.runs?Math.round(s.success_rate*100)+'%':'—'; }}
async function loadResults() {{ const data=await (await fetch('/api/results')).json(); updateSummary(data.summary); drawChart(data.results.slice(-100)); }}
function drawChart(rows) {{ const c=$('chart'), ctx=c.getContext('2d'); ctx.clearRect(0,0,c.width,c.height); ctx.fillStyle='#94a3b8'; ctx.fillText('Last '+rows.length+' saved runs: cyan=time seconds, purple=links visited',20,24); if(!rows.length) return; const maxT=Math.max(...rows.map(r=>r.total_seconds),1), maxL=Math.max(...rows.map(r=>r.links_visited),1); rows.forEach((r,i)=>{{ const x=40+i*((c.width-80)/Math.max(rows.length-1,1)); ctx.fillStyle='#38bdf8'; ctx.fillRect(x-3,c.height-30-(r.total_seconds/maxT)*(c.height-70),6,(r.total_seconds/maxT)*(c.height-70)); ctx.fillStyle='#a78bfa'; ctx.fillRect(x+4,c.height-30-(r.links_visited/maxL)*(c.height-70),6,(r.links_visited/maxL)*(c.height-70)); }}); }}
$('runOne').onclick=()=>start('single'); $('runBatch').onclick=()=>start('batch'); $('loadResults').onclick=loadResults; loadResults();
</script></body></html>"""


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


if __name__ == "__main__":
    main()
