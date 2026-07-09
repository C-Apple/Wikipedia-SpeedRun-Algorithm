"""Small browser UI for running the project's pytest suite.

Run with ``python -m src.test_frontend`` and open the printed localhost URL.
The server intentionally uses only the Python standard library so the dashboard
works before optional web-framework dependencies are installed.
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"


@dataclass(frozen=True)
class TestRunResult:
    """JSON-serializable summary of a pytest run."""

    command: list[str]
    exit_code: int
    duration_seconds: float
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return self.exit_code == 0

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


def discover_test_files(tests_dir: Path = TESTS_DIR) -> list[str]:
    """Return repository-relative pytest files for the dashboard selector."""

    if not tests_dir.exists():
        return []
    return [str(path.relative_to(REPO_ROOT)) for path in sorted(tests_dir.glob("test_*.py"))]


def safe_test_targets(requested_targets: list[str], tests_dir: Path = TESTS_DIR) -> list[str]:
    """Filter requested test targets to known files under ``tests_dir``.

    Unknown or unsafe values are ignored so browser requests cannot execute
    arbitrary local files or pytest expressions outside this repository.
    """

    allowed = set(discover_test_files(tests_dir))
    selected = [target for target in requested_targets if target in allowed]
    return selected or ["tests"]


def run_pytest(targets: list[str] | None = None, extra_args: list[str] | None = None) -> TestRunResult:
    """Run pytest and capture output for API responses."""

    selected_targets = safe_test_targets(targets or [])
    args = [sys.executable, "-m", "pytest", *selected_targets, *(extra_args or [])]
    started = time.perf_counter()
    completed = subprocess.run(args, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    duration = time.perf_counter() - started
    return TestRunResult(args, completed.returncode, duration, completed.stdout, completed.stderr)


def render_index(test_files: list[str] | None = None) -> str:
    """Render the self-contained HTML/JS test dashboard."""

    files = test_files if test_files is not None else discover_test_files()
    checkboxes = "\n".join(
        f'<label><input type="checkbox" name="target" value="{html.escape(path)}" checked> {html.escape(path)}</label>'
        for path in files
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Wikipedia SpeedRun Test Runner</title>
  <style>
    :root {{ color-scheme: light dark; font-family: Inter, system-ui, sans-serif; }}
    body {{ margin: 0; background: #0f172a; color: #e2e8f0; }}
    main {{ max-width: 980px; margin: 0 auto; padding: 32px; }}
    .card {{ background: #111827; border: 1px solid #334155; border-radius: 18px; padding: 24px; box-shadow: 0 20px 60px #0006; }}
    h1 {{ margin-top: 0; font-size: clamp(2rem, 5vw, 3.4rem); }}
    p {{ color: #94a3b8; }}
    .grid {{ display: grid; gap: 10px; margin: 20px 0; }}
    label {{ background: #1e293b; border-radius: 12px; padding: 12px; }}
    button {{ border: 0; border-radius: 999px; background: #38bdf8; color: #082f49; font-weight: 800; padding: 14px 22px; cursor: pointer; }}
    button:disabled {{ cursor: wait; opacity: .7; }}
    #status {{ margin-left: 12px; font-weight: 700; }}
    pre {{ white-space: pre-wrap; overflow-x: auto; background: #020617; border-radius: 12px; padding: 18px; min-height: 180px; }}
    .pass {{ color: #86efac; }} .fail {{ color: #fca5a5; }}
  </style>
</head>
<body>
  <main>
    <section class="card">
      <h1>Wikipedia SpeedRun Test Runner</h1>
      <p>Select test files, run pytest from your browser, and inspect the exact command output.</p>
      <form id="runner">
        <div class="grid">{checkboxes or '<em>No test files found.</em>'}</div>
        <button type="submit">Run selected tests</button><span id="status">Idle</span>
      </form>
      <h2>Output</h2>
      <pre id="output">Click “Run selected tests” to start.</pre>
    </section>
  </main>
  <script>
    const form = document.querySelector('#runner');
    const statusEl = document.querySelector('#status');
    const output = document.querySelector('#output');
    form.addEventListener('submit', async (event) => {{
      event.preventDefault();
      const button = form.querySelector('button');
      const targets = [...form.querySelectorAll('input[name="target"]:checked')].map(input => input.value);
      button.disabled = true; statusEl.textContent = 'Running…'; output.textContent = '';
      try {{
        const response = await fetch('/api/run-tests', {{
          method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify({{targets}})
        }});
        const result = await response.json();
        statusEl.textContent = result.passed ? `Passed in ${{result.duration_seconds.toFixed(2)}}s` : `Failed with code ${{result.exit_code}}`;
        statusEl.className = result.passed ? 'pass' : 'fail';
        output.textContent = `$ ${{result.command.join(' ')}}\n\n${{result.stdout}}${{result.stderr ? '\nSTDERR:\n' + result.stderr : ''}}`;
      }} catch (error) {{ statusEl.textContent = 'Request failed'; statusEl.className = 'fail'; output.textContent = String(error); }}
      finally {{ button.disabled = false; }}
    }});
  </script>
</body>
</html>"""


class TestFrontendHandler(BaseHTTPRequestHandler):
    """HTTP handler for the dashboard and JSON test-run API."""

    server_version = "WikipediaSpeedRunTestFrontend/1.0"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(render_index())
        elif parsed.path == "/api/tests":
            self._send_json({"tests": discover_test_files()})
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        if parsed.path != "/api/run-tests":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        targets = self._read_targets()
        result = run_pytest(targets, extra_args=["-q"])
        status = HTTPStatus.OK if result.passed else HTTPStatus.INTERNAL_SERVER_ERROR
        self._send_json(result.to_json(), status=status)

    def _read_targets(self) -> list[str]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else ""
        if self.headers.get("Content-Type", "").startswith("application/json"):
            try:
                data = json.loads(raw or "{}")
            except json.JSONDecodeError:
                return []
            return [str(item) for item in data.get("targets", [])]
        parsed = parse_qs(raw)
        return parsed.get("target", [])

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
    return ThreadingHTTPServer((host, port), TestFrontendHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a browser UI for running project tests")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    server = build_server(args.host, args.port)
    print(f"Test frontend running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down test frontend")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
