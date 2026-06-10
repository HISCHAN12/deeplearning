"""Local web advisor for project demos.

This server is intentionally dependency-free. It trains the prototype model on
synthetic logs, then serves a small browser UI that refreshes recommendations.
"""

from __future__ import annotations

import base64
import json
import os
import ssl
import subprocess
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .recommender import TFTAdvisor
from .state_encoder import archetype_deck_vector, board_from_style
from .synthetic_data import ACTIONS, ARCHETYPES, generate_matches


HOST = "127.0.0.1"
PORT = int(os.environ.get("TFT_ADVISOR_PORT", "8000"))
APP: "LiveAdvisorApp | None" = None


class LiveAdvisorApp:
    def __init__(self) -> None:
        records = generate_matches(n_matches=1800, seed=42)
        self.advisor = TFTAdvisor(n_clusters=4)
        self.metrics = self.advisor.fit(records)
        self.started_at = time.time()

    def recommend(self, archetype: str, board_style: str) -> dict[str, Any]:
        if archetype not in ARCHETYPES:
            archetype = "fast_8_carry"
        deck_vector = archetype_deck_vector(archetype)
        board_grid = board_from_style(board_style)
        recommendation = self.advisor.recommend(deck_vector, board_grid)
        return {
            "source": self.client_status(),
            "archetype": archetype,
            "board_style": board_style,
            "metrics": self.metrics,
            "actions": ACTIONS,
            "recommendation": {
                "nearest_cluster": recommendation.cluster_id,
                "action": recommendation.action,
                "expected_placement": round(recommendation.expected_placement, 2),
                "top4_rate": round(recommendation.top4_rate, 3),
                "predicted_placement": recommendation.predicted_placement,
                "placement_probabilities": [round(value, 4) for value in recommendation.placement_probabilities],
            },
            "updated_at": time.strftime("%H:%M:%S"),
        }

    def client_status(self) -> dict[str, Any]:
        status = read_riot_client_status()
        if status is None:
            return {
                "mode": "demo_manual",
                "label": "Manual demo mode",
                "detail": "TFT/League client lockfile was not detected, so the overlay uses selected sample state.",
            }
        return status


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path, _, query = self.path.partition("?")
        if path == "/":
            self._send_text(render_html(), "text/html; charset=utf-8")
        elif path == "/api/recommend":
            params = parse_query(query)
            payload = get_app().recommend(params.get("archetype", "fast_8_carry"), params.get("board", "frontline"))
            self._send_json(payload)
        else:
            self.send_error(404)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_query(query: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in query.split("&"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        result[key] = value.replace("+", " ")
    return result


def get_app() -> LiveAdvisorApp:
    if APP is None:
        raise RuntimeError("Live advisor app has not been initialized.")
    return APP


def read_riot_client_status() -> dict[str, Any] | None:
    lockfile = find_lockfile()
    if lockfile is None:
        return None
    try:
        name, _pid, port, password, protocol = lockfile.read_text(encoding="utf-8").strip().split(":")
        token = base64.b64encode(f"riot:{password}".encode("utf-8")).decode("ascii")
        request = urllib.request.Request(
            f"{protocol}://127.0.0.1:{port}/lol-gameflow/v1/gameflow-phase",
            headers={"Authorization": f"Basic {token}"},
        )
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(request, context=context, timeout=0.4) as response:
            phase = response.read().decode("utf-8").strip('"')
        return {
            "mode": "riot_client_detected",
            "label": "Riot client detected",
            "detail": f"{name} gameflow phase: {phase}",
        }
    except Exception as exc:
        return {
            "mode": "riot_client_detected",
            "label": "Riot client detected",
            "detail": f"Lockfile found, but local API request failed: {exc.__class__.__name__}",
        }


def find_lockfile() -> Path | None:
    candidates = [
        Path("/Applications/League of Legends.app/Contents/LoL/lockfile"),
        Path.home() / "Applications/League of Legends.app/Contents/LoL/lockfile",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    try:
        output = subprocess.check_output(["pgrep", "-fl", "LeagueClient"], text=True, timeout=0.4)
    except Exception:
        return None
    for line in output.splitlines():
        for token in line.split():
            path = Path(token)
            if path.name == "LeagueClient" and path.parent.joinpath("lockfile").exists():
                return path.parent / "lockfile"
    return None


def render_html() -> str:
    archetype_options = "\n".join(f'<option value="{name}">{name}</option>' for name in ARCHETYPES)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TFT Live Advisor</title>
  <style>
    :root {{ font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #f8fafc; background: #111827; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; min-height: 100vh; background: #111827; }}
    main {{ width: min(1120px, calc(100% - 32px)); margin: 0 auto; padding: 28px 0; }}
    header {{ display: flex; justify-content: space-between; gap: 18px; align-items: end; margin-bottom: 20px; }}
    h1 {{ margin: 0; font-size: clamp(2rem, 5vw, 4.2rem); line-height: 1; letter-spacing: 0; }}
    .status {{ color: #bae6fd; font-weight: 700; }}
    .layout {{ display: grid; grid-template-columns: 320px 1fr; gap: 16px; }}
    section, aside {{ border: 1px solid rgba(255,255,255,.14); border-radius: 8px; background: #172033; padding: 18px; }}
    label {{ display: block; margin: 0 0 12px; color: #cbd5e1; font-size: .92rem; }}
    select {{ width: 100%; height: 42px; margin-top: 6px; border-radius: 6px; border: 1px solid #334155; color: #f8fafc; background: #0f172a; padding: 0 10px; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-bottom: 14px; }}
    .metric {{ min-height: 84px; border: 1px solid rgba(255,255,255,.12); border-radius: 8px; padding: 12px; background: #0f172a; }}
    .metric span {{ display: block; color: #94a3b8; font-size: .8rem; }}
    .metric strong {{ display: block; margin-top: 8px; font-size: 1.65rem; }}
    .recommend {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: stretch; }}
    .action {{ border-radius: 8px; background: #f59e0b; color: #111827; padding: 18px; }}
    .action span {{ display: block; font-weight: 800; font-size: .84rem; text-transform: uppercase; }}
    .action strong {{ display: block; margin-top: 10px; font-size: clamp(1.8rem, 4vw, 3.4rem); line-height: 1; }}
    .bars {{ display: grid; gap: 8px; }}
    .bar {{ display: grid; grid-template-columns: 34px 1fr 48px; gap: 8px; align-items: center; color: #cbd5e1; }}
    .track {{ height: 12px; border-radius: 999px; overflow: hidden; background: #263244; }}
    .fill {{ height: 100%; background: #38bdf8; }}
    .board {{ display: grid; grid-template-columns: repeat(7, 1fr); gap: 6px; margin-top: 14px; }}
    .cell {{ aspect-ratio: 1; border: 1px solid rgba(148,163,184,.22); border-radius: 6px; background: #0f172a; }}
    .cell.unit {{ background: #38bdf8; box-shadow: 0 0 18px rgba(56,189,248,.45); }}
    .note {{ color: #94a3b8; line-height: 1.5; margin: 12px 0 0; }}
    @media (max-width: 780px) {{ header, .layout, .recommend {{ grid-template-columns: 1fr; display: grid; }} .metric-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>TFT Live Advisor</h1>
        <p class="note">Local project demo for placement prediction and next-action recommendation.</p>
      </div>
      <div class="status" id="status">Training model...</div>
    </header>
    <div class="layout">
      <aside>
        <label>Current meta-deck
          <select id="archetype">{archetype_options}</select>
        </label>
        <label>Board positioning
          <select id="board">
            <option value="frontline">frontline carry setup</option>
            <option value="backline">backline carry setup</option>
            <option value="corner_carry">corner carry setup</option>
            <option value="spread">spread board setup</option>
          </select>
        </label>
        <div class="board" id="boardGrid"></div>
        <p class="note" id="sourceDetail">Waiting for advisor.</p>
      </aside>
      <section>
        <div class="metric-grid">
          <div class="metric"><span>Predicted place</span><strong id="place">-</strong></div>
          <div class="metric"><span>Top 4 rate</span><strong id="top4">-</strong></div>
          <div class="metric"><span>Cluster</span><strong id="cluster">-</strong></div>
          <div class="metric"><span>Model Top 4 acc.</span><strong id="acc">-</strong></div>
        </div>
        <div class="recommend">
          <div class="action">
            <span>Recommended next action</span>
            <strong id="action">-</strong>
          </div>
          <div class="bars" id="bars"></div>
        </div>
      </section>
    </div>
  </main>
  <script>
    const archetype = document.getElementById('archetype');
    const board = document.getElementById('board');
    const boardGrid = document.getElementById('boardGrid');
    const boardPatterns = {{
      frontline: [1,3,5,23,25],
      backline: [9,11,22,24,26],
      corner_carry: [2,3,4,21,22,27],
      spread: [2,4,15,17,19,23,25]
    }};

    function drawBoard() {{
      const active = new Set(boardPatterns[board.value] || []);
      boardGrid.innerHTML = '';
      for (let i = 0; i < 28; i++) {{
        const cell = document.createElement('div');
        cell.className = active.has(i) ? 'cell unit' : 'cell';
        boardGrid.appendChild(cell);
      }}
    }}

    async function refresh() {{
      const params = new URLSearchParams({{ archetype: archetype.value, board: board.value }});
      const response = await fetch('/api/recommend?' + params.toString());
      const data = await response.json();
      document.getElementById('status').textContent = data.source.label + ' · ' + data.updated_at;
      document.getElementById('sourceDetail').textContent = data.source.detail;
      document.getElementById('place').textContent = '#' + data.recommendation.predicted_placement;
      document.getElementById('top4').textContent = Math.round(data.recommendation.top4_rate * 100) + '%';
      document.getElementById('cluster').textContent = data.recommendation.nearest_cluster;
      document.getElementById('acc').textContent = Math.round(data.metrics.top4_accuracy * 100) + '%';
      document.getElementById('action').textContent = data.recommendation.action;

      const bars = document.getElementById('bars');
      bars.innerHTML = '';
      data.recommendation.placement_probabilities.forEach((prob, index) => {{
        const row = document.createElement('div');
        row.className = 'bar';
        row.innerHTML = `<span>#${{index + 1}}</span><div class="track"><div class="fill" style="width:${{Math.round(prob * 100)}}%"></div></div><span>${{Math.round(prob * 100)}}%</span>`;
        bars.appendChild(row);
      }});
    }}

    archetype.addEventListener('change', refresh);
    board.addEventListener('change', () => {{ drawBoard(); refresh(); }});
    drawBoard();
    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>"""


def main() -> None:
    global APP
    print("Starting TFT live advisor...", flush=True)
    print("Training local prototype model. This usually takes a few seconds.", flush=True)
    APP = LiveAdvisorApp()
    print("Model ready.", flush=True)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"TFT live advisor running at http://{HOST}:{PORT}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
