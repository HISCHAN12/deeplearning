"""Offline TFT recording analysis and placement-probability reporting."""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .game_state import GameState
from .recommender import TFTAdvisor
from .state_encoder import archetype_deck_vector
from .synthetic_data import ARCHETYPES, generate_matches


@dataclass(frozen=True)
class VideoPrediction:
    timestamp_seconds: float
    progress: float
    predicted_placement: int
    top4_probability: float
    board_strength: float
    board_activity: float
    occupied_cells: int
    placement_probabilities: list[float]


def extract_board_grid(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Extract a visual 4x7 occupancy proxy from the center TFT board area."""
    height, width = frame.shape[:2]
    left, right = int(width * 0.16), int(width * 0.84)
    top, bottom = int(height * 0.24), int(height * 0.73)
    board = frame[top:bottom, left:right]
    if board.size == 0:
        raise ValueError("Could not extract the board area from the frame.")

    hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 150)
    cell_height = board.shape[0] / 4
    cell_width = board.shape[1] / 7
    raw_scores = np.zeros((4, 7), dtype=np.float64)

    for row in range(4):
        for column in range(7):
            y1, y2 = int(row * cell_height), int((row + 1) * cell_height)
            x1, x2 = int(column * cell_width), int((column + 1) * cell_width)
            cell_hsv = hsv[y1:y2, x1:x2]
            cell_edges = edges[y1:y2, x1:x2]
            saturation = float(cell_hsv[:, :, 1].mean()) / 255.0
            brightness = float(cell_hsv[:, :, 2].mean()) / 255.0
            edge_density = float(np.mean(cell_edges > 0))
            raw_scores[row, column] = 0.45 * saturation + 0.25 * brightness + 0.30 * edge_density

    low, high = np.percentile(raw_scores, [20, 90])
    scale = max(high - low, 1e-6)
    normalized = np.clip((raw_scores - low) / scale, 0.0, 1.0)
    normalized[normalized < 0.42] = 0.0
    return normalized, (left, top, right, bottom)


def average_deck_vector() -> np.ndarray:
    vectors = [archetype_deck_vector(name) for name in ARCHETYPES]
    return np.mean(vectors, axis=0)


def visual_game_state(
    progress: float,
    board_grid: np.ndarray,
    activity: float,
) -> GameState:
    occupied = int(np.count_nonzero(board_grid > 0.2))
    visual_strength = float(board_grid.sum()) / max(occupied, 1)
    board_strength = int(np.clip(round(2.5 + occupied * 0.42 + visual_strength * 2.4), 1, 10))
    stage = round(2.1 + progress * 4.4, 1)
    level = int(np.clip(round(4.5 + progress * 4.0), 4, 10))
    return GameState(
        stage=stage,
        health=70,
        gold=30,
        level=level,
        streak=0,
        board_strength=board_strength,
        unspent_items=1,
        board_strength_delta=int(np.clip(round(activity * 5), -5, 5)),
    )


class RecordingAnalyzer:
    def __init__(self) -> None:
        self.advisor = TFTAdvisor(n_clusters=4)
        self.metrics = self.advisor.fit(generate_matches(n_matches=1800, seed=42))
        self.deck_vector = average_deck_vector()

    def analyze(
        self,
        video_path: Path,
        interval_seconds: float = 5.0,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[VideoPrediction]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if frame_count > 0 else 0.0
        step = max(1, int(fps * interval_seconds))
        predictions: list[VideoPrediction] = []
        previous_gray: np.ndarray | None = None

        for frame_index in range(0, frame_count, step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue

            board_grid, (left, top, right, bottom) = extract_board_grid(frame)
            board_crop = cv2.resize(frame[top:bottom, left:right], (280, 160))
            current_gray = cv2.cvtColor(board_crop, cv2.COLOR_BGR2GRAY)
            activity = 0.0
            if previous_gray is not None:
                activity = float(np.mean(cv2.absdiff(current_gray, previous_gray))) / 255.0
            previous_gray = current_gray

            timestamp = frame_index / fps
            progress = timestamp / duration if duration > 0 else 0.0
            game_state = visual_game_state(progress, board_grid, activity)
            recommendation = self.advisor.recommend(
                self.deck_vector,
                board_grid,
                game_state.to_vector(),
            )
            predictions.append(
                VideoPrediction(
                    timestamp_seconds=round(timestamp, 2),
                    progress=round(progress, 4),
                    predicted_placement=recommendation.predicted_placement,
                    top4_probability=round(recommendation.predicted_top4_probability, 4),
                    board_strength=float(game_state.board_strength),
                    board_activity=round(activity, 4),
                    occupied_cells=int(np.count_nonzero(board_grid > 0.2)),
                    placement_probabilities=[
                        round(value, 4) for value in recommendation.placement_probabilities
                    ],
                )
            )
            if progress_callback:
                progress_callback(min(progress, 1.0))

        capture.release()
        if not predictions:
            raise ValueError("No frames could be analyzed from the video.")
        return predictions


def save_report(
    video_path: Path,
    predictions: list[VideoPrediction],
    output_dir: Path,
    metrics: dict[str, float],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "video_analysis.json"
    html_path = output_dir / "video_analysis.html"
    payload = {
        "video": str(video_path),
        "model_metrics": metrics,
        "limitations": [
            "Offline recording analysis only.",
            "Board state is estimated from visual color and edge features.",
            "Health, gold, champions, items, and augments are not OCR-recognized.",
            "Results are prototype estimates trained on synthetic data.",
        ],
        "timeline": [asdict(prediction) for prediction in predictions],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    html_path.write_text(_render_html(video_path, predictions, metrics), encoding="utf-8")
    return json_path, html_path


def _render_html(
    video_path: Path,
    predictions: list[VideoPrediction],
    metrics: dict[str, float],
) -> str:
    width, height = 960, 320
    margin = 46
    usable_width = width - margin * 2
    usable_height = height - margin * 2
    points = []
    for index, prediction in enumerate(predictions):
        x = margin + usable_width * (index / max(len(predictions) - 1, 1))
        y = margin + usable_height * (1.0 - prediction.top4_probability)
        points.append(f"{x:.1f},{y:.1f}")

    rows = "\n".join(
        (
            "<tr>"
            f"<td>{_format_time(item.timestamp_seconds)}</td>"
            f"<td>#{item.predicted_placement}</td>"
            f"<td>{item.top4_probability * 100:.1f}%</td>"
            f"<td>{item.board_strength:.0f}/10</td>"
            f"<td>{item.occupied_cells}</td>"
            "</tr>"
        )
        for item in predictions
    )
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>TFT 녹화 영상 승부 예측</title>
  <style>
    body {{ margin: 0; background: #0f172a; color: #e2e8f0; font-family: system-ui, sans-serif; }}
    main {{ width: min(1080px, calc(100% - 32px)); margin: 0 auto; padding: 32px 0; }}
    section {{ background: #172033; border: 1px solid #334155; border-radius: 12px; padding: 20px; margin: 14px 0; }}
    h1, h2 {{ margin-top: 0; }}
    .accent {{ color: #38bdf8; }}
    svg {{ width: 100%; height: auto; background: #111827; border-radius: 8px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 9px; border-bottom: 1px solid #334155; text-align: left; }}
    .warning {{ color: #fbbf24; line-height: 1.6; }}
  </style>
</head>
<body>
<main>
  <h1>TFT 녹화 영상 승부 예측</h1>
  <p>{html.escape(video_path.name)}</p>
  <section>
    <h2>시간대별 Top 4 확률</h2>
    <svg viewBox="0 0 {width} {height}" role="img">
      <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#64748b" />
      <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#64748b" />
      <text x="8" y="{margin + 5}" fill="#94a3b8">100%</text>
      <text x="18" y="{height - margin + 5}" fill="#94a3b8">0%</text>
      <polyline points="{" ".join(points)}" fill="none" stroke="#38bdf8" stroke-width="4" />
    </svg>
  </section>
  <section>
    <h2>분석 결과</h2>
    <table>
      <thead><tr><th>영상 시점</th><th>예측 등수</th><th>Top 4 확률</th><th>시각 보드 강도</th><th>활성 칸</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </section>
  <section>
    <h2>모델 평가</h2>
    <p>합성 hold-out exact placement: <span class="accent">{metrics['placement_accuracy'] * 100:.1f}%</span></p>
    <p>합성 hold-out Top 4: <span class="accent">{metrics['top4_accuracy'] * 100:.1f}%</span></p>
    <p class="warning">이 분석은 녹화 영상의 중앙 보드 영역에서 색상·윤곽선·프레임 변화를 추출한 연구용 prototype입니다. 챔피언, 체력, 골드, 아이템을 정확히 인식하는 OCR/객체 탐지 모델은 포함하지 않으며 실제 TFT 성능을 보장하지 않습니다.</p>
  </section>
</main>
</body>
</html>"""


def _format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    remaining = int(seconds % 60)
    return f"{minutes:02d}:{remaining:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a recorded TFT video offline.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/video_analysis"))
    args = parser.parse_args()

    analyzer = RecordingAnalyzer()
    predictions = analyzer.analyze(args.video, args.interval)
    json_path, html_path = save_report(args.video, predictions, args.output, analyzer.metrics)
    print(f"Analyzed {len(predictions)} frames.")
    print(f"JSON: {json_path}")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
