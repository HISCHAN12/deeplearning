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
    recommended_action: str
    recommended_action_label: str
    recommendation_reason: str
    expected_placement_improvement: float


ACTION_LABELS = {
    "level_up": "레벨 업",
    "roll_down": "리롤",
    "hold_economy": "골드 유지",
    "slam_item": "아이템 즉시 장착",
    "reposition_carry": "캐리 위치 변경",
}


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

    # Use a stable absolute mapping. Per-frame percentile normalization made
    # every screen look populated, including menus and low-information scenes.
    normalized = np.clip((raw_scores - 0.34) / 0.20, 0.0, 1.0)
    normalized[normalized < 0.38] = 0.0
    return normalized, (left, top, right, bottom)


def average_deck_vector() -> np.ndarray:
    vectors = [archetype_deck_vector(name) for name in ARCHETYPES]
    return np.mean(vectors, axis=0)


def visual_game_state(
    progress: float,
    relative_strength: float,
    activity: float,
    strength_delta: float,
) -> GameState:
    board_strength = int(np.clip(round(2.0 + relative_strength * 7.0), 1, 10))
    stage = round(2.1 + progress * 4.4, 1)
    level = int(np.clip(round(4.5 + progress * 4.0), 4, 10))
    estimated_health = int(np.clip(round(30 + relative_strength * 55), 20, 90))
    estimated_gold = int(np.clip(round(18 + relative_strength * 40), 10, 60))
    streak = int(np.clip(round(strength_delta * 8), -3, 3))
    return GameState(
        stage=stage,
        # These visual proxy values are only for a reference recommendation.
        health=estimated_health,
        gold=estimated_gold,
        level=level,
        streak=streak,
        board_strength=board_strength,
        unspent_items=1,
        board_strength_delta=int(np.clip(round(strength_delta * 5), -5, 5)),
    )


def calibrate_probabilities(probabilities: np.ndarray, temperature: float = 1.8) -> np.ndarray:
    """Reduce overconfidence when applying a synthetic-data model to video."""
    safe = np.clip(probabilities, 1e-8, 1.0)
    calibrated = safe ** (1.0 / temperature)
    return calibrated / calibrated.sum()


def select_video_action(
    model_action: str,
    progress: float,
    relative_strength: float,
    strength_delta: float,
    relative_activity: float,
) -> tuple[str, str]:
    """Combine the cluster action with transparent recording-only heuristics."""
    if strength_delta <= -0.18:
        return (
            "reposition_carry",
            "직전 분석 시점보다 시각적 보드 강도가 크게 하락해 캐리 배치 점검을 권장합니다.",
        )
    if relative_strength <= 0.25:
        return (
            "roll_down",
            "영상 내 상대 보드 강도가 낮아 리롤을 통한 즉시 전력 보강을 권장합니다.",
        )
    if relative_activity >= 0.78 and strength_delta >= 0.08:
        return (
            "slam_item",
            "화면 활동량과 보드 강도가 함께 증가해 보유 아이템의 즉시 활용을 참고 행동으로 제안합니다.",
        )
    if relative_strength >= 0.72:
        return (
            "hold_economy",
            "영상 내 상대 보드 강도가 높아 현재 전력을 유지하며 골드를 모으는 선택을 권장합니다.",
        )
    if progress <= 0.72:
        return (
            "level_up",
            "중반 이전의 보통 보드 강도로 판단되어 유닛 수 확장을 위한 레벨 업을 권장합니다.",
        )
    return (
        model_action,
        "후반 시점에는 시각 규칙보다 학습된 meta-deck 군집의 평균 성적을 우선했습니다.",
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
        observations: list[dict[str, object]] = []
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
            occupied = int(np.count_nonzero(board_grid > 0.38))
            active_values = board_grid[board_grid > 0]
            cell_strength = float(active_values.mean()) if len(active_values) else 0.0
            visual_signal = 0.58 * (occupied / 12.0) + 0.42 * cell_strength
            observations.append(
                {
                    "timestamp": timestamp,
                    "progress": progress,
                    "board_grid": board_grid,
                    "activity": activity,
                    "occupied": occupied,
                    "visual_signal": visual_signal,
                }
            )
            if progress_callback:
                progress_callback(min(progress * 0.55, 0.55))

        capture.release()
        if not observations:
            raise ValueError("No frames could be analyzed from the video.")

        visual_signals = np.array(
            [float(observation["visual_signal"]) for observation in observations],
            dtype=np.float64,
        )
        low, high = np.percentile(visual_signals, [10, 90])
        signal_scale = max(float(high - low), 1e-6)
        relative_strengths = np.clip((visual_signals - low) / signal_scale, 0.0, 1.0)
        activities = np.array(
            [float(observation["activity"]) for observation in observations],
            dtype=np.float64,
        )
        activity_low, activity_high = np.percentile(activities, [10, 90])
        activity_scale = max(float(activity_high - activity_low), 1e-6)
        relative_activities = np.clip(
            (activities - activity_low) / activity_scale,
            0.0,
            1.0,
        )

        predictions: list[VideoPrediction] = []
        previous_strength = float(relative_strengths[0])
        for index, observation in enumerate(observations):
            relative_strength = float(relative_strengths[index])
            strength_delta = relative_strength - previous_strength
            previous_strength = relative_strength
            board_grid = np.asarray(observation["board_grid"], dtype=np.float64)
            activity = float(observation["activity"])
            progress = float(observation["progress"])
            game_state = visual_game_state(
                progress,
                relative_strength,
                activity,
                strength_delta,
            )
            recommendation = self.advisor.recommend(
                self.deck_vector,
                board_grid,
                game_state.to_vector(),
            )
            selected_action, action_reason = select_video_action(
                recommendation.action,
                progress,
                relative_strength,
                strength_delta,
                float(relative_activities[index]),
            )
            action_stats = self.advisor.action_stats[recommendation.cluster_id][selected_action]
            action_adjustments, _ = self.advisor._state_action_adjustments(
                game_state.to_vector()
            )
            expected_improvement = (
                action_stats["expected_placement_improvement"]
                + action_adjustments[selected_action]
            )
            calibrated = calibrate_probabilities(
                np.array(recommendation.placement_probabilities, dtype=np.float64)
            )
            predictions.append(
                VideoPrediction(
                    timestamp_seconds=round(float(observation["timestamp"]), 2),
                    progress=round(progress, 4),
                    predicted_placement=int(calibrated.argmax() + 1),
                    top4_probability=round(float(calibrated[:4].sum()), 4),
                    board_strength=float(game_state.board_strength),
                    board_activity=round(activity, 4),
                    occupied_cells=int(observation["occupied"]),
                    placement_probabilities=[round(float(value), 4) for value in calibrated],
                    recommended_action=selected_action,
                    recommended_action_label=ACTION_LABELS.get(
                        selected_action,
                        selected_action,
                    ),
                    recommendation_reason=action_reason,
                    expected_placement_improvement=round(
                        expected_improvement,
                        3,
                    ),
                )
            )
            if progress_callback:
                progress_callback(0.55 + 0.45 * ((index + 1) / len(observations)))
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
            "Action recommendations use visual proxy state and are reference estimates only.",
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
            f"<td><strong>{html.escape(item.recommended_action_label)}</strong></td>"
            f"<td>{html.escape(item.recommendation_reason)}</td>"
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
      <thead><tr><th>영상 시점</th><th>예측 등수</th><th>Top 4 확률</th><th>시각 보드 강도</th><th>활성 칸</th><th>참고 행동</th><th>추천 근거</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </section>
  <section>
    <h2>모델 평가</h2>
    <p>합성 hold-out exact placement: <span class="accent">{metrics['placement_accuracy'] * 100:.1f}%</span></p>
    <p>합성 hold-out Top 4: <span class="accent">{metrics['top4_accuracy'] * 100:.1f}%</span></p>
    <p class="warning">이 분석은 녹화 영상의 중앙 보드 영역에서 색상·윤곽선·프레임 변화를 추출한 연구용 prototype입니다. 참고 행동은 시각적 보드 강도와 변화량으로 추정한 상태를 사용합니다. 챔피언, 체력, 골드, 아이템을 정확히 인식하는 OCR/객체 탐지 모델은 포함하지 않으므로 실제 게임 상태를 확인한 뒤 판단해야 합니다.</p>
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
