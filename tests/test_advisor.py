import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

from src.tft_advisor import TFTAdvisor
from src.tft_advisor.game_state import GameState, with_flow
from src.tft_advisor.live_advisor import find_lockfile
from src.tft_advisor.overlay_advisor import display_values
from src.tft_advisor.synthetic_data import ACTIONS, generate_matches
from src.tft_advisor.video_analyzer import (
    RecordingAnalyzer,
    extract_board_grid,
    save_report,
)


class AdvisorTest(unittest.TestCase):
    def test_advisor_recommends_known_action(self) -> None:
        records = generate_matches(n_matches=240, seed=3)
        advisor = TFTAdvisor(n_clusters=4)
        metrics = advisor.fit(records)
        recommendation = advisor.recommend(records[0].deck_vector, records[0].board_grid)

        self.assertGreaterEqual(metrics["top4_accuracy"], 0.0)
        self.assertLessEqual(metrics["top4_accuracy"], 1.0)
        self.assertEqual(metrics["train_samples"], 192.0)
        self.assertEqual(metrics["test_samples"], 48.0)
        self.assertIn(recommendation.action, ACTIONS)
        self.assertGreaterEqual(recommendation.predicted_placement, 1)
        self.assertLessEqual(recommendation.predicted_placement, 8)
        self.assertTrue(np.isclose(sum(recommendation.placement_probabilities), 1.0))
        self.assertTrue(
            np.isclose(
                recommendation.predicted_top4_probability,
                sum(recommendation.placement_probabilities[:4]),
            )
        )

    def test_configured_riot_lockfile_is_detected(self) -> None:
        with TemporaryDirectory() as directory:
            lockfile = Path(directory) / "lockfile"
            lockfile.write_text("LeagueClient:1:1234:password:https", encoding="utf-8")
            with patch.dict("os.environ", {"TFT_RIOT_LOCKFILE": str(lockfile)}):
                self.assertEqual(find_lockfile(), lockfile)

    def test_overlay_display_values_translate_action(self) -> None:
        payload = {
            "source": {"label": "수동 데모 모드", "detail": "샘플 상태 사용"},
            "updated_at": "12:34:56",
            "metrics": {"top4_accuracy": 0.708},
            "recommendation": {
                "predicted_placement": 4,
                "predicted_top4_probability": 0.614,
                "nearest_cluster": 2,
                "action": "hold_economy",
                "expected_placement_improvement": 0.47,
            },
        }
        values = display_values(payload)

        self.assertEqual(values["placement"], "#4")
        self.assertEqual(values["top4"], "61%")
        self.assertEqual(values["accuracy"], "71%")
        self.assertEqual(values["action"], "골드 유지")

    def test_game_state_tracks_changes(self) -> None:
        previous = GameState(health=82, gold=50, level=6, board_strength=7)
        current = GameState(health=61, gold=18, level=7, board_strength=5)
        flowed = with_flow(current, previous)

        self.assertEqual(flowed.health_delta, -21)
        self.assertEqual(flowed.gold_delta, -32)
        self.assertEqual(flowed.level_delta, 1)
        self.assertEqual(flowed.board_strength_delta, -2)
        self.assertEqual(len(flowed.to_vector()), 11)

    def test_low_health_prioritizes_roll_over_economy(self) -> None:
        advisor = TFTAdvisor()
        low_health = GameState(health=22, gold=55, health_delta=-18).to_vector()
        adjustments, reasons = advisor._state_action_adjustments(low_health)

        self.assertGreater(adjustments["roll_down"], adjustments["hold_economy"])
        self.assertIn("체력", reasons["roll_down"])

    def test_video_board_grid_has_expected_shape(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.rectangle(frame, (350, 260), (440, 360), (0, 180, 255), -1)
        cv2.rectangle(frame, (700, 400), (790, 510), (255, 80, 20), -1)

        board_grid, bounds = extract_board_grid(frame)

        self.assertEqual(board_grid.shape, (4, 7))
        self.assertGreaterEqual(np.count_nonzero(board_grid), 1)
        self.assertEqual(len(bounds), 4)

    def test_recording_analyzer_creates_timeline_and_report(self) -> None:
        with TemporaryDirectory() as directory:
            directory_path = Path(directory)
            video_path = directory_path / "sample.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                5.0,
                (640, 360),
            )
            for index in range(15):
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.circle(
                    frame,
                    (180 + index * 10, 190),
                    28,
                    (20, 140 + index * 5, 240),
                    -1,
                )
                writer.write(frame)
            writer.release()

            analyzer = RecordingAnalyzer()
            predictions = analyzer.analyze(video_path, interval_seconds=1.0)
            json_path, html_path = save_report(
                video_path,
                predictions,
                directory_path / "report",
                analyzer.metrics,
            )

            self.assertGreaterEqual(len(predictions), 2)
            self.assertTrue(json_path.is_file())
            self.assertTrue(html_path.is_file())
            self.assertIn("Top 4", html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
