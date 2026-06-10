import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from src.tft_advisor import TFTAdvisor
from src.tft_advisor.live_advisor import find_lockfile
from src.tft_advisor.overlay_advisor import display_values
from src.tft_advisor.synthetic_data import ACTIONS, generate_matches


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


if __name__ == "__main__":
    unittest.main()
