import unittest

import numpy as np

from src.tft_advisor import TFTAdvisor
from src.tft_advisor.synthetic_data import ACTIONS, generate_matches


class AdvisorTest(unittest.TestCase):
    def test_advisor_recommends_known_action(self) -> None:
        records = generate_matches(n_matches=240, seed=3)
        advisor = TFTAdvisor(n_clusters=4)
        metrics = advisor.fit(records)
        recommendation = advisor.recommend(records[0].deck_vector, records[0].board_grid)

        self.assertGreaterEqual(metrics["top4_accuracy"], 0.0)
        self.assertLessEqual(metrics["top4_accuracy"], 1.0)
        self.assertIn(recommendation.action, ACTIONS)
        self.assertGreaterEqual(recommendation.predicted_placement, 1)
        self.assertLessEqual(recommendation.predicted_placement, 8)
        self.assertTrue(np.isclose(sum(recommendation.placement_probabilities), 1.0))


if __name__ == "__main__":
    unittest.main()
