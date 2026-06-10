"""Train and demonstrate the TFT advisor on reproducible sample data."""

from __future__ import annotations

import json
from pathlib import Path

from .recommender import TFTAdvisor
from .synthetic_data import generate_matches


def main() -> None:
    records = generate_matches(n_matches=1800, seed=42)
    advisor = TFTAdvisor(n_clusters=4)
    metrics = advisor.fit(records)
    sample = records[17]
    recommendation = advisor.recommend(sample.deck_vector, sample.board_grid)

    result = {
        "metrics": metrics,
        "sample_archetype": sample.archetype,
        "sample_observed_action": sample.action,
        "sample_observed_placement": sample.placement,
        "recommendation": {
            "nearest_cluster": recommendation.cluster_id,
            "action": recommendation.action,
            "expected_placement": recommendation.expected_placement,
            "expected_placement_improvement": recommendation.expected_placement_improvement,
            "top4_rate": recommendation.top4_rate,
            "predicted_top4_probability": recommendation.predicted_top4_probability,
            "predicted_placement": recommendation.predicted_placement,
            "placement_probabilities": recommendation.placement_probabilities,
        },
    }

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "demo_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("TFT Advisor Demo")
    print("================")
    print(f"Autoencoder reconstruction loss: {metrics['reconstruction_loss']:.4f}")
    print(f"Placement classifier loss:       {metrics['classifier_loss']:.4f}")
    print(f"Hold-out placement accuracy:     {metrics['placement_accuracy']:.3f}")
    print(f"Hold-out Top 4 accuracy:         {metrics['top4_accuracy']:.3f}")
    print(f"Train / test samples:            {int(metrics['train_samples'])} / {int(metrics['test_samples'])}")
    print()
    print(f"Sample archetype:                {sample.archetype}")
    print(f"Observed action / placement:     {sample.action} / {sample.placement}")
    print(f"Nearest meta-deck cluster:       {recommendation.cluster_id}")
    print(f"Predicted placement:             {recommendation.predicted_placement}")
    print(f"Predicted Top 4 probability:     {recommendation.predicted_top4_probability:.3f}")
    print(f"Recommended next action:         {recommendation.action}")
    print(f"Cluster expected placement:      {recommendation.expected_placement:.2f}")
    print(f"Expected placement improvement:  {recommendation.expected_placement_improvement:+.2f}")
    print(f"Cluster Top 4 rate:              {recommendation.top4_rate:.3f}")
    print("Saved outputs/demo_result.json")


if __name__ == "__main__":
    main()

