#!/usr/bin/env python3
"""Diagnose an existing paired GSM8K report without changing its registered verdict.

This analysis is explicitly secondary. It separates end-to-end effects from
conditional cache-policy effects and reports trajectory divergence, execution
order sensitivity, and matched-response subsets. It never overwrites or
reclassifies the primary pre-registered result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Callable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0.0, min(1.0, probability)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_mean_ci(
    values: list[float],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    rng = random.Random(seed)
    count = len(values)
    means = [
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    ]
    return percentile(means, 0.025), percentile(means, 0.975)


def relative_improvement(reference: float, candidate: float) -> float:
    if reference <= 0.0:
        return 0.0
    return (reference - candidate) / reference


def run_value(pair: dict[str, Any], mode: str, key: str) -> float:
    return float(pair[mode][key])


def paired_values(
    pairs: list[dict[str, Any]],
    function: Callable[[dict[str, Any]], float],
) -> list[float]:
    return [function(pair) for pair in pairs]


def summarize_values(values: list[float], samples: int, seed: int) -> dict[str, Any]:
    low, high = bootstrap_mean_ci(values, samples, seed)
    return {
        "samples": len(values),
        "mean": mean(values),
        "minimum": min(values) if values else 0.0,
        "maximum": max(values) if values else 0.0,
        "bootstrap_95_percent_ci": [low, high],
    }


def subset_summary(
    pairs: list[dict[str, Any]],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    latency = paired_values(
        pairs,
        lambda pair: relative_improvement(
            run_value(pair, "always_refresh", "elapsed_seconds"),
            run_value(pair, "frozen_ensemble", "elapsed_seconds"),
        ),
    )
    refresh_compute = paired_values(
        pairs,
        lambda pair: relative_improvement(
            run_value(pair, "always_refresh", "refresh_cost"),
            run_value(pair, "frozen_ensemble", "refresh_cost"),
        ),
    )
    quality_delta = [
        int(pair["frozen_ensemble"]["correct"])
        - int(pair["always_refresh"]["correct"])
        for pair in pairs
    ]
    return {
        "pairs": len(pairs),
        "latency_relative_improvement": summarize_values(latency, samples, seed),
        "refresh_compute_relative_improvement": summarize_values(
            refresh_compute,
            samples,
            seed + 1,
        ),
        "mean_accuracy_delta": mean([float(value) for value in quality_delta]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()

    source = json.loads(args.report.read_text(encoding="utf-8"))
    if source.get("status") != "frozen_policy_paired_end_to_end_gsm8k":
        raise SystemExit("unexpected source report status")
    pairs = source.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise SystemExit("source report contains no paired observations")

    exact_response_pairs = [
        pair
        for pair in pairs
        if pair["always_refresh"]["response"]
        == pair["frozen_ensemble"]["response"]
    ]
    same_prediction_pairs = [
        pair
        for pair in pairs
        if pair["always_refresh"]["prediction"]
        == pair["frozen_ensemble"]["prediction"]
    ]
    same_correctness_pairs = [
        pair
        for pair in pairs
        if bool(pair["always_refresh"]["correct"])
        == bool(pair["frozen_ensemble"]["correct"])
    ]

    decision_ratios = paired_values(
        pairs,
        lambda pair: run_value(pair, "frozen_ensemble", "decisions")
        / max(run_value(pair, "always_refresh", "decisions"), 1.0),
    )
    possible_cost_ratios = paired_values(
        pairs,
        lambda pair: run_value(pair, "frozen_ensemble", "possible_refresh_cost")
        / max(run_value(pair, "always_refresh", "possible_refresh_cost"), 1e-15),
    )
    conditional_savings = paired_values(
        pairs,
        lambda pair: 1.0
        - run_value(pair, "frozen_ensemble", "conditional_refresh_fraction"),
    )
    per_decision_cost_improvements = paired_values(
        pairs,
        lambda pair: relative_improvement(
            run_value(pair, "always_refresh", "refresh_cost")
            / max(run_value(pair, "always_refresh", "decisions"), 1.0),
            run_value(pair, "frozen_ensemble", "refresh_cost")
            / max(run_value(pair, "frozen_ensemble", "decisions"), 1.0),
        ),
    )
    latency_per_decision_improvements = paired_values(
        pairs,
        lambda pair: relative_improvement(
            run_value(pair, "always_refresh", "elapsed_seconds")
            / max(run_value(pair, "always_refresh", "decisions"), 1.0),
            run_value(pair, "frozen_ensemble", "elapsed_seconds")
            / max(run_value(pair, "frozen_ensemble", "decisions"), 1.0),
        ),
    )

    always_first = [
        pair for pair in pairs if pair.get("execution_order", [None])[0] == "always"
    ]
    ensemble_first = [
        pair for pair in pairs if pair.get("execution_order", [None])[0] == "ensemble"
    ]

    decisions_more = sum(
        run_value(pair, "frozen_ensemble", "decisions")
        > run_value(pair, "always_refresh", "decisions")
        for pair in pairs
    )
    decisions_equal = sum(
        run_value(pair, "frozen_ensemble", "decisions")
        == run_value(pair, "always_refresh", "decisions")
        for pair in pairs
    )
    decisions_fewer = len(pairs) - decisions_more - decisions_equal

    negative_latency = sorted(
        pairs,
        key=lambda pair: float(pair["latency_improvement"]),
    )[:5]
    negative_compute = sorted(
        pairs,
        key=lambda pair: float(pair["refresh_compute_improvement"]),
    )[:5]

    report = {
        "schema_version": 1,
        "status": "secondary_end_to_end_diagnostic",
        "source_report": str(args.report),
        "source_report_sha256": sha256_file(args.report),
        "primary_registered_verdict": {
            "end_to_end_success": source.get("end_to_end_success"),
            "quality": source.get("quality"),
            "latency": source.get("latency"),
            "refresh_compute": source.get("refresh_compute"),
            "scientific_conclusion": source.get("scientific_conclusion"),
            "unchanged_by_this_analysis": True,
        },
        "trajectory_divergence": {
            "pairs": len(pairs),
            "exact_response_matches": len(exact_response_pairs),
            "exact_response_match_rate": len(exact_response_pairs) / len(pairs),
            "same_prediction": len(same_prediction_pairs),
            "same_prediction_rate": len(same_prediction_pairs) / len(pairs),
            "same_correctness": len(same_correctness_pairs),
            "same_correctness_rate": len(same_correctness_pairs) / len(pairs),
            "ensemble_decisions_more_often": decisions_more,
            "ensemble_decisions_equal": decisions_equal,
            "ensemble_decisions_fewer_often": decisions_fewer,
            "ensemble_to_always_decision_ratio": summarize_values(
                decision_ratios,
                args.bootstrap_samples,
                args.seed,
            ),
            "ensemble_to_always_possible_refresh_cost_ratio": summarize_values(
                possible_cost_ratios,
                args.bootstrap_samples,
                args.seed + 1,
            ),
        },
        "secondary_policy_diagnostics": {
            "conditional_refresh_cost_saving": summarize_values(
                conditional_savings,
                args.bootstrap_samples,
                args.seed + 2,
            ),
            "refresh_cost_per_decision_improvement": summarize_values(
                per_decision_cost_improvements,
                args.bootstrap_samples,
                args.seed + 3,
            ),
            "latency_per_decision_improvement": summarize_values(
                latency_per_decision_improvements,
                args.bootstrap_samples,
                args.seed + 4,
            ),
        },
        "matched_subsets": {
            "exact_response": subset_summary(
                exact_response_pairs,
                args.bootstrap_samples,
                args.seed + 10,
            ),
            "same_prediction": subset_summary(
                same_prediction_pairs,
                args.bootstrap_samples,
                args.seed + 20,
            ),
            "same_correctness": subset_summary(
                same_correctness_pairs,
                args.bootstrap_samples,
                args.seed + 30,
            ),
        },
        "execution_order": {
            "always_first": subset_summary(
                always_first,
                args.bootstrap_samples,
                args.seed + 40,
            ),
            "ensemble_first": subset_summary(
                ensemble_first,
                args.bootstrap_samples,
                args.seed + 50,
            ),
        },
        "largest_negative_latency_pairs": [
            {
                "dataset_index": pair["dataset_index"],
                "latency_improvement": pair["latency_improvement"],
                "refresh_compute_improvement": pair[
                    "refresh_compute_improvement"
                ],
                "always_decisions": pair["always_refresh"]["decisions"],
                "ensemble_decisions": pair["frozen_ensemble"]["decisions"],
                "same_response": pair["always_refresh"]["response"]
                == pair["frozen_ensemble"]["response"],
            }
            for pair in negative_latency
        ],
        "largest_negative_refresh_compute_pairs": [
            {
                "dataset_index": pair["dataset_index"],
                "refresh_compute_improvement": pair[
                    "refresh_compute_improvement"
                ],
                "latency_improvement": pair["latency_improvement"],
                "always_refresh_cost": pair["always_refresh"]["refresh_cost"],
                "ensemble_refresh_cost": pair["frozen_ensemble"][
                    "refresh_cost"
                ],
                "always_decisions": pair["always_refresh"]["decisions"],
                "ensemble_decisions": pair["frozen_ensemble"]["decisions"],
                "same_response": pair["always_refresh"]["response"]
                == pair["frozen_ensemble"]["response"],
            }
            for pair in negative_compute
        ],
        "interpretation": (
            "The primary registered end-to-end verdict is preserved. Secondary "
            "metrics diagnose whether total latency and refresh-cost results are "
            "driven by cache decisions, divergent generation paths, or execution "
            "order. They must not be used to retroactively redefine success."
        ),
    }

    output = args.output or args.report.with_name(
        "dream_frozen_ensemble_gsm8k_diagnostic.json"
    )
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nRapport diagnostique: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
