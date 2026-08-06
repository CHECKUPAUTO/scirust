#!/usr/bin/env python3
"""Create a compact, fail-closed summary of a repeated crossover report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    return value


def require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} must be finite")
    return result


def metric_summary(report: dict[str, Any], key: str) -> dict[str, Any]:
    metric = require_mapping(report.get(key), key)
    interval = metric.get("bootstrap_95_percent_ci")
    if not isinstance(interval, list) or len(interval) != 2:
        raise RuntimeError(f"{key}.bootstrap_95_percent_ci must contain two values")
    return {
        "samples": int(require_number(metric.get("samples"), f"{key}.samples")),
        "mean": require_number(metric.get("mean"), f"{key}.mean"),
        "median": require_number(metric.get("median"), f"{key}.median"),
        "bootstrap_95_percent_ci": [
            require_number(interval[0], f"{key}.bootstrap_95_percent_ci[0]"),
            require_number(interval[1], f"{key}.bootstrap_95_percent_ci[1]"),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("status") != "exploratory_repeated_counterbalanced_gsm8k":
        raise SystemExit("unexpected repeated crossover report status")

    source = require_mapping(report.get("source_registered_report"), "source_registered_report")
    determinism = require_mapping(report.get("within_mode_determinism"), "within_mode_determinism")
    quality = require_mapping(report.get("quality"), "quality")
    accuracy = require_mapping(quality.get("accuracy_delta"), "quality.accuracy_delta")

    summary = {
        "schema_version": 1,
        "status": "repeated_crossover_compact_summary",
        "source_report": str(args.report),
        "source_report_sha256": sha256_file(args.report),
        "registered_end_to_end_verdict": {
            "end_to_end_success": source.get("end_to_end_success"),
            "verdict_preserved": source.get("verdict_preserved") is True,
        },
        "design": report.get("design"),
        "within_mode_determinism": {
            "all_questions_deterministic": determinism.get("all_questions_deterministic") is True,
            "always_response_mismatches": int(
                require_number(
                    determinism.get("always_response_mismatches"),
                    "within_mode_determinism.always_response_mismatches",
                )
            ),
            "ensemble_response_mismatches": int(
                require_number(
                    determinism.get("ensemble_response_mismatches"),
                    "within_mode_determinism.ensemble_response_mismatches",
                )
            ),
        },
        "quality": {
            "accuracy_delta_mean": require_number(accuracy.get("mean"), "quality.accuracy_delta.mean"),
            "accuracy_delta_bootstrap_95_percent_ci": accuracy.get("bootstrap_95_percent_ci"),
            "same_prediction_rate": require_number(
                quality.get("same_prediction_rate"), "quality.same_prediction_rate"
            ),
            "exact_response_match_rate": require_number(
                quality.get("exact_response_match_rate"), "quality.exact_response_match_rate"
            ),
        },
        "end_to_end_latency": metric_summary(report, "end_to_end_latency"),
        "end_to_end_refresh_cost": metric_summary(report, "end_to_end_refresh_cost"),
        "conditional_refresh_cost_saving": metric_summary(
            report, "conditional_refresh_cost_saving"
        ),
        "refresh_cost_per_decision_improvement": metric_summary(
            report, "refresh_cost_per_decision_improvement"
        ),
        "latency_per_decision_improvement": metric_summary(
            report, "latency_per_decision_improvement"
        ),
        "ensemble_to_always_decision_ratio": metric_summary(
            report, "ensemble_to_always_decision_ratio"
        ),
        "controlled_latency_signal_positive": report.get(
            "controlled_latency_signal_positive"
        )
        is True,
        "controlled_refresh_signal_positive": report.get(
            "controlled_refresh_signal_positive"
        )
        is True,
        "exploratory_controlled_signal": report.get("exploratory_controlled_signal") is True,
        "scientific_conclusion": report.get("scientific_conclusion"),
        "evidence_boundary": report.get("evidence_boundary"),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nRésumé: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
