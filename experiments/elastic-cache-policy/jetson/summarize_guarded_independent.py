#!/usr/bin/env python3
"""Create a compact, fail-closed summary of guarded independent GSM8K validation."""

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


def require_interval(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise RuntimeError(f"{label} must contain exactly two values")
    return [
        require_number(value[0], f"{label}[0]"),
        require_number(value[1], f"{label}[1]"),
    ]


def metric_summary(report: dict[str, Any], key: str) -> dict[str, Any]:
    metric = require_mapping(report.get(key), key)
    result = {
        "samples": int(require_number(metric.get("samples"), f"{key}.samples")),
        "mean": require_number(metric.get("mean"), f"{key}.mean"),
        "median": require_number(metric.get("median"), f"{key}.median"),
        "bootstrap_95_percent_ci": require_interval(
            metric.get("bootstrap_95_percent_ci"),
            f"{key}.bootstrap_95_percent_ci",
        ),
    }
    if "pass" in metric:
        result["pass"] = metric.get("pass") is True
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("status") != "frozen_guard_independent_counterbalanced_gsm8k":
        raise SystemExit("unexpected guarded independent report status")

    source = require_mapping(report.get("source_registered_report"), "source_registered_report")
    guard = require_mapping(report.get("guard_selection"), "guard_selection")
    dataset = require_mapping(report.get("dataset"), "dataset")
    determinism = require_mapping(report.get("within_mode_determinism"), "within_mode_determinism")
    quality = require_mapping(report.get("quality"), "quality")
    accuracy = require_mapping(quality.get("accuracy_delta"), "quality.accuracy_delta")

    summary = {
        "schema_version": 1,
        "status": "guarded_independent_compact_summary",
        "source_report": str(args.report),
        "source_report_sha256": sha256_file(args.report),
        "registered_end_to_end_verdict": {
            "end_to_end_success": source.get("end_to_end_success"),
            "verdict_preserved": source.get("verdict_preserved") is True,
        },
        "guard": {
            "parameters": guard.get("guard"),
            "frozen_before_task_evaluation": guard.get("frozen_before_task_evaluation") is True,
            "task_outcomes_used_for_selection": guard.get("task_outcomes_used_for_selection") is True,
        },
        "dataset": {
            "samples": int(require_number(dataset.get("samples"), "dataset.samples")),
            "selection_seed": int(
                require_number(dataset.get("selection_seed"), "dataset.selection_seed")
            ),
            "disjoint_from_registered_indices": dataset.get(
                "disjoint_from_registered_indices"
            )
            is True,
        },
        "design": report.get("design"),
        "pre_registered_criteria": report.get("pre_registered_criteria"),
        "within_mode_determinism": {
            "all_questions_deterministic": determinism.get("all_questions_deterministic") is True,
            "always_response_mismatches": int(
                require_number(
                    determinism.get("always_response_mismatches"),
                    "within_mode_determinism.always_response_mismatches",
                )
            ),
            "guarded_response_mismatches": int(
                require_number(
                    determinism.get("guarded_response_mismatches"),
                    "within_mode_determinism.guarded_response_mismatches",
                )
            ),
        },
        "quality": {
            "accuracy_delta_mean": require_number(
                accuracy.get("mean"), "quality.accuracy_delta.mean"
            ),
            "accuracy_delta_median": require_number(
                accuracy.get("median"), "quality.accuracy_delta.median"
            ),
            "accuracy_delta_bootstrap_95_percent_ci": require_interval(
                accuracy.get("bootstrap_95_percent_ci"),
                "quality.accuracy_delta.bootstrap_95_percent_ci",
            ),
            "same_prediction_rate": require_number(
                quality.get("same_prediction_rate"), "quality.same_prediction_rate"
            ),
            "exact_response_match_rate": require_number(
                quality.get("exact_response_match_rate"), "quality.exact_response_match_rate"
            ),
            "pass": quality.get("pass") is True,
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
        "guarded_to_always_decision_ratio": metric_summary(
            report, "guarded_to_always_decision_ratio"
        ),
        "independent_guard_validation_success": report.get(
            "independent_guard_validation_success"
        )
        is True,
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
