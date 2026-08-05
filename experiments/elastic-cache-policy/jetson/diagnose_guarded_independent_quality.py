#!/usr/bin/env python3
"""Diagnose quality and trajectory divergences in the guarded GSM8K validation.

This is a secondary, post-hoc analysis. It preserves the independent validation
verdict and never changes the frozen policy, guard, prompts, or success criteria.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
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


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be an array")
    return value


def require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} must be finite")
    return result


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "mean": mean(values),
        "median": median(values),
        "minimum": min(values) if values else 0.0,
        "maximum": max(values) if values else 0.0,
    }


def compact_run(run: dict[str, Any], label: str) -> dict[str, Any]:
    return {
        "correct": bool(run.get("representative_correct")),
        "prediction": run.get("representative_prediction"),
        "gold": run.get("gold"),
        "median_elapsed_seconds": require_number(
            run.get("median_elapsed_seconds"), f"{label}.median_elapsed_seconds"
        ),
        "median_decisions": require_number(
            run.get("median_decisions"), f"{label}.median_decisions"
        ),
        "median_refreshes": require_number(
            run.get("median_refreshes"), f"{label}.median_refreshes"
        ),
        "median_refresh_cost": require_number(
            run.get("median_refresh_cost"), f"{label}.median_refresh_cost"
        ),
    }


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pairs": len(rows),
        "latency_improvement": summarize(
            [float(row["latency_improvement"]) for row in rows]
        ),
        "refresh_cost_improvement": summarize(
            [float(row["refresh_cost_improvement"]) for row in rows]
        ),
        "guarded_to_always_decision_ratio": summarize(
            [float(row["decision_ratio"]) for row in rows]
        ),
        "guarded_skips": summarize([float(row["guarded_skips"]) for row in rows]),
        "decision_delta": summarize([float(row["decision_delta"]) for row in rows]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("status") != "frozen_guard_independent_counterbalanced_gsm8k":
        raise SystemExit("unexpected guarded independent report status")
    if report.get("independent_guard_validation_success") is not False:
        raise SystemExit("quality-failure diagnostic expects a negative primary verdict")

    dataset = require_mapping(report.get("dataset"), "dataset")
    guard_selection = require_mapping(report.get("guard_selection"), "guard_selection")
    guard = require_mapping(guard_selection.get("guard"), "guard_selection.guard")
    pairs = require_list(report.get("pairs"), "pairs")
    expected_samples = int(require_number(dataset.get("samples"), "dataset.samples"))
    if len(pairs) != expected_samples:
        raise SystemExit(
            f"pair count mismatch: expected {expected_samples}, found {len(pairs)}"
        )
    if dataset.get("disjoint_from_registered_indices") is not True:
        raise SystemExit("evaluation indices are not certified disjoint")
    if guard_selection.get("frozen_before_task_evaluation") is not True:
        raise SystemExit("guard was not certified frozen before evaluation")
    if guard_selection.get("task_outcomes_used_for_selection") is not False:
        raise SystemExit("guard selection used task outcomes")

    rows: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    changed_predictions: list[dict[str, Any]] = []
    both_wrong_changed: list[dict[str, Any]] = []

    for offset, raw_pair in enumerate(pairs):
        pair = require_mapping(raw_pair, f"pairs[{offset}]")
        index = int(require_number(pair.get("dataset_index"), f"pairs[{offset}].dataset_index"))
        always = compact_run(
            require_mapping(pair.get("always_refresh"), f"pairs[{offset}].always_refresh"),
            f"pairs[{offset}].always_refresh",
        )
        guarded = compact_run(
            require_mapping(pair.get("guarded_ensemble"), f"pairs[{offset}].guarded_ensemble"),
            f"pairs[{offset}].guarded_ensemble",
        )
        latency = require_number(
            pair.get("latency_improvement"), f"pairs[{offset}].latency_improvement"
        )
        refresh = require_number(
            pair.get("refresh_cost_improvement"),
            f"pairs[{offset}].refresh_cost_improvement",
        )
        ratio = require_number(
            pair.get("guarded_to_always_decision_ratio"),
            f"pairs[{offset}].guarded_to_always_decision_ratio",
        )
        guarded_skips = guarded["median_decisions"] - guarded["median_refreshes"]
        decision_delta = guarded["median_decisions"] - always["median_decisions"]
        prediction_changed = always["prediction"] != guarded["prediction"]
        response_changed = pair.get("responses_identical_between_modes") is not True
        regression = bool(always["correct"]) and not bool(guarded["correct"])
        improvement = not bool(always["correct"]) and bool(guarded["correct"])

        row = {
            "dataset_index": index,
            "always": always,
            "guarded": guarded,
            "prediction_changed": prediction_changed,
            "response_changed": response_changed,
            "quality_regression": regression,
            "quality_improvement": improvement,
            "latency_improvement": latency,
            "refresh_cost_improvement": refresh,
            "decision_ratio": ratio,
            "decision_delta": decision_delta,
            "guarded_skips": guarded_skips,
        }
        rows.append(row)
        if regression:
            regressions.append(row)
        if improvement:
            improvements.append(row)
        if prediction_changed:
            changed_predictions.append(row)
        if prediction_changed and not always["correct"] and not guarded["correct"]:
            both_wrong_changed.append(row)

    max_skips_raw = guard.get("max_skips_per_layer")
    max_skips = None if max_skips_raw is None else int(max_skips_raw)
    unchanged_predictions = [row for row in rows if not row["prediction_changed"]]
    same_decision_count = [row for row in rows if row["decision_delta"] == 0.0]
    more_decisions = [row for row in rows if row["decision_delta"] > 0.0]
    fewer_decisions = [row for row in rows if row["decision_delta"] < 0.0]

    def compact_divergence(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "dataset_index": row["dataset_index"],
            "gold": row["always"]["gold"],
            "always_prediction": row["always"]["prediction"],
            "guarded_prediction": row["guarded"]["prediction"],
            "always_correct": row["always"]["correct"],
            "guarded_correct": row["guarded"]["correct"],
            "guarded_skips": row["guarded_skips"],
            "decision_delta": row["decision_delta"],
            "decision_ratio": row["decision_ratio"],
            "latency_improvement": row["latency_improvement"],
            "refresh_cost_improvement": row["refresh_cost_improvement"],
        }

    report_quality = require_mapping(report.get("quality"), "quality")
    summary = {
        "schema_version": 1,
        "status": "guarded_independent_quality_failure_diagnostic",
        "source_report": str(args.report),
        "source_report_sha256": sha256_file(args.report),
        "primary_verdict": {
            "independent_guard_validation_success": False,
            "quality_pass": report_quality.get("pass") is True,
            "latency_pass": require_mapping(
                report.get("end_to_end_latency"), "end_to_end_latency"
            ).get("pass")
            is True,
            "refresh_cost_pass": require_mapping(
                report.get("end_to_end_refresh_cost"), "end_to_end_refresh_cost"
            ).get("pass")
            is True,
            "unchanged_by_this_analysis": True,
        },
        "guard": guard,
        "dataset": {
            "samples": expected_samples,
            "selection_seed": dataset.get("selection_seed"),
            "disjoint_from_registered_indices": True,
        },
        "quality_transitions": {
            "always_correct_guarded_wrong": len(regressions),
            "always_wrong_guarded_correct": len(improvements),
            "net_accuracy_delta_count": len(improvements) - len(regressions),
            "prediction_changed": len(changed_predictions),
            "both_wrong_with_different_predictions": len(both_wrong_changed),
            "same_prediction": len(unchanged_predictions),
        },
        "prediction_divergences": [
            compact_divergence(row) for row in changed_predictions
        ],
        "quality_regressions": [compact_divergence(row) for row in regressions],
        "quality_improvements": [compact_divergence(row) for row in improvements],
        "subsets": {
            "prediction_changed": group_summary(changed_predictions),
            "prediction_unchanged": group_summary(unchanged_predictions),
            "quality_regressions": group_summary(regressions),
            "more_guarded_decisions": group_summary(more_decisions),
            "same_decision_count": group_summary(same_decision_count),
            "fewer_guarded_decisions": group_summary(fewer_decisions),
        },
        "guard_cap_diagnostic": {
            "max_skips_per_layer": max_skips,
            "regressions_at_skip_cap": (
                sum(int(row["guarded_skips"] >= max_skips) for row in regressions)
                if max_skips is not None
                else None
            ),
            "changed_predictions_at_skip_cap": (
                sum(
                    int(row["guarded_skips"] >= max_skips)
                    for row in changed_predictions
                )
                if max_skips is not None
                else None
            ),
            "all_regressions_at_skip_cap": (
                bool(regressions)
                and max_skips is not None
                and all(row["guarded_skips"] >= max_skips for row in regressions)
            ),
        },
        "scientific_interpretation": (
            "The local stability guard remains deterministic and reduces per-decision work, "
            "but task-level non-inferiority is not established. The divergent predictions are "
            "reported for diagnosis only and must not be used to retune the frozen guard on "
            "these evaluation prompts."
        ),
        "next_protocol_boundary": (
            "Any redesigned policy must use new development evidence and a new untouched task "
            "evaluation set. The current 60 prompts are now evaluation history, not tuning data."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nDiagnostic: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
