#!/usr/bin/env python3
"""Run deterministic trajectory-balanced cross-validation on an existing Dream trace."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any


FLOAT = r"(?:inf|-inf|nan|[0-9.eE+-]+)"


def parse_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"invalid boolean: {value}")


def parse_float(value: str) -> float | str:
    parsed = float(value)
    if math.isfinite(parsed):
        return parsed
    return value


def require(pattern: str, text: str, label: str) -> re.Match[str]:
    match = re.search(pattern, text)
    if not match:
        raise RuntimeError(f"missing {label} in SciRust output")
    return match


def parse_fold(text: str) -> dict[str, Any]:
    split = require(
        r"split folds=(\d+) validation_fold=(\d+) test_fold=(\d+)",
        text,
        "split",
    )
    validation = require(
        rf"validation trajectory mean_quality_loss=({FLOAT}) "
        rf"tail_quality_loss=({FLOAT}) worst_quality_loss=({FLOAT}) "
        rf"mean_compute=({FLOAT}) mean_refresh_rate=({FLOAT})",
        text,
        "validation trajectory metrics",
    )
    learned = require(
        rf"robust test learned quality_loss=({FLOAT}) mean_quality_loss=({FLOAT}) "
        rf"tail_quality_loss=({FLOAT}) worst_quality_loss=({FLOAT}) "
        rf"compute=({FLOAT}) mean_compute=({FLOAT}) refresh_rate=({FLOAT})",
        text,
        "robust learned metrics",
    )
    baseline = require(
        rf"robust test best_gamma gamma=({FLOAT}) quality_loss=({FLOAT}) "
        rf"mean_quality_loss=({FLOAT}) tail_quality_loss=({FLOAT}) "
        rf"worst_quality_loss=({FLOAT}) compute=({FLOAT}) "
        rf"mean_compute=({FLOAT}) refresh_rate=({FLOAT})",
        text,
        "robust gamma metrics",
    )
    status = require(
        r"robust learned_meets_budget=(true|false) "
        r"fixed_gamma_meets_budget=(true|false) constrained_better=(true|false)",
        text,
        "robust status",
    )
    comparison = require(
        rf"robust relative_compute_improvement=({FLOAT}) pareto_dominates=(true|false)",
        text,
        "robust comparison",
    )
    weights = require(r"weights=\[(.*?)\]", text, "weights")
    threshold = require(rf"threshold=({FLOAT})", text, "threshold")

    return {
        "folds": int(split.group(1)),
        "validation_fold": int(split.group(2)),
        "test_fold": int(split.group(3)),
        "weights": [float(item.strip()) for item in weights.group(1).split(",")],
        "threshold": parse_float(threshold.group(1)),
        "validation": {
            "mean_quality_loss": float(validation.group(1)),
            "tail_quality_loss": float(validation.group(2)),
            "worst_quality_loss": float(validation.group(3)),
            "mean_compute": float(validation.group(4)),
            "mean_refresh_rate": float(validation.group(5)),
        },
        "learned_test": {
            "quality_loss": float(learned.group(1)),
            "mean_quality_loss": float(learned.group(2)),
            "tail_quality_loss": float(learned.group(3)),
            "worst_quality_loss": float(learned.group(4)),
            "compute": float(learned.group(5)),
            "mean_compute": float(learned.group(6)),
            "refresh_rate": float(learned.group(7)),
        },
        "best_fixed_gamma_test": {
            "gamma": parse_float(baseline.group(1)),
            "quality_loss": float(baseline.group(2)),
            "mean_quality_loss": float(baseline.group(3)),
            "tail_quality_loss": float(baseline.group(4)),
            "worst_quality_loss": float(baseline.group(5)),
            "compute": float(baseline.group(6)),
            "mean_compute": float(baseline.group(7)),
            "refresh_rate": float(baseline.group(8)),
        },
        "learned_meets_budget": parse_bool(status.group(1)),
        "fixed_gamma_meets_budget": parse_bool(status.group(2)),
        "constrained_better": parse_bool(status.group(3)),
        "relative_compute_improvement": float(comparison.group(1)),
        "pareto_dominates": parse_bool(comparison.group(2)),
    }


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--quality-budget", type=float, default=0.05)
    parser.add_argument("--calibration-budget-fraction", type=float, default=0.5)
    parser.add_argument("--tail-quality-quantile", type=float, default=0.9)
    parser.add_argument("--tail-penalty-weight", type=float, default=4.0)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    if not args.binary.is_file():
        raise SystemExit(f"binary not found: {args.binary}")
    if not args.trace.is_file():
        raise SystemExit(f"trace not found: {args.trace}")
    if args.folds < 3:
        raise SystemExit("--folds must be at least 3")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_results: list[dict[str, Any]] = []

    for test_fold in range(args.folds):
        command = [
            str(args.binary),
            "--trace",
            str(args.trace),
            "--seed",
            str(args.seed + test_fold),
            "--steps",
            str(args.steps),
            "--max-quality-loss",
            str(args.quality_budget),
            "--calibration-budget-fraction",
            str(args.calibration_budget_fraction),
            "--trajectory-balanced",
            "--tail-quality-quantile",
            str(args.tail_quality_quantile),
            "--tail-penalty-weight",
            str(args.tail_penalty_weight),
            "--folds",
            str(args.folds),
            "--test-fold",
            str(test_fold),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        combined = completed.stdout + "\n" + completed.stderr
        output_path = args.output_dir / f"robust_fold_{test_fold}.txt"
        output_path.write_text(combined, encoding="utf-8")
        print(combined, flush=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"SciRust robust fold {test_fold} failed with exit code "
                f"{completed.returncode}; see {output_path}"
            )
        fold_results.append(parse_fold(combined))

    learned_meets = [fold["learned_meets_budget"] for fold in fold_results]
    constrained = [fold["constrained_better"] for fold in fold_results]
    improvements = [
        float(fold["relative_compute_improvement"]) for fold in fold_results
    ]
    learned_compute = [
        float(fold["learned_test"]["compute"]) for fold in fold_results
    ]
    learned_quality = [
        float(fold["learned_test"]["quality_loss"]) for fold in fold_results
    ]
    learned_tail = [
        float(fold["learned_test"]["tail_quality_loss"]) for fold in fold_results
    ]

    all_folds_meet_budget = all(learned_meets)
    all_folds_constrained_better = all(constrained)
    report = {
        "schema_version": 1,
        "status": "exploratory_trajectory_balanced_cross_validation",
        "trace": str(args.trace),
        "configuration": {
            "seed": args.seed,
            "steps_per_fold": args.steps,
            "quality_budget": args.quality_budget,
            "calibration_budget_fraction": args.calibration_budget_fraction,
            "calibration_budget": (
                args.quality_budget * args.calibration_budget_fraction
            ),
            "tail_quality_quantile": args.tail_quality_quantile,
            "tail_penalty_weight": args.tail_penalty_weight,
            "folds": args.folds,
        },
        "aggregate": {
            "all_folds_meet_budget": all_folds_meet_budget,
            "folds_meeting_budget": sum(learned_meets),
            "all_folds_constrained_better": all_folds_constrained_better,
            "folds_constrained_better": sum(constrained),
            "mean_relative_compute_improvement": mean(improvements),
            "minimum_relative_compute_improvement": min(improvements),
            "maximum_relative_compute_improvement": max(improvements),
            "mean_learned_compute": mean(learned_compute),
            "mean_learned_quality_loss": mean(learned_quality),
            "maximum_learned_quality_loss": max(learned_quality),
            "mean_learned_tail_quality_loss": mean(learned_tail),
            "maximum_learned_tail_quality_loss": max(learned_tail),
        },
        "strict_exploratory_success": (
            all_folds_meet_budget and all_folds_constrained_better
        ),
        "evidence_boundary": (
            "Exploratory out-of-fold evaluation on the already observed 30-trajectory "
            "Dream trace. A new independently generated trace is required for a "
            "confirmatory claim."
        ),
        "fold_results": fold_results,
    }
    report_path = args.output_dir / "dream_robust_cross_validation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nRapport: {report_path}")
    return 0 if all_folds_meet_budget else 2


if __name__ == "__main__":
    raise SystemExit(main())
