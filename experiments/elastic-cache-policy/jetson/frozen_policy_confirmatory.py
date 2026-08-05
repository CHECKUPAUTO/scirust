#!/usr/bin/env python3
"""Evaluate the frozen five-policy bundle on an independent Dream trace."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Callable


FEATURE_NAMES = [
    "drift",
    "worsening",
    "head_std",
    "cache_age",
    "untracked_mass",
    "layer_fraction",
    "drift_age",
    "refresh_cost",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_rank(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = math.ceil(max(0.0, min(1.0, quantile)) * len(ordered))
    return ordered[max(0, min(len(ordered) - 1, rank - 1))]


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0.0, min(1.0, probability)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def row_features(row: dict[str, float | int]) -> list[float]:
    drift = 1.0 - float(row["similarity"])
    worsening = max(-float(row["similarity_delta"]), 0.0)
    return [
        drift,
        worsening,
        math.sqrt(max(float(row["head_variance"]), 0.0)),
        float(row["cache_age"]),
        1.0 - float(row["attention_mass"]),
        float(row["layer_fraction"]),
        drift * float(row["cache_age"]),
        float(row["refresh_cost"]),
    ]


def read_trace(path: Path) -> list[dict[str, float | int]]:
    required = {
        "trajectory_id",
        "step",
        "layer_id",
        "similarity",
        "similarity_delta",
        "head_variance",
        "cache_age",
        "attention_mass",
        "layer_fraction",
        "refresh_cost",
        "stale_loss",
    }
    rows: list[dict[str, float | int]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            raise RuntimeError(
                f"unexpected trace header: {reader.fieldnames}; expected {sorted(required)}"
            )
        for raw in reader:
            row: dict[str, float | int] = {
                "trajectory_id": int(raw["trajectory_id"]),
                "step": int(raw["step"]),
                "layer_id": int(raw["layer_id"]),
            }
            for key in required - {"trajectory_id", "step", "layer_id"}:
                value = float(raw[key])
                if not math.isfinite(value):
                    raise RuntimeError(f"non-finite {key} in trace")
                row[key] = value
            rows.append(row)
    if not rows:
        raise RuntimeError("empty trace")
    return rows


def evaluate(
    rows: list[dict[str, float | int]],
    refresh: Callable[[dict[str, float | int]], bool],
    tail_quantile: float,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    by_trajectory: dict[int, dict[str, float | int]] = {}
    total_loss = 0.0
    total_cost = 0.0
    incurred_loss = 0.0
    refresh_cost = 0.0
    refreshes = 0

    for row in rows:
        trajectory_id = int(row["trajectory_id"])
        accumulator = by_trajectory.setdefault(
            trajectory_id,
            {
                "trajectory_id": trajectory_id,
                "rows": 0,
                "refreshes": 0,
                "total_loss": 0.0,
                "incurred_loss": 0.0,
                "total_cost": 0.0,
                "refresh_cost": 0.0,
            },
        )
        stale_loss = float(row["stale_loss"])
        cost = float(row["refresh_cost"])
        decision = refresh(row)

        total_loss += stale_loss
        total_cost += cost
        accumulator["rows"] = int(accumulator["rows"]) + 1
        accumulator["total_loss"] = float(accumulator["total_loss"]) + stale_loss
        accumulator["total_cost"] = float(accumulator["total_cost"]) + cost
        if decision:
            refreshes += 1
            refresh_cost += cost
            accumulator["refreshes"] = int(accumulator["refreshes"]) + 1
            accumulator["refresh_cost"] = float(accumulator["refresh_cost"]) + cost
        else:
            incurred_loss += stale_loss
            accumulator["incurred_loss"] = float(accumulator["incurred_loss"]) + stale_loss

    trajectory_metrics: list[dict[str, float | int]] = []
    for trajectory_id in sorted(by_trajectory):
        accumulator = by_trajectory[trajectory_id]
        trajectory_total_loss = max(float(accumulator["total_loss"]), float.fromhex("0x1.0p-1022"))
        trajectory_total_cost = max(float(accumulator["total_cost"]), float.fromhex("0x1.0p-1022"))
        trajectory_rows = max(int(accumulator["rows"]), 1)
        trajectory_metrics.append(
            {
                "trajectory_id": trajectory_id,
                "rows": int(accumulator["rows"]),
                "quality_loss": float(accumulator["incurred_loss"]) / trajectory_total_loss,
                "compute": float(accumulator["refresh_cost"]) / trajectory_total_cost,
                "refresh_rate": int(accumulator["refreshes"]) / trajectory_rows,
            }
        )

    qualities = [float(item["quality_loss"]) for item in trajectory_metrics]
    computes = [float(item["compute"]) for item in trajectory_metrics]
    refresh_rates = [float(item["refresh_rate"]) for item in trajectory_metrics]
    aggregate = {
        "rows": len(rows),
        "trajectories": len(trajectory_metrics),
        "quality_loss": incurred_loss / max(total_loss, float.fromhex("0x1.0p-1022")),
        "compute": refresh_cost / max(total_cost, float.fromhex("0x1.0p-1022")),
        "refresh_rate": refreshes / len(rows),
        "mean_trajectory_quality_loss": sum(qualities) / len(qualities),
        "tail_trajectory_quality_loss": nearest_rank(qualities, tail_quantile),
        "worst_trajectory_quality_loss": max(qualities),
        "mean_trajectory_compute": sum(computes) / len(computes),
        "mean_trajectory_refresh_rate": sum(refresh_rates) / len(refresh_rates),
    }
    return aggregate, trajectory_metrics


def bootstrap_mean_ci(
    values: list[float],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    rng = random.Random(seed)
    means: list[float] = []
    count = len(values)
    for _ in range(samples):
        means.append(sum(values[rng.randrange(count)] for _ in range(count)) / count)
    return percentile(means, 0.025), percentile(means, 0.975)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-report", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vote-threshold", type=int, default=3)
    parser.add_argument("--quality-budget", type=float, default=0.05)
    parser.add_argument("--tail-quality-quantile", type=float, default=0.90)
    parser.add_argument("--minimum-compute-improvement", type=float, default=0.005)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260805)
    args = parser.parse_args()

    policy_report = json.loads(args.policy_report.read_text(encoding="utf-8"))
    trace_manifest = json.loads(args.trace_manifest.read_text(encoding="utf-8"))
    fold_results = policy_report.get("fold_results")
    if not isinstance(fold_results, list) or len(fold_results) != 5:
        raise SystemExit("the frozen policy report must contain exactly five folds")
    if policy_report.get("strict_exploratory_success") is not True:
        raise SystemExit("the policy bundle did not pass strict exploratory cross-validation")
    if not 1 <= args.vote_threshold <= len(fold_results):
        raise SystemExit("vote threshold must lie within the policy bundle size")
    if trace_manifest.get("policy_fitting_performed") is not False:
        raise SystemExit("trace manifest does not certify trace-only collection")

    source_trace = Path(str(policy_report.get("trace", "")))
    confirmatory_trace_sha256 = sha256_file(args.trace)
    source_trace_sha256 = sha256_file(source_trace) if source_trace.is_file() else None
    independent_trace = (
        args.trace.resolve() != source_trace.resolve()
        and (
            source_trace_sha256 is None
            or confirmatory_trace_sha256 != source_trace_sha256
        )
    )
    if not independent_trace:
        raise SystemExit("confirmatory trace is not independent from the fitting trace")
    if trace_manifest.get("trace_sha256") != confirmatory_trace_sha256:
        raise SystemExit("trace SHA-256 does not match its manifest")

    policies: list[tuple[list[float], float]] = []
    for fold in fold_results:
        weights = [float(value) for value in fold["weights"]]
        threshold = float(fold["threshold"])
        if len(weights) != len(FEATURE_NAMES):
            raise SystemExit("invalid frozen policy feature count")
        policies.append((weights, threshold))

    rows = read_trace(args.trace)
    vote_histogram = {str(votes): 0 for votes in range(len(policies) + 1)}

    def frozen_ensemble_refresh(row: dict[str, float | int]) -> bool:
        features = row_features(row)
        votes = 0
        for weights, threshold in policies:
            risk = sum(weight * feature for weight, feature in zip(weights, features))
            votes += int(risk >= threshold)
        vote_histogram[str(votes)] += 1
        return votes >= args.vote_threshold

    learned, per_trajectory = evaluate(
        rows,
        frozen_ensemble_refresh,
        args.tail_quality_quantile,
    )
    always_refresh, _ = evaluate(rows, lambda _row: True, args.tail_quality_quantile)

    trajectory_improvements = [
        1.0 - float(item["compute"]) for item in per_trajectory
    ]
    ci_low, ci_high = bootstrap_mean_ci(
        trajectory_improvements,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    aggregate_improvement = 1.0 - float(learned["compute"])
    mean_improvement = 1.0 - float(learned["mean_trajectory_compute"])

    budget_pass = (
        float(learned["quality_loss"]) <= args.quality_budget + 1e-12
        and float(learned["mean_trajectory_quality_loss"])
        <= args.quality_budget + 1e-12
        and float(learned["tail_trajectory_quality_loss"])
        <= args.quality_budget + 1e-12
    )
    compute_pass = (
        mean_improvement >= args.minimum_compute_improvement
        and ci_low > 0.0
    )
    confirmatory_success = budget_pass and compute_pass

    report = {
        "schema_version": 1,
        "status": "frozen_policy_independent_confirmatory_evaluation",
        "model": trace_manifest.get("model"),
        "hardware": trace_manifest.get("device"),
        "policy_bundle": {
            "source_report": str(args.policy_report),
            "source_report_sha256": sha256_file(args.policy_report),
            "policies": len(policies),
            "aggregation": "refresh_when_at_least_k_of_five_policies_vote_refresh",
            "vote_threshold": args.vote_threshold,
            "feature_names": FEATURE_NAMES,
            "frozen_before_confirmatory_evaluation": True,
        },
        "trace": {
            "path": str(args.trace),
            "sha256": confirmatory_trace_sha256,
            "manifest": str(args.trace_manifest),
            "seed": trace_manifest.get("seed"),
            "rows": len(rows),
            "trajectories": learned["trajectories"],
            "independent_from_fitting_trace": independent_trace,
            "policy_fitting_performed": False,
        },
        "pre_registered_criteria": {
            "quality_budget": args.quality_budget,
            "tail_quality_quantile": args.tail_quality_quantile,
            "minimum_mean_compute_improvement": args.minimum_compute_improvement,
            "bootstrap_confidence": 0.95,
            "bootstrap_samples": args.bootstrap_samples,
            "require_lower_compute_improvement_ci_above_zero": True,
        },
        "frozen_ensemble": learned,
        "always_refresh_baseline": always_refresh,
        "compute_improvement": {
            "aggregate": aggregate_improvement,
            "mean_per_trajectory": mean_improvement,
            "bootstrap_95_percent_ci": [ci_low, ci_high],
        },
        "vote_histogram": vote_histogram,
        "budget_pass": budget_pass,
        "compute_pass": compute_pass,
        "confirmatory_success": confirmatory_success,
        "scientific_conclusion": (
            "The frozen ensemble satisfies the independent confirmatory criteria."
            if confirmatory_success
            else "The frozen ensemble does not satisfy all independent confirmatory criteria."
        ),
        "evidence_boundary": (
            "Independent local attention-output counterfactual evidence only. "
            "This is not yet an end-to-end task-accuracy or wall-clock latency claim."
        ),
        "per_trajectory": per_trajectory,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "dream_frozen_policy_confirmatory_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nRapport: {report_path}")
    return 0 if confirmatory_success else 2


if __name__ == "__main__":
    raise SystemExit(main())
