#!/usr/bin/env python3
"""Select a conservative runtime guard for the frozen Dream cache ensemble.

This is an exploratory development step. It reuses the independent local
counterfactual trace only to choose runtime-implementable guard parameters. The
selected guard must later be frozen and evaluated on previously unseen task
prompts. No GSM8K result is used by this selector.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from frozen_policy_confirmatory import (
    FEATURE_NAMES,
    evaluate,
    nearest_rank,
    read_trace,
    row_features,
    sha256_file,
)


@dataclass(frozen=True)
class ScoredRow:
    row: dict[str, float | int]
    base_refresh: bool
    skip_margin: float


@dataclass(frozen=True)
class Guard:
    minimum_skip_margin: float
    minimum_refresh_cost: float
    max_skips_per_layer: int | None
    cooldown_decisions: int


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


def unique_quantiles(values: list[float], probabilities: list[float]) -> list[float]:
    return sorted({percentile(values, probability) for probability in probabilities})


def load_policies(path: Path) -> list[tuple[list[float], float]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    folds = report.get("fold_results")
    if report.get("strict_exploratory_success") is not True:
        raise RuntimeError("policy report did not pass strict exploratory validation")
    if not isinstance(folds, list) or len(folds) != 5:
        raise RuntimeError("policy report must contain exactly five folds")
    policies: list[tuple[list[float], float]] = []
    for fold in folds:
        weights = [float(value) for value in fold["weights"]]
        threshold = float(fold["threshold"])
        if len(weights) != len(FEATURE_NAMES):
            raise RuntimeError("invalid policy feature count")
        if not all(math.isfinite(value) for value in weights + [threshold]):
            raise RuntimeError("non-finite policy parameter")
        policies.append((weights, threshold))
    return policies


def score_rows(
    rows: list[dict[str, float | int]],
    policies: list[tuple[list[float], float]],
    vote_threshold: int,
) -> list[ScoredRow]:
    scored: list[ScoredRow] = []
    for row in rows:
        features = row_features(row)
        risks = [
            sum(weight * feature for weight, feature in zip(weights, features))
            for weights, _threshold in policies
        ]
        votes = sum(
            int(risk >= threshold)
            for risk, (_weights, threshold) in zip(risks, policies)
        )
        skip_margins = [
            threshold - risk
            for risk, (_weights, threshold) in zip(risks, policies)
            if risk < threshold
        ]
        scored.append(
            ScoredRow(
                row=row,
                base_refresh=votes >= vote_threshold,
                skip_margin=min(skip_margins) if skip_margins else 0.0,
            )
        )
    return scored


def guarded_decisions(
    scored: list[ScoredRow],
    guard: Guard,
) -> dict[int, bool]:
    state: dict[tuple[int, int], dict[str, int]] = {}
    decisions: dict[int, bool] = {}
    for index, item in enumerate(scored):
        row = item.row
        key = (int(row["trajectory_id"]), int(row["layer_id"]))
        layer = state.setdefault(key, {"skips": 0, "cooldown": 0})
        refresh = item.base_refresh
        if not refresh:
            within_budget = (
                guard.max_skips_per_layer is None
                or layer["skips"] < guard.max_skips_per_layer
            )
            allowed = (
                item.skip_margin + 1e-15 >= guard.minimum_skip_margin
                and float(row["refresh_cost"]) + 1e-15 >= guard.minimum_refresh_cost
                and within_budget
                and layer["cooldown"] == 0
            )
            refresh = not allowed
            if allowed:
                layer["skips"] += 1
                layer["cooldown"] = guard.cooldown_decisions
        if refresh and layer["cooldown"] > 0:
            layer["cooldown"] -= 1
        decisions[index] = refresh
    return decisions


def skipped_distribution(
    scored: list[ScoredRow], decisions: dict[int, bool]
) -> dict[str, float | int]:
    counts: dict[int, int] = {}
    for index, item in enumerate(scored):
        if decisions[index]:
            continue
        trajectory_id = int(item.row["trajectory_id"])
        counts[trajectory_id] = counts.get(trajectory_id, 0) + 1
    all_ids = sorted({int(item.row["trajectory_id"]) for item in scored})
    values = [counts.get(trajectory_id, 0) for trajectory_id in all_ids]
    return {
        "total": sum(values),
        "mean_per_trajectory": sum(values) / len(values),
        "tail_90_per_trajectory": nearest_rank([float(value) for value in values], 0.90),
        "maximum_per_trajectory": max(values),
    }


def guard_to_json(guard: Guard) -> dict[str, float | int | None]:
    return {
        "minimum_skip_margin": guard.minimum_skip_margin,
        "minimum_refresh_cost": guard.minimum_refresh_cost,
        "max_skips_per_layer": guard.max_skips_per_layer,
        "cooldown_decisions": guard.cooldown_decisions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-report", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vote-threshold", type=int, default=3)
    parser.add_argument("--quality-budget", type=float, default=0.01)
    parser.add_argument("--minimum-mean-compute-improvement", type=float, default=0.008)
    parser.add_argument("--tail-quality-quantile", type=float, default=0.90)
    args = parser.parse_args()

    policies = load_policies(args.policy_report)
    if not 1 <= args.vote_threshold <= len(policies):
        raise SystemExit("invalid vote threshold")
    rows = read_trace(args.trace)
    scored = score_rows(rows, policies, args.vote_threshold)
    skipped = [item for item in scored if not item.base_refresh]
    if not skipped:
        raise SystemExit("base ensemble has no skipped decisions")

    margin_values = [item.skip_margin for item in skipped]
    cost_values = [float(item.row["refresh_cost"]) for item in skipped]
    margin_thresholds = unique_quantiles(margin_values, [0.0, 0.25, 0.50, 0.75, 0.90])
    cost_thresholds = unique_quantiles(cost_values, [0.0, 0.25, 0.50, 0.75])
    caps: list[int | None] = [1, 2, 4, None]
    cooldowns = [0, 1, 2]

    candidates: list[dict[str, Any]] = []
    for margin in margin_thresholds:
        for cost in cost_thresholds:
            for cap in caps:
                for cooldown in cooldowns:
                    guard = Guard(margin, cost, cap, cooldown)
                    decisions = guarded_decisions(scored, guard)
                    cursor = 0

                    def refresh(_row: dict[str, float | int]) -> bool:
                        nonlocal cursor
                        decision = decisions[cursor]
                        cursor += 1
                        return decision

                    aggregate, _per_trajectory = evaluate(
                        rows,
                        refresh,
                        args.tail_quality_quantile,
                    )
                    mean_improvement = 1.0 - float(aggregate["mean_trajectory_compute"])
                    quality_pass = (
                        float(aggregate["quality_loss"]) <= args.quality_budget + 1e-12
                        and float(aggregate["mean_trajectory_quality_loss"])
                        <= args.quality_budget + 1e-12
                        and float(aggregate["tail_trajectory_quality_loss"])
                        <= args.quality_budget + 1e-12
                    )
                    compute_pass = (
                        mean_improvement + 1e-12
                        >= args.minimum_mean_compute_improvement
                    )
                    candidates.append(
                        {
                            "guard": guard_to_json(guard),
                            "metrics": aggregate,
                            "mean_compute_improvement": mean_improvement,
                            "skipped": skipped_distribution(scored, decisions),
                            "quality_pass": quality_pass,
                            "compute_pass": compute_pass,
                            "eligible": quality_pass and compute_pass,
                        }
                    )

    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        raise SystemExit("no stability guard satisfies the registered local constraints")
    eligible.sort(
        key=lambda candidate: (
            float(candidate["skipped"]["mean_per_trajectory"]),
            float(candidate["skipped"]["maximum_per_trajectory"]),
            -float(candidate["mean_compute_improvement"]),
            float(candidate["metrics"]["tail_trajectory_quality_loss"]),
        )
    )
    selected = eligible[0]

    base_decisions = {index: item.base_refresh for index, item in enumerate(scored)}
    cursor = 0

    def base_refresh(_row: dict[str, float | int]) -> bool:
        nonlocal cursor
        decision = base_decisions[cursor]
        cursor += 1
        return decision

    base_metrics, _ = evaluate(rows, base_refresh, args.tail_quality_quantile)
    report = {
        "schema_version": 1,
        "status": "exploratory_trajectory_stability_guard_selection",
        "policy_report": {
            "path": str(args.policy_report),
            "sha256": sha256_file(args.policy_report),
        },
        "development_trace": {
            "path": str(args.trace),
            "sha256": sha256_file(args.trace),
            "rows": len(rows),
            "trajectories": base_metrics["trajectories"],
            "repurposed_for_guard_development": True,
            "independent_confirmation_required": True,
        },
        "selection_constraints": {
            "quality_budget": args.quality_budget,
            "tail_quality_quantile": args.tail_quality_quantile,
            "minimum_mean_compute_improvement": args.minimum_mean_compute_improvement,
            "ranking": "minimize skipped decisions and concentration, then maximize compute improvement",
        },
        "base_ensemble": {
            "metrics": base_metrics,
            "skipped": skipped_distribution(scored, base_decisions),
        },
        "grid": {
            "minimum_skip_margin_values": margin_thresholds,
            "minimum_refresh_cost_values": cost_thresholds,
            "max_skips_per_layer_values": caps,
            "cooldown_decision_values": cooldowns,
            "candidates": len(candidates),
            "eligible_candidates": len(eligible),
        },
        "selected": selected,
        "top_eligible_candidates": eligible[:20],
        "scientific_conclusion": (
            "A conservative runtime-implementable guard was selected without using task-level benchmark outcomes."
        ),
        "evidence_boundary": (
            "Exploratory guard development on a previously observed local counterfactual trace. "
            "The guard must be frozen before evaluation on new task prompts."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nRapport: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
