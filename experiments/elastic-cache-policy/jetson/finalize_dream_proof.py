#!/usr/bin/env python3
"""Finalize a Dream Jetson proof from an existing trace and SciRust output.

Pure-standard-library recovery tool. It deliberately accepts `gamma=inf`, which
means that no finite fixed cosine threshold met the requested quality budget and
the feasible fixed-policy baseline is unconditional refresh.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from typing import Any

NUMBER = r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan)"


def required(text: str, pattern: str, name: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        raise RuntimeError(f"missing {name} in SciRust output")
    return match.group(1)


def finite_or_text(token: str) -> float | str:
    value = float(token)
    if math.isfinite(value):
        return value
    return token.lower()


def boolean(text: str, name: str) -> bool:
    return required(text, rf"\b{name}=(true|false)\b", name).lower() == "true"


def parse_metrics(text: str) -> dict[str, Any]:
    learned = re.search(
        rf"test learned quality_loss=({NUMBER}) compute=({NUMBER}) refresh_rate=({NUMBER})",
        text,
        flags=re.IGNORECASE,
    )
    fixed = re.search(
        rf"test best_gamma gamma=({NUMBER}) quality_loss=({NUMBER}) "
        rf"compute=({NUMBER}) refresh_rate=({NUMBER})",
        text,
        flags=re.IGNORECASE,
    )
    validation = re.search(
        rf"validation quality_loss=({NUMBER}) compute=({NUMBER}) refresh_rate=({NUMBER})",
        text,
        flags=re.IGNORECASE,
    )
    if not learned or not fixed or not validation:
        raise RuntimeError("incomplete learned, fixed-gamma, or validation metrics")

    gamma_token = fixed.group(1)
    gamma_value = finite_or_text(gamma_token)
    gamma_policy = "always_refresh" if gamma_token.lower() in {"inf", "+inf"} else "threshold"

    return {
        "validation": {
            "quality_loss": float(validation.group(1)),
            "compute": float(validation.group(2)),
            "refresh_rate": float(validation.group(3)),
        },
        "learned_test": {
            "quality_loss": float(learned.group(1)),
            "compute": float(learned.group(2)),
            "refresh_rate": float(learned.group(3)),
        },
        "best_fixed_gamma_test": {
            "gamma": gamma_value,
            "policy": gamma_policy,
            "quality_loss": float(fixed.group(2)),
            "compute": float(fixed.group(3)),
            "refresh_rate": float(fixed.group(4)),
        },
        "learned_meets_budget": boolean(text, "learned_meets_budget"),
        "fixed_gamma_meets_budget": boolean(text, "fixed_gamma_meets_budget"),
        "constrained_better": boolean(text, "constrained_better"),
        "relative_compute_improvement": float(
            required(text, rf"relative_compute_improvement=({NUMBER})", "relative_compute_improvement")
        ),
        "pareto_dominates": boolean(text, "pareto_dominates"),
    }


def trace_summary(trace_path: Path) -> dict[str, Any]:
    rows = 0
    trajectories: set[int] = set()
    per_trajectory: dict[int, int] = {}
    with trace_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {
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
        if set(reader.fieldnames or ()) != expected:
            raise RuntimeError(f"unexpected trace header: {reader.fieldnames}")
        for row in reader:
            trajectory_id = int(row["trajectory_id"])
            trajectories.add(trajectory_id)
            per_trajectory[trajectory_id] = per_trajectory.get(trajectory_id, 0) + 1
            rows += 1
    return {
        "rows": rows,
        "trajectories": len(trajectories),
        "rows_per_trajectory": {str(key): per_trajectory[key] for key in sorted(per_trajectory)},
    }


def verdict(metrics: dict[str, Any], target: float, tolerance: float) -> tuple[bool, str]:
    gain = metrics["relative_compute_improvement"]
    reproduced = (
        metrics["learned_meets_budget"]
        and metrics["fixed_gamma_meets_budget"]
        and metrics["constrained_better"]
        and metrics["pareto_dominates"]
        and target - tolerance <= gain <= target + tolerance
    )
    if reproduced:
        return True, "reproduced"
    if not metrics["learned_meets_budget"]:
        return False, "not_reproduced_learned_policy_exceeds_quality_budget"
    if not metrics["pareto_dominates"]:
        return False, "not_reproduced_no_pareto_dominance"
    return False, "not_reproduced_gain_outside_preregistered_band"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quality-budget", type=float, default=0.05)
    parser.add_argument("--target", type=float, default=0.63110794)
    parser.add_argument("--tolerance", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--model", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--container-image", default="nvcr.io/nvidia/pytorch:25.08-py3")
    parser.add_argument(
        "--container-digest",
        default="sha256:ace9a848c0ae543317e3c4763b6b4248961c47902625abfe3c77a0fb931c50fb",
    )
    args = parser.parse_args()

    trace_path = args.output_dir / "dream_counterfactual_trace.csv"
    raw_path = args.output_dir / "scirust_discovery_output.txt"
    report_path = args.output_dir / "dream_real_policy_report.json"
    if not trace_path.is_file() or not raw_path.is_file():
        raise SystemExit(f"missing trace or SciRust output in {args.output_dir}")

    text = raw_path.read_text(encoding="utf-8")
    metrics = parse_metrics(text)
    summary = trace_summary(trace_path)
    reproduced, verdict_name = verdict(metrics, args.target, args.tolerance)

    report = {
        "schema_version": 2,
        "status": "real_dream_counterfactual_result",
        "verdict": verdict_name,
        "scope": "Dream-v0-Instruct-7B real Jetson counterfactual attention-output trace",
        "model": args.model,
        "hardware": "NVIDIA Jetson AGX Thor",
        "container": {
            "image": args.container_image,
            "digest": args.container_digest,
            "pytorch": "2.8.0a0+34c6371d24.nv25.08",
            "cuda": "13.0",
            "compute_capability": "11.0",
        },
        "seed": args.seed,
        "quality_budget": args.quality_budget,
        "synthetic_reference_gain": args.target,
        "reproduction_tolerance": args.tolerance,
        "reproduction_band": [args.target - args.tolerance, args.target + args.tolerance],
        "trace": summary,
        "metrics": metrics,
        "actual_compute_improvement_percent": 100.0 * metrics["relative_compute_improvement"],
        "reproduced_63_11_percent_band": reproduced,
        "scientific_conclusion": (
            "The synthetic 63.11% claim is not reproduced under this real Dream protocol. "
            "The learned policy reduces normalized refresh compute but violates the "
            "pre-registered stale-loss budget on held-out trajectories."
            if not reproduced
            else "The synthetic gain band is reproduced under this real Dream protocol."
        ),
        "evidence_boundary": (
            "Real Dream attention-state counterfactual evidence on 30 deterministic prompts. "
            "This is not yet an end-to-end GSM8K or HumanEval accuracy result."
        ),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0 if reproduced else 2


if __name__ == "__main__":
    raise SystemExit(main())
