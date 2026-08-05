#!/usr/bin/env python3
"""Repeated counterbalanced GSM8K benchmark for the frozen Dream cache ensemble.

This exploratory follow-up preserves the original registered end-to-end verdict.
It reuses exactly the same GSM8K indices, runs each mode twice per question in
ABBA/BAAB order, and compares per-question medians to separate execution-order
noise from generation-path divergence.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import sys
import types
from typing import Any

import torch
from transformers import AutoTokenizer

from benchmark_frozen_ensemble import (
    Example,
    RunResult,
    bootstrap_mean_ci,
    generate_one,
    load_frozen_policies,
    normalize_number,
    set_determinism,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elastic-cache", type=Path, required=True)
    parser.add_argument("--policy-report", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--gsm8k-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--warmup-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--window-length", type=int, default=32)
    parser.add_argument("--decoding-threshold", type=float, default=0.9)
    parser.add_argument("--vote-threshold", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    return parser.parse_args()


def load_selected_examples(path: Path, indices: list[int]) -> list[Example]:
    requested = set(indices)
    examples: dict[int, Example] = {}
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index not in requested:
                continue
            row = json.loads(line)
            answer = normalize_number(str(row["answer"]).split("####")[-1].strip())
            if answer is None:
                raise RuntimeError(f"cannot parse GSM8K answer at row {index}")
            examples[index] = Example(index, str(row["question"]), answer)
    missing = [index for index in indices if index not in examples]
    if missing:
        raise RuntimeError(f"missing GSM8K indices: {missing}")
    return [examples[index] for index in indices]


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def summarize_runs(runs: list[RunResult]) -> dict[str, Any]:
    if len(runs) != 2:
        raise RuntimeError(f"expected exactly two repeats, got {len(runs)}")
    elapsed = [run.elapsed_seconds for run in runs]
    refresh_cost = [run.refresh_cost for run in runs]
    possible_cost = [run.possible_refresh_cost for run in runs]
    decisions = [run.decisions for run in runs]
    refreshes = [run.refreshes for run in runs]
    conditional = [run.conditional_refresh_fraction for run in runs]
    return {
        "repeats": 2,
        "responses_identical": runs[0].response == runs[1].response,
        "predictions_identical": runs[0].prediction == runs[1].prediction,
        "correctness_identical": runs[0].correct == runs[1].correct,
        "decisions_identical": runs[0].decisions == runs[1].decisions,
        "refreshes_identical": runs[0].refreshes == runs[1].refreshes,
        "representative_correct": runs[0].correct,
        "representative_prediction": runs[0].prediction,
        "gold": runs[0].gold,
        "median_elapsed_seconds": median(elapsed),
        "median_refresh_cost": median(refresh_cost),
        "median_possible_refresh_cost": median(possible_cost),
        "median_decisions": median([float(value) for value in decisions]),
        "median_refreshes": median([float(value) for value in refreshes]),
        "median_conditional_refresh_fraction": median(conditional),
        "runs": [asdict(run) for run in runs],
    }


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def relative_improvement(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference if reference > 0.0 else 0.0


def summarize_metric(values: list[float], samples: int, seed: int) -> dict[str, Any]:
    low, high = bootstrap_mean_ci(values, samples, seed)
    return {
        "samples": len(values),
        "mean": sum(values) / len(values) if values else 0.0,
        "median": median(values) if values else 0.0,
        "minimum": min(values) if values else 0.0,
        "maximum": max(values) if values else 0.0,
        "bootstrap_95_percent_ci": [low, high],
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.warmup_samples < 1:
        raise SystemExit("at least one warmup sample is required")

    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    indices = source.get("dataset", {}).get("indices")
    if not isinstance(indices, list) or len(indices) < 20:
        raise SystemExit("source report must contain at least 20 fixed GSM8K indices")
    indices = [int(value) for value in indices]
    examples = load_selected_examples(args.gsm8k_test, indices)
    policies = load_frozen_policies(args.policy_report)
    if not 1 <= args.vote_threshold <= len(policies):
        raise SystemExit("invalid vote threshold")

    dream_dir = args.elastic_cache / "dream"
    if not (dream_dir / "model" / "modeling_dream.py").is_file():
        raise SystemExit(f"invalid Elastic-Cache checkout: {dream_dir}")
    sys.path.insert(0, str(dream_dir))
    from model.generation_utils_elastic import DreamGenerationMixin
    from model.modeling_dream import DreamModel

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    set_determinism(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = DreamModel.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval().to("cuda")
    model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
    model._sample = types.MethodType(DreamGenerationMixin._sample, model)

    print(
        json.dumps(
            {
                "event": "environment",
                "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "samples": len(examples),
                "sequence": "ABBA/BAAB",
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for offset, example in enumerate(examples[: args.warmup_samples]):
        mode = "always" if offset % 2 == 0 else "ensemble"
        generate_one(
            model,
            tokenizer,
            example,
            mode,
            policies,
            args,
            args.seed + 900000 + example.dataset_index,
        )
        print(json.dumps({"event": "warmup", "mode": mode, "index": offset + 1}), flush=True)

    pair_reports: list[dict[str, Any]] = []
    latency_improvements: list[float] = []
    refresh_cost_improvements: list[float] = []
    conditional_savings: list[float] = []
    cost_per_decision_improvements: list[float] = []
    latency_per_decision_improvements: list[float] = []
    decision_ratios: list[float] = []
    partial_path = args.output.with_suffix(args.output.suffix + ".partial")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for offset, example in enumerate(examples):
        sequence = (
            ["always", "ensemble", "ensemble", "always"]
            if offset % 2 == 0
            else ["ensemble", "always", "always", "ensemble"]
        )
        runs: dict[str, list[RunResult]] = {"always": [], "ensemble": []}
        generation_seed = args.seed + example.dataset_index
        for position, mode in enumerate(sequence):
            run = generate_one(
                model,
                tokenizer,
                example,
                mode,
                policies,
                args,
                generation_seed,
            )
            runs[mode].append(run)
            print(
                json.dumps(
                    {
                        "event": "run",
                        "completed_question": offset + 1,
                        "questions": len(examples),
                        "dataset_index": example.dataset_index,
                        "position": position + 1,
                        "mode": mode,
                        "seconds": run.elapsed_seconds,
                        "decisions": run.decisions,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        always = summarize_runs(runs["always"])
        ensemble = summarize_runs(runs["ensemble"])
        latency_gain = relative_improvement(
            float(always["median_elapsed_seconds"]),
            float(ensemble["median_elapsed_seconds"]),
        )
        refresh_gain = relative_improvement(
            float(always["median_refresh_cost"]),
            float(ensemble["median_refresh_cost"]),
        )
        conditional_gain = 1.0 - float(ensemble["median_conditional_refresh_fraction"])
        always_cost_per_decision = ratio(
            float(always["median_refresh_cost"]),
            float(always["median_decisions"]),
        )
        ensemble_cost_per_decision = ratio(
            float(ensemble["median_refresh_cost"]),
            float(ensemble["median_decisions"]),
        )
        always_latency_per_decision = ratio(
            float(always["median_elapsed_seconds"]),
            float(always["median_decisions"]),
        )
        ensemble_latency_per_decision = ratio(
            float(ensemble["median_elapsed_seconds"]),
            float(ensemble["median_decisions"]),
        )
        cost_per_decision_gain = relative_improvement(
            always_cost_per_decision,
            ensemble_cost_per_decision,
        )
        latency_per_decision_gain = relative_improvement(
            always_latency_per_decision,
            ensemble_latency_per_decision,
        )
        decision_ratio = ratio(
            float(ensemble["median_decisions"]),
            float(always["median_decisions"]),
        )

        pair_report = {
            "dataset_index": example.dataset_index,
            "sequence": sequence,
            "always_refresh": always,
            "frozen_ensemble": ensemble,
            "latency_improvement": latency_gain,
            "refresh_cost_improvement": refresh_gain,
            "conditional_refresh_cost_saving": conditional_gain,
            "refresh_cost_per_decision_improvement": cost_per_decision_gain,
            "latency_per_decision_improvement": latency_per_decision_gain,
            "ensemble_to_always_decision_ratio": decision_ratio,
            "accuracy_delta": int(bool(ensemble["representative_correct"]))
            - int(bool(always["representative_correct"])),
            "responses_identical_between_modes": (
                runs["always"][0].response == runs["ensemble"][0].response
            ),
        }
        pair_reports.append(pair_report)
        latency_improvements.append(latency_gain)
        refresh_cost_improvements.append(refresh_gain)
        conditional_savings.append(conditional_gain)
        cost_per_decision_improvements.append(cost_per_decision_gain)
        latency_per_decision_improvements.append(latency_per_decision_gain)
        decision_ratios.append(decision_ratio)
        partial_path.write_text(
            json.dumps(pair_reports, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    accuracy_deltas = [float(pair["accuracy_delta"]) for pair in pair_reports]
    latency_summary = summarize_metric(
        latency_improvements, args.bootstrap_samples, args.seed + 1
    )
    refresh_summary = summarize_metric(
        refresh_cost_improvements, args.bootstrap_samples, args.seed + 2
    )
    conditional_summary = summarize_metric(
        conditional_savings, args.bootstrap_samples, args.seed + 3
    )
    cost_per_decision_summary = summarize_metric(
        cost_per_decision_improvements, args.bootstrap_samples, args.seed + 4
    )
    latency_per_decision_summary = summarize_metric(
        latency_per_decision_improvements, args.bootstrap_samples, args.seed + 5
    )
    decision_ratio_summary = summarize_metric(
        decision_ratios, args.bootstrap_samples, args.seed + 6
    )
    accuracy_summary = summarize_metric(
        accuracy_deltas, args.bootstrap_samples, args.seed + 7
    )

    deterministic_within_mode = all(
        bool(pair["always_refresh"]["responses_identical"])
        and bool(pair["frozen_ensemble"]["responses_identical"])
        and bool(pair["always_refresh"]["decisions_identical"])
        and bool(pair["frozen_ensemble"]["decisions_identical"])
        for pair in pair_reports
    )
    controlled_latency_signal_positive = (
        float(latency_summary["mean"]) > 0.0
        and float(latency_summary["bootstrap_95_percent_ci"][0]) > 0.0
    )
    controlled_refresh_signal_positive = (
        float(refresh_summary["mean"]) > 0.0
        and float(refresh_summary["bootstrap_95_percent_ci"][0]) > 0.0
    )

    report = {
        "schema_version": 1,
        "status": "exploratory_repeated_counterbalanced_gsm8k",
        "source_registered_report": {
            "path": str(args.source_report),
            "sha256": sha256_file(args.source_report),
            "end_to_end_success": source.get("end_to_end_success"),
            "verdict_preserved": True,
        },
        "policy": {
            "path": str(args.policy_report),
            "sha256": sha256_file(args.policy_report),
            "policies": len(policies),
            "vote_threshold": args.vote_threshold,
            "fitted_or_calibrated_here": False,
        },
        "dataset": {
            "path": str(args.gsm8k_test),
            "sha256": sha256_file(args.gsm8k_test),
            "indices": indices,
            "samples": len(indices),
            "same_indices_as_registered_run": True,
        },
        "design": {
            "sequence": "ABBA for even positions, BAAB for odd positions",
            "repeats_per_mode_per_question": 2,
            "same_generation_seed_within_question": True,
            "per_question_summary": "median of two repeats per mode",
            "bootstrap_samples": args.bootstrap_samples,
            "exploratory_post_hoc": True,
        },
        "within_mode_determinism": {
            "all_questions_deterministic": deterministic_within_mode,
            "always_response_mismatches": sum(
                not bool(pair["always_refresh"]["responses_identical"])
                for pair in pair_reports
            ),
            "ensemble_response_mismatches": sum(
                not bool(pair["frozen_ensemble"]["responses_identical"])
                for pair in pair_reports
            ),
        },
        "quality": {
            "accuracy_delta": accuracy_summary,
            "same_prediction_rate": sum(
                pair["always_refresh"]["representative_prediction"]
                == pair["frozen_ensemble"]["representative_prediction"]
                for pair in pair_reports
            )
            / len(pair_reports),
            "exact_response_match_rate": sum(
                bool(pair["responses_identical_between_modes"])
                for pair in pair_reports
            )
            / len(pair_reports),
        },
        "end_to_end_latency": latency_summary,
        "end_to_end_refresh_cost": refresh_summary,
        "conditional_refresh_cost_saving": conditional_summary,
        "refresh_cost_per_decision_improvement": cost_per_decision_summary,
        "latency_per_decision_improvement": latency_per_decision_summary,
        "ensemble_to_always_decision_ratio": decision_ratio_summary,
        "controlled_latency_signal_positive": controlled_latency_signal_positive,
        "controlled_refresh_signal_positive": controlled_refresh_signal_positive,
        "exploratory_controlled_signal": (
            deterministic_within_mode
            and controlled_latency_signal_positive
            and controlled_refresh_signal_positive
        ),
        "scientific_conclusion": (
            "The repeated counterbalanced run isolates a positive latency and refresh-cost signal."
            if deterministic_within_mode
            and controlled_latency_signal_positive
            and controlled_refresh_signal_positive
            else "The repeated counterbalanced run does not isolate both positive end-to-end signals."
        ),
        "evidence_boundary": (
            "Exploratory post-hoc crossover analysis on the same 60 GSM8K questions. "
            "It cannot replace or retroactively alter the registered negative end-to-end verdict."
        ),
        "pairs": pair_reports,
    }
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    partial_path.unlink(missing_ok=True)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nRapport: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
