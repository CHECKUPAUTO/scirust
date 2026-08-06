#!/usr/bin/env python3
"""Independent repeated GSM8K validation for the frozen guarded Dream policy.

The runtime guard is loaded from the local counterfactual selection report and
is never fitted or recalibrated here. Evaluation uses 60 GSM8K indices disjoint
from the earlier registered benchmark and two repeats per mode in ABBA/BAAB
order. The earlier negative end-to-end verdict remains preserved.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import random
import sys
import time
import types
from typing import Any

import torch
from transformers import AutoTokenizer

from benchmark_frozen_ensemble import (
    Example,
    RunResult,
    attention_modules,
    collect_runtime_metrics,
    load_frozen_policies,
    normalize_number,
    prompt_text,
    set_determinism,
    sha256_file,
)
from benchmark_repeated_crossover import (
    ratio,
    relative_improvement,
    summarize_metric,
    summarize_runs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elastic-cache", type=Path, required=True)
    parser.add_argument("--policy-report", type=Path, required=True)
    parser.add_argument("--guard-selection", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--gsm8k-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--warmup-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--window-length", type=int, default=32)
    parser.add_argument("--decoding-threshold", type=float, default=0.9)
    parser.add_argument("--vote-threshold", type=int, default=3)
    parser.add_argument("--quality-noninferiority-margin", type=float, default=0.05)
    parser.add_argument("--minimum-latency-improvement", type=float, default=0.005)
    parser.add_argument("--minimum-refresh-cost-improvement", type=float, default=0.005)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    return parser.parse_args()


def load_guard(path: Path, policy_report: Path) -> dict[str, float | int | None]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "exploratory_trajectory_stability_guard_selection":
        raise RuntimeError("unexpected guard-selection report status")
    selected = report.get("selected")
    if not isinstance(selected, dict) or selected.get("eligible") is not True:
        raise RuntimeError("guard-selection report has no eligible selected guard")
    policy_metadata = report.get("policy_report", {})
    if policy_metadata.get("sha256") != sha256_file(policy_report):
        raise RuntimeError("guard selection was produced for a different policy report")
    guard = selected.get("guard")
    if not isinstance(guard, dict):
        raise RuntimeError("selected guard is missing")
    required = {
        "minimum_skip_margin",
        "minimum_refresh_cost",
        "max_skips_per_layer",
        "cooldown_decisions",
    }
    if set(guard) != required:
        raise RuntimeError(f"unexpected guard fields: {sorted(guard)}")
    minimum_margin = float(guard["minimum_skip_margin"])
    minimum_cost = float(guard["minimum_refresh_cost"])
    max_skips_raw = guard["max_skips_per_layer"]
    max_skips = None if max_skips_raw is None else int(max_skips_raw)
    cooldown = int(guard["cooldown_decisions"])
    values = [minimum_margin, minimum_cost, float(cooldown)]
    if max_skips is not None:
        values.append(float(max_skips))
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("guard contains non-finite values")
    if minimum_margin < 0.0 or not 0.0 <= minimum_cost <= 1.0:
        raise RuntimeError("guard thresholds are outside their valid ranges")
    if max_skips is not None and max_skips < 0:
        raise RuntimeError("guard max skips must be non-negative or null")
    if cooldown < 0:
        raise RuntimeError("guard cooldown must be non-negative")
    return {
        "minimum_skip_margin": minimum_margin,
        "minimum_refresh_cost": minimum_cost,
        "max_skips_per_layer": max_skips,
        "cooldown_decisions": cooldown,
    }


def load_independent_examples(
    path: Path,
    excluded_indices: set[int],
    seed: int,
    warmup_samples: int,
    samples: int,
) -> tuple[list[Example], list[Example]]:
    examples: list[Example] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            answer = normalize_number(str(row["answer"]).split("####")[-1].strip())
            if answer is None:
                raise RuntimeError(f"cannot parse GSM8K answer at row {index}")
            examples.append(Example(index, str(row["question"]), answer))
    candidates = [index for index in range(len(examples)) if index not in excluded_indices]
    required = warmup_samples + samples
    if len(candidates) < required:
        raise RuntimeError(
            f"only {len(candidates)} independent GSM8K examples remain; need {required}"
        )
    random.Random(seed).shuffle(candidates)
    selected = [examples[index] for index in candidates[:required]]
    return selected[:warmup_samples], selected[warmup_samples:]


def configure_runtime(
    model: Any,
    mode: str,
    policies: list[tuple[tuple[float, ...], float]],
    vote_threshold: int,
    guard: dict[str, float | int | None],
) -> None:
    layers = attention_modules(model)
    for module in layers:
        module.scirust_policy_mode = mode
        module.scirust_policy_ensemble = policies
        module.scirust_policy_vote_threshold = vote_threshold
        module.scirust_layer_count = len(layers)
        module.scirust_guard_minimum_skip_margin = float(
            guard["minimum_skip_margin"]
        )
        module.scirust_guard_minimum_refresh_cost = float(
            guard["minimum_refresh_cost"]
        )
        module.scirust_guard_max_skips_per_layer = guard["max_skips_per_layer"]
        module.scirust_guard_cooldown_decisions = int(guard["cooldown_decisions"])
        module.scirust_guard_skip_count = 0
        module.scirust_guard_cooldown_remaining = 0
        module.scirust_previous_similarity = None
        module.scirust_cache_age_steps = 0
        module.scirust_decisions = 0
        module.scirust_refreshes = 0
        module.scirust_refresh_cost = 0.0
        module.scirust_possible_refresh_cost = 0.0
        module.scirust_risk_sum = 0.0
        module.scirust_trace_enabled = False
        module.scirust_trace = []


@torch.inference_mode()
def generate_one(
    model: Any,
    tokenizer: Any,
    example: Example,
    mode: str,
    policies: list[tuple[tuple[float, ...], float]],
    guard: dict[str, float | int | None],
    args: argparse.Namespace,
    generation_seed: int,
) -> RunResult:
    set_determinism(generation_seed)
    configure_runtime(model, mode, policies, args.vote_threshold, guard)
    rendered = prompt_text(tokenizer, example.question)
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded.input_ids.to("cuda")
    attention_mask = encoded.attention_mask.to("cuda")
    steps = max(1, math.ceil(args.max_new_tokens / args.window_length))

    torch.cuda.synchronize()
    started = time.perf_counter()
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        output_history=False,
        return_dict_in_generate=True,
        steps=steps,
        temperature=0.0,
        top_p=None,
        top_k=None,
        alg="confidence_threshold",
        threshold=args.decoding_threshold,
        gamma=0.9,
        window_length=args.window_length,
        track_num=1,
        block_caching=True,
        tokens_per_iter=1,
        eos_id=tokenizer.eos_token_id,
        bos_id=tokenizer.bos_token_id,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    generated = output.sequences[0, input_ids.shape[1] :]
    response = tokenizer.decode(generated.tolist(), skip_special_tokens=True)
    from benchmark_frozen_ensemble import extract_prediction

    prediction = extract_prediction(response)
    decisions, refreshes, refresh_cost, possible = collect_runtime_metrics(model)
    result = RunResult(
        mode=mode,
        correct=prediction == example.answer,
        prediction=prediction,
        gold=example.answer,
        elapsed_seconds=elapsed,
        decisions=decisions,
        refreshes=refreshes,
        refresh_rate=refreshes / decisions if decisions else 0.0,
        refresh_cost=refresh_cost,
        possible_refresh_cost=possible,
        conditional_refresh_fraction=refresh_cost / possible if possible else 0.0,
        response=response,
    )
    del input_ids, attention_mask, output, generated
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.samples < 20:
        raise SystemExit("at least 20 evaluation samples are required")
    if args.warmup_samples < 1:
        raise SystemExit("at least one warmup sample is required")

    policies = load_frozen_policies(args.policy_report)
    if not 1 <= args.vote_threshold <= len(policies):
        raise SystemExit("invalid vote threshold")
    guard = load_guard(args.guard_selection, args.policy_report)
    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    source_indices_raw = source.get("dataset", {}).get("indices")
    if not isinstance(source_indices_raw, list) or len(source_indices_raw) < 20:
        raise SystemExit("source report does not contain fixed GSM8K indices")
    excluded_indices = {int(value) for value in source_indices_raw}
    warmup, examples = load_independent_examples(
        args.gsm8k_test,
        excluded_indices,
        args.seed,
        args.warmup_samples,
        args.samples,
    )
    evaluation_indices = {example.dataset_index for example in examples}
    if evaluation_indices & excluded_indices:
        raise SystemExit("independent evaluation overlaps the registered GSM8K indices")

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
                "guard": guard,
                "sequence": "ABBA/BAAB",
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for offset, example in enumerate(warmup):
        mode = "always" if offset % 2 == 0 else "guarded_ensemble"
        generate_one(
            model,
            tokenizer,
            example,
            mode,
            policies,
            guard,
            args,
            args.seed + 900000 + example.dataset_index,
        )
        print(
            json.dumps(
                {"event": "warmup", "mode": mode, "index": offset + 1},
                sort_keys=True,
            ),
            flush=True,
        )

    pair_reports: list[dict[str, Any]] = []
    latency_improvements: list[float] = []
    refresh_cost_improvements: list[float] = []
    conditional_savings: list[float] = []
    cost_per_decision_improvements: list[float] = []
    latency_per_decision_improvements: list[float] = []
    decision_ratios: list[float] = []
    accuracy_deltas: list[float] = []
    partial_path = args.output.with_suffix(args.output.suffix + ".partial")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for offset, example in enumerate(examples):
        sequence = (
            ["always", "guarded_ensemble", "guarded_ensemble", "always"]
            if offset % 2 == 0
            else ["guarded_ensemble", "always", "always", "guarded_ensemble"]
        )
        runs: dict[str, list[RunResult]] = {"always": [], "guarded_ensemble": []}
        generation_seed = args.seed + example.dataset_index
        for position, mode in enumerate(sequence):
            run = generate_one(
                model,
                tokenizer,
                example,
                mode,
                policies,
                guard,
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
        guarded = summarize_runs(runs["guarded_ensemble"])
        latency_gain = relative_improvement(
            float(always["median_elapsed_seconds"]),
            float(guarded["median_elapsed_seconds"]),
        )
        refresh_gain = relative_improvement(
            float(always["median_refresh_cost"]),
            float(guarded["median_refresh_cost"]),
        )
        conditional_gain = 1.0 - float(guarded["median_conditional_refresh_fraction"])
        always_cost_per_decision = ratio(
            float(always["median_refresh_cost"]),
            float(always["median_decisions"]),
        )
        guarded_cost_per_decision = ratio(
            float(guarded["median_refresh_cost"]),
            float(guarded["median_decisions"]),
        )
        always_latency_per_decision = ratio(
            float(always["median_elapsed_seconds"]),
            float(always["median_decisions"]),
        )
        guarded_latency_per_decision = ratio(
            float(guarded["median_elapsed_seconds"]),
            float(guarded["median_decisions"]),
        )
        cost_per_decision_gain = relative_improvement(
            always_cost_per_decision,
            guarded_cost_per_decision,
        )
        latency_per_decision_gain = relative_improvement(
            always_latency_per_decision,
            guarded_latency_per_decision,
        )
        decision_ratio = ratio(
            float(guarded["median_decisions"]),
            float(always["median_decisions"]),
        )
        accuracy_delta = int(bool(guarded["representative_correct"])) - int(
            bool(always["representative_correct"])
        )
        pair_report = {
            "dataset_index": example.dataset_index,
            "sequence": sequence,
            "always_refresh": always,
            "guarded_ensemble": guarded,
            "latency_improvement": latency_gain,
            "refresh_cost_improvement": refresh_gain,
            "conditional_refresh_cost_saving": conditional_gain,
            "refresh_cost_per_decision_improvement": cost_per_decision_gain,
            "latency_per_decision_improvement": latency_per_decision_gain,
            "guarded_to_always_decision_ratio": decision_ratio,
            "accuracy_delta": accuracy_delta,
            "responses_identical_between_modes": (
                runs["always"][0].response == runs["guarded_ensemble"][0].response
            ),
        }
        pair_reports.append(pair_report)
        latency_improvements.append(latency_gain)
        refresh_cost_improvements.append(refresh_gain)
        conditional_savings.append(conditional_gain)
        cost_per_decision_improvements.append(cost_per_decision_gain)
        latency_per_decision_improvements.append(latency_per_decision_gain)
        decision_ratios.append(decision_ratio)
        accuracy_deltas.append(float(accuracy_delta))
        partial_path.write_text(
            json.dumps(pair_reports, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

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
        and bool(pair["guarded_ensemble"]["responses_identical"])
        and bool(pair["always_refresh"]["decisions_identical"])
        and bool(pair["guarded_ensemble"]["decisions_identical"])
        for pair in pair_reports
    )
    quality_pass = (
        float(accuracy_summary["mean"]) >= -args.quality_noninferiority_margin
        and float(accuracy_summary["bootstrap_95_percent_ci"][0])
        >= -args.quality_noninferiority_margin
    )
    latency_pass = (
        float(latency_summary["mean"]) >= args.minimum_latency_improvement
        and float(latency_summary["bootstrap_95_percent_ci"][0]) > 0.0
    )
    refresh_pass = (
        float(refresh_summary["mean"]) >= args.minimum_refresh_cost_improvement
        and float(refresh_summary["bootstrap_95_percent_ci"][0]) > 0.0
    )
    validation_success = (
        deterministic_within_mode and quality_pass and latency_pass and refresh_pass
    )

    report = {
        "schema_version": 1,
        "status": "frozen_guard_independent_counterbalanced_gsm8k",
        "source_registered_report": {
            "path": str(args.source_report),
            "sha256": sha256_file(args.source_report),
            "end_to_end_success": source.get("end_to_end_success"),
            "verdict_preserved": True,
            "excluded_indices": sorted(excluded_indices),
        },
        "policy": {
            "path": str(args.policy_report),
            "sha256": sha256_file(args.policy_report),
            "policies": len(policies),
            "vote_threshold": args.vote_threshold,
            "fitted_or_calibrated_here": False,
        },
        "guard_selection": {
            "path": str(args.guard_selection),
            "sha256": sha256_file(args.guard_selection),
            "guard": guard,
            "frozen_before_task_evaluation": True,
            "task_outcomes_used_for_selection": False,
        },
        "dataset": {
            "path": str(args.gsm8k_test),
            "sha256": sha256_file(args.gsm8k_test),
            "selection_seed": args.seed,
            "warmup_indices": [example.dataset_index for example in warmup],
            "indices": [example.dataset_index for example in examples],
            "samples": len(examples),
            "disjoint_from_registered_indices": True,
        },
        "design": {
            "sequence": "ABBA for even positions, BAAB for odd positions",
            "repeats_per_mode_per_question": 2,
            "same_generation_seed_within_question": True,
            "per_question_summary": "median of two repeats per mode",
            "bootstrap_samples": args.bootstrap_samples,
            "independent_task_validation": True,
        },
        "pre_registered_criteria": {
            "quality_noninferiority_margin": args.quality_noninferiority_margin,
            "minimum_mean_latency_improvement": args.minimum_latency_improvement,
            "minimum_mean_refresh_cost_improvement": args.minimum_refresh_cost_improvement,
            "require_latency_ci_lower_bound_above_zero": True,
            "require_refresh_cost_ci_lower_bound_above_zero": True,
            "require_within_mode_determinism": True,
        },
        "within_mode_determinism": {
            "all_questions_deterministic": deterministic_within_mode,
            "always_response_mismatches": sum(
                not bool(pair["always_refresh"]["responses_identical"])
                for pair in pair_reports
            ),
            "guarded_response_mismatches": sum(
                not bool(pair["guarded_ensemble"]["responses_identical"])
                for pair in pair_reports
            ),
        },
        "quality": {
            "accuracy_delta": accuracy_summary,
            "same_prediction_rate": sum(
                pair["always_refresh"]["representative_prediction"]
                == pair["guarded_ensemble"]["representative_prediction"]
                for pair in pair_reports
            )
            / len(pair_reports),
            "exact_response_match_rate": sum(
                bool(pair["responses_identical_between_modes"])
                for pair in pair_reports
            )
            / len(pair_reports),
            "pass": quality_pass,
        },
        "end_to_end_latency": {**latency_summary, "pass": latency_pass},
        "end_to_end_refresh_cost": {**refresh_summary, "pass": refresh_pass},
        "conditional_refresh_cost_saving": conditional_summary,
        "refresh_cost_per_decision_improvement": cost_per_decision_summary,
        "latency_per_decision_improvement": latency_per_decision_summary,
        "guarded_to_always_decision_ratio": decision_ratio_summary,
        "independent_guard_validation_success": validation_success,
        "scientific_conclusion": (
            "The frozen stability guard satisfies the independent GSM8K quality, latency, and refresh-cost criteria."
            if validation_success
            else "The frozen stability guard does not satisfy every independent GSM8K criterion."
        ),
        "evidence_boundary": (
            "Independent 60-question GSM8K validation on one model, decoding configuration, hardware platform, "
            "and guard selected from local counterfactual evidence. Broader workloads remain necessary."
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
    return 0 if validation_success else 2


if __name__ == "__main__":
    raise SystemExit(main())
