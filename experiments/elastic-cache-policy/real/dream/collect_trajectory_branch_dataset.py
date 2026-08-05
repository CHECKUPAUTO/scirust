#!/usr/bin/env python3
"""Collect causal generation labels for individual guard-eligible cache skips."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
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
    extract_prediction,
    load_frozen_policies,
    normalize_number,
    prompt_text,
    set_determinism,
    sha256_file,
)
from benchmark_guarded_independent import load_guard


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--elastic-cache", type=Path, required=True)
    p.add_argument("--policy-report", type=Path, required=True)
    p.add_argument("--guard-selection", type=Path, required=True)
    p.add_argument("--gsm8k-train", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="Dream-org/Dream-v0-Instruct-7B")
    p.add_argument("--seed", type=int, default=20260809)
    p.add_argument("--samples", type=int, default=40)
    p.add_argument("--warmup-samples", type=int, default=2)
    p.add_argument("--max-candidates", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--window-length", type=int, default=32)
    p.add_argument("--decoding-threshold", type=float, default=0.9)
    p.add_argument("--vote-threshold", type=int, default=3)
    p.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    return p.parse_args()


def load_examples(path: Path, seed: int, warmup: int, samples: int):
    rows: list[Example] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            raw = json.loads(line)
            answer = normalize_number(str(raw["answer"]).split("####")[-1].strip())
            if answer is None:
                raise RuntimeError(f"unparseable answer at train row {index}")
            rows.append(Example(index, str(raw["question"]), answer))
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    selected = [rows[index] for index in indices[: warmup + samples]]
    if len(selected) != warmup + samples:
        raise RuntimeError("insufficient GSM8K train rows")
    return selected[:warmup], selected[warmup:]


def configure(model, mode, policies, vote_threshold, guard, target):
    state: dict[str, Any] = {
        "candidate_ordinal": 0,
        "candidates": [],
        "eligible_per_layer": {},
        "target_ordinal": 0 if target is None else int(target),
        "applied": False,
        "applied_candidate": None,
    }
    layers = attention_modules(model)
    for module in layers:
        module.scirust_policy_mode = mode
        module.scirust_policy_ensemble = policies
        module.scirust_policy_vote_threshold = vote_threshold
        module.scirust_layer_count = len(layers)
        module.scirust_guard_minimum_skip_margin = float(guard["minimum_skip_margin"])
        module.scirust_guard_minimum_refresh_cost = float(guard["minimum_refresh_cost"])
        module.scirust_guard_max_skips_per_layer = guard["max_skips_per_layer"]
        module.scirust_guard_cooldown_decisions = int(guard["cooldown_decisions"])
        module.scirust_probe_state = state
        module.scirust_previous_similarity = None
        module.scirust_cache_age_steps = 0
        module.scirust_decisions = 0
        module.scirust_refreshes = 0
        module.scirust_refresh_cost = 0.0
        module.scirust_possible_refresh_cost = 0.0
        module.scirust_risk_sum = 0.0
        module.scirust_trace_enabled = False
        module.scirust_trace = []
    return state


@torch.inference_mode()
def run_one(model, tokenizer, example, mode, policies, guard, args, seed, target=None):
    set_determinism(seed)
    state = configure(model, mode, policies, args.vote_threshold, guard, target)
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
    prediction = extract_prediction(response)
    decisions, refreshes, cost, possible = collect_runtime_metrics(model)
    result = RunResult(
        mode=mode,
        correct=prediction == example.answer,
        prediction=prediction,
        gold=example.answer,
        elapsed_seconds=elapsed,
        decisions=decisions,
        refreshes=refreshes,
        refresh_rate=refreshes / decisions if decisions else 0.0,
        refresh_cost=cost,
        possible_refresh_cost=possible,
        conditional_refresh_fraction=cost / possible if possible else 0.0,
        response=response,
    )
    del input_ids, attention_mask, output, generated
    gc.collect()
    torch.cuda.empty_cache()
    return result, state


def response_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compact(run: RunResult):
    data = asdict(run)
    data["response_sha256"] = response_sha256(run.response)
    data["response_chars"] = len(run.response)
    data.pop("response")
    return data


def rel(base: float, candidate: float) -> float:
    return (base - candidate) / base if base > 0.0 else 0.0


def same_candidate(expected, observed) -> bool:
    if expected["ordinal"] != observed["ordinal"] or expected["layer_id"] != observed["layer_id"]:
        return False
    keys = ("skip_margin", "refresh_cost")
    if any(not math.isclose(float(expected[k]), float(observed[k]), rel_tol=1e-9, abs_tol=1e-12) for k in keys):
        return False
    return all(
        math.isclose(float(v), float(observed["features"][k]), rel_tol=1e-9, abs_tol=1e-12)
        for k, v in expected["features"].items()
    )


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.samples < 10 or args.warmup_samples < 1 or not 1 <= args.max_candidates <= 16:
        raise SystemExit("invalid sample, warmup, or candidate count")
    policies = load_frozen_policies(args.policy_report)
    guard = load_guard(args.guard_selection, args.policy_report)
    if int(guard["cooldown_decisions"]) != 0:
        raise SystemExit("single-skip probe requires a zero-cooldown guard")
    warmup, examples = load_examples(
        args.gsm8k_train, args.seed, args.warmup_samples, args.samples
    )

    dream_dir = args.elastic_cache / "dream"
    sys.path.insert(0, str(dream_dir))
    from model.generation_utils_elastic import DreamGenerationMixin
    from model.modeling_dream import DreamModel

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    set_determinism(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = DreamModel.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True
    ).eval().to("cuda")
    model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
    model._sample = types.MethodType(DreamGenerationMixin._sample, model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "dream_single_skip_trajectory_candidates.jsonl"
    report_path = args.output_dir / "dream_single_skip_trajectory_report.json"
    rows_path.write_text("", encoding="utf-8")

    for example in warmup:
        run_one(
            model, tokenizer, example, "probe_enumerate", policies, guard, args,
            args.seed + 900000 + example.dataset_index,
        )

    rows = []
    prompt_summaries = []
    for offset, example in enumerate(examples):
        seed = args.seed + example.dataset_index
        baseline, baseline_state = run_one(
            model, tokenizer, example, "probe_enumerate", policies, guard, args, seed
        )
        if baseline.decisions != baseline.refreshes or baseline_state["applied"]:
            raise RuntimeError("enumeration did not preserve always-refresh behavior")
        candidates = list(baseline_state["candidates"])[: args.max_candidates]
        for candidate in candidates:
            ordinal = int(candidate["ordinal"])
            branch, state = run_one(
                model, tokenizer, example, "probe_single_skip", policies, guard,
                args, seed, ordinal,
            )
            applied = state.get("applied_candidate")
            if not state.get("applied") or not isinstance(applied, dict):
                raise RuntimeError(f"candidate {ordinal} was not applied")
            if branch.decisions - branch.refreshes != 1 or not same_candidate(candidate, applied):
                raise RuntimeError(f"invalid single-skip branch for candidate {ordinal}")
            row = {
                "schema_version": 1,
                "split": "gsm8k_train",
                "dataset_index": example.dataset_index,
                "generation_seed": seed,
                "candidate": candidate,
                "baseline": compact(baseline),
                "single_skip": compact(branch),
                "labels": {
                    "exact_response_invariant": baseline.response == branch.response,
                    "prediction_invariant": baseline.prediction == branch.prediction,
                    "correctness_invariant": baseline.correct == branch.correct,
                    "quality_regression": baseline.correct and not branch.correct,
                    "quality_improvement": not baseline.correct and branch.correct,
                    "decision_count_invariant": baseline.decisions == branch.decisions,
                },
                "effects": {
                    "decision_delta": branch.decisions - baseline.decisions,
                    "decision_ratio": branch.decisions / baseline.decisions,
                    "latency_improvement": rel(baseline.elapsed_seconds, branch.elapsed_seconds),
                    "refresh_cost_improvement": rel(baseline.refresh_cost, branch.refresh_cost),
                },
            }
            rows.append(row)
            with rows_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        prompt_summaries.append(
            {
                "dataset_index": example.dataset_index,
                "candidates": len(candidates),
                "baseline_prediction": baseline.prediction,
                "baseline_correct": baseline.correct,
            }
        )
        print(json.dumps({"event": "prompt_complete", "completed": offset + 1, "prompts": len(examples), "candidates": len(candidates)}, sort_keys=True), flush=True)

    labels = [row["labels"] for row in rows]
    effects = [row["effects"] for row in rows]
    report = {
        "schema_version": 1,
        "status": "single_skip_generation_trajectory_development_dataset",
        "model": args.model,
        "hardware": torch.cuda.get_device_name(0),
        "dataset": {
            "split": "gsm8k_train",
            "path": str(args.gsm8k_train),
            "sha256": sha256_file(args.gsm8k_train),
            "selection_seed": args.seed,
            "indices": [example.dataset_index for example in examples],
            "prompts": len(examples),
            "test_prompts_used": False,
        },
        "policy_sha256": sha256_file(args.policy_report),
        "guard_sha256": sha256_file(args.guard_selection),
        "guard": guard,
        "design": {
            "baseline": "always refresh while enumerating eligible skips",
            "intervention": "exactly one skip; all other decisions refresh",
            "same_generation_seed_within_prompt": True,
            "maximum_candidates_per_prompt": args.max_candidates,
            "development_only": True,
        },
        "candidate_dataset": {
            "path": str(rows_path),
            "sha256": sha256_file(rows_path),
            "rows": len(rows),
        },
        "labels": {
            "exact_response_invariant_rate": mean([float(x["exact_response_invariant"]) for x in labels]),
            "prediction_invariant_rate": mean([float(x["prediction_invariant"]) for x in labels]),
            "correctness_invariant_rate": mean([float(x["correctness_invariant"]) for x in labels]),
            "decision_count_invariant_rate": mean([float(x["decision_count_invariant"]) for x in labels]),
            "quality_regressions": sum(int(x["quality_regression"]) for x in labels),
            "quality_improvements": sum(int(x["quality_improvement"]) for x in labels),
        },
        "effects": {
            "mean_decision_delta": mean([float(x["decision_delta"]) for x in effects]),
            "mean_decision_ratio": mean([float(x["decision_ratio"]) for x in effects]),
            "mean_latency_improvement": mean([float(x["latency_improvement"]) for x in effects]),
            "mean_refresh_cost_improvement": mean([float(x["refresh_cost_improvement"]) for x in effects]),
        },
        "prompt_summaries": prompt_summaries,
        "scientific_conclusion": "Individual skips are causally labelled for generation-trajectory safety; no confirmation claim is made.",
        "evidence_boundary": "Only GSM8K train prompts are used. Prior test prompts remain evaluation history.",
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nRapport: {report_path}\nCandidats: {rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
