#!/usr/bin/env python3
"""Paired end-to-end GSM8K benchmark for the frozen Dream cache ensemble.

The five policies and their thresholds are read from the exploratory report and
are never fitted or recalibrated here. Each GSM8K example is generated once with
always-refresh and once with the frozen majority ensemble. Execution order is
alternated to reduce systematic warmup and thermal bias.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import time
import types
from typing import Any

import torch
from transformers import AutoTokenizer


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
BOX_RE = re.compile(r"\\boxed\{([^{}]+)\}")


@dataclass(frozen=True)
class Example:
    dataset_index: int
    question: str
    answer: str


@dataclass
class RunResult:
    mode: str
    correct: bool
    prediction: str | None
    gold: str
    elapsed_seconds: float
    decisions: int
    refreshes: int
    refresh_rate: float
    refresh_cost: float
    possible_refresh_cost: float
    conditional_refresh_fraction: float
    response: str


@dataclass
class PairResult:
    dataset_index: int
    execution_order: list[str]
    always_refresh: RunResult
    frozen_ensemble: RunResult
    latency_improvement: float
    refresh_compute_improvement: float
    accuracy_delta: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elastic-cache", type=Path, required=True)
    parser.add_argument("--policy-report", type=Path, required=True)
    parser.add_argument("--gsm8k-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--warmup-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--window-length", type=int, default=32)
    parser.add_argument("--decoding-threshold", type=float, default=0.9)
    parser.add_argument("--vote-threshold", type=int, default=3)
    parser.add_argument("--quality-noninferiority-margin", type=float, default=0.05)
    parser.add_argument("--minimum-latency-improvement", type=float, default=0.005)
    parser.add_argument("--minimum-refresh-compute-improvement", type=float, default=0.005)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0.0, min(1.0, probability)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(
    values: list[float],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    rng = random.Random(seed)
    count = len(values)
    means = [
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    ]
    return percentile(means, 0.025), percentile(means, 0.975)


def normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().replace(",", "").replace("$", "")
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return cleaned.lower()
    if number == number.to_integral():
        return str(number.quantize(Decimal(1)))
    return format(number.normalize(), "f")


def extract_prediction(text: str) -> str | None:
    boxed = BOX_RE.findall(text)
    if boxed:
        boxed_numbers = NUMBER_RE.findall(boxed[-1])
        if boxed_numbers:
            return normalize_number(boxed_numbers[-1])
    numbers = NUMBER_RE.findall(text)
    return normalize_number(numbers[-1]) if numbers else None


def load_examples(
    path: Path,
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
            examples.append(
                Example(
                    dataset_index=index,
                    question=str(row["question"]),
                    answer=answer,
                )
            )
    required = warmup_samples + samples
    if len(examples) < required:
        raise RuntimeError(
            f"GSM8K file has {len(examples)} examples, need at least {required}"
        )
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    selected = [examples[index] for index in indices[:required]]
    return selected[:warmup_samples], selected[warmup_samples:]


def load_frozen_policies(path: Path) -> list[tuple[tuple[float, ...], float]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("strict_exploratory_success") is not True:
        raise RuntimeError("frozen policy report did not pass exploratory validation")
    folds = report.get("fold_results")
    if not isinstance(folds, list) or len(folds) != 5:
        raise RuntimeError("frozen policy report must contain exactly five folds")
    policies: list[tuple[tuple[float, ...], float]] = []
    for fold in folds:
        weights = tuple(float(value) for value in fold["weights"])
        threshold = float(fold["threshold"])
        if len(weights) != 8 or not all(math.isfinite(value) for value in weights):
            raise RuntimeError("invalid frozen policy weights")
        if not math.isfinite(threshold):
            raise RuntimeError("invalid frozen policy threshold")
        policies.append((weights, threshold))
    return policies


def set_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False


def prompt_text(tokenizer: Any, question: str) -> str:
    instruction = (
        "Solve the problem carefully. End with the final numeric answer inside "
        "\\boxed{...}.\n\nQuestion: " + question
    )
    messages = [{"role": "user", "content": instruction}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return (tokenizer.bos_token or "") + instruction


def attention_modules(model: Any) -> list[Any]:
    return [layer.self_attn for layer in model.model.layers]


def configure_policy(
    model: Any,
    mode: str,
    policies: list[tuple[tuple[float, ...], float]],
    vote_threshold: int,
) -> None:
    layers = attention_modules(model)
    for module in layers:
        module.scirust_policy_mode = mode
        module.scirust_policy_ensemble = policies
        module.scirust_policy_vote_threshold = vote_threshold
        module.scirust_layer_count = len(layers)
        module.scirust_previous_similarity = None
        module.scirust_cache_age_steps = 0
        module.scirust_decisions = 0
        module.scirust_refreshes = 0
        module.scirust_refresh_cost = 0.0
        module.scirust_possible_refresh_cost = 0.0
        module.scirust_risk_sum = 0.0
        module.scirust_trace_enabled = False
        module.scirust_trace = []


def collect_runtime_metrics(model: Any) -> tuple[int, int, float, float]:
    decisions = 0
    refreshes = 0
    refresh_cost = 0.0
    possible = 0.0
    for module in attention_modules(model):
        decisions += int(getattr(module, "scirust_decisions", 0))
        refreshes += int(getattr(module, "scirust_refreshes", 0))
        refresh_cost += float(getattr(module, "scirust_refresh_cost", 0.0))
        possible += float(getattr(module, "scirust_possible_refresh_cost", 0.0))
    return decisions, refreshes, refresh_cost, possible


@torch.inference_mode()
def generate_one(
    model: Any,
    tokenizer: Any,
    example: Example,
    mode: str,
    policies: list[tuple[tuple[float, ...], float]],
    args: argparse.Namespace,
    generation_seed: int,
) -> RunResult:
    set_determinism(generation_seed)
    configure_policy(model, mode, policies, args.vote_threshold)
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


def pair_result(
    example: Example,
    order: list[str],
    model: Any,
    tokenizer: Any,
    policies: list[tuple[tuple[float, ...], float]],
    args: argparse.Namespace,
    generation_seed: int,
) -> PairResult:
    runs: dict[str, RunResult] = {}
    for mode in order:
        runs[mode] = generate_one(
            model,
            tokenizer,
            example,
            mode,
            policies,
            args,
            generation_seed,
        )
    always = runs["always"]
    ensemble = runs["ensemble"]
    latency_improvement = (
        (always.elapsed_seconds - ensemble.elapsed_seconds) / always.elapsed_seconds
        if always.elapsed_seconds > 0.0
        else 0.0
    )
    refresh_compute_improvement = (
        (always.refresh_cost - ensemble.refresh_cost) / always.refresh_cost
        if always.refresh_cost > 0.0
        else 0.0
    )
    return PairResult(
        dataset_index=example.dataset_index,
        execution_order=order,
        always_refresh=always,
        frozen_ensemble=ensemble,
        latency_improvement=latency_improvement,
        refresh_compute_improvement=refresh_compute_improvement,
        accuracy_delta=int(ensemble.correct) - int(always.correct),
    )


def aggregate_mode(results: list[PairResult], field: str) -> dict[str, float | int]:
    runs = [getattr(pair, field) for pair in results]
    samples = len(runs)
    correct = sum(int(run.correct) for run in runs)
    elapsed = sum(run.elapsed_seconds for run in runs)
    decisions = sum(run.decisions for run in runs)
    refreshes = sum(run.refreshes for run in runs)
    refresh_cost = sum(run.refresh_cost for run in runs)
    possible = sum(run.possible_refresh_cost for run in runs)
    return {
        "samples": samples,
        "correct": correct,
        "accuracy": correct / samples if samples else 0.0,
        "elapsed_seconds": elapsed,
        "mean_seconds": elapsed / samples if samples else 0.0,
        "decisions": decisions,
        "refreshes": refreshes,
        "refresh_rate": refreshes / decisions if decisions else 0.0,
        "refresh_cost": refresh_cost,
        "possible_refresh_cost": possible,
        "conditional_refresh_fraction": refresh_cost / possible if possible else 0.0,
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.samples < 20:
        raise SystemExit("at least 20 GSM8K samples are required")
    if args.warmup_samples < 1:
        raise SystemExit("at least one warmup sample is required")

    policies = load_frozen_policies(args.policy_report)
    if not 1 <= args.vote_threshold <= len(policies):
        raise SystemExit("invalid vote threshold")
    warmup, examples = load_examples(
        args.gsm8k_test,
        args.seed,
        args.warmup_samples,
        args.samples,
    )

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
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "dtype": args.dtype,
                "policies": len(policies),
                "vote_threshold": args.vote_threshold,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for offset, example in enumerate(warmup):
        order = ["always", "ensemble"] if offset % 2 == 0 else ["ensemble", "always"]
        pair_result(
            example,
            order,
            model,
            tokenizer,
            policies,
            args,
            args.seed + 100000 + example.dataset_index,
        )
        print(
            json.dumps(
                {"event": "warmup", "index": offset + 1, "dataset_index": example.dataset_index},
                sort_keys=True,
            ),
            flush=True,
        )

    results: list[PairResult] = []
    partial_path = args.output.with_suffix(args.output.suffix + ".partial")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for offset, example in enumerate(examples):
        order = ["always", "ensemble"] if offset % 2 == 0 else ["ensemble", "always"]
        pair = pair_result(
            example,
            order,
            model,
            tokenizer,
            policies,
            args,
            args.seed + example.dataset_index,
        )
        results.append(pair)
        partial_path.write_text(
            json.dumps([asdict(item) for item in results], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "event": "pair",
                    "completed": offset + 1,
                    "total": len(examples),
                    "dataset_index": example.dataset_index,
                    "order": order,
                    "always_correct": pair.always_refresh.correct,
                    "ensemble_correct": pair.frozen_ensemble.correct,
                    "always_seconds": pair.always_refresh.elapsed_seconds,
                    "ensemble_seconds": pair.frozen_ensemble.elapsed_seconds,
                    "latency_improvement": pair.latency_improvement,
                    "refresh_compute_improvement": pair.refresh_compute_improvement,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    always = aggregate_mode(results, "always_refresh")
    ensemble = aggregate_mode(results, "frozen_ensemble")
    accuracy_deltas = [float(pair.accuracy_delta) for pair in results]
    latency_improvements = [pair.latency_improvement for pair in results]
    compute_improvements = [pair.refresh_compute_improvement for pair in results]
    accuracy_ci = bootstrap_mean_ci(
        accuracy_deltas,
        args.bootstrap_samples,
        args.seed + 1,
    )
    latency_ci = bootstrap_mean_ci(
        latency_improvements,
        args.bootstrap_samples,
        args.seed + 2,
    )
    compute_ci = bootstrap_mean_ci(
        compute_improvements,
        args.bootstrap_samples,
        args.seed + 3,
    )
    accuracy_delta = float(ensemble["accuracy"]) - float(always["accuracy"])
    mean_latency_improvement = sum(latency_improvements) / len(latency_improvements)
    mean_compute_improvement = sum(compute_improvements) / len(compute_improvements)

    paired_outcomes = {
        "both_correct": sum(
            pair.always_refresh.correct and pair.frozen_ensemble.correct for pair in results
        ),
        "always_only_correct": sum(
            pair.always_refresh.correct and not pair.frozen_ensemble.correct for pair in results
        ),
        "ensemble_only_correct": sum(
            pair.frozen_ensemble.correct and not pair.always_refresh.correct for pair in results
        ),
        "both_incorrect": sum(
            not pair.always_refresh.correct and not pair.frozen_ensemble.correct for pair in results
        ),
    }

    quality_pass = (
        accuracy_delta >= -args.quality_noninferiority_margin
        and accuracy_ci[0] >= -args.quality_noninferiority_margin
    )
    latency_pass = (
        mean_latency_improvement >= args.minimum_latency_improvement
        and latency_ci[0] > 0.0
    )
    refresh_compute_pass = (
        mean_compute_improvement >= args.minimum_refresh_compute_improvement
        and compute_ci[0] > 0.0
    )
    end_to_end_success = quality_pass and latency_pass and refresh_compute_pass

    report = {
        "schema_version": 1,
        "status": "frozen_policy_paired_end_to_end_gsm8k",
        "model": args.model,
        "hardware": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "policy": {
            "report": str(args.policy_report),
            "report_sha256": sha256_file(args.policy_report),
            "policies": len(policies),
            "vote_threshold": args.vote_threshold,
            "fitted_or_calibrated_on_gsm8k": False,
        },
        "dataset": {
            "path": str(args.gsm8k_test),
            "sha256": sha256_file(args.gsm8k_test),
            "selection_seed": args.seed,
            "warmup_samples": args.warmup_samples,
            "evaluated_samples": args.samples,
            "indices": [example.dataset_index for example in examples],
        },
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "window_length": args.window_length,
            "decoding_threshold": args.decoding_threshold,
            "dtype": args.dtype,
            "alternating_execution_order": True,
        },
        "pre_registered_criteria": {
            "quality_noninferiority_margin": args.quality_noninferiority_margin,
            "minimum_mean_latency_improvement": args.minimum_latency_improvement,
            "minimum_mean_refresh_compute_improvement": args.minimum_refresh_compute_improvement,
            "bootstrap_confidence": 0.95,
            "bootstrap_samples": args.bootstrap_samples,
            "require_latency_ci_lower_bound_above_zero": True,
            "require_refresh_compute_ci_lower_bound_above_zero": True,
        },
        "always_refresh": always,
        "frozen_ensemble": ensemble,
        "paired_outcomes": paired_outcomes,
        "quality": {
            "accuracy_delta": accuracy_delta,
            "bootstrap_95_percent_ci": list(accuracy_ci),
            "pass": quality_pass,
        },
        "latency": {
            "mean_relative_improvement": mean_latency_improvement,
            "bootstrap_95_percent_ci": list(latency_ci),
            "pass": latency_pass,
        },
        "refresh_compute": {
            "mean_relative_improvement": mean_compute_improvement,
            "bootstrap_95_percent_ci": list(compute_ci),
            "pass": refresh_compute_pass,
        },
        "end_to_end_success": end_to_end_success,
        "scientific_conclusion": (
            "The frozen ensemble satisfies the paired GSM8K quality, refresh-compute, and wall-clock criteria."
            if end_to_end_success
            else "The frozen ensemble does not satisfy every paired GSM8K end-to-end criterion."
        ),
        "evidence_boundary": (
            "Paired GSM8K exact-match and Jetson wall-clock evidence for one model, decoding configuration, "
            "and hardware platform. Broader workloads and repeated system-level runs remain necessary."
        ),
        "pairs": [asdict(pair) for pair in results],
    }
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    partial_path.unlink(missing_ok=True)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nRapport: {args.output}")
    return 0 if end_to_end_success else 2


if __name__ == "__main__":
    raise SystemExit(main())
