#!/usr/bin/env python3
"""Real-checkpoint Dream benchmark for SciRust versus fixed gamma.

The model is loaded once. Candidate thresholds are selected on complete
validation prompts, then the frozen candidates are compared on held-out prompts.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import time
import types
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from model.generation_utils_elastic import DreamGenerationMixin
from model.modeling_dream import DreamModel

DEFAULT_WEIGHTS = (
    5.755952374691927,
    0.7882582865936595,
    4.897209095046155,
    -1.2589910896098846,
    3.570827209603688,
    -2.65516332465057,
    0.5164815206184813,
    -3.853401529744073,
)
DEFAULT_THRESHOLD = 1.0058506570900936
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
BOX_RE = re.compile(r"\\boxed\{([^{}]+)\}")


@dataclass(frozen=True)
class Example:
    example_id: str
    question: str
    answer: str


@dataclass
class SampleResult:
    example_id: str
    correct: bool
    prediction: str | None
    gold: str
    elapsed_seconds: float
    decisions: int
    refreshes: int
    refresh_cost: float
    possible_refresh_cost: float
    response: str


@dataclass
class Aggregate:
    mode: str
    parameter: float
    samples: int
    correct: int
    accuracy: float
    elapsed_seconds: float
    mean_seconds: float
    decisions: int
    refreshes: int
    refresh_rate: float
    refresh_cost: float
    possible_refresh_cost: float
    conditional_refresh_fraction: float
    results: list[SampleResult]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--validation-size", type=int, default=8)
    parser.add_argument("--test-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--window-length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--gamma-grid", default="0.80,0.85,0.90,0.95,0.98")
    parser.add_argument("--policy-quantiles", default="0.20,0.40,0.60,0.75,0.88")
    parser.add_argument("--probe-prompts", type=int, default=2)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    return parser.parse_args()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


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


def load_gsm8k(validation_size: int, test_size: int, seed: int) -> tuple[list[Example], list[Example]]:
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    selected = indices[: validation_size + test_size]
    examples = []
    for index in selected:
        row = dataset[index]
        gold = normalize_number(row["answer"].split("####")[-1].strip())
        if gold is None:
            raise RuntimeError(f"cannot parse GSM8K gold answer at index {index}")
        examples.append(Example(str(index), row["question"], gold))
    return examples[:validation_size], examples[validation_size:]


def prompt_text(tokenizer: AutoTokenizer, question: str) -> str:
    instruction = (
        "Solve the problem carefully. End with the final numeric answer inside "
        "\\boxed{...}.\n\nQuestion: " + question
    )
    messages = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        continue_final_message=False,
    )


def attention_modules(model: DreamModel) -> list[Any]:
    return [layer.self_attn for layer in model.model.layers]


def configure_policy(
    model: DreamModel,
    mode: str,
    parameter: float,
    trace_enabled: bool = False,
) -> None:
    layers = attention_modules(model)
    for module in layers:
        module.scirust_policy_mode = mode
        module.scirust_policy_weights = DEFAULT_WEIGHTS
        module.scirust_policy_threshold = parameter if mode == "linear" else DEFAULT_THRESHOLD
        module.scirust_layer_count = len(layers)
        module.scirust_previous_similarity = None
        module.scirust_cache_age_steps = 0
        module.scirust_decisions = 0
        module.scirust_refreshes = 0
        module.scirust_refresh_cost = 0.0
        module.scirust_possible_refresh_cost = 0.0
        module.scirust_risk_sum = 0.0
        module.scirust_trace_enabled = trace_enabled
        module.scirust_trace = []


def collect_runtime_metrics(model: DreamModel) -> tuple[int, int, float, float, list[float]]:
    decisions = 0
    refreshes = 0
    refresh_cost = 0.0
    possible = 0.0
    risks: list[float] = []
    for module in attention_modules(model):
        decisions += int(getattr(module, "scirust_decisions", 0))
        refreshes += int(getattr(module, "scirust_refreshes", 0))
        refresh_cost += float(getattr(module, "scirust_refresh_cost", 0.0))
        possible += float(getattr(module, "scirust_possible_refresh_cost", 0.0))
        risks.extend(float(row["risk"]) for row in getattr(module, "scirust_trace", []))
    return decisions, refreshes, refresh_cost, possible, risks


@torch.inference_mode()
def generate_one(
    model: DreamModel,
    tokenizer: AutoTokenizer,
    example: Example,
    *,
    mode: str,
    parameter: float,
    max_new_tokens: int,
    window_length: int,
    decoding_threshold: float,
    seed: int,
    trace_enabled: bool = False,
) -> tuple[SampleResult, list[float]]:
    set_determinism(seed)
    configure_policy(model, mode, parameter, trace_enabled=trace_enabled)
    prompt = prompt_text(tokenizer, example.question)
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(model.device)
    attention_mask = encoded.attention_mask.to(model.device)
    steps = max(1, math.ceil(max_new_tokens / window_length))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        output_history=False,
        return_dict_in_generate=True,
        steps=steps,
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        alg="confidence_threshold",
        alg_temp=0.0,
        threshold=decoding_threshold,
        gamma=parameter if mode == "gamma" else 0.9,
        window_length=window_length,
        track_num=1,
        block_caching=True,
        tokens_per_iter=1,
        eos_id=tokenizer.eos_token_id,
        bos_id=tokenizer.bos_token_id,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    sequence = output.sequences[0]
    generated = sequence[input_ids.shape[1] :]
    response = tokenizer.decode(generated.tolist(), skip_special_tokens=False)
    if tokenizer.eos_token and tokenizer.eos_token in response:
        response = response.split(tokenizer.eos_token, 1)[0]
    prediction = extract_prediction(response)
    decisions, refreshes, refresh_cost, possible, risks = collect_runtime_metrics(model)
    return (
        SampleResult(
            example_id=example.example_id,
            correct=prediction == example.answer,
            prediction=prediction,
            gold=example.answer,
            elapsed_seconds=elapsed,
            decisions=decisions,
            refreshes=refreshes,
            refresh_cost=refresh_cost,
            possible_refresh_cost=possible,
            response=response,
        ),
        risks,
    )


def evaluate(
    model: DreamModel,
    tokenizer: AutoTokenizer,
    examples: Iterable[Example],
    *,
    mode: str,
    parameter: float,
    args: argparse.Namespace,
    trace_enabled: bool = False,
) -> tuple[Aggregate, list[float]]:
    results: list[SampleResult] = []
    all_risks: list[float] = []
    for offset, example in enumerate(examples):
        sample, risks = generate_one(
            model,
            tokenizer,
            example,
            mode=mode,
            parameter=parameter,
            max_new_tokens=args.max_new_tokens,
            window_length=args.window_length,
            decoding_threshold=args.threshold,
            seed=args.seed + offset,
            trace_enabled=trace_enabled,
        )
        results.append(sample)
        all_risks.extend(risks)
        print(
            json.dumps(
                {
                    "event": "sample",
                    "mode": mode,
                    "parameter": parameter,
                    "example_id": sample.example_id,
                    "correct": sample.correct,
                    "prediction": sample.prediction,
                    "gold": sample.gold,
                    "seconds": sample.elapsed_seconds,
                    "refresh_cost": sample.refresh_cost,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    correct = sum(result.correct for result in results)
    elapsed = sum(result.elapsed_seconds for result in results)
    decisions = sum(result.decisions for result in results)
    refreshes = sum(result.refreshes for result in results)
    refresh_cost = sum(result.refresh_cost for result in results)
    possible = sum(result.possible_refresh_cost for result in results)
    count = len(results)
    aggregate = Aggregate(
        mode=mode,
        parameter=parameter,
        samples=count,
        correct=correct,
        accuracy=correct / count if count else 0.0,
        elapsed_seconds=elapsed,
        mean_seconds=elapsed / count if count else 0.0,
        decisions=decisions,
        refreshes=refreshes,
        refresh_rate=refreshes / decisions if decisions else 0.0,
        refresh_cost=refresh_cost,
        possible_refresh_cost=possible,
        conditional_refresh_fraction=refresh_cost / possible if possible else 0.0,
        results=results,
    )
    print(json.dumps({"event": "aggregate", **aggregate_to_json(aggregate)}, sort_keys=True), flush=True)
    return aggregate, all_risks


def aggregate_to_json(aggregate: Aggregate) -> dict[str, Any]:
    data = asdict(aggregate)
    data["results"] = [asdict(result) for result in aggregate.results]
    return data


def quantile(values: list[float], q: float) -> float:
    if not values:
        return DEFAULT_THRESHOLD
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def choose_candidate(candidates: list[Aggregate], reference_accuracy: float, margin: float) -> Aggregate:
    eligible = [candidate for candidate in candidates if candidate.accuracy + margin >= reference_accuracy]
    if not eligible:
        return max(candidates, key=lambda candidate: (candidate.accuracy, -candidate.refresh_cost))
    return min(eligible, key=lambda candidate: (candidate.refresh_cost, candidate.mean_seconds, -candidate.accuracy))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required for the real Dream benchmark")
    if args.validation_size < 2 or args.test_size < 2:
        raise ValueError("validation-size and test-size must both be >= 2")
    set_determinism(args.seed)
    validation, test = load_gsm8k(args.validation_size, args.test_size, args.seed)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print(json.dumps({"event": "device", "name": torch.cuda.get_device_name(), "dtype": args.dtype}))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = DreamModel.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval().to("cuda")
    model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
    model._sample = types.MethodType(DreamGenerationMixin._sample, model)

    probe_count = min(args.probe_prompts, len(validation))
    _, probe_risks = evaluate(
        model,
        tokenizer,
        validation[:probe_count],
        mode="gamma",
        parameter=0.9,
        args=args,
        trace_enabled=True,
    )
    requested_quantiles = [float(value) for value in args.policy_quantiles.split(",")]
    policy_thresholds = sorted({quantile(probe_risks, q) for q in requested_quantiles})
    gamma_values = [float(value) for value in args.gamma_grid.split(",")]

    gamma_validation = [
        evaluate(model, tokenizer, validation, mode="gamma", parameter=value, args=args)[0]
        for value in gamma_values
    ]
    policy_validation = [
        evaluate(model, tokenizer, validation, mode="linear", parameter=value, args=args)[0]
        for value in policy_thresholds
    ]
    best_observed_accuracy = max(
        candidate.accuracy for candidate in gamma_validation + policy_validation
    )
    margin = 1.0 / len(validation)
    selected_gamma = choose_candidate(gamma_validation, best_observed_accuracy, margin)
    selected_policy = choose_candidate(policy_validation, best_observed_accuracy, margin)

    gamma_test = evaluate(
        model,
        tokenizer,
        test,
        mode="gamma",
        parameter=selected_gamma.parameter,
        args=args,
    )[0]
    policy_test = evaluate(
        model,
        tokenizer,
        test,
        mode="linear",
        parameter=selected_policy.parameter,
        args=args,
    )[0]

    compute_gain = (
        (gamma_test.refresh_cost - policy_test.refresh_cost) / gamma_test.refresh_cost
        if gamma_test.refresh_cost > 0.0
        else 0.0
    )
    latency_gain = (
        (gamma_test.elapsed_seconds - policy_test.elapsed_seconds) / gamma_test.elapsed_seconds
        if gamma_test.elapsed_seconds > 0.0
        else 0.0
    )
    quality_delta = policy_test.accuracy - gamma_test.accuracy
    result = {
        "schema_version": 1,
        "scope": "real Dream-v0-Instruct-7B checkpoint; preliminary held-out GSM8K benchmark",
        "model": args.model,
        "device": torch.cuda.get_device_name(),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "window_length": args.window_length,
        "validation_size": len(validation),
        "test_size": len(test),
        "weights": list(DEFAULT_WEIGHTS),
        "probe_risk": {
            "count": len(probe_risks),
            "min": min(probe_risks) if probe_risks else None,
            "median": statistics.median(probe_risks) if probe_risks else None,
            "max": max(probe_risks) if probe_risks else None,
        },
        "gamma_validation": [aggregate_to_json(value) for value in gamma_validation],
        "policy_validation": [aggregate_to_json(value) for value in policy_validation],
        "selected_gamma": selected_gamma.parameter,
        "selected_policy_threshold": selected_policy.parameter,
        "gamma_test": aggregate_to_json(gamma_test),
        "policy_test": aggregate_to_json(policy_test),
        "quality_delta": quality_delta,
        "relative_refresh_compute_gain": compute_gain,
        "relative_latency_gain": latency_gain,
        "reproduces_synthetic_63_11_percent": compute_gain >= 0.60 and quality_delta >= 0.0,
        "evidence_boundary": (
            "This is a real-checkpoint run. The initial workflow is a small-sample smoke benchmark; "
            "a publication-grade claim requires the larger preregistered run and confidence intervals."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "final", **{k: result[k] for k in (
        "selected_gamma",
        "selected_policy_threshold",
        "quality_delta",
        "relative_refresh_compute_gain",
        "relative_latency_gain",
        "reproduces_synthetic_63_11_percent",
    )}}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
