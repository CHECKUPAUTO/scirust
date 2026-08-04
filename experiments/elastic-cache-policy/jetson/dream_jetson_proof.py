#!/usr/bin/env python3
"""Collect real Dream counterfactual cache traces and invoke SciRust discovery.

This script is intentionally offline-training-only: every reuse decision also computes
an exact full-refresh attention output for the same layer input, then records the
resulting local stale-cache loss. Deployed inference never performs this dual path.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer


HEADER = [
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
]


@dataclass
class TraceController:
    writer: csv.DictWriter
    file_handle: Any
    max_cache_age: int = 16
    trajectory_id: int = 0
    decode_step: int = -1
    behavior_gamma: float = 0.9
    rows: int = 0
    previous_similarity: dict[int, float] = field(default_factory=dict)
    cache_age_by_layer: dict[int, int] = field(default_factory=dict)
    step_triggered: bool = False
    normalized_refresh_compute: float = 0.0

    def begin_trajectory(self, trajectory_id: int, behavior_gamma: float) -> None:
        self.trajectory_id = trajectory_id
        self.behavior_gamma = behavior_gamma
        self.decode_step = -1
        self.previous_similarity.clear()
        self.cache_age_by_layer.clear()
        self.step_triggered = False
        self.normalized_refresh_compute = 0.0

    def begin_model_step(self) -> None:
        self.decode_step += 1
        self.step_triggered = False

    def cache_refreshed(self, layer_id: int) -> None:
        self.cache_age_by_layer[layer_id] = 0

    def cache_reused(self, layer_id: int) -> float:
        age = self.cache_age_by_layer.get(layer_id, 0) + 1
        self.cache_age_by_layer[layer_id] = age
        return min(age / max(self.max_cache_age, 1), 1.0)

    def record(
        self,
        *,
        layer_id: int,
        similarity: float,
        head_variance: float,
        cache_age: float,
        attention_mass: float,
        layer_fraction: float,
        refresh_cost: float,
        stale_loss: float,
    ) -> bool:
        previous = self.previous_similarity.get(layer_id, similarity)
        similarity_delta = max(-1.0, min(1.0, similarity - previous))
        self.previous_similarity[layer_id] = similarity
        row = {
            "trajectory_id": self.trajectory_id,
            "step": self.decode_step,
            "layer_id": layer_id,
            "similarity": clamp01(similarity),
            "similarity_delta": similarity_delta,
            "head_variance": clamp01(head_variance),
            "cache_age": clamp01(cache_age),
            "attention_mass": clamp01(attention_mass),
            "layer_fraction": clamp01(layer_fraction),
            "refresh_cost": clamp01(refresh_cost),
            "stale_loss": clamp01(stale_loss),
        }
        self.writer.writerow(row)
        self.rows += 1
        if self.rows % 128 == 0:
            self.file_handle.flush()

        trigger = similarity < self.behavior_gamma
        if trigger and not self.step_triggered:
            self.normalized_refresh_compute += refresh_cost
            self.step_triggered = True
        return trigger


def clamp01(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite trace value: {value}")
    return max(0.0, min(1.0, value))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_exact(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    return tensor * cos.unsqueeze(1) + rotate_half(tensor) * sin.unsqueeze(1)


def output_stale_loss(reuse: torch.Tensor, refresh: torch.Tensor) -> float:
    reuse_f = reuse.float().reshape(1, -1)
    refresh_f = refresh.float().reshape(1, -1)
    cosine = F.cosine_similarity(reuse_f, refresh_f, dim=1).clamp(-1.0, 1.0)
    cosine_loss = 0.5 * (1.0 - cosine)
    mse = (reuse_f - refresh_f).square().mean()
    denom = refresh_f.square().mean().clamp_min(1e-12)
    relative_mse = mse / denom
    bounded_mse = relative_mse / (1.0 + relative_mse)
    return float((0.5 * cosine_loss + 0.5 * bounded_mse).item())


def install_attention_probe(modeling: Any, controller: TraceController) -> None:
    repeat_kv = modeling.repeat_kv
    original_class = modeling.DreamSdpaAttention

    def probed_forward(
        self,
        hidden_states: torch.Tensor,
        input_layernorm,
        attention_mask=None,
        positions=None,
        lengths=None,
        output_attentions=False,
        use_cache=False,
        rotary_pos=None,
        block_idx=None,
    ):
        del output_attentions, use_cache, attention_mask
        query_position, _full_position, track_position, query_masked_position, masked_position = positions
        _key_len, start_reset, _gamma, track_num = lengths
        num_layers = int(self.config.num_hidden_layers)

        if block_idx == 0:
            controller.begin_model_step()

        if block_idx > start_reset:
            self.x_cache = hidden_states
        else:
            self.x_cache[:, query_position, :] = hidden_states
            if block_idx == start_reset:
                hidden_states = self.x_cache

        normed_hidden_states = input_layernorm(hidden_states)
        batch_size, query_len, _ = normed_hidden_states.size()

        query_states = self.q_proj(normed_hidden_states)
        key_states = self.k_proj(normed_hidden_states)
        value_states = self.v_proj(normed_hidden_states)

        query_states = query_states.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if query_states.shape[-2] == rotary_pos[0].shape[1]:
            query_states = apply_rope_exact(query_states, rotary_pos[0], rotary_pos[1])
            key_states = apply_rope_exact(key_states, rotary_pos[0], rotary_pos[1])
        else:
            query_states = apply_rope_exact(query_states, rotary_pos[2], rotary_pos[3])
            key_states = apply_rope_exact(key_states, rotary_pos[2], rotary_pos[3])

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        reuse_branch = block_idx < start_reset
        if not reuse_branch:
            self.q_cache = query_states
            self.k_cache = key_states
            self.v_cache = value_states
            controller.cache_refreshed(block_idx)
            past_k = None
            past_q = None
        else:
            cache_age = controller.cache_reused(block_idx)
            past_k = self.k_cache.clone()
            self.k_cache[:, :, query_position, :] = key_states
            self.v_cache[:, :, query_position, :] = value_states
            past_q = self.q_cache[:, :, track_position, :].clone()
            self.q_cache[:, :, query_position, :] = query_states
            key_states = self.k_cache
            value_states = self.v_cache

        scores = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        attention = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attention = nn.functional.dropout(attention, p=self.attention_dropout, training=self.training)
        attention_output = torch.matmul(attention, value_states)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, query_len, self.hidden_size)
        attention_output = self.o_proj(attention_output)

        if not reuse_branch:
            masked_attention = attention[:, :, query_masked_position, :]
        else:
            masked_attention = attention[:, :, -query_masked_position.shape[0] :, :]
            tracked_attention = attention[:, :, : track_position.shape[0], :]
            scale_factor = 1.0 / math.sqrt(past_q.size(-1))
            past_attention = torch.softmax(past_q @ past_k.transpose(-2, -1) * scale_factor, dim=-1)

            similarity = float(F.cosine_similarity(past_attention, tracked_attention, dim=1).mean().item())

            if past_attention.numel() and tracked_attention.numel():
                past_flat = past_attention.float().flatten(2)
                tracked_flat = tracked_attention.float().flatten(2)
                per_head = F.cosine_similarity(past_flat, tracked_flat, dim=2).clamp(-1.0, 1.0)
                head_variance = float(per_head.var(unbiased=False).item())
            else:
                head_variance = 0.0

            if track_position.numel() > 0:
                tracked_mass = masked_attention[..., track_position].sum()
                total_mass = masked_attention.sum().clamp_min(torch.finfo(masked_attention.dtype).tiny)
                attention_mass = float((tracked_mass / total_mass).float().item())
            else:
                attention_mass = 0.0

            full_norm = input_layernorm(self.x_cache)
            query_norm = full_norm[:, query_position, :]
            full_query = self.q_proj(query_norm)
            full_key = self.k_proj(full_norm)
            full_value = self.v_proj(full_norm)
            full_query = full_query.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
            full_key = full_key.view(batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            full_value = full_value.view(batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            full_query = apply_rope_exact(full_query, rotary_pos[2], rotary_pos[3])
            full_key = apply_rope_exact(full_key, rotary_pos[0], rotary_pos[1])
            full_key = repeat_kv(full_key, self.num_key_value_groups)
            full_value = repeat_kv(full_value, self.num_key_value_groups)
            full_scores = torch.matmul(full_query, full_key.transpose(2, 3)) / math.sqrt(self.head_dim)
            full_attention = nn.functional.softmax(full_scores, dim=-1, dtype=torch.float32).to(full_query.dtype)
            full_output = torch.matmul(full_attention, full_value)
            full_output = full_output.transpose(1, 2).contiguous().view(batch_size, query_len, self.hidden_size)
            full_output = self.o_proj(full_output)
            stale_loss = output_stale_loss(attention_output, full_output)

            layer_fraction = block_idx / max(num_layers - 1, 1)
            refresh_cost = (num_layers - (block_idx + 1)) / max(num_layers, 1)
            if controller.record(
                layer_id=block_idx,
                similarity=similarity,
                head_variance=head_variance,
                cache_age=cache_age,
                attention_mass=attention_mass,
                layer_fraction=layer_fraction,
                refresh_cost=refresh_cost,
                stale_loss=stale_loss,
            ):
                lengths[1] = block_idx + 1

        masked_summary = masked_attention.sum(dim=(0, 1, 2))
        masked_summary[masked_position] = 0.0
        effective_track_num = min(int(track_num), int(masked_summary.numel()))
        self.track_token = masked_summary.topk(k=max(effective_track_num, 1), dim=0, largest=True)[1]
        return hidden_states, attention_output

    original_class.forward = probed_forward


def build_prompts(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    prompts: list[str] = []
    templates = [
        "Calculate {a} * {b} + {c}. Explain briefly, then give the final integer.",
        "A warehouse has {a} boxes with {b} items each and receives {c} more items. How many items are there?",
        "Find the next number in this sequence and explain the rule: {a}, {a2}, {a3}, ...",
        "Write a concise Rust function that returns the greatest common divisor of {a} and {b}.",
        "Compare deterministic caching and random eviction in exactly three short sentences.",
        "Solve for x: {m}x + {c} = {rhs}. Show the main step.",
    ]
    for index in range(count):
        a = rng.randint(7, 97)
        b = rng.randint(3, 41)
        c = rng.randint(1, 53)
        m = rng.randint(2, 13)
        x = rng.randint(2, 19)
        template = templates[index % len(templates)]
        prompts.append(
            template.format(
                a=a,
                b=b,
                c=c,
                a2=a + b,
                a3=a + 2 * b,
                m=m,
                rhs=m * x + c,
            )
        )
    return prompts


def format_prompt(tokenizer: Any, prompt: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prefix = tokenizer.bos_token or ""
        return prefix + prompt


def parse_discovery_output(text: str) -> dict[str, Any]:
    patterns = {
        "learned_quality_loss": r"test learned quality_loss=([0-9.eE+-]+)",
        "learned_compute": r"test learned quality_loss=[0-9.eE+-]+ compute=([0-9.eE+-]+)",
        "gamma": r"test best_gamma gamma=([0-9.eE+-]+)",
        "gamma_quality_loss": r"test best_gamma gamma=[0-9.eE+-]+ quality_loss=([0-9.eE+-]+)",
        "gamma_compute": r"test best_gamma gamma=[0-9.eE+-]+ quality_loss=[0-9.eE+-]+ compute=([0-9.eE+-]+)",
        "relative_compute_improvement": r"relative_compute_improvement=([0-9.eE+-]+)",
        "pareto_dominates": r"pareto_dominates=(true|false)",
    }
    result: dict[str, Any] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            raise RuntimeError(f"missing {key} in SciRust output")
        value = match.group(1)
        result[key] = value == "true" if value in {"true", "false"} else float(value)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elastic-cache", type=Path, required=True)
    parser.add_argument("--scirust", type=Path, required=True)
    parser.add_argument("--model", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectories", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--window-length", type=int, default=16)
    parser.add_argument("--quality-budget", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--steps", type=int, default=1200)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; this proof must run on the Jetson GPU")
    if args.trajectories < 10:
        raise SystemExit("at least 10 trajectories are required for train/validation/test splits")

    dream_dir = args.elastic_cache / "dream"
    if not (dream_dir / "model" / "modeling_dream.py").is_file():
        raise SystemExit(f"invalid Elastic-Cache checkout: {dream_dir}")
    sys.path.insert(0, str(dream_dir))

    from model.generation_utils_elastic import DreamGenerationMixin
    from model.modeling_dream import DreamModel
    import model.modeling_dream as modeling
    import types

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "dream_counterfactual_trace.csv"
    report_path = args.output_dir / "dream_real_policy_report.json"
    raw_output_path = args.output_dir / "scirust_discovery_output.txt"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = DreamModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval().to("cuda")
    model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
    model._sample = types.MethodType(DreamGenerationMixin._sample, model)

    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        controller = TraceController(writer=writer, file_handle=handle)
        install_attention_probe(modeling, controller)
        prompts = build_prompts(args.trajectories, args.seed)
        behavior_gammas = [0.80, 0.85, 0.90, 0.94, 0.97]
        trajectory_summaries = []

        for trajectory_id, prompt in enumerate(prompts):
            gamma = behavior_gammas[trajectory_id % len(behavior_gammas)]
            controller.begin_trajectory(trajectory_id, gamma)
            rendered = format_prompt(tokenizer, prompt)
            encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=True)
            input_ids = encoded.input_ids.to("cuda")
            attention_mask = encoded.attention_mask.to("cuda")
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                result = model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    output_history=False,
                    return_dict_in_generate=True,
                    steps=max(1, math.ceil(args.max_new_tokens / args.window_length)),
                    temperature=0.0,
                    top_p=None,
                    top_k=None,
                    alg="confidence_threshold",
                    threshold=0.9,
                    gamma=gamma,
                    window_length=args.window_length,
                    track_num=1,
                    block_caching=True,
                    tokens_per_iter=1,
                    eos_id=tokenizer.eos_token_id,
                    bos_id=tokenizer.bos_token_id,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            generated = result.sequences[0, input_ids.shape[1] :]
            response = tokenizer.decode(generated.tolist(), skip_special_tokens=True)
            trajectory_summaries.append(
                {
                    "trajectory_id": trajectory_id,
                    "behavior_gamma": gamma,
                    "seconds": elapsed,
                    "rows_total": controller.rows,
                    "response_prefix": response[:160],
                }
            )
            print(
                f"trajectory={trajectory_id + 1}/{len(prompts)} gamma={gamma:.2f} "
                f"rows={controller.rows} seconds={elapsed:.3f}",
                flush=True,
            )
            del input_ids, attention_mask, result, generated
            gc.collect()
            torch.cuda.empty_cache()

    if controller.rows < 100:
        raise RuntimeError(f"insufficient trace rows: {controller.rows}")

    cargo_cmd = [
        "cargo",
        "+nightly-2026-07-02",
        "run",
        "--release",
        "--manifest-path",
        str(args.scirust / "experiments/elastic-cache-policy/Cargo.toml"),
        "--",
        "--trace",
        str(trace_path),
        "--seed",
        str(args.seed),
        "--steps",
        str(args.steps),
        "--max-quality-loss",
        str(args.quality_budget),
    ]
    completed = subprocess.run(cargo_cmd, text=True, capture_output=True, check=False)
    combined = completed.stdout + "\n" + completed.stderr
    raw_output_path.write_text(combined, encoding="utf-8")
    print(combined)
    if completed.returncode != 0:
        raise RuntimeError(f"SciRust discovery failed with exit code {completed.returncode}")

    metrics = parse_discovery_output(combined)
    target = 0.63110794
    tolerance = 0.08
    reproduced = (
        metrics["pareto_dominates"]
        and metrics["relative_compute_improvement"] >= target - tolerance
        and metrics["relative_compute_improvement"] <= target + tolerance
    )
    report = {
        "schema_version": 1,
        "scope": "Dream-v0-Instruct-7B real Jetson counterfactual attention-output trace",
        "model": args.model,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "seed": args.seed,
        "trajectories": args.trajectories,
        "trace_rows": controller.rows,
        "max_new_tokens": args.max_new_tokens,
        "window_length": args.window_length,
        "quality_budget": args.quality_budget,
        "synthetic_reference_gain": target,
        "reproduction_tolerance": tolerance,
        "metrics": metrics,
        "reproduced_63_11_percent_band": reproduced,
        "evidence_boundary": (
            "Real Dream hidden-state/attention counterfactual evidence. "
            "This is not yet a GSM8K/HumanEval end-to-end accuracy claim."
        ),
        "trajectory_summaries": trajectory_summaries,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if reproduced else 2


if __name__ == "__main__":
    raise SystemExit(main())
