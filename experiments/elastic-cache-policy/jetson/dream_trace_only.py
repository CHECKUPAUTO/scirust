#!/usr/bin/env python3
"""Collect an independent Dream counterfactual trace without fitting a policy."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import types
from typing import Any

import torch
from transformers import AutoTokenizer

import dream_jetson_proof as proof


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elastic-cache", type=Path, required=True)
    parser.add_argument("--model", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectories", type=int, default=60)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--window-length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; the trace must run on the Jetson GPU")
    if args.trajectories < 20:
        raise SystemExit("at least 20 independent trajectories are required")

    dream_dir = args.elastic_cache / "dream"
    if not (dream_dir / "model" / "modeling_dream.py").is_file():
        raise SystemExit(f"invalid Elastic-Cache checkout: {dream_dir}")
    sys.path.insert(0, str(dream_dir))

    from model.generation_utils_elastic import DreamGenerationMixin
    from model.modeling_dream import DreamModel
    import model.modeling_dream as modeling

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "dream_counterfactual_trace.csv"
    manifest_path = args.output_dir / "dream_trace_manifest.json"

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

    behavior_gammas = [0.80, 0.85, 0.90, 0.94, 0.97]
    trajectory_summaries: list[dict[str, Any]] = []
    prompts = proof.build_prompts(args.trajectories, args.seed)

    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=proof.HEADER)
        writer.writeheader()
        controller = proof.TraceController(writer=writer, file_handle=handle)
        proof.install_attention_probe(modeling, controller)

        for trajectory_id, prompt in enumerate(prompts):
            gamma = behavior_gammas[trajectory_id % len(behavior_gammas)]
            controller.begin_trajectory(trajectory_id, gamma)
            rendered = proof.format_prompt(tokenizer, prompt)
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
                f"confirmatory_trajectory={trajectory_id + 1}/{len(prompts)} "
                f"gamma={gamma:.2f} rows={controller.rows} seconds={elapsed:.3f}",
                flush=True,
            )
            del input_ids, attention_mask, result, generated
            gc.collect()
            torch.cuda.empty_cache()

    if controller.rows < 100:
        raise RuntimeError(f"insufficient trace rows: {controller.rows}")

    manifest = {
        "schema_version": 1,
        "status": "independent_confirmatory_trace",
        "model": args.model,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "seed": args.seed,
        "trajectories": args.trajectories,
        "trace_rows": controller.rows,
        "max_new_tokens": args.max_new_tokens,
        "window_length": args.window_length,
        "behavior_gammas": behavior_gammas,
        "prompt_suite": "dream-counterfactual-v1-independent-seed",
        "trace_sha256": sha256_file(trace_path),
        "trace_path": str(trace_path),
        "policy_fitting_performed": False,
        "trajectory_summaries": trajectory_summaries,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nTrace: {trace_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
