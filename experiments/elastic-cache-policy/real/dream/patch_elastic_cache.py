#!/usr/bin/env python3
"""Fail-closed patcher adding SciRust cache policies to Elastic-Cache Dream.

The runtime feature extraction mirrors the real counterfactual trace collector:
cosine drift, similarity worsening, per-head standard deviation, normalized
cache age, untracked attention mass, layer fraction, drift-age interaction, and
normalized downstream refresh cost.
"""
from __future__ import annotations

import argparse
from pathlib import Path

HELPER = r'''

_SCIRUST_DEFAULT_WEIGHTS = (
    5.755952374691927,
    0.7882582865936595,
    4.897209095046155,
    -1.2589910896098846,
    3.570827209603688,
    -2.65516332465057,
    0.5164815206184813,
    -3.853401529744073,
)
_SCIRUST_DEFAULT_THRESHOLD = 1.0058506570900936


def _scirust_linear_score(weights, features):
    if len(weights) != len(features):
        raise ValueError(
            f"SciRust policy feature mismatch: {len(weights)} weights for {len(features)} features"
        )
    return sum(float(weight) * feature for weight, feature in zip(weights, features))


def _scirust_cache_decision(
    module,
    similarity,
    per_head_similarity,
    tracked_attention_mass,
    block_idx,
    gamma,
):
    """Return a deterministic refresh decision and record runtime evidence."""
    similarity_value = float(similarity.detach().float().item())
    head_std = float(
        per_head_similarity.detach().float().std(unbiased=False).clamp_min(0.0).item()
    )
    attention_mass = float(
        tracked_attention_mass.detach().float().clamp(0.0, 1.0).item()
    )
    previous_similarity = getattr(module, "scirust_previous_similarity", None)
    similarity_delta = (
        0.0 if previous_similarity is None else similarity_value - previous_similarity
    )
    module.scirust_previous_similarity = similarity_value

    age_steps = int(getattr(module, "scirust_cache_age_steps", 0)) + 1
    module.scirust_cache_age_steps = age_steps
    cache_age = min(age_steps / 16.0, 1.0)
    layer_count = max(int(getattr(module, "scirust_layer_count", 28)), 1)
    layer_fraction = block_idx / max(layer_count - 1, 1)
    refresh_cost = max(layer_count - block_idx - 1, 0) / layer_count
    drift = max(0.0, min(1.0, 1.0 - similarity_value))
    worsening = max(-similarity_delta, 0.0)
    untracked_mass = max(0.0, min(1.0, 1.0 - attention_mass))
    features = (
        drift,
        worsening,
        max(0.0, min(1.0, head_std)),
        cache_age,
        untracked_mass,
        max(0.0, min(1.0, layer_fraction)),
        drift * cache_age,
        max(0.0, min(1.0, refresh_cost)),
    )

    mode = getattr(module, "scirust_policy_mode", "gamma")
    weights = tuple(
        getattr(module, "scirust_policy_weights", _SCIRUST_DEFAULT_WEIGHTS)
    )
    threshold = float(
        getattr(module, "scirust_policy_threshold", _SCIRUST_DEFAULT_THRESHOLD)
    )
    score = _scirust_linear_score(weights, features)
    votes = None

    if mode == "linear":
        refresh = score >= threshold
    elif mode == "ensemble":
        policies = tuple(getattr(module, "scirust_policy_ensemble", ()))
        if not policies:
            raise ValueError("SciRust ensemble mode requires at least one frozen policy")
        vote_threshold = int(
            getattr(module, "scirust_policy_vote_threshold", (len(policies) // 2) + 1)
        )
        if not 1 <= vote_threshold <= len(policies):
            raise ValueError(
                f"invalid SciRust ensemble vote threshold {vote_threshold} for {len(policies)} policies"
            )
        policy_scores = []
        votes = 0
        for policy_weights, policy_threshold in policies:
            policy_score = _scirust_linear_score(tuple(policy_weights), features)
            policy_scores.append(policy_score)
            votes += int(policy_score >= float(policy_threshold))
        score = sum(policy_scores) / len(policy_scores)
        refresh = votes >= vote_threshold
    elif mode == "always":
        refresh = True
    elif mode == "never":
        refresh = False
    elif mode == "gamma":
        refresh = similarity_value < float(gamma)
    else:
        raise ValueError(f"unsupported SciRust cache policy mode: {mode}")

    module.scirust_decisions = int(getattr(module, "scirust_decisions", 0)) + 1
    module.scirust_risk_sum = float(getattr(module, "scirust_risk_sum", 0.0)) + score
    module.scirust_possible_refresh_cost = (
        float(getattr(module, "scirust_possible_refresh_cost", 0.0)) + refresh_cost
    )
    if refresh:
        module.scirust_refreshes = int(getattr(module, "scirust_refreshes", 0)) + 1
        module.scirust_refresh_cost = (
            float(getattr(module, "scirust_refresh_cost", 0.0)) + refresh_cost
        )

    if bool(getattr(module, "scirust_trace_enabled", False)):
        trace = getattr(module, "scirust_trace", None)
        if trace is None:
            trace = []
            module.scirust_trace = trace
        trace.append(
            {
                "layer_id": int(block_idx),
                "similarity": similarity_value,
                "similarity_delta": similarity_delta,
                "head_std": head_std,
                "cache_age": cache_age,
                "attention_mass": attention_mass,
                "layer_fraction": layer_fraction,
                "refresh_cost": refresh_cost,
                "risk": score,
                "votes": votes,
                "refresh": bool(refresh),
            }
        )

    return refresh
'''

OLD_DECISION = r'''            sim = F.cosine_similarity(past_att_weight, att_weight, dim=1).mean()
            if sim < gamma:
                lengths[1] = block_idx + 1
'''

NEW_DECISION = r'''            sim = F.cosine_similarity(past_att_weight, att_weight, dim=1).mean()
            past_flat = past_att_weight.float().flatten(2)
            tracked_flat = att_weight.float().flatten(2)
            per_head_sim = F.cosine_similarity(past_flat, tracked_flat, dim=2)
            if track_position.numel() > 0:
                tracked_mass = masked_att_weight[..., track_position].sum()
                total_mass = masked_att_weight.sum().clamp_min(
                    torch.finfo(masked_att_weight.dtype).tiny
                )
                tracked_attention_mass = tracked_mass / total_mass
            else:
                tracked_attention_mass = att_weight.new_tensor(0.0)
            if _scirust_cache_decision(
                self,
                sim,
                per_head_sim,
                tracked_attention_mass,
                block_idx,
                gamma,
            ):
                lengths[1] = block_idx + 1
'''

OLD_UPDATE = r'''        if block_idx >= start_reset: # Cache update
            self.q_cache = query_states
'''

NEW_UPDATE = r'''        if block_idx >= start_reset: # Cache update
            self.scirust_cache_age_steps = 0
            self.q_cache = query_states
'''


def replace_exact(text: str, old: str, new: str, expected: int, label: str) -> str:
    count = text.count(old)
    if count != expected:
        raise RuntimeError(f"{label}: expected {expected} exact matches, found {count}")
    return text.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("elastic_cache_root", type=Path)
    args = parser.parse_args()

    target = args.elastic_cache_root / "dream" / "model" / "modeling_dream.py"
    text = target.read_text(encoding="utf-8")
    marker = (
        "    return hidden_states.reshape(batch, num_key_value_heads * n_rep, "
        "slen, head_dim)\n"
    )
    if text.count(marker) != 1:
        raise RuntimeError("repeat_kv insertion marker is not unique")
    if "_SCIRUST_DEFAULT_WEIGHTS" in text:
        raise RuntimeError("target already contains the SciRust patch")
    text = text.replace(marker, marker + HELPER, 1)
    text = replace_exact(text, OLD_UPDATE, NEW_UPDATE, 2, "cache-update hook")
    text = replace_exact(text, OLD_DECISION, NEW_DECISION, 2, "decision hook")
    target.write_text(text, encoding="utf-8")
    print(f"patched {target}")


if __name__ == "__main__":
    main()
