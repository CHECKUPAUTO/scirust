#!/usr/bin/env python3
"""Extend the patched Dream runtime with fail-closed single-skip probe modes.

This patch is intentionally separate from ``patch_elastic_cache.py`` so the
historical phase-4 to phase-7 runtime remains byte-for-byte reproducible. Apply
it only after the base SciRust patch.
"""
from __future__ import annotations

import argparse
from pathlib import Path


PROBE_HELPER = r'''


def _scirust_register_probe_candidate(
    module,
    skip_margin,
    refresh_cost,
    block_idx,
    features,
    votes,
):
    """Register an eligible baseline-trajectory skip opportunity.

    Probe modes always refresh before the selected opportunity. Consequently,
    candidate ordinals are evaluated on the exact always-refresh trajectory up
    to the intervention point. A shared state object is attached to every
    attention module by the trajectory-branch benchmark.
    """
    state = getattr(module, "scirust_probe_state", None)
    if not isinstance(state, dict):
        raise ValueError("SciRust probe mode requires a shared dictionary state")

    minimum_margin = float(getattr(module, "scirust_guard_minimum_skip_margin", 0.0))
    minimum_cost = float(getattr(module, "scirust_guard_minimum_refresh_cost", 0.0))
    max_skips = getattr(module, "scirust_guard_max_skips_per_layer", None)
    if max_skips is not None:
        max_skips = int(max_skips)
        if max_skips < 0:
            raise ValueError("SciRust guard max skips must be non-negative or None")
    cooldown = int(getattr(module, "scirust_guard_cooldown_decisions", 0))
    if cooldown != 0:
        raise ValueError(
            "single-skip probe currently requires a zero-cooldown frozen guard"
        )

    layer_id = int(block_idx)
    eligible_per_layer = state.setdefault("eligible_per_layer", {})
    previous = int(eligible_per_layer.get(layer_id, 0))
    within_budget = max_skips is None or previous < max_skips
    eligible = (
        skip_margin + 1e-15 >= minimum_margin
        and refresh_cost + 1e-15 >= minimum_cost
        and within_budget
    )
    if not eligible:
        return None

    eligible_per_layer[layer_id] = previous + 1
    ordinal = int(state.get("candidate_ordinal", 0)) + 1
    state["candidate_ordinal"] = ordinal
    feature_names = (
        "drift",
        "worsening",
        "head_std",
        "cache_age",
        "untracked_mass",
        "layer_fraction",
        "drift_age",
        "refresh_cost",
    )
    candidate = {
        "ordinal": ordinal,
        "layer_id": layer_id,
        "skip_margin": float(skip_margin),
        "refresh_cost": float(refresh_cost),
        "votes": None if votes is None else int(votes),
        "features": {
            name: float(value) for name, value in zip(feature_names, features)
        },
    }
    state.setdefault("candidates", []).append(candidate)
    return ordinal
'''

MODE_OLD = '    elif mode in ("ensemble", "guarded_ensemble"):\n'
MODE_NEW = (
    '    elif mode in (\n'
    '        "ensemble",\n'
    '        "guarded_ensemble",\n'
    '        "probe_enumerate",\n'
    '        "probe_single_skip",\n'
    '    ):\n'
)

INITIALIZERS_OLD = r'''    guard_allowed = None
    guard_forced_refresh = False
'''

INITIALIZERS_NEW = r'''    guard_allowed = None
    guard_forced_refresh = False
    probe_candidate_ordinal = None
    probe_applied = False
'''

GUARD_BLOCK_OLD = r'''        if mode == "guarded_ensemble" and not refresh:
            guard_allowed = _scirust_guarded_skip(module, skip_margin, refresh_cost)
            refresh = not guard_allowed
            guard_forced_refresh = refresh
'''

GUARD_BLOCK_NEW = r'''        if mode == "guarded_ensemble" and not refresh:
            guard_allowed = _scirust_guarded_skip(module, skip_margin, refresh_cost)
            refresh = not guard_allowed
            guard_forced_refresh = refresh
        elif mode in ("probe_enumerate", "probe_single_skip") and not refresh:
            probe_candidate_ordinal = _scirust_register_probe_candidate(
                module,
                skip_margin,
                refresh_cost,
                block_idx,
                features,
                votes,
            )
            if probe_candidate_ordinal is None or mode == "probe_enumerate":
                refresh = True
            else:
                state = getattr(module, "scirust_probe_state", None)
                if not isinstance(state, dict):
                    raise ValueError(
                        "SciRust probe mode requires a shared dictionary state"
                    )
                target = int(state.get("target_ordinal", 0))
                probe_applied = (
                    not bool(state.get("applied", False))
                    and probe_candidate_ordinal == target
                )
                refresh = not probe_applied
                if probe_applied:
                    state["applied"] = True
                    state["applied_candidate"] = dict(state["candidates"][-1])
'''

TRACE_OLD = r'''                "guard_skip_count": int(getattr(module, "scirust_guard_skip_count", 0)),
                "refresh": bool(refresh),
'''

TRACE_NEW = r'''                "guard_skip_count": int(getattr(module, "scirust_guard_skip_count", 0)),
                "probe_candidate_ordinal": probe_candidate_ordinal,
                "probe_applied": probe_applied,
                "refresh": bool(refresh),
'''


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one exact match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("elastic_cache_root", type=Path)
    args = parser.parse_args()

    target = args.elastic_cache_root / "dream" / "model" / "modeling_dream.py"
    text = target.read_text(encoding="utf-8")
    if "_SCIRUST_DEFAULT_WEIGHTS" not in text:
        raise RuntimeError("base SciRust cache-policy patch is missing")
    if "_scirust_register_probe_candidate" in text:
        raise RuntimeError("target already contains the trajectory-probe patch")

    marker = "\ndef _scirust_cache_decision(\n"
    if text.count(marker) != 1:
        raise RuntimeError("cache-decision insertion marker is not unique")
    text = text.replace(marker, PROBE_HELPER + marker, 1)
    text = replace_exact(text, MODE_OLD, MODE_NEW, "ensemble mode extension")
    text = replace_exact(
        text, INITIALIZERS_OLD, INITIALIZERS_NEW, "probe state initializers"
    )
    text = replace_exact(text, GUARD_BLOCK_OLD, GUARD_BLOCK_NEW, "probe decision")
    text = replace_exact(text, TRACE_OLD, TRACE_NEW, "probe trace fields")
    target.write_text(text, encoding="utf-8")
    print(f"trajectory-probe patched {target}")


if __name__ == "__main__":
    main()
