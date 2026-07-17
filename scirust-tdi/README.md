# scirust-tdi

Prospective **dynamic-information analysis**, ported from the *TDI* ("Dynamic
Information Theory") research project. It gives SciRust an exact, deterministic
toolkit for asking whether the **structure of accessible futures** carries
predictive information that scalar summaries such as Shannon entropy do not
preserve.

Everything here is **exact**: finite-state dynamics with arbitrary-precision
**rational** probabilities (via `num-bigint`, already in the workspace), no
floating-point rounding, deterministic generators. This is a faithful,
test-for-test re-homing of the reference `tdi-core` crate.

## What it provides

| Layer | API |
|---|---|
| **Exact finite-state dynamics** | `TableSystem` / `TransitionSystem`, `State`, `Action`, `explore` (reachability) |
| **Future-structure descriptors** | `uniform_future_block_distribution`, `uniform_branching_path_distribution`, `uniform_branching_state_distribution`, and the flagship `distribution_overlap` (intervention-conditioned overlap) |
| **Honest baselines** | `uniform_future_block_entropy_bits` (Shannon), `analyze_orbit` (orbital), `analyze_recovery` / `analyze_branching_recovery` (perturbation recovery) |
| **Exact arithmetic** | `ExactRatio`, `TdiSignature` |

## Quick start

```bash
cargo test -p scirust-tdi     # 40 exact tests (39 unit + adversarial recovery)
```

## Design

- **Exact & deterministic**: rational arithmetic throughout; identical inputs
  give identical results, bit-for-bit.
- **No unsafe** (`#![forbid(unsafe_code)]`).
- Dependencies (`num-bigint`, `num-integer`, `num-traits`) are pure Rust and
  already present in the SciRust workspace.

## Provenance & scope

A faithful port of TDI's `tdi-core`. As the reference project reports honestly,
its results **challenge entropy-only sufficiency** within the tested synthetic
families but do **not** establish a universal law, invariance across system
sizes, or superiority over every dynamical baseline — TDI-1's signal was fully
subsumed by the orbital baseline (incremental gain `0`), and the width-4
out-of-distribution holdout was poorly calibrated. What transfers to SciRust is
the deterministic, exact machinery and the honest evaluation discipline.

> Note: the upstream TDI project has not yet selected a license.
