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

A faithful port of TDI's `tdi-core`, kept semantically identical: the twelve
source modules differ from the frozen upstream only by SciRust's `rustfmt.toml`
(`control_brace_style = "AlwaysNextLine"`, `match_block_trailing_comma`) and by
this crate's own documentation. Ten of the twelve are token-for-token identical
once whitespace is normalised. **The upstream confirmatory results therefore
describe this crate's behaviour directly.**

### What the reference project has since established

The upstream program has moved well past TDI-1. Each experiment below is
preregistered — design frozen and SHA-256-pinned *before* execution, run exactly
once behind a human-gated confirmation token, reproducing byte-exactly — and the
figures come from real confirmatory runs, not pilots:

| Experiment | Control the overlaps had to beat | Outcome |
|---|---|---|
| **TDI-5.5** | exact contraction descriptors (Dobrushin δ, δ̄) | Beneficial at every horizon U₃…U₈ |
| **TDI-5.6** | δ, δ̄ **+** exact spectral moments s₂, s₃ | Beneficial at every horizon; the moments are themselves informative, so the control is not inert |
| **TDI-5.7** | the same, across four structurally distinct generator families | replicates in all four — the single-generator limitation is substantially closed |
| **TDI-6.1** | the **literal** spectral gap `1 − \|λ₂\|` and the ε-mixing time | Beneficial at both focal horizons — the signal is not a proxy for classical mixing structure |
| **TDI-5.9** | δ, δ̄ **+** s₂, s₃, s₄ | Beneficial at every horizon, no redundancy horizon — *and* the exact descriptor ladder **saturates** (s₄ buys 14 % / 26 % of what s₂+s₃ bought, disjoint CIs) while the signal does not |

The short version: the structure of accessible futures carries predictive
information that scalar summaries of the same system — entropy, contraction,
spectral moments, and the literal spectral gap — do not preserve, and that gap
does **not** close as the scalar summary is enriched.

### What is still not established

TDI-1's original signal **was** fully subsumed by the orbital baseline
(incremental gain `0`), and the width-4 out-of-distribution holdout was poorly
calibrated. Those negative results stand, and they are a large part of why the
positive ones above are worth anything. Beyond them the following remain open —
evaluators are built and frozen upstream, but the confirmatory runs have not
been executed:

- **cross-width invariance at scale** (TDI-5.8) — widths 3–4 only so far;
- **causal / interventional effect** (TDI-6.4) — everything above is predictive,
  not interventional;
- **sufficiency under nonlinear or non-parametric learners** (TDI-6.2) — the
  models used throughout are linear ridge;
- **information decomposition** of what O₁ and O₂ each contribute (TDI-6.3);
- **generator-family robustness under the non-exact discipline** (TDI-6.5).

No universal law is claimed. Nothing here has been tested outside small
synthetic finite-state families, and none of it has been evaluated against
learned representations.

> Note: the upstream TDI project has not yet selected a license.
