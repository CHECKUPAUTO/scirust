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
| **TDI-5.8** | the same, across widths 3, 4 **and** 5 | Beneficial in **all 18** width × horizon cells; effect size nearly width-invariant (0.61 pp spread at U₆). But cross-width **calibration fails** (every R² far below zero) while **rank ordering survives** — Spearman 0.695 through the overlaps, −0.040 through the exact descriptors |
| **TDI-6.2** | the spectral baseline **under a degree-2 interaction ridge** | The stronger model helps the baseline — and the overlaps' advantage **grows** rather than shrinks, at all six horizons (54.2 % vs 47.0 % at U₃). The signal is not a linear-modeling artifact |
| **TDI-6.5** | the literal-spectral control, across the four families, non-exactly | Beneficial at both focal horizons in all four families, **24/24** grid cells — but the effect size ranges over **39.8 pp**, and in the one family where the literal descriptors are *censored* (`\|λ₂\| = 1` exactly for a large fraction) they add nothing at U₃ |
| **TDI-6.3** | — (descriptive: Gaussian/MMI information decomposition) | Total information and redundancy both decay with horizon while **synergy is the only component that grows** (0.005 % → 6.6 %). Note `Unique(O₁) = 0` throughout is *definitional* under MMI, not an empirical finding |
| **TDI-6.4** | — (descriptive: node-exchangeability causal probe) | The recovery trajectory **does** depend on which node is perturbed, but the heterogeneity falls from 23.3 % to 11.0 % with horizon and the early→late coupling is node-invariant. The intervention target changes the *magnitude*, not the *relationship* |

The short version: the structure of accessible futures carries predictive
information that scalar summaries of the same system — entropy, contraction,
spectral moments, and the literal spectral gap — do not preserve; that gap does
**not** close as the scalar summary is enriched; and it does not close under a
nonlinear model either.

### The transfer failure, and two refuted repairs

The most informative results in the series are negative ones.

A model fitted in one domain does **not** predict absolute deficit levels in a
new one — neither across widths (TDI-5.8) nor across generators (TDI-6.5). What
transfers is *rank ordering*. Two label-free repairs were then preregistered and
tested, and both were refuted:

| Repair | Outcome | Mechanism |
|---|---|---|
| **TDI-6.6** — re-standardize the features with the target domain's own statistics | ***Harmful*** in all four confirmatory cells | Centring on the target's mean annihilates, *by construction*, the domain displacement that was carrying the level. The oracle arm — which also replaces the target scaler — *does* repair it, locating the residual failure in the target's **deficit level**, the very quantity being predicted |
| **TDI-6.7** — add the observable shift `Δ = μ₂ᵀ − μ₂ˢ` to the intercept (`O₂` is a feature, so `Δ` needs no labels) | ***Harmful*** in all four confirmatory cells | An additive constant moves **only** the bias — residual spread invariant to `3e-12` across 144 blocks — and the frozen model has already carried the level shift through the features. A *perfect* `Δ` would help in **fewer** cells (9/24) than the imperfect one (12/24) |

Refutations with identified mechanisms bound the family of possible remedies,
rather than merely failing to find one.

### What is still not established

TDI-1's original signal **was** fully subsumed by the orbital baseline
(incremental gain `0`), and the width-4 out-of-distribution holdout was poorly
calibrated. Those negative results stand, and they are a large part of why the
positive ones above are worth anything.

Beyond them, the following are **not** established:

- **a universal law.** Nothing here has been tested outside small synthetic
  finite-state branching families at widths 3–5, and none of it has been
  evaluated against learned representations;
- **transportable calibration.** This is now a *measured* failure rather than an
  untested question, with two candidate repairs refuted (TDI-6.6, TDI-6.7);
- **a transportable effect size.** Nearly width-invariant (0.61 pp spread) but
  strongly generator-dependent (39.8 pp) — no single number is "the" effect;
- **superiority over an arbitrarily expressive learner.** A degree-2 interaction
  ridge is the strongest model tested; kernel methods and tree ensembles are
  untested;
- **causal structure** beyond TDI-6.4's node-exchangeability probe, or a
  decomposition under any PID definition other than Gaussian/MMI.

One experiment is **built but not run**: TDI-6.8 asks whether transferred
*ordering* survives where the level does not — the reading offered four times
across the series and tested zero times. Its preregistration is frozen and its
evaluator complete; the confirmatory run is a deliberate human action behind a
token that no test, commit or CI supplies. No result is claimed for it here.

> Licensing: TDI is dual-licensed on the same model as SciRust — free for
> noncommercial and personal use under the PolyForm Noncommercial License 1.0.0,
> commercial use under separate terms. The full preregistrations, results
> reports and hash manifests are copied verbatim under
> [`docs/tdi-upstream/`](../docs/tdi-upstream/).
