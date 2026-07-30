# TDI-6.4 — The Causal Probe: Confirmatory Results

## Status

This document reports the single, real, preregistered TDI-6.4 run. The design
was frozen before execution
(`docs/TDI-6.4-CAUSAL-PROBE-PREREGISTRATION.md`, SHA-256
`8933a389586ee22af9e1cddf7ed9d30d6205b7d10682fbde362a9a01e680333b`) and the
run was a deliberate one-time human action under the exact confirmation token.
No summary below may be rewritten (preregistration Section 20). None of
TDI-6.4A/B/C is a pass/fail classification; all three are descriptive by
design.

**Headline.** The recovery trajectory **does depend on which node is
perturbed** — the intervention target is not exchangeable. Aggregate
node-to-node heterogeneity `range(h)` averages 1.01–1.46 bits with bootstrap
intervals far from zero at every horizon. But two qualifications matter as
much as the finding itself. First, the heterogeneity **shrinks in relative
terms**: it grows in absolute bits with horizon while falling monotonically
from 23.3 % to 11.0 % of the typical deficit level, so *which* node matters
most at short horizons and washes out at long ones. Second, the
**early→late coupling is node-invariant** — `corr(O_i, U_i(6))` is
statistically indistinguishable across every node at both widths. The
intervention target changes the *magnitude* of the deficit, not the
*relationship* between early overlap and late deficit.

## 1. Provenance and integrity

| Item | Value |
|---|---|
| Run git commit | `4b5255df50dccb44820543d5719d192866510b99` |
| Evaluator (v64) SHA-256 | `1c569c3d46eb7ef766f084c5051d3d437290e5357aa523305a61b5d63c9f51b5` |
| Preregistration SHA-256 | `8933a389586ee22af9e1cddf7ed9d30d6205b7d10682fbde362a9a01e680333b` |
| Scientific-manifest SHA-256 | `832cc352fc0ccfba1bab56a7e9f1d1874cd6d2bcb834726ad50fa4edd9cb8110` |
| Result log SHA-256 | `d8d5450aa2fc64fcb595a4facf80c0a1ce495021684ad7f24ca00a82e44af197` |
| Toolchain | rustc 1.97.1 (8bab26f4f 2026-07-14), cargo 1.97.1 (c980f4866 2026-06-30) |
| Host | Linux `tarek`, aarch64 (Jetson, ARM64) |
| Start / end (UTC) | 2026-07-29T17:02:37Z / 2026-07-29T18:50:09Z (~1 h 48) |

`scripts/reproduce-tdi6.4.sh` verified the **full frozen chain** — twelve
ancestors (TDI-5.1 → 5.8, 6.1, 6.2, 6.3, 6.5) plus TDI-6.4's own three
manifests, every entry including all of `tdi-core` — before any generation,
and refuses a dirty repository. The run therefore used the exact reviewed,
frozen scientific code.

Reproduction is **byte-exact**, not tolerance-based. Despite being filed under
the roadmap's non-exact "Track B", TDI-6.4 stays entirely on the bit-exact
rational track (preregistration Section 1.2): it calls the already-exact
`tdi-core` primitives more times rather than replacing them with a
floating-point computation. The declared regime is IEEE-754 binary64,
single-threaded, fixed operation order, no FMA.

## 2. Populations and the one changed factor

Three fresh seed blocks **V / W / X** (bases 9.0e9 / 9.1e9 / 9.2e9),
**40,000 accepted records per block, 120,000 total**, no OOD populations.
120,238 candidates were attempted for 120,000 accepted — **238 preregistered
exclusions** (V 78, W 79, X 81), all at width 3 (236
`observation-fully-recovered`, 2 `target-fully-recovered-h3`); width 4
produced zero. Generation ran far inside its budgets (worst population 15,063
of 960,000 permitted attempts, 1.6 %).

The single changed factor versus the frozen TDI-5.6 ancestor is the
**perturbation protocol**. Every prior TDI-5.x/6.x experiment perturbs exactly
one node — the historical `Action::Flip{node: width-1}`. TDI-6.4 instead
analyses **every** node `0..width` through the same frozen
`analyze_branching_recovery`, making the comparison a genuine
intervention/counterfactual contrast in the Neyman–Rubin sense rather than a
new predictive model.

## 3. Criterion TDI-6.4A — node-to-node heterogeneity

For each accepted system, `range(h) = max_i U_i(h) − min_i U_i(h)` over all
analysed nodes. Aggregate over all three blocks:

| Horizon | median `range(h)` | mean | 95 % CI (mean) | typical `U_i(h)` | mean / typical |
|---|---:|---:|---|---:|---:|
| U₃ | 0.9090 | 1.0117 | [1.0084, 1.0150] | 4.3374 | **23.32 %** |
| U₄ | 1.0073 | 1.1331 | [1.1293, 1.1369] | 6.1418 | 18.45 % |
| U₅ | 1.0888 | 1.2362 | [1.2319, 1.2406] | 7.9335 | 15.58 % |
| U₆ | 1.1520 | 1.3210 | [1.3164, 1.3259] | 9.7122 | 13.60 % |
| U₇ | 1.2051 | 1.3953 | [1.3903, 1.4005] | 11.4689 | 12.17 % |
| U₈ | 1.2494 | 1.4569 | [1.4516, 1.4625] | 13.2088 | **11.03 %** |

Every interval excludes zero by a wide margin: **the node perturbed matters.**
The distribution is right-skewed (mean above median throughout; aggregate
maxima reach 6.45 bits at U₃ and 23.23 at U₈), so a minority of systems are
far more node-sensitive than the median one.

The absolute-versus-relative contrast is the part that does not read off the
log directly and deserves emphasis. `range(h)` **grows monotonically in
absolute bits** with horizon, which in isolation would suggest node identity
matters more at longer horizons. Normalised by the typical deficit level at
the same horizon it **falls monotonically**, from 23.3 % at U₃ to 11.0 % at
U₈. Both trends are monotone across all six horizons. The reading that
respects both: node identity is a first-order effect early, and is
progressively diluted as the deficit itself grows.

The preregistration's descriptive threshold (the population median `U_i(h)`)
is exceeded by a very small fraction of systems — 0.082 % at U₃ falling to
0.010 % at U₈ — confirming that extreme node-sensitivity is rare rather than
typical.

## 4. Criterion TDI-6.4B — the coupling is node-invariant

Per-node correlation between early overlap and the U₆ deficit, aggregate,
reported separately per width:

| Width | Node | `corr(O_i(1), U_i(6))` | `corr(O_i(2), U_i(6))` |
|---|---:|---:|---:|
| 3 | 0 | 0.4499 [0.4434, 0.4564] | 0.7117 [0.7073, 0.7162] |
| 3 | 1 | 0.4578 [0.4509, 0.4644] | 0.7170 [0.7126, 0.7216] |
| 3 | 2 | 0.4615 [0.4551, 0.4678] | 0.7170 [0.7127, 0.7215] |
| 4 | 0 | 0.4180 [0.4113, 0.4247] | 0.6563 [0.6508, 0.6618] |
| 4 | 1 | 0.4244 [0.4178, 0.4313] | 0.6610 [0.6555, 0.6663] |
| 4 | 2 | 0.4192 [0.4125, 0.4257] | 0.6622 [0.6566, 0.6680] |
| 4 | 3 | 0.4263 [0.4198, 0.4330] | 0.6625 [0.6572, 0.6678] |

**`stable(O₁) = true` and `stable(O₂) = true` at both widths**: every node's
bootstrap interval overlaps that of the historical node `i* = width − 1`, for
both overlaps, with no exceptions.

This is the result that carries the most weight for the causal question. The
perturbation target shifts *how large* the resulting deficit is (Section 3),
but leaves the *predictive relationship* between early overlap and late
deficit unchanged. A relationship that survives relocation of the intervention
is more robust than one established under a single fixed intervention — which
is all every prior experiment in the series had.

Two secondary observations, consistent with earlier results: `O₂` couples far
more strongly than `O₁` throughout (≈0.71 vs ≈0.45 at width 3), echoing
TDI-5.4's finding that `O₁`'s marginal value decays; and width-3 couplings
exceed width-4 ones (0.717 vs 0.662 for `O₂`), a cross-width difference that
TDI-5.8 is designed to address and this experiment does not.

## 5. Criterion TDI-6.4C — what predicts heterogeneity

Aggregate correlation of `range(6)` with each exact descriptor:

| Descriptor | `corr(range(6), ·)` | 95 % CI | variance explained |
|---|---:|---|---:|
| `s₂` | 0.2373 | [0.2298, 0.2447] | 5.63 % |
| `s₃` | 0.1370 | [0.1266, 0.1477] | 1.88 % |
| δ | 0.0669 | [0.0615, 0.0724] | 0.45 % |
| δ̄ | 0.0524 | [0.0460, 0.0589] | 0.28 % |

All four are positive with intervals excluding zero, and all four are modest.
The ordering is informative: the **spectral moments predict node-sensitivity
better than the contraction descriptors do**, by roughly a factor of four. But
even the best of them accounts for under 6 % of the variance, so node
heterogeneity is largely *not* explained by the exact descriptor set the
programme has accumulated. Whatever makes some systems node-sensitive is
mostly not visible to δ, δ̄, s₂ or s₃.

## 6. Internal verifications that fired

Two guards designed into the evaluator before it was built were exercised on
real data, and both behaved as specified.

**Section 19 — historical-node consistency: SUCCESS on all 120,000 accepted
records.** A runtime `assert_eq!` triple inside `analyze_seed` compares the
recomputed `NodeAnalysis` entry for the historical node against the value
inherited from the TDI-5.6 computation path. Any divergence between the two
paths would have panicked the run rather than producing a quietly inconsistent
result.

**Section 5.1 — full-recovery exclusion.** Non-historical nodes can reach
`O_i(h) = 1` (giving `U_i(h) = −log2(1−1) = +∞`) without the record being
rejected, unlike the historical node. The rule requires excluding those from
aggregates and *counting* the exclusions. The counts confirm the wiring:
499 aggregate exclusions at U₄–U₈ (498 at U₃), and in the per-node breakdown
the historical node `width − 1` has **exactly zero** exclusions at both widths
while other nodes reach 245 and 255 — precisely what the design predicts,
since full recovery at the historical node causes upstream record rejection.
Width-3 systems account for essentially all of it (nodes 0 and 1 at width 3),
while width 4 produced at most one, consistent with the smaller state space
recovering fully more often.

## 7. Interpretation and boundaries

TDI-6.4 answers its preregistered question — *does recovery depend on which
node is perturbed, or only that a perturbation happened?* — with **which**.
Combined with the node-invariance of the coupling, the picture is that the
system's response magnitude is intervention-specific while the overlap→deficit
law is not.

What this does **not** establish, and should not be read as establishing:

- **It does not show that the overlap signal is causal for recovery.** The
  experiment varies the *intervention target* and observes that trajectories
  differ; it does not identify a causal effect of `O` on `U`. The
  node-invariance of TDI-6.4B is evidence of robustness under relocation of
  the intervention, which is stronger than pure association but well short of
  causal identification.
- **It uses `Flip` only.** `Action::Clamp` — a state-independent intervention,
  arguably the cleaner `do()` operator — is available in `tdi-core` and was
  deliberately deferred (preregistration Section 4.1). Single-node
  interventions only; no joint or multi-node perturbations.
- Widths 3–4 only (TDI-5.8); single generator (TDI-5.7 addressed this for a
  different question); linear ridge machinery inherited unchanged; no
  nonlinear or non-parametric learners (TDI-6.2); no information decomposition
  (TDI-6.3). No universal law is claimed, and nothing here was tested outside
  small synthetic finite-state families.

The TDI-6.4A / TDI-6.4B / TDI-6.4C summaries are frozen as reported.

## 8. Reproduction

    TDI64_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI64_FREEZE_RULE \
      bash scripts/reproduce-tdi6.4.sh

The script refuses without the exact token, refuses a dirty repository,
verifies the full frozen hash chain before any generation, executes the
evaluator once with `--full`, verifies the final criterion lines, and writes
read-only result, metadata, hash and completion artifacts under
`results/tdi6.4-causal-probe/`. Determinism is exact: the log SHA-256
`d8d5450aa2fc64fcb595a4facf80c0a1ce495021684ad7f24ca00a82e44af197` is
reproduced by any faithful re-run on the same toolchain and architecture.
