# TDI-5.7 — Generator Robustness: Confirmatory Results

## Status

This document reports the single, real, preregistered TDI-5.7 run. The design
was frozen before execution
(`docs/TDI-5.7-GENERATOR-ROBUSTNESS-PREREGISTRATION.md`) and the run was a
deliberate one-time human action under the exact confirmation token. No
classification below may be rewritten (preregistration Section 22).

**Headline.** The primary finding of TDI-5.6 — the marginal value of the early
overlaps `{O_1, O_2}` beyond the full exact descriptor set (Dobrushin δ, δ̄ **and**
the exact spectral moments s₂, s₃; the SKT-vs-SK comparison) — **replicates
across all four structurally distinct exact generator families** at the focal
horizons U₃ and U₆. Criterion **TDI-5.7A is met: replicated = yes**, decisively
and with 3/3 seed blocks confirming in every family. The single-generator
limitation repeated since TDI-5.2 is thereby substantially closed: within the
exact scope, the overlap signal is a property of exact branching dynamics
**broadly**, not an artifact of the one base generator. Two honest
qualifications travel with this: the effect **size** is strongly heterogeneous
across families (5.7B), and the *same fitted* signal transports across
generators only in **direction**, not calibration (5.7C).

## 1. Provenance and integrity

| Item | Value |
|---|---|
| Run git commit | `c7375f652a90c4ab27e44db26087995c9cf07d7a` |
| Evaluator (v57) SHA-256 | `900031bc27a35e327038911d93f10d74458f913e64d9644b225963df699049ae` |
| Preregistration SHA-256 | `2ca7d1a674d451e642beb5b01f8a0d8f08f8fadcf7f91032370e7fd5e3d91476` |
| Scientific-manifest SHA-256 | `25c2175780e9498334939bb3d1d217e24b58d8c83840da4277fb9bdd86c06d5e` |
| Result log SHA-256 | `6e8b8d89853ea674d97337bf34f213329456586987baac0d8318af71daf4d5d7` |
| Toolchain | rustc 1.97.0 (2d8144b78 2026-07-07), cargo 1.97.0 (c980f4866 2026-06-30) |
| Host | Linux `tarek`, aarch64 (Jetson, ARM64) |
| Start / end (UTC) | 2026-07-23T07:29:24Z / 2026-07-23T08:23:55Z (~54 min) |

The run's recorded evaluator hash equals the committed frozen v57 evaluator
and its `EVALUATOR.sha256` manifest (`900031bc…`); the preregistration
(`2ca7d1a6…`) and scientific-manifest (`25c21757…`) hashes likewise match the
committed frozen files. The run was executed at commit `c7375f6` — the current
`main`, which carries the post-review §17-compliance correction — so the run's
own provenance printout correctly self-reports the v57 hash `900031bc…`, and
the required per-family per-horizon grid (U₃…U₈, Section 5 below) is emitted.
The recorded frozen-ancestor hashes (TDI-5.5 evaluator, TDI-5.6 evaluator and
preregistration) match their committed values, and the full frozen chain
TDI-5.1 → TDI-5.6 verifies clean. The committed result log has been reverified
independently to hash to
`6e8b8d89853ea674d97337bf34f213329456586987baac0d8318af71daf4d5d7`. The run
therefore used the exact reviewed, frozen scientific code.

## 2. Populations and generator families

Four generator families, each holding the entire TDI-5.6 measurement apparatus
fixed and varying **only** the successor-mask construction rule (preregistration
Section 5); every rule is a deterministic function of the candidate seed via the
inherited `splitmix64` chain, assembled by the unchanged frozen `build_system`,
and guarantees a non-empty successor set (a total, exact `Noop` kernel):

- **F0Base** — the inherited uniform generator (reproduces the TDI-5.6 candidate
  distribution under fresh seeds);
- **F1Sparse** — low out-degree `d ∈ {1, 2}`, `d` distinct successors by
  rejection;
- **F2Dense** — all states minus `e ∈ {0, 1}` excluded bit;
- **F3Local** — Hamming-≤1 neighbourhood `{s, s⊕1, s⊕2, …}`, self forced on an
  empty draw.

Per family: three fresh seed blocks, base width-3 + width-4 composition, 40,000
accepted records per block — **120,000 per family, 480,000 total** across the
48 pairwise-disjoint seed reservations (verified disjoint from the TDI-5.6
blocks J/K/L and all earlier blocks). Models were fitted on each block's
combined training population and every criterion evaluated on that block's
combined holdout, standardized-U space primary. The confirmatory layouts are
inherited unchanged: **CK** (baseline + δ + δ̄), **SK** (CK + s₂ + s₃), **SKT**
(SK + O₁ + O₂).

## 3. Criterion TDI-5.7A — replication across generators (primary)

SKT-vs-SK four-way classification (symmetric 2% relative-MSE margin) at the
focal horizons, per family:

| Family | U₃ class. | U₃ rel-reduction | U₃ 95% CI | U₆ class. | U₆ rel-reduction | U₆ 95% CI |
|---|---|---:|---|---|---:|---|
| **F0Base** | Beneficial | 47.02% | [0.4507, 0.4695] | Beneficial | 22.31% | [0.2133, 0.2334] |
| **F1Sparse** | Beneficial | 56.70% | [0.5579, 0.5761] | Beneficial | 26.19% | [0.2525, 0.2713] |
| **F2Dense** | Beneficial | 40.87% | [0.4008, 0.4164] | Beneficial | 19.43% | [0.1863, 0.2028] |
| **F3Local** | Beneficial | 77.23% | [0.7665, 0.7777] | Beneficial | 52.33% | [0.5140, 0.5324] |

**Replicated = yes** — Beneficial at both U₃ and U₆ for **all four** families,
each with 3/3 blocks confirming, every aggregate reduction beyond the 2% margin
by more than an order of magnitude, and every aggregate bootstrap lower bound
strictly positive. There is **no located non-replication**. Representative
aggregate error reductions: F3Local U₃ SK MSE 0.7462 → SKT 0.1699 (R² 0.253 →
0.830); F0Base U₃ 0.3364 → 0.1816 (R² 0.664 → 0.819).

## 4. Criterion TDI-5.7B — effect-size heterogeneity

Across-family spread of the aggregate SKT-vs-SK reduction, per focal horizon:

| Horizon | Minimum (family) | Maximum (family) | Range | All four exceed 2% |
|---|---:|---:|---:|:--:|
| **U₃** | 40.86% (F2Dense) | 77.23% (F3Local) | 36.36 pp | **yes** |
| **U₆** | 19.43% (F2Dense) | 52.34% (F3Local) | 32.89 pp | **yes** |

The **direction** of the effect is universal (all four Beneficial, all above
2%), but the **magnitude** swings by roughly 2–4× across families. This is the
preregistered transportability caveat: the overlap signal generalizes robustly
in sign but not in size — a skeptic reading only F2Dense would see a modest
effect, one reading F3Local a dominant one.

## 5. Per-family per-horizon reductions across the grid (U₃…U₈)

Descriptive (Section 12): the aggregate SKT-vs-SK relative-MSE reduction at every
horizon of the dense grid, per family. **All 24 points classify Beneficial.**

| Family | U₃ | U₄ | U₅ | U₆ | U₇ | U₈ |
|---|---:|---:|---:|---:|---:|---:|
| F0Base | 46.00% | 34.99% | 27.28% | 22.30% | 18.25% | 15.46% |
| F1Sparse | 56.71% | 41.38% | 31.74% | 26.20% | 21.27% | 18.75% |
| F2Dense | 40.86% | 28.62% | 24.29% | 19.44% | 15.70% | 14.03% |
| F3Local | 77.23% | 69.03% | 58.17% | 52.34% | 45.98% | 42.43% |

Every family shows the same qualitative decay with horizon (largest near U₃,
shrinking but never negligible by U₈), with F3Local's marginal overlap value
staying far above the others at every horizon.

## 6. Criterion TDI-5.7C — cross-generator transfer (descriptive)

F0Base's fitted SK and SKT models evaluated on F1Sparse's combined holdout:

| Horizon | Classification | Rel-reduction | 95% CI | SK R² | SKT R² |
|---|---|---:|---|---:|---:|
| U₃ | Beneficial | 27.33% | [0.2603, 0.2861] | −2.582 | −1.603 |
| U₆ | Beneficial | 13.81% | [0.1276, 0.1487] | −4.499 | −3.740 |

Two things hold at once, and both matter. **The overlaps still help** across
generators — SKT beats SK on F1's holdout, Beneficial at both horizons — so the
*ranking / direction* of the fitted signal transports. But the **absolute**
predictive quality collapses: both layouts have strongly **negative R²** on the
foreign holdout (worse than predicting the mean), because F0's calibration does
not fit F1's very different dynamics. The honest reading: the overlap signal
**exists within each generator** (5.7A, strong) more robustly than the **same
fitted model transports across generators** (5.7C, direction-only). This is a
single ordered pair (F0 → F1), not a general transfer claim.

## 7. Criterion TDI-5.7D — descriptor drift (descriptive)

Per-family holdout means of the four exact descriptors, and their across-family
range — the context that makes 5.7A demanding or easy in each family:

| Family | δ | δ̄ | s₂ | s₃ |
|---|---:|---:|---:|---:|
| F0Base | 0.9486 | 0.5824 | 1.1211 | 1.0180 |
| F1Sparse | 1.0000 | 0.8843 | 1.6609 | 1.4270 |
| F2Dense | 0.1046 | 0.0736 | 1.0059 | 0.9994 |
| F3Local | 1.0000 | 0.8641 | 3.6373 | 2.0672 |
| **range** | 0.8954 | 0.8106 | 2.6314 | 1.0678 |

The families span an enormous range of exact structure, which sharpens the
5.7A result:

- **F2Dense** has near-zero contraction (δ ≈ 0.10, δ̄ ≈ 0.07): its kernels are
  near-uniform, so the SK baseline has little exact structure to exploit and its
  5.7A test is effectively *SKT-vs-baseline*. The overlaps are still Beneficial
  (19–41%).
- **F3Local** has maximal Dobrushin contraction (δ = 1.0) **and** by far the
  richest spectral moments (s₂ = 3.64, s₃ = 2.07): the **most demanding** SK
  baseline of the four. Yet the overlaps are **most** valuable there (52–77%) —
  they carry information the strong exact descriptors miss precisely where those
  descriptors are richest.

So the signal is not an artifact of weak baselines: it survives, and is largest
under, the most structured generator.

## 8. Interpretation and boundaries

Within the exact scope, this is the strongest generality evidence in the
series. The `{O_1, O_2}`-beyond-`{contraction + spectral}` signal is Beneficial
at both focal horizons in **every** one of four structurally distinct exact
generators — sparse, dense, locally-structured, and the base — spanning
near-uniform to strongly-contracting, low- to high-spectral kernels. The
most-repeated open limitation of TDI-5.2 … 5.6 (single generator) is
substantially closed.

The result establishes exactly this and no more (preregistration Section 20).
It does **not** establish:

- **effect-size transportability** — the magnitude is strongly heterogeneous
  (5.7B), so "how much the overlaps help" is generator-dependent even though
  "whether they help" is not;
- **model transfer beyond direction** — 5.7C shows the *same fitted* F0 model
  helps on F1 only in ranking (negative absolute R²), and covers only the single
  F0 → F1 pair;
- **control against the literal spectral gap / mixing time** — s₂, s₃ are exact
  rational *moments*; the second eigenvalue `|λ₂|` and ε-mixing time remain
  transcendental/iterative and deferred to the non-exact **TDI-6** track;
- robustness across **widths** (single base width-3 + width-4 composition;
  roadmap TDI-5.8), sufficiency under **nonlinear / non-parametric** model
  families, **causal** effects, or external empirical validity.

The TDI-5.7A / TDI-5.7B / TDI-5.7C / TDI-5.7D summaries are frozen as reported.

## 9. Reproduction

    TDI57_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI57_FREEZE_RULE \
      bash scripts/reproduce-tdi5.7.sh

The script refuses without the exact token, refuses a dirty repository, verifies
the full frozen hash chain (TDI-5.1 → TDI-5.6 plus the TDI-5.7 manifests) before
any generation, executes the evaluator once with `--full`, verifies the final
criterion lines, and writes read-only result, metadata, hash and completion
artifacts under `results/tdi5.7-generator-robustness/`. Determinism is exact:
the log SHA-256
`6e8b8d89853ea674d97337bf34f213329456586987baac0d8318af71daf4d5d7` is
reproduced by any faithful re-run of commit
`c7375f652a90c4ab27e44db26087995c9cf07d7a` (frozen v57 evaluator hash
`900031bc27a35e327038911d93f10d74458f913e64d9644b225963df699049ae`) on the same
toolchain and architecture.
