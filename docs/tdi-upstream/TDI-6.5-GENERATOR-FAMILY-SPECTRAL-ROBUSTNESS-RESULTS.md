# TDI-6.5 — Generator-Family Spectral Robustness: Confirmatory Results

## Status

This document reports the single, real, preregistered TDI-6.5 run. The design
was frozen before execution
(`docs/TDI-6.5-GENERATOR-FAMILY-SPECTRAL-ROBUSTNESS-PREREGISTRATION.md`, SHA-256
`f44eb21446ffdc6897c76818f4d4b22ecf266cf4f2707a4a8d995b0479acd589`) and the run
was a deliberate one-time human action under the exact confirmation token. No
classification below may be rewritten.

**Headline.** TDI-6.5 asked whether TDI-6.1's result — the overlaps beat a
baseline that already contains the **literal** spectral gap and mixing time —
survives when the generator is changed. It does, without exception:
**TDI-6.5A replicates fully**, *Beneficial* at both focal horizons in all four
generator families, and all **24** cells of the family × horizon grid are
*Beneficial*. But the effect size is **strongly family-dependent** — a factor of
three between families (U₃: 37.4 % for F2-dense against 77.2 % for F3-local) —
and the reason turns out to invert the preregistration's own a-priori
reasoning: the families where the literal spectral descriptors are *least*
useful are those where `|λ₂| = 1` exactly for a large fraction of candidates, so
`g` and `τ_ε` are **censored**, not merely small. The single non-*Beneficial*
cell in the entire run is a 6.5D diagnostic, F3-local at U₃ — the one place
where the literal descriptors add nothing measurable beyond the exact moments.

## 1. Provenance and integrity

| Item | Value |
|---|---|
| Run git commit | `2876f473d5d5516b2b59f8667ef83c43a6f2a892` |
| Evaluator | `tdi-bench/src/bin/tdi-independent-overlap-ablation-v65.rs` |
| Evaluator SHA-256 | `75bd5198486e7e3c6072deebbdebd256aa3152a7b43b60054349f8e181c200f0` |
| Preregistration SHA-256 | `f44eb21446ffdc6897c76818f4d4b22ecf266cf4f2707a4a8d995b0479acd589` |
| Scientific-manifest SHA-256 | `f5eb6cf8c3a150af06eaf8e2362518751f33825184ff7d8d1bf523010dc964d1` |
| Result log SHA-256 | `8e065007c93e0a72b63cb0a422ecb895f6ff95def4dd9e88d770bc5590da6e04` |
| Toolchain | rustc 1.97.1 (8bab26f4f 2026-07-14), cargo 1.97.1 (c980f4866 2026-06-30) |
| Host | Linux `tarek`, aarch64 (Jetson, ARM64) |
| Start / end (UTC) | 2026-07-29T22:25:20Z / 2026-07-29T23:31:06Z (1 h 05 min 46 s) |

The committed log has been independently rehashed in this working tree to
`8e065007c93e0a72b63cb0a422ecb895f6ff95def4dd9e88d770bc5590da6e04`, matching the
value the run recorded for itself. The frozen TDI-5.7 and TDI-6.1 evaluators and
preregistrations were verified before any generation. Reproduction is
**tolerance-based**: `g = 1 − |λ₂|` and `τ_ε/T_max` are the only non-exact
quantities, computed in IEEE-754 binary64, single-threaded, fixed operation
order; the ±2 % classifier margin dwarfs the eigensolver tolerance (η = 1e-12).

The model is the **linear** ridge (λ = 1), unchanged from TDI-6.1 — CK 15
features, SK 17, GK 19, GKT 21. The single changed factor is the generator.

## 2. The four families and their populations

| Family | Rule |
|---|---|
| **F0-base** | uniform over all non-empty successor subsets (TDI-5.6's generator, unchanged) |
| **F1-sparse** | low out-degree `d ∈ {1,2}`, distinct successors drawn by rejection |
| **F2-dense** | high out-degree: all states minus `e ∈ {0,1}` excluded bit(s) |
| **F3-local** | local neighbourhood (Hamming ≤ 1): subset of `{s, s⊕1, s⊕2, …}`, self forced if empty |

Three fresh seed blocks per family, 40,000 accepted records each →
**120,000 per family, 480,000 total** — the largest population in the series.
548,441 candidates were attempted, so **68,441 preregistered exclusions**, and
their distribution is a finding in its own right:

| Family | Exclusions | Rate | `observation-fully-recovered` | `target-fully-recovered-h*` | `invalid-target-geometry-h8` |
|---|---:|---:|---:|---:|---:|
| F0-base | 259 | 0.22 % | 258 | 1 | 0 |
| F1-sparse | 8,050 | 6.29 % | 5,527 | 2,523 | 0 |
| F2-dense | **58,282** | **32.69 %** | 56,873 | 1,405 | 4 |
| F3-local | 1,850 | 1.52 % | 1,780 | 70 | 0 |

F2-dense rejects roughly one candidate in three: when almost every state
transitions to almost every other, the observation is fully recovered and the
candidate is degenerate. F1-sparse is the only family where *target* recovery is
a major category (2,523, reaching out to h₈) — with out-degree 1–2 the dynamics
frequently collapse onto a cycle. The `invalid-target-geometry-h8` category,
dead in every prior experiment, fires 4 times here, all in F2-dense.

## 3. Criterion TDI-6.5A — full replication across generators

GKT vs GK on each family's combined holdout, at the focal horizons:

| Family | U₃ | U₆ |
|---|:--:|:--:|
| F0-base | **Beneficial** | **Beneficial** |
| F1-sparse | **Beneficial** | **Beneficial** |
| F2-dense | **Beneficial** | **Beneficial** |
| F3-local | **Beneficial** | **Beneficial** |

**`replication = yes`** — Beneficial at both focal horizons for all four
families. With effect sizes and 95 % bootstrap intervals:

| Family | U₃ reduction (95 % CI) | U₆ reduction (95 % CI) |
|---|---|---|
| F0-base | 47.76 % [46.88, 48.60] | 28.04 % [27.10, 28.95] |
| F1-sparse | 57.81 % [56.95, 58.68] | 31.01 % [30.07, 31.91] |
| F2-dense | 37.35 % [36.58, 38.09] | 16.83 % [16.07, 17.53] |
| F3-local | **77.17 %** [76.62, 77.71] | **53.42 %** [52.53, 54.31] |

Standardized-U `R²` (GK → GKT) shows how differently hard the four problems
are: F0-base 0.680 → 0.833 at U₃; F1-sparse 0.326 → 0.715; F2-dense 0.811 →
0.882; F3-local 0.266 → 0.832. In F3-local the GK baseline barely works at all
(`R²` 0.266, Spearman 0.532) and adding the two overlaps takes it to 0.832 /
0.924.

**An unplanned replication of TDI-6.1.** F0-base is TDI-6.1's generator under
the same linear model, differing only in seed blocks (M/N/O there, F0-base/b0–b2
here). TDI-6.1 measured 46.98 % at U₃ and 27.46 % at U₆; TDI-6.5 measures
47.76 % and 28.04 % on fresh seeds. That agreement to within ~0.8 pp is not a
preregistered criterion of either experiment, but it is a genuine independent
re-measurement of 6.1's headline, and it came out the same.

## 4. Criterion TDI-6.5B — the effect size is *not* family-invariant

| Horizon | min | max | range | all four above the 2 % margin |
|---|---:|---:|---:|:---:|
| U₃ | 37.35 % (F2-dense) | 77.17 % (F3-local) | **39.82 pp** | yes |
| U₆ | 16.83 % (F2-dense) | 53.42 % (F3-local) | **36.59 pp** | yes |

This is the preregistered caveat firing. Compare TDI-5.8, where the same
quantity across *widths* agreed to 0.61 pp at U₆: across *generators* the spread
is sixty times larger. The direction is consistent (F3-local > F1-sparse >
F0-base > F2-dense at both horizons), and every family decays monotonically
across the full grid:

| Horizon | F0-base | F1-sparse | F2-dense | F3-local |
|---|---:|---:|---:|---:|
| U₃ | 47.76 % | 57.81 % | 37.35 % | 77.17 % |
| U₄ | 37.57 % | 43.91 % | 24.93 % | 69.30 % |
| U₅ | 31.94 % | 36.04 % | 21.54 % | 58.95 % |
| U₆ | 28.04 % | 31.01 % | 16.83 % | 53.42 % |
| U₇ | 24.86 % | 26.51 % | 12.71 % | 47.49 % |
| U₈ | 22.41 % | 24.07 % | 11.05 % | 44.20 % |

All **24 cells Beneficial**; no redundancy horizon in any family. So the
*existence* of the signal is generator-robust; its *magnitude* is not, and
nothing here licenses transporting an effect size measured on one generator to
another.

## 5. Criterion TDI-6.5D — descriptor drift, and why the prereg's reasoning inverted

Holdout means of the six descriptors per family:

| Family | δ | δ̄ | s₂ | s₃ | `g` | `τ_ε` |
|---|---:|---:|---:|---:|---:|---:|
| F0-base | 0.949434 | 0.582330 | 1.121374 | 1.018715 | 0.640727 | 0.001165 |
| F1-sparse | 1.000000 | 0.883850 | 1.651142 | 1.435296 | 0.134245 | 0.370652 |
| F2-dense | 0.104561 | 0.073360 | 1.005902 | 0.999413 | 0.925750 | 0.000244 |
| F3-local | 1.000000 | 0.863726 | 3.638077 | 2.069930 | 0.091443 | 0.398642 |
| **range** | **0.895439** | **0.810490** | **2.632175** | **1.070516** | **0.834307** | **0.398398** |

The drift is enormous — `s₂` spans 1.006 to 3.638, δ spans 0.105 to exactly
1.000. These are, in descriptor space, four genuinely different worlds, which is
what makes §3's unanimous replication meaningful.

The per-family GK-vs-SK diagnostic — the marginal value of the *literal*
spectral descriptors beyond the exact moments, i.e. the per-family analogue of
TDI-6.1B:

| Family | U₃ | U₆ |
|---|---|---|
| F0-base | 6.01 % **Beneficial** | 35.52 % **Beneficial** |
| F1-sparse | 3.86 % **Beneficial** | 23.42 % **Beneficial** |
| F2-dense | **13.71 %** **Beneficial** | **50.45 %** **Beneficial** |
| F3-local | **1.46 %** — ***Equivalent*** | 6.23 % **Beneficial** |

**F3-local at U₃ is the only non-*Beneficial* cell in the entire run.** There,
the literal `|λ₂|` and `τ_ε` add nothing measurable beyond the exact moments, so
6.5A's F3-local/U₃ test is effectively GKT-vs-SK rather than the intended
GKT-vs-GK — which is exactly the caveat the preregistration said to look for,
and it is the cell with the largest overlap gain (77.17 %). That is not a
coincidence to be waved away: where the added baseline block is inert, the
overlaps have more room.

**The preregistration's a-priori reasoning is inverted by the data, and the
inversion is informative.** Section 19 predicted that a *near-uniform* family
(δ, δ̄, s₂ small, `g` large) would "offer the descriptors little to work with",
and that a *strong-contraction / slow-mixing* family (`g` small, `τ_ε` large)
would make 6.5A demanding. The observed ordering is the opposite on both counts:
F2-dense is the near-uniform family (δ 0.105, s₂ 1.006, `g` 0.926) and it is
where the literal descriptors help *most* (13.71 % / 50.45 %); F3-local and
F1-sparse are the slow-mixing families (`g` 0.091 / 0.134, `τ_ε` 0.399 / 0.371)
and they are where the descriptors help *least*.

The spectral cross-validation table (§6) supplies a consistent mechanism: what
matters is not the *level* of `g` but whether it is **censored**. Of the six
sampled F3-local candidates, four have `|λ₂| = 1.000000000` exactly — so
`g = 0` and `τ_ε/T_max = 1.000000`, meaning the ε-mixing criterion was never met
within the 4096-iteration ceiling. Four of six F1-sparse candidates are the
same. For those candidates `g` and `τ_ε` are not small measurements but
saturated ones: they record "does not mix", with no resolution beyond that. The
family-mean `τ_ε` values (0.399, 0.371) are consistent with roughly 37–40 % of
each population sitting at the ceiling. F2-dense, by contrast, shows no censored
candidate in the sample and `|λ₂|` varying over 0.017–0.143 — genuinely
informative variation.

This is offered as a **consistent explanation, not a demonstrated cause**: the
experiment measures the drift, the censoring in a 6-candidate-per-family
diagnostic sample, and the GK-vs-SK values, but it does not test the link
between them. A design that varied `T_max` and re-measured would.

## 6. The spectral cross-validation, and a sharper version of TDI-6.2's caveat

| Family | max trace residual, method 1 |
|---|---:|
| F0-base | 1.26e-13 |
| F1-sparse | 1.85e-12 |
| F2-dense | 2.07e-14 |
| F3-local | 1.23e-12 |

The rigorous witness is again the trace identity `Σλᵏ = trace(Pᵏ)`, at machine
level in all four families. The method-1 ↔ method-2 disagreement reaches
**4.31e-1** here — far worse than TDI-6.2's 7.48e-2 — and TDI-6.5 makes the
reason **checkable rather than merely argued**: in four sampled candidates,
deflated power iteration (method 2) returns `|λ₂|` values of **1.046, 1.225,
1.007 and 1.042**. For a row-stochastic matrix every eigenvalue satisfies
`|λ| ≤ 1`, so those values are impossible. Where the two methods disagree, it is
**method 2 that is wrong** — this is not an appeal to which method one trusts.
Method 1 never exceeds 1.000000000 anywhere in the table.

Method 2 remains a useful cross-check on real-spectrum kernels, where the
known-spectrum battery in the test suite verifies 1 ↔ 2 agreement to 1e-9 with
method 3 (reference crate) as the third leg. On non-symmetric candidates with a
complex `λ₂` it is simply out of scope, and this run makes that visible.

## 7. Criterion TDI-6.5C — cross-generator transfer: calibration dies, ordering survives

The GK and GKT models fitted on F0-base, evaluated on F1-sparse's holdout:

| Horizon | Layout | MSE (std-U) | `R²` (std-U) | Spearman (std-U) | `R²` (recon-O) | Spearman (recon-O) |
|---|---|---:|---:|---:|---:|---:|
| U₃ | GK | 1.709216 | **−2.174** | 0.404 | −0.626 | 0.403 |
| U₃ | GKT | 1.219813 | **−1.265** | **0.694** | **+0.340** | **0.771** |
| U₆ | GK | 1.084354 | **−1.659** | 0.427 | −1.426 | 0.405 |
| U₆ | GKT | 0.886480 | **−1.174** | **0.596** | −0.746 | 0.593 |

Both cells classify as *Beneficial* (relative MSE reduction 28.61 %
[27.38, 29.82] at U₃, 18.22 % [17.13, 19.32] at U₆), **and that label needs the
same care TDI-5.8B needed.** Every standardized-U `R²` is below zero: both
layouts are worse than predicting the target's mean, so the *fitted scale* does
not carry from F0-base to F1-sparse. Reporting "Beneficial" without that context
would invert the finding.

What the Spearman column adds is that transfer is not uniformly dead. The
overlaps carry substantial **ordering** information across generators (0.694 at
U₃, 0.596 at U₆) where the baseline carries much less (0.404, 0.427).

**One difference from TDI-5.8B worth stating precisely.** In TDI-5.8's
cross-width transfer *every* `R²` was negative, in both spaces. Here, in
reconstructed-O space at U₃, GKT's `R²` is **positive (+0.340)** — the only
transfer cell in either experiment that beats the mean. Cross-generator transfer
of the overlap model is therefore *degraded but not uniformly worthless* at the
short horizon, which is a slightly better outcome than cross-width transfer, not
merely the same shape. At U₆ it is negative again (−0.746), so the improvement
does not persist with horizon.

## 8. Interpretation and boundaries

TDI-6.5 does for generators what TDI-5.8 did for widths, and lands in the same
two-part shape: **the signal replicates completely; the calibration does not
transport.** It additionally establishes that the *magnitude* of the effect is
strongly generator-dependent, and it locates a family (F3-local) where the
literal spectral control is inert at the short horizon and must be reported as
such.

What this does **not** establish:

- **transportable effect sizes.** A 77 % reduction on F3-local and a 37 %
  reduction on F2-dense are the same qualitative finding and very different
  quantitative ones. No number here should be quoted as "the" effect size;
- **that the censoring explains the 6.5D ordering** — §5 gives a consistent
  mechanism, not a tested one;
- four hand-specified families are not "generators in general"; widths 3–4 only;
  linear ridge (the nonlinear control is TDI-6.2, on F0-base only); one-step
  Noop kernel; `ε = 1/4`, `T_max = 4096`;
- no causal claim (TDI-6.4), no information decomposition (TDI-6.3), no
  cross-width claim (TDI-5.8);
- nothing here was tested outside small synthetic finite-state families, and no
  universal law is claimed.

The TDI-6.5A / TDI-6.5B / TDI-6.5C / TDI-6.5D summaries are frozen as reported.

## 9. Reproduction

    TDI65_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI65_FREEZE_RULE \
      bash scripts/reproduce-tdi6.5.sh

The script refuses without the exact token, refuses a dirty repository, verifies
the frozen hash chain before any generation, executes the evaluator once with
`--full`, verifies the final criterion lines, and writes read-only artifacts
under `results/tdi6.5-family-spectral-robustness/`. F2-dense's 33 % rejection
rate dominates the wall-clock; budget roughly 1 h 15.
