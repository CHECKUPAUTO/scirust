# TDI-5.9 — Higher Exact Moments and Descriptor Saturation: Confirmatory Results

## Status

This document reports the single, real, preregistered TDI-5.9 run. The
design was frozen before execution
(`docs/TDI-5.9-SPECTRAL-MOMENT-SATURATION-PREREGISTRATION.md`,
SHA-256 `1fd2db07cfcad98c3c56b99270239a2cc5297fe3401846eb1a633b645177cab8`)
and the run was a deliberate one-time human action under the exact
confirmation token. No classification below may be rewritten
(preregistration Section 21).

**Headline.** The run answers the roadmap's "exact-side ceiling" question
with a clean asymmetry: **the exact spectral descriptor ladder is
saturating, and the overlap signal is not.** Adding the fourth exact moment
`s_4 = trace(P^4)` to the exact descriptor set buys far less than the second
and third moments did — 14% as much at U₃ and 26% as much at U₆, with
non-overlapping 95% bootstrap intervals at both focal horizons (criterion
TDI-5.9D, `saturating = true` at both). Yet the early overlaps `{O_1, O_2}`
remain **Beneficial at every horizon U₃ … U₈** against that strictly richer
control, with **no redundancy horizon** and a decay profile essentially
identical to TDI-5.6's (criteria TDI-5.9A, TDI-5.9C). The confound is
running out of road; the signal is not.

## 1. Provenance and integrity

| Item | Value |
|---|---|
| Run git commit | `79d7f11ff7e7991b60abb5bfbee5af9b5b17e049` |
| Evaluator (v59) SHA-256 | `ffa55883c7a0f87e864f817e40518d3021281ae4886d7366c5f7318b1985be8e` |
| Preregistration SHA-256 | `1fd2db07cfcad98c3c56b99270239a2cc5297fe3401846eb1a633b645177cab8` |
| Scientific-manifest SHA-256 | `0522e5543121bd116b55f351b36ab694e3646cd1b8f84fa827183ab43a4a2419` |
| Result log SHA-256 | `4a74257ecf08803ae57fd5b436e878a8c3285a0c83eb1fb696dfa9d9b8ccc7b7` |
| Toolchain | rustc 1.97.1 (8bab26f4f 2026-07-14), cargo 1.97.1 (c980f4866 2026-06-30) |
| Host | Linux `tarek` 6.8.12-tegra, aarch64 (Jetson, ARM64) |
| Start / end (UTC) | 2026-07-28T21:28:50Z / 2026-07-28T21:55:29Z (~27 min) |

The run git commit `79d7f11f` is exactly the merge commit of the pull request
that froze this experiment's evaluator and manifests, so the run executed the
merged, reviewed tree with nothing added on top.

The evaluator hash recorded by the run equals the committed frozen v59
evaluator and the frozen `EVALUATOR.sha256` manifest (`ffa55883…`), the
preregistration hash equals the committed frozen document (`1fd2db07…`), and
the scientific-manifest hash equals the committed
`docs/TDI-5.9-SCIENTIFIC-CODE.sha256` (`0522e554…`) — all three reverified
independently in this working tree against the run's own metadata. The
committed log has been independently rehashed to
`4a74257ecf08803ae57fd5b436e878a8c3285a0c83eb1fb696dfa9d9b8ccc7b7`, matching
the value the run recorded for itself. Because
`scripts/reproduce-tdi5.9.sh` verifies the **entire** frozen chain — all
thirteen ancestors (TDI-5.1 → 5.8, 6.1, 6.2, 6.3, 6.5, 6.4) plus TDI-5.9's
own three manifests — before any generation, and refuses a dirty repository,
the run necessarily used the exact reviewed, frozen scientific code. TDI-5.9
additionally verifies that chain **as a hard runtime gate inside the
evaluator** in both `--preflight` and `--full` (new in this experiment; every
prior evaluator only printed the chain), so the freeze was enforced twice
over, independently, at run time.

Reproduction is **byte-exact**, not tolerance-based (preregistration Sections
4.1, 20): TDI-5.9 stays entirely on the bit-exact `ExactRatio` track, so the
log SHA-256 above is reproduced exactly by any faithful re-run on the same
toolchain and architecture.

## 2. Populations and the one changed factor

Single generator (base width-3 + width-4 in-distribution composition), three
fresh seed blocks **Y / Z / AA** — pairwise disjoint and disjoint from every
prior block (TDI-6.4 consumes seeds to ≈ 9.23×10⁹; TDI-5.9 starts at
1.0×10¹⁰) — with **40,000 accepted records per block, 120,000 total**, no OOD
populations (preregistration Section 8).

All twelve populations reached their requested count exactly (15,000 /
5,000 / 15,000 / 5,000 per block); 120,242 candidates were attempted for
120,000 accepted, i.e. **242 preregistered exclusions** (Y 89, Z 70, AA 83).
Every exclusion occurred at width 3 — 241 `observation-fully-recovered` and a
single `target-fully-recovered-h3` (block Y, training-w3); width 4 produced
**zero** exclusions in all three blocks. Generation ran far inside its
deterministic budgets: the worst-case population consumed 15,066 of 960,000
permitted attempts (1.57%), so no result is near a termination limit. Each
population's consumed seed range sits well inside its 10,000,000-wide
reservation (largest span 15,066), so the Y/Z/AA reservations are pairwise
disjoint with a very large margin, as the evaluator independently verifies at
run time.

The single changed factor versus the frozen TDI-5.6 ancestor is one
descriptor: the fourth exact spectral moment, added as one more nested
closed-walk loop over the same already-exact machinery,

    s_4 = trace(P^4) = sum over ordered quadruples (i,j,k,l) with j a
          successor of i, k of j, l of k and i of l, of 1/(d_i d_j d_k d_l),

accumulated as an exact rational and rounded to `f64` in a single final step.
Like `s_2` and `s_3`, it is a power sum of the spectrum (`s_4 = Σ_i λ_i^4`) —
genuine spectral information obtained without ever computing an eigenvalue.

## 3. Four nested layouts

| Layout | Features | Count | Role |
|---|---|---:|---|
| **CK** | baseline + δ + δ̄ | 15 | contraction baseline (5.9D(a) baseline) |
| **SK** | baseline + δ + δ̄ + s₂ + s₃ | 17 | TDI-5.6's spectral baseline (5.9B baseline) |
| **SK4** | baseline + δ + δ̄ + s₂ + s₃ + s₄ | 18 | full exact spectral baseline (5.9A/C baseline) |
| **SKT4** | SK4 + O₁ + O₂ | 20 | full model |

The aggregate holdout MSE nests cleanly and strictly, `CK ≥ SK ≥ SK4 ≥ SKT4`,
at both focal horizons:

| Horizon | CK MSE | SK MSE | SK4 MSE | SKT4 MSE |
|---|---:|---:|---:|---:|
| U₃ | 0.347263 | 0.338423 | 0.337208 | 0.176784 |
| U₆ | 0.290546 | 0.248585 | 0.239127 | 0.183297 |

Each successive increment is smaller than the one before it — the saturation
that criterion TDI-5.9D quantifies formally in Section 6.

## 4. Criterion TDI-5.9A — signal beyond the full exact spectral profile

**SKT4** versus **SK4** on combined holdout at the focal horizons, four-way
classification with the symmetric 2% relative-MSE margin:

| Horizon | Classification | Aggregate rel-MSE reduction | Aggregate 95% CI | Blocks confirming |
|---|---|---:|---|---:|
| **U₃** | **Beneficial** | 47.5741% | [46.6374%, 48.4929%] | 3 / 3 |
| **U₆** | **Beneficial** | 23.3474% | [22.3244%, 24.3198%] | 3 / 3 |

At both focal horizons all three blocks individually confirm the benefit, the
aggregate relative improvement exceeds the 2% margin by an order of
magnitude, and the aggregate bootstrap lower bound is strictly positive
(U₃ aggregate SK4 MSE 0.337208 → SKT4 0.176784; U₆ SK4 0.239127 → SKT4
0.183297, R² 0.7660 → 0.8206).

**The overlaps carry predictive information that neither the exact
contraction descriptors nor any of the three exact spectral moments
express.** This is the strongest exact-track control the series has applied,
and the signal survives it intact.

## 5. Criterion TDI-5.9B — marginal value of the fourth moment

**SK4** versus **SK** at the focal horizons — the control that determines how
demanding TDI-5.9A actually is:

| Horizon | Classification | Aggregate rel-MSE reduction | Aggregate 95% CI |
|---|---|---:|---|
| **U₃** | **Equivalent** | 0.3589% | [0.1826%, 0.5203%] |
| **U₆** | **Beneficial** | 3.8049% | [3.1883%, 4.3934%] |

The U₃ result deserves precision, because "Equivalent" here does **not** mean
"no effect". The bootstrap interval excludes zero — the fourth moment
produces a real, statistically detectable improvement — but the entire
interval lies inside the ±2% equivalence band, so the effect is practically
negligible and the preregistered classifier correctly returns Equivalent.
This is exactly the distinction the margin exists to draw.

The fourth moment's value **grows with horizon** (0.36% → 3.80%), the same
qualitative pattern TDI-5.6B found for the `{s_2, s_3}` pair (2.35% → 14.27%)
— but uniformly, and substantially, smaller. That contrast is the saturation
signature, measured formally next.

## 6. Criterion TDI-5.9D — descriptor saturation

The experiment's genuinely new criterion. Both comparisons below are computed
on the **identical** TDI-5.9 holdout populations under the identical
deterministic bootstrap, so the contrast is not confounded by sampling
variation across different seed blocks (preregistration Section 4.4):

| Horizon | (a) SK over CK — the `{s₂,s₃}` pair | (b) SK4 over SK — `s₄` alone | (b)/(a) | `saturating` | intervals overlap |
|---|---:|---:|---:|:---:|:---:|
| **U₃** | 2.5457% [2.1154, 3.0006] | 0.3589% [0.1826, 0.5203] | 0.141 | **true** | **no** |
| **U₆** | 14.4419% [13.3687, 15.5173] | 3.8049% [3.1883, 4.3934] | 0.264 | **true** | **no** |

`saturating = true` at both focal horizons, and — the stronger statement —
the two 95% bootstrap intervals are **disjoint** at both horizons, so the
inequality is not an artifact of noise. The fourth moment delivers roughly
one seventh of the pair's value at U₃ and one quarter at U₆.

**Post-hoc note (not a preregistered quantity).** TDI-5.9D is deliberately an
asymmetric comparison — a two-feature increment against a one-feature
increment — and the preregistration says so explicitly rather than hiding it.
Normalising (a) per added feature is arithmetic on preregistered quantities,
not a new measurement, and the conclusion survives it: (a)/2 = 1.27% versus
(b) = 0.36% at U₃, and 7.22% versus 3.80% at U₆. Saturation holds on either
reading, though the margin narrows at U₆.

## 7. Criterion TDI-5.9C — decay law, and an unplanned replication of TDI-5.6

SKT4-vs-SK4 across the dense grid, beside TDI-5.6's independently-measured
SKT-vs-SK profile on different seed blocks:

| Horizon | TDI-5.9 (SKT4 vs SK4) | Aggregate 95% CI | TDI-5.6 (SKT vs SK) | Difference | Classification |
|---|---:|---|---:|---:|---|
| U₃ | 47.5741% | [46.6374%, 48.4929%] | 47.08% | +0.49 pp | Beneficial |
| U₄ | 36.3379% | [35.3591%, 37.3192%] | 35.92% | +0.42 pp | Beneficial |
| U₅ | 28.3884% | [27.3443%, 29.3988%] | 28.25% | +0.14 pp | Beneficial |
| U₆ | 23.3474% | [22.3244%, 24.3198%] | 22.78% | +0.57 pp | Beneficial |
| U₇ | 19.3296% | [18.3348%, 20.2904%] | 18.70% | +0.63 pp | Beneficial |
| U₈ | 16.4629% | [15.5346%, 17.3918%] | 16.01% | +0.45 pp | Beneficial |

- **`monotone_non_increasing` = true.**
- **redundancy horizon `h★` = none** — Beneficial at every horizon in U₃…U₈;
  the marginal value never enters the ±2% Equivalent band.
- successive ratios `r_{h+1}/r_h` = [0.7638, 0.7812, 0.8224, 0.8279, 0.8517]
  — increasing toward 1: a **decelerating** decay that plateaus far above the
  margin, rather than collapsing to negligibility.

Two independent replications fall out of this run, neither of them planned as
a criterion:

1. **The decay profile reproduces.** Every one of TDI-5.6's six point
   estimates falls **inside** the corresponding TDI-5.9 95% bootstrap
   interval — all six horizons, on entirely fresh seed blocks, against a
   richer baseline. The uniformly positive offsets (+0.14 to +0.63 pp) are
   **not** evidence that a richer baseline *raises* the overlaps' value: the
   two experiments used different seed blocks, so this is an uncontrolled
   cross-experiment comparison, and since every 5.6 estimate lies within
   5.9's interval the differences are not separable from block-to-block
   variation. The honest reading is that the profile is unchanged.
2. **TDI-5.6B reproduces.** Criterion 5.9D(a) is a fresh re-measurement of
   TDI-5.6B's SK-vs-CK comparison on blocks Y/Z/AA: it returns 2.5457% at U₃
   and 14.4419% at U₆, against TDI-5.6B's 2.35% and 14.27% measured on blocks
   J/K/L. Each point estimate sits inside the other experiment's confidence
   interval, in **both** directions, at both horizons. A year-defining result
   this is not — but it is a clean, independent confirmation that the
   TDI-5.6B finding was not a seed-block artifact, and it is what licenses
   using the fresh 5.9D(a) measurement as the saturation baseline in
   Section 6.

## 8. Interpretation and boundaries

The exact track set out to answer whether TDI's overlap signal is a
repackaging of exact-computable structure. TDI-5.9 closes that line with the
sharpest available answer: the overlaps survive the richest exact spectral
control the program can construct, at every horizon, while the control itself
is visibly exhausting its returns. Extrapolating the saturation trend, a
fifth or sixth exact moment would be expected to buy less still — which is
why the preregistration placed moments of order ≥ 5 explicitly out of scope
(Section 4.2) rather than leaving an open-ended ladder.

The result establishes exactly this and no more (preregistration Section 21).
It does **not** establish:

- **control against the *literal* spectral gap or mixing time.** `s_2, s_3,
  s_4` are exact rational *moments* — a partial proxy. The literal second
  eigenvalue and the ε-threshold mixing time are transcendental / iterative
  and belong to the non-exact TDI-6.1 track. TDI-5.9 strengthens the exact
  proxy and shows it saturating; it does not close the literal question;
- control against spectral moments of order ≥ 5 (Section 4.2);
- sufficiency under nonlinear or non-parametric model families (TDI-6.2); a
  formal information decomposition (TDI-6.3); causal effects (TDI-6.4);
  robustness to generator changes (TDI-5.7, TDI-6.5); cross-width invariance
  (TDI-5.8); universal validity across dynamical systems; or external
  empirical validity.

One process caveat, recorded for the reader rather than buried: TDI-5.9's
independent adversarial review was cut short by an account spend limit before
it completed, and the remaining review was performed by the same agent that
commissioned the build. Self-review is weaker than independent review. The
frozen-hash chain, the bit-exact reproduction guarantee, and the two
unplanned replications in Section 7 all constrain what could have gone
undetected, but the asymmetry is real and is not claimed away.

The TDI-5.9A / TDI-5.9B / TDI-5.9C / TDI-5.9D summaries are frozen as
reported.

## 9. Reproduction

    TDI59_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI59_FREEZE_RULE \
      bash scripts/reproduce-tdi5.9.sh

The script refuses without the exact token, refuses a dirty repository,
verifies the full frozen hash chain (thirteen ancestors plus TDI-5.9) before
any generation, executes the evaluator once with `--full`, verifies the final
criterion lines, and writes read-only result, metadata, hash and completion
artifacts under `results/tdi5.9-spectral-moment-saturation/`. Determinism is
exact: the log SHA-256
`4a74257ecf08803ae57fd5b436e878a8c3285a0c83eb1fb696dfa9d9b8ccc7b7` is
reproduced by any faithful re-run on the same toolchain and architecture.
