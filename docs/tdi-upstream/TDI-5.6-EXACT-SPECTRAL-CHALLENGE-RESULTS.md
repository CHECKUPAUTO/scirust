# TDI-5.6 — The Exact Spectral Challenge: Confirmatory Results

## Status

This document reports the single, real, preregistered TDI-5.6 run. The
design was frozen before execution
(`docs/TDI-5.6-EXACT-SPECTRAL-CHALLENGE-PREREGISTRATION.md`) and the run was
a deliberate one-time human action under the exact confirmation token. No
classification below may be rewritten (preregistration Section 20).

**Headline.** Within the exact scope of the design, the early overlaps
`{O_1, O_2}` carry **substantial, robust predictive signal beyond an exact
contraction baseline *augmented with two exact spectral moments*** — the
Dobrushin coefficient δ, the mean pairwise total variation δ̄, **and**
`s_2 = trace(P^2)`, `s_3 = trace(P^3)` of the one-step `Noop` kernel — at
**every** horizon `U_3 … U_8`. Crucially, the spectral moments are
themselves informative (criterion TDI-5.6B), so the overlaps had to beat a
strictly stronger, genuinely predictive baseline than in TDI-5.5, and they
did — decisively and at every horizon, replicated across three fresh seed
blocks. This is the strongest evidence in the exact series that TDI's
overlap signal is not a repackaging of exact-computable contraction **or**
low-order spectral structure.

## 1. Provenance and integrity

| Item | Value |
|---|---|
| Run git commit | `f5dcb878f666d3e65139c0b9ddc8ae953a9a7a0f` |
| Evaluator (v56) SHA-256 | `0820274b3edb58a6e123c612dbed8dd8a1725221240365f142d9510404e1d1b2` |
| Preregistration SHA-256 | `59e3375b82d0bb7aad7be0591b9d1eac074d4b194678dfe0e06e73c8aac89807` |
| Scientific-manifest SHA-256 | `b34a90a6dff311422cef46f2b213880d85d2edc7c3a78f9563c059a91479fa4c` |
| Result log SHA-256 | `010b3045222221e563dc8befa10966a98b161018ee884d0f8e8c44e08ef5986e` |
| Toolchain | rustc 1.97.0 (2d8144b78 2026-07-07), cargo 1.97.0 (c980f4866 2026-06-30) |
| Host | Linux `tarek`, aarch64 (Jetson, ARM64) |
| Start / end (UTC) | 2026-07-22T22:05:34Z / 2026-07-22T22:24:38Z (~19 min) |

The run's recorded evaluator hash equals the committed frozen v56 evaluator
and the frozen `EVALUATOR.sha256` manifest (`0820274b…`); the preregistration
(`59e3375b…`) and scientific-manifest (`b34a90a6…`) hashes likewise match the
committed frozen files, and the full frozen scientific chain TDI-5.1 → TDI-5.6
(31 files, including all of `tdi-core`) verifies clean via `sha256sum -c
docs/TDI-5.6-SCIENTIFIC-CODE.sha256`. The run therefore used the exact
reviewed, frozen scientific code.

The run's read-only artifacts are committed under
`results/tdi5.6-exact-spectral-challenge/` (`…v56.log`, `…v56.log.sha256`,
`…v56.complete`, `…v56.metadata.txt`); the committed `…metadata.txt` records
the run git commit, start/end timestamps, `rustc`/`cargo` versions, `uname`,
and the cross-checked frozen hashes above. The committed log has been
reverified independently to hash to
`010b3045222221e563dc8befa10966a98b161018ee884d0f8e8c44e08ef5986e`.

`scripts/reproduce-tdi5.6.sh` verifies the entire frozen hash chain **before
any generation** and refuses a dirty repository, so the frozen scientific code
was intact at run time. Determinism is exact: the log SHA-256 above is
reproduced by any faithful re-run of commit
`f5dcb878f666d3e65139c0b9ddc8ae953a9a7a0f` on the same toolchain and
architecture.

## 2. Populations

Single generator (base width-3 + width-4 in-distribution composition),
three fresh seed blocks **J / K / L** — pairwise disjoint and disjoint from
the TDI-5.5 blocks G/H/I and all earlier blocks — with **40,000 accepted
records per block, 120,000 total**, no OOD populations (preregistration
Section 8). Models were fitted on each block's combined width-3 + width-4
training population and every criterion evaluated on that block's combined
holdout. Standardized-U space is the primary confirmatory domain; per block
and horizon a single target scaler is shared across the three layouts.

The exact contraction descriptors (Dobrushin δ, mean pairwise total
variation δ̄) and the two **exact spectral moments** were computed per
candidate system as exact rationals and converted to `f64` in a single
rounding step. The moments are the traces of the matrix powers of the
one-step `Noop` kernel `P`, written as finite closed-walk sums of the
unit-fraction transition probabilities,

    s_2 = trace(P^2) = sum_{i,j} P_ij P_ji,
    s_3 = trace(P^3) = sum_{i,j,k} P_ij P_jk P_ki,

so, unlike the second eigenvalue or the mixing time, they are **exact
rationals** — never transcendental or iterative (preregistration Sections 4,
5). `s_2 = sum_i λ_i^2` and `s_3 = sum_i λ_i^3` are power-sum functions of
the spectrum: they are a genuine, if partial, exact proxy for spectral
structure without ever computing an eigenvalue.

## 3. Three nested layouts

| Layout | Features | Count | Role |
|---|---|---:|---|
| **CK** | baseline + δ + δ̄ | 15 | contraction baseline (TDI-5.6B baseline) |
| **SK** | baseline + δ + δ̄ + s₂ + s₃ | 17 | spectral baseline (TDI-5.6A/C baseline) |
| **SKT** | baseline + δ + δ̄ + s₂ + s₃ + O₁ + O₂ | 19 | full model |

`SK − CK` isolates the marginal value of the exact spectral moments after
contraction (criterion TDI-5.6B). `SKT − SK` isolates the marginal value of
the overlaps `{O_1, O_2}` after **both** contraction and the spectral
moments (criteria TDI-5.6A, TDI-5.6C). The aggregate holdout MSE nests
cleanly, `CK ≥ SK ≥ SKT`, at both focal horizons:

| Horizon | CK MSE | SK MSE | SKT MSE |
|---|---:|---:|---:|
| U₃ | 0.342922 | 0.334825 | 0.177201 |
| U₆ | 0.289827 | 0.248475 | 0.191864 |

## 4. Criterion TDI-5.6A — signal beyond the spectral moments

**SKT** (baseline + δ + δ̄ + s₂ + s₃ + O₁ + O₂) versus **SK** (baseline +
δ + δ̄ + s₂ + s₃) on combined holdout at the focal horizons, four-way
classification with the symmetric 2% relative-MSE margin:

| Horizon | Classification | Aggregate rel-MSE reduction | Aggregate 95% CI | Blocks confirming |
|---|---|---:|---|---:|
| **U₃** | **Beneficial** | 47.08% | [0.4616, 0.4800] | 3 / 3 |
| **U₆** | **Beneficial** | 22.78% | [0.2177, 0.2382] | 3 / 3 |

At both focal horizons all three blocks individually confirm the benefit,
the aggregate relative improvement exceeds the 2% margin by an order of
magnitude, and the aggregate bootstrap lower bound is strictly positive
(U₃ aggregate SK MSE 0.334825 → SKT 0.177201, R² 0.6595 → 0.8198; U₆ SK
0.248475 → SKT 0.191864, R² 0.7501 → 0.8070). **Adding `{O_1, O_2}` on top
of the exact contraction descriptors *and* the exact spectral moments
reduces error substantially**; the overlaps are not redundant with either
descriptor family.

## 5. Criterion TDI-5.6B — marginal value of the spectral moments

**SK** versus **CK** at the focal horizons — the test that makes TDI-5.6A
demanding rather than easy:

| Horizon | Classification | Aggregate rel-MSE reduction | Aggregate 95% CI | Blocks confirming |
|---|---|---:|---|---:|
| **U₃** | **Beneficial** | 2.35% | [0.0191, 0.0280] | 3 / 3 |
| **U₆** | **Beneficial** | 14.27% | [0.1313, 0.1537] | 3 / 3 |

The exact spectral moments are **themselves informative**: adding `{s_2,
s_3}` to the contraction baseline reduces holdout error, Beneficial at both
focal horizons, 3/3 blocks, with strictly positive bootstrap lower bounds
(U₃ CK MSE 0.342922 → SK 0.334825; U₆ CK 0.289827 → SK 0.248475). This is
the decisive control for reading TDI-5.6A: the overlaps did not beat an
inert baseline padded with useless features — they beat a baseline that
already extracts real, exact spectral information, and the spectral moments'
own contribution **grows with horizon** (2.35% at U₃ to 14.27% at U₆) as
mixing behaviour becomes the dominant driver of the long-horizon deficit.

## 6. Criterion TDI-5.6C — decay law and redundancy horizon

SKT-vs-SK across the dense grid `U_3 … U_8` (base generator):

| Horizon | Aggregate rel-MSE reduction | Aggregate 95% CI | Classification |
|---|---:|---|---|
| U₃ | 47.08% | [0.4616, 0.4800] | Beneficial |
| U₄ | 35.92% | [0.3489, 0.3693] | Beneficial |
| U₅ | 28.25% | [0.2723, 0.2929] | Beneficial |
| U₆ | 22.78% | [0.2177, 0.2382] | Beneficial |
| U₇ | 18.70% | [0.1772, 0.1968] | Beneficial |
| U₈ | 16.01% | [0.1507, 0.1693] | Beneficial |

- **`monotone_non_increasing` = true** — the marginal value of `{O_1, O_2}`
  beyond contraction-plus-spectral decreases monotonically with horizon.
- **redundancy horizon `h★` = none** — the classification is **Beneficial
  at every horizon** in U₃…U₈; the marginal value never enters the ±2%
  Equivalent band in the tested range.
- successive ratios `r_{h+1}/r_h` = [0.763, 0.786, 0.807, 0.821, 0.856] —
  increasing toward 1: a **decelerating** decay that plateaus far above the
  2% margin within this range, rather than collapsing to negligibility.

The decay is slightly steeper than TDI-5.5's CKT-vs-CK profile (5.5 ratios
[0.749 … 0.858]) — expected, because part of what the overlaps supplied over
the pure contraction baseline in 5.5 is now supplied by the spectral moments
inside SK (Section 5, and see TDI-5.6B) — yet the overlaps' incremental
value remains Beneficial throughout.

## 7. Relationship to TDI-5.5 and TDI-5.4

TDI-5.5 established `{O_1, O_2}` beyond the **exact contraction** descriptors
(δ, δ̄) and a naive persistence competitor, Beneficial at every horizon
U₃…U₈. TDI-5.6 **raises the baseline**: it adds the two exact spectral
moments `s_2, s_3` — power sums of the kernel spectrum — to that contraction
baseline, and the joint overlaps survive the strengthened control, still
Beneficial at every horizon. TDI-5.6B confirms the added moments were **not
inert**: they genuinely reduce error (increasingly so at longer horizons),
so TDI-5.6A's survival is a real strengthening of TDI-5.5, not a technical
artifact of adding redundant columns.

TDI-5.4 studied **O₁ alone** under a nonlinear basis and found its marginal
value **decayed into Equivalence by U₅–U₆**. That remains consistent: O₁'s
*incremental* nonlinear contribution over O₂ fades quickly, whereas the
*joint* overlap pair carries persistent information that neither the exact
contraction descriptors nor the exact spectral moments supply across
U₃…U₈.

## 8. Interpretation and boundaries

Within the exact scope, this is the strongest evidence in the series that
TDI's overlap signal is a **candidate independent informational dimension**
rather than a compact estimator of already-known structure: it survives the
best *exact-computable* contraction descriptors **and** two exact spectral
moments that are themselves demonstrably informative, decisively and at
every horizon, replicated across three fresh seed blocks.

The result establishes exactly this and no more (preregistration Section
20). It does **not** establish:

- **control against the *literal* spectral gap or mixing time.** `s_2` and
  `s_3` are exact rational *moments* of the spectrum, a partial proxy; the
  literal second eigenvalue `|λ_2|` (spectral gap `1 − |λ_2|`) and the
  ε-threshold mixing time are transcendental / iterative and were
  deliberately deferred to the non-exact **TDI-6** track (preregistration
  Section 21; roadmap TDI-6.1). A skeptic cannot yet rule out that a
  literal spectral-gap feature would explain the signal — that is precisely
  the decisive test TDI-6.1 is meant to run. The claim here is "beyond the
  exact contraction descriptors and the exact spectral moments `s_2, s_3`,"
  and TDI-6 (Section 5) shows those moments are a genuine, if partial,
  spectral control rather than a null one;
- sufficiency under nonlinear or non-parametric model families;
  **robustness to generator changes** (single generator only — the target of
  TDI-5.7); causal effects; cross-width invariance at scale; universal
  validity across dynamical systems; or external empirical validity.

The TDI-5.6A / TDI-5.6B / TDI-5.6C summaries are frozen as reported.

## 9. Reproduction

    TDI56_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI56_FREEZE_RULE \
      bash scripts/reproduce-tdi5.6.sh

The script refuses without the exact token, refuses a dirty repository,
verifies the full frozen hash chain (TDI-5.1 → TDI-5.6) before any
generation, executes the evaluator once with `--full`, verifies the final
criterion lines, and writes read-only result, metadata, hash and completion
artifacts under `results/tdi5.6-exact-spectral-challenge/`. Determinism is
exact: the log SHA-256
`010b3045222221e563dc8befa10966a98b161018ee884d0f8e8c44e08ef5986e` is
reproduced by any faithful re-run of commit
`f5dcb878f666d3e65139c0b9ddc8ae953a9a7a0f` (frozen v56 evaluator hash
`0820274b3edb58a6e123c612dbed8dd8a1725221240365f142d9510404e1d1b2`) on the
same toolchain and architecture.
