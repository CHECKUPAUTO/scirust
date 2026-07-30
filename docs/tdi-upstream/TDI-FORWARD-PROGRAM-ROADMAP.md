# TDI — Forward Program Roadmap (DRAFT for review)

> **Status: DRAFT.** This document is a design roadmap, not a preregistration
> and not frozen. It exists so the next experiments can be chosen and scoped
> before any is frozen or built. Nothing here authorizes a run; every
> experiment below still requires its own frozen preregistration, evaluator,
> manifests, CI and the deliberate human `--full` action before it produces a
> result.

## 1. What the exact series has established (TDI-5.2 → TDI-5.6)

Within the frozen exact discipline — bit-exact rational arithmetic,
deterministic seed-block generation, ridge `lambda = 1.0`, paired +
stratified-aggregate bootstrap, the symmetric ±2% four-way classifier — and
within the **single base generator** (splitmix64 successor masks drawn
uniformly over all non-empty successor subsets, widths 3 and 4):

| Exp | Question | Headline (per its frozen RESULTS) |
|---|---|---|
| 5.2 | Joint `{O_1,O_2}` beyond an entropy/topology baseline | Beneficial at the width-3 population |
| 5.3 | Independent activation of `O_1` / `O_2` | (per its frozen result) |
| 5.4 | Nonlinear-basis sufficiency; `O_1`-alone decay | `O_1`'s marginal value decays into Equivalence by U₅–U₆ |
| 5.5 | Beyond the **exact contraction** descriptors (Dobrushin δ, δ̄) **and** naive persistence | Beneficial at **every** horizon U₃…U₈ |
| 5.6 | Beyond the **exact spectral moments** `s₂,s₃` (on top of δ, δ̄) | (awaiting the real run) |

The recurring, explicitly-stated limitations (every prereg Section 20) are the
map of what remains:

1. **Single generator** — no robustness to generator changes is established.
2. **Non-exact controls** — the *literal* spectral gap `1−|λ₂|` and the
   ε-mixing time are transcendental / iterative and were deferred.
3. **Nonlinear / non-parametric model families** — deferred.
4. **Causal effects** — never tested (everything is predictive/associational).
5. **Cross-width invariance** — width-3↔4 transfer was weak (5.2D); widths
   5–6 untested at scale.

## 2. Two tracks forward

The remaining frontiers split cleanly by whether they preserve the bit-exact
invariant that defines the current program.

### Track A — the **exact** continuation (stays TDI-5.x)

These keep bit-exact rational arithmetic and the frozen determinism
discipline. They are the low-risk, high-confidence continuations.

- **TDI-5.7 — Generator robustness** *(recommended next build; full draft
  preregistration in `docs/TDI-5.7-GENERATOR-ROBUSTNESS-PREREGISTRATION.md`)*.
  Does the `{O_1,O_2}`-beyond-`{contraction+spectral}` signal (5.6's
  SKT-vs-SK) replicate across a **family of distinct exact generators**, not
  just the single base generator? Attacks limitation (1) directly. Fully
  buildable on the frozen `tdi-core` primitives — only the (non-frozen)
  mask-generation rule changes per family.
- **TDI-5.8 — Cross-width invariance (exact).** A dedicated, honest test of
  whether the signal and its calibration transfer across widths 3→4→5→6,
  turning the weak 5.2D OOD observation into a first-class criterion. Attacks
  limitation (5). Heavier (width-5/6 generation), but exact.
- **TDI-5.9 — Higher exact moments / exact descriptor saturation.** Add
  `s₄ = trace(P⁴)` (and the exact return-probability profile) to SK; measure
  whether the overlaps' marginal value shrinks as the exact descriptor set is
  saturated — the exact-side ceiling of the "is TDI redundant?" question.

### Track B — the **non-exact** frontier (TDI-6, a new identifier & discipline)

TDI-6 is **not** a TDI-5.x experiment. It abandons bit-exactness for the
descriptors/models the exact track cannot express, and therefore needs its own
**non-exact determinism discipline**: fixed training seeds, a declared
floating-point + threading regime, tolerance-based (not byte-exact)
reproduction, and — where required — new dependencies (an eigensolver, a
tree/kernel library). Each TDI-6 sub-experiment carries its own frozen
preregistration under the TDI-6 identifier.

- **TDI-6.1 — Literal spectral gap & mixing time.** The decisive skeptic test
  5.6 could only approximate: does `{O_1,O_2}` survive the *actual* second
  eigenvalue `|λ₂|` (spectral gap `1−|λ₂|`) and the ε-threshold mixing time as
  features? Needs an eigensolver (new dep) and a declared FP tolerance.
  Attacks limitation (2).
- **TDI-6.2 — Non-parametric model families.** Replace ridge with gradient-
  boosted trees / kernel ridge / a small MLP: is the overlap signal a linear
  artifact, or does it survive a flexible learner (and does a flexible learner
  on the baseline subsume it)? Attacks limitation (3).
- **TDI-6.3 — Information decomposition (PID).** A formal
  unique/redundant/synergistic split of what `O_1` and `O_2` each contribute
  about `U_h` (Williams–Beer / Bertschinger PID, or transfer entropy). Turns
  the "which overlap carries what" question from ridge coefficients into an
  information-theoretic decomposition.
- **TDI-6.4 — Causal probe.** Move from prediction to intervention: use the
  existing exact perturbation machinery to ask whether the overlap-associated
  structure is causal for recovery, or merely predictive. The hardest and
  most valuable frontier; needs its own careful design.

## 3. Recommended sequence

1. **TDI-5.7 (generator robustness)** — highest value per unit risk: it stays
   exact, reuses everything frozen, and closes the single most-repeated
   limitation. Draft preregistration is ready for review now.
2. **TDI-6.1 (literal spectral gap)** — the decisive non-exact control the
   series has been explicitly pointing at since 5.5/5.6. Start its
   preregistration + the non-exact determinism discipline once 5.7's result is
   in. This is the first TDI-6 experiment and sets the TDI-6 conventions.
3. Then **TDI-5.8 / TDI-6.2 / TDI-6.3 / TDI-6.4** as the questions each prior
   result sharpens.

## 4. Standing invariants (unchanged for the exact track)

- Frozen files (TDI-5.1…5.6 evaluators, preregistrations, all SHA-256
  manifests, and all of `tdi-core`) are never modified. New exact experiments
  derive a fresh evaluator, add fresh disjoint seed blocks and bootstrap
  seeds, and pin the full frozen ancestor chain.
- No `#[allow(...)]`; clippy `-D warnings` and `cargo fmt --check` pass; dead
  code is removed, not suppressed.
- The real `--full` run is always a deliberate human action behind an exact
  confirmation token; no commit, test, or CI supplies it, and the authoring
  agent never runs it.
- TDI-6 relaxes **only** bit-exactness, and only under an explicit,
  preregistered non-exact determinism discipline — never silently.

## 5. What needs your decision

- Confirm **TDI-5.7** as the next build (or redirect to another Track A / B
  item first).
- For **TDI-6**, confirm the discipline shift is acceptable in principle
  (new dependencies such as an eigensolver; tolerance-based reproduction
  instead of byte-exact) before its first preregistration is frozen.
