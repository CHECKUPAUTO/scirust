# Numerical-Representation Research Program (TSA → ATRA → CANR → ANEE) — Closing Synthesis

**Date:** 2026-07-18
**Status:** Program **closed**. This document is the entry point and final record for the
four-phase investigation; it consolidates, and does not modify, the phase documents it links.
**Phase documents (this directory):**
`TSA_TRANSFORMED_SCALAR_ALGORITHMS_2026-07-16.md` **[TSA]**,
`ATRA_ADAPTIVE_TRANSFORMED_REPRESENTATIONS_2026-07-16.md` **[ATRA]**,
`CANR_CERTIFIED_ADAPTIVE_REPRESENTATIONS_2026-07-16.md` **[CANR]**,
`ANEE_ADAPTIVE_NUMERICAL_EXECUTION_ENGINE_2026-07-17.md` **[ANEE]** (verdict + Addenda 1–3).
**Merged pull requests:** #627 (ANEE evaluation), #635 (Phase C kernel 1), #653 (kernel 2
replication), #662 (Addendum 3 self-falsification round).

---

## 1. The arc, in one paragraph

Over four phases, one initial instinct — *"maybe transformed/adaptive numerical representations
are a new field"* — was progressively narrowed by deliberate falsification until what remained
was small, precise, tested, and useful. TSA proved the invertible-transform core is classical
conjugation, already mature in ~24 named fields. ATRA proved the unrestricted adaptive form is
logically vacuous and every restriction lands in an existing field. CANR found the one
defensible niche (certified, jointly-selected representation/operator pairs — "a new tool
niche," novelty level 6 of 8) and shipped it as engineering. ANEE tested whether scaling that
niche to a 7-axis "execution engine" changes the verdict: it does not (the formulation is Rice
1976; the architecture is FFTW/ATLAS/SPIRAL/AutoTVM; "composition as the optimization object"
is equality saturation) — but its one unclosed candidate (representation search with a
distribution-keyed plan cache) was then **built, validated on one kernel, bounded by a failed
replication on a second, distilled into a boundary heuristic, and that heuristic prospectively
validated (13/15) while the program's own "graph" framing was self-falsified down to
dictionary densification.** Four phases, four consecutive "narrow real value, not a field"
conclusions, and a closed loop from conjecture to operational rule.

## 2. Phase verdicts

| Phase | Hypothesis tested | Verdict | Killing/deciding evidence |
|---|---|---|---|
| TSA | Conjugating algorithms by invertible scalar transforms (`φ⁻¹∘A∘φ`) is a new paradigm | **No** — classical conjugation/change-of-variables; mapped onto 24 existing fields; Γ-family transforms are a dead end (non-injective, ill-conditioned, no intertwining identity) | Propositions 1–5 + experiments E1–E9b, incl. an honest negative (VST loses in 3 regimes at naive thresholding) |
| ATRA | Adaptively choosing an encoder/operator/decoder triple is a new field | **No** — the unrestricted form is vacuous (Lemma 1); every restriction is owned (transform coding, quantization theory, R–D optimization, precision autotuning); 2 implementation contributions survive (Box–Cox shift toxicity; expm1 fixes) | X1–X5, incl. the chordal-beats-Karcher crossover later reused by ANEE kernel 2 |
| CANR | *Certified* selection over (representation, operator) pairs under accuracy/reproducibility constraints is new | **Level 6 only** — "a new tool niche" (FFTW is speed-only, Precimonious types-only, Herbie expressions-only); known theorems repackaged; engineering module recommended, research program explicitly rejected | Y1–Y6 (certificates held 123/123; selector passed held-out 3/3), determinism ladder D0–D3 |
| ANEE | Joint 7-axis `P=(R,O,A,T,Q,M,H)` optimization + plan artifacts + contracts + learned cache is a new abstraction | **No as stated** (Rice 1976/CASH; FFTW/ATLAS/SPIRAL/AutoTVM/Ansor/MLIR-Transform own the architecture; egg/Diospyros own composition; plan portability is an *open problem for the whole field*, not a feature to inherit) — one candidate left unclosed: distribution-keyed representation-plan caching, absent from all ~25 systems verified against primary sources | §§1–15 + experiments Z1–Z4; certificate composition confirmed open (2026 PhD-thesis territory); precision-as-search-axis confirmed absent from every compiler/library surveyed, twice in the systems' own words |

## 3. The Phase C empirical loop (ANEE Addenda 1–3)

| Round | Question | Pre-registered bar | Result |
|---|---|---|---|
| Kernel 1 (#635) | Does joint (R,A) search beat sequential per-axis selection on a compress-then-aggregate pipeline? | ≥20% held-out error reduction on ≥2/3 workload families | **Met 3/3** (23.1% / 99.3% / 99.2%); mechanism identified: the cost-based S1 tie-break is objective-blind; distribution-mismatch demo: reusing a plan across distributions costs **3793×** — the argued cache-key gap, quantified |
| Kernel 2 (#653) | Does the finding replicate on quaternion orientation averaging (ATRA X5's task, real `scirust-simd` quaternions)? | Same bar, same protocol, deliberately not re-tuned | **Not met 1/3** — genuine non-replication; at low/medium noise the cheap default is already right, so there is no (R,A) interaction to exploit; implementation cross-validated against ATRA X5's independent Karcher reference (9.58–9.92° vs. 9.959°) |
| Addendum 3, avenue 1 (#662) | Is the resulting boundary heuristic ("cheap single-axis ablation first") a valid *prospective* decision rule? | Predictor/outcome agreement ≥12/15 cells over a 5-point quantizer-level dose grid | **Met 13/15** — zero false negatives; both failures are false-PAYS at the noise floor → **absolute-error-floor guard adopted**; 2 of 3 secondary author predictions failed and are recorded (one was mis-specified against already-published data — an authoring error, kept on the record) |
| Addendum 3, avenue 2 (#662) | Is the §4 "representation graph" ever useful *as a graph* (multi-hop paths)? | Existential: ≥1/6 cells where a two-hop composition beats the best single hop by ≥20%; author's declared prior: zero | **Met 2/6 — the author's prior was falsified — then deflated by a labeled post-hoc diagnostic:** both wins are dictionary densification (`power(½)∘power(½)` *is* `power(¼)`; the densified flat dictionary matches bit-for-bit or beats the two-hop winners). Path structure has no demonstrated value; composition survives as a *generator of certified dictionary members* |

## 4. What survives — operational artifacts in `master`

- **A validated decision rule** (the program's most useful single output): *before any joint
  multi-axis search, run the cheap single-axis ablation on dev data; invest in joint search
  only where the default representation's gap is large **and** its absolute error is above the
  target floor.* Prospectively validated at 13/15 with zero false negatives; the guard
  eliminates the only observed failure mode.
- **Code** (`scirust-core`): `representation_graph.rs` — certified representation dictionary
  with `Identity` and `Composed` members (κ via the exact product law of Proposition ANEE-2,
  unit-tested to 10⁻¹²), encode-based safety gate, joint + sequential search over the existing
  `autotune_by` harness, level-parameterized pipeline, and the
  `(kernel, DistributionSummary, BackendKind)`-keyed `PlanCache` with confidence/history;
  `representation_graph_quaternion.rs` (feature `portable-simd`) — chart/accumulation search
  over real quaternions. Supporting: `UniformQuantizer` widened to `pub(crate)`; `BackendKind`
  gained `Hash`.
- **Benchmarks as committed, reproducible artifacts** (fixed seeds): `anee_phase_c_prototype`,
  `anee_phase_c_kernel2_quaternion`, `anee_phase_c_dose_response`, `anee_phase_c_two_hop`
  (examples), plus the four phases' Python experiment scripts
  (`tsa_experiments/`, `atra_experiments/`, `canr_experiments/`, `anee_experiments/`).
- **Earlier-phase engineering already shipped** ([CANR] Phase A, before this program's final
  phase): `certified_numerics.rs` (certified transform pairs + certificate-driven summation),
  `transform_search.rs`, `transform_autotune.rs`, `autotune_accumulate.rs`,
  `scirust-signal`'s VST autotuner.
- **Formal results of record** (known mathematics, precisely stated and validated for this
  stack): exact multiplicative composition of κ_rt (elasticity chain rule; Z3 at Decimal
  prec-60 + Rust unit test); determinism composes as a lattice meet (Z4); non-separability +
  combinatorial infeasibility of exhaustive joint search (Z1/Z2: `k⁷` vs `7k`).

## 5. What is closed, and stays closed

Γ/ζ/Bessel/Airy scalar transforms; hypercomplex transform algebra (isomorphism-rigid); "TSA
filters" as a family; ATRA as an unrestricted class; a general 7-axis planner; a general
numerical-contract type system spanning arbitrary plans; an "ANEE" crate/runtime/paper;
**shortest-path machinery over a representation graph** (deflated to dictionary generation by
our own avenue-2 experiment); **generalizing kernel 1's joint-search win** (bounded by kernel
2's non-replication — the win is conditional on an objective-blind default, which is now a
testable precondition, not an assumption). Each closure carries its evidence in the phase
documents; none was closed by opinion.

## 6. Methodology as a first-class result

What made four phases of mostly-negative results cumulative rather than circular: kill
criteria and author priors **locked into committed sources before every run** (which is what
allowed three author predictions to fail *visibly* in Addendum 3 — P2's regime model, P4's
mis-specification, avenue 2's zero-win prior — and be kept on the record as results);
same-bar replication (kernel 2 reused kernel 1's exact criterion, un-tuned); multi-seed
held-out validation (which caught kernel 1's benign family flipping sign on a single draw —
the difference between reporting 23.1% and reporting −68.6%); post-hoc analysis permitted but
**labeled** (avenue 2's densification diagnostic); negative results retained in-tree with the
same care as positive ones, per the convention [TSA] established.

## 7. Going forward

- **No further research phases.** The remaining untested avenues from the Addendum-3 menu
  (cache bucket-collision attack, determinism as a selection constraint, certificate-
  conservatism measurement) are diagnostics of a prototype this program has decided not to
  expand; expected value does not justify reopening the frame. This mirrors, deliberately, the
  closing discipline of the three prior phases.
- **Ordinary engineering that needs no research frame** (from [ANEE §13] Phases A–B, still
  valid): unify the three uncoordinated hardware-dispatch abstractions
  (`scirust-simd::BackendKind`, `scirust-gpu::RawComputeBackend`, `scirust-cuda`'s feature
  gates); re-attempt [CANR §9]'s shared benchmark-result schema as a **compile-time-enforced
  crate** rather than a design document (the design-only version demonstrably failed to be
  adopted).
- **The one standing falsifier:** [ANEE §12.2]'s claim that no existing autotuning cache keys
  on data *distribution* is an absence-of-evidence finding from a ~25-system primary-source
  sweep. Discovery of prior art would falsify it and should be recorded in an addendum if
  found. Nothing else in this program rests on that claim, so no active monitoring is
  warranted — only honesty if the evidence surfaces.

## 8. Closing statement

The program set out to find a new field and, by refusing to find one where none existed, ended
with something smaller and better: a handful of certified, tested, in-tree tools; one
prospectively validated decision rule with a known guard; a precise map of what is owned by
whom in the surrounding literature; and a four-document record in which every claim — including
the authors' own failed predictions — can be checked against committed code. **Closed.**
