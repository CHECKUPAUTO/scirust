# ANEE Phase D — pre-registration (criteria and priors locked before any run)

**Date:** 2026-07-18. **Status:** committed BEFORE any Phase D experiment was
implemented or executed; the run results will be reported in a separate
document and must be judged against the criteria below, unmodified.

## 0. Why this phase exists

The program's closing synthesis
([`ANEE_PROGRAM_SYNTHESIS_2026-07-18.md`](ANEE_PROGRAM_SYNTHESIS_2026-07-18.md) §7)
recommended **no further research phases** and enumerated exactly what
remained: three untested diagnostics from the Addendum-3 menu, one unclosed
novelty candidate under a standing falsifier, two field-scale open problems
out of this repository's reach, and three ordinary-engineering complements.
On 2026-07-18 the program's principal explicitly directed that **all** of
those remaining avenues be explored. This phase is that exploration, opened
by principal decision against the synthesis default — recorded as such, not
as a change in the evidence.

Methodology is unchanged from phases 1–4: kill criteria and author priors
locked in this committed document before any run; multi-seed held-out
validation; post-hoc analysis permitted but labeled; negative results
retained with the same care as positive ones.

The two field-scale problems (certificate composition across transformation
passes; cross-hardware plan portability) are **not** attacked at field scale
— that would repeat the pattern this program falsified four times. They are
probed **in-repo, bounded, honestly labeled** as D5 and D6 below, with the
scope limitation stated inside each design.

## 1. Shared materials

Identical to the Phase C dose-response protocol
(`scirust-core/examples/anee_phase_c_dose_response.rs`) unless a section
says otherwise:

- **Workload families** (generators reused verbatim): `benign`
  (uniform 0.5..1.5), `wide-range` (log-uniform over 12 decades),
  `stagnation-prone` (80% ≈1e-3 / 20% ≈1e3).
- **Batches:** n = 8192; dev seed 1, eval seed 2, fresh held-out seeds
  {13, 14, 15}; reported outcomes are 3-fresh-seed means.
- **Dictionaries:** `default_representation_dictionary()` (5 single hops)
  and `default_accumulators()` (5 methods); two-hop dictionary
  `two_hop_dictionary()` (20 members) where stated.
- **Searches:** `sequential_baseline_with_levels` / `joint_search_with_levels`
  as committed in `scirust-core/src/representation_graph.rs`, unmodified.
- **Level grid:** stated per experiment; L = 64 is the default when a single
  level count is used.
- **Experiment binaries:** `anee_phase_d_cache.rs` (D1+D4),
  `anee_phase_d_determinism_transfer.rs` (D2+D6),
  `anee_phase_d_certificates.rs` (D3+D5) — all emitting
  `scirust-bench-schema` JSONL records alongside human-readable output.

## 2. Experiments

### D1 — PlanCache bucket-collision attack

**Question.** `DistributionSummary` compresses a batch to
`(log10_range_decades, stagnation_risk)`. Is that 2-field key coarse enough
that two workloads with **identical summaries** have **materially different
optimal plans** — i.e., can the cache be made to serve a wrong plan while
its keying says "same distribution"?

**Protocol.** Four constructed pairs (W_x, W_y), each designed to collide
(collision verified at runtime by `summarize(dev_x) == summarize(dev_y)`;
a pair that fails verification is discarded and reported as such). Attack
direction: joint-search a plan on W_x (dev/eval), serve it to W_y; regret =
mean 3-fresh-seed error of the served plan on W_y ÷ same-seed error of
W_y's own joint-searched plan. L = 64. Pair designs (fixed now): (P1)
log-uniform 6 decades vs endpoint-bimodal 6 decades, both risk-free; (P2)
stagnation-prone vs a 6-decade batch with 10.5% small mass concentrated
just under the 1e-3·max threshold; (P3) benign-with-one-outlier (decades 6,
risk false) vs log-uniform 6 decades; (P4) two stagnation batches at equal
decades whose small-mass fractions (15% vs 60%) both set `stagnation_risk`
but should favor different accumulators.

**Decisive criterion.** The attack **succeeds** if ≥ 1 pair has a verified
summary collision AND regret ≥ 2.0×. **Author prior: the attack succeeds**
(the 2-field summary is knowably lossy; ANEE §12.2 already flagged bucket
design as the weak point of the candidate). Descriptive follow-up (not
decisive): report whether the wrongly-served component is R or A.

### D2 — determinism as a selection constraint

**Question.** What does bit-reproducibility-under-parallel-chunking cost in
accuracy, and can the R axis buy the cost back?

**Constraint definition (operational, not by rung label).** An accumulation
method is *chunk-invariant* at n = 8192 if splitting the input into P equal
chunks, accumulating each chunk with the method, and combining the P
partials with the same method is **bit-identical** to the serial result for
P ∈ {2, 4, 16}. The experiment first *measures* invariance for all 5
methods (expectation: only `PairwiseF32` passes, its fixed halving tree
coinciding with power-of-two chunking; if the measured constraint set
differs, the measured set is used and the discrepancy reported). The
constrained search is `joint_search_with_levels` with the A dictionary
restricted to the measured chunk-invariant set.

**Protocol.** Full 15-cell grid (3 families × L ∈ {8, 16, 64, 256, 1024}).
ρ = (constrained joint 3-seed mean error) ÷ (unconstrained joint 3-seed
mean error) per cell.

**Decisive criterion.** "Determinism is cheap" if ρ ≤ 2.0 in ≥ 12/15
cells. **Author prior: MET**, with failures (if any) concentrated in
stagnation-prone × low L, where compensated accumulation is the whole win
and no representation substitutes for it. Also recorded (descriptive):
cells where the constrained winner's R differs from the unconstrained
winner's R — cross-axis compensation, if it exists, is visible exactly
there.

### D3 — certificate conservatism

**Question.** How loose are the certified round-trip bounds relative to
observed error — and how much of the *pipeline* error do they not even
claim to cover?

**Protocol.** Two separate measurements, kept apart because they answer
different questions:

1. **Slack within scope.** For every representation the searches actually
   select (per family, L = 64) and for every member of the two-hop
   dictionary admissible on the family's support: slack =
   (certified `roundtrip_bound(support).ulps` × UNIT) ÷ (observed max
   relative round-trip error of encode∘decode over dev ∪ eval). By
   construction slack ≥ 1 means the bound holds; the size is the
   conservatism.
2. **Coverage gap (category honesty).** The certificate bounds round-trip
   error only — not quantization, not accumulation. On wide-range at
   L = 8, compare the certified relative round-trip bound against the
   observed full-pipeline error of the joint winner.

**Criteria (descriptive prior, not kill).** **Author prior:** median
within-scope slack ∈ [2, 100]; and the L = 8 wide-range pipeline error
exceeds the round-trip bound by ≥ 10× (the certificate is not a pipeline
certificate; quantifying the gap is the deliverable). A single observed
round-trip error *above* its certified bound would be a **soundness bug**
and overrides everything else in D3.

### D4 — distribution-keyed caching at scale (the unclosed candidate)

**Question.** ANEE §12.2's one unclosed novelty candidate: does keying a
plan cache on a data-distribution summary actually pay, against the
baseline every surveyed system uses (kernel+hardware key only) and against
an oracle that re-searches every batch?

**Protocol.** One stream of T = 240 batches (n = 4096), seeded and fixed:
three blocks of 80 (benign with range drifting 0.5..1.5 → 0.1..10;
wide-range with decades drifting 4 → 8; stagnation-prone with small-mass
fraction drifting 0.6 → 0.9), and within every block each batch switches
with probability 0.2 (seeded) to a random *other* family (regime noise).
Per batch b: dev = batch, held-out = fresh seed 10_000 + b. Policies, all
using `joint_search_with_levels` at L = 64 as their search primitive:

- **kernel-only:** search once on the first batch, serve that plan forever
  (the FFTW/TopHub-shaped baseline: keyed on kernel+hardware, both
  constant here);
- **distribution-keyed:** the committed `PlanCache`; search on miss, serve
  on hit, key = (kernel, `summarize(dev)`, backend);
- **oracle:** search every batch.

Metrics: per-policy mean held-out error; regret = policy mean ÷ oracle
mean; cost = number of searches.

**Decisive criteria.** The candidate **pays** if (i) distribution-keyed
regret ≤ 1.25× oracle at ≤ 20% of oracle's search count, AND (ii)
kernel-only regret ≥ 2× distribution-keyed regret. The candidate **dies**
(empirically, in-repo) if kernel-only regret ≤ 1.1× distribution-keyed —
i.e., distribution keying adds nothing a frozen plan doesn't already give.
Between those bands: report as inconclusive with the numbers. **Author
prior: pays on (i); (ii) is driven by the block transitions and the regime
noise.** Hazard, declared: D1's collision attack, if it succeeds, bounds
how much trust (ii) can carry — both results must be reported together.

### D5 — certificate-composition slack (scoped probe)

**Scope limitation, stated up front.** Certificate composition across
transformation passes is an open problem at field scale; this probe
examines the one compositional certificate this repository actually has —
`Composed::kappa_rt` is the *exact* product of hop κ's (Proposition
ANEE-2) — and asks a bounded question about slack, not a field answer.

**Question.** For two-hop compositions, is the certified bound's slack
(D3's measure) roughly the product of the hops' individual slacks — i.e.,
does conservatism compound multiplicatively even when the κ composition
itself is exact?

**Protocol.** For each of the 20 two-hop members admissible on each
family's support (L-independent): r = slack(composed) ÷
(slack(hop₁) × slack(hop₂ on hop₁'s image support)).

**Criterion (descriptive prior).** **Author prior: median r ≈ 1.0**
(within [0.75, 1.33]): the exact κ product should make slack compose
multiplicatively, no better. Median r < 0.75 would be a small positive
surprise (composition tightens relative to naive compounding); median
r > 1.33 would mean composition *adds* conservatism beyond its parts —
the pessimistic field intuition reproduced in miniature.

### D6 — plan transfer across execution regimes (scoped probe)

**Scope limitation, stated up front.** This container has one CPU and no
second hardware target; *true* cross-hardware portability is not testable
here and this probe does not claim to test it. The varied axis is the
**execution model** (ANEE's M axis): data-parallel chunked reduction with
P partials (chunks accumulated with the plan's method, partials combined
with the same method), P ∈ {1, 4, 16, 64} — the same variation that
distinguishes serial from multi-threaded deployment of one plan on one
machine.

**Question.** Does the (R, A) plan tuned serially (P = 1) remain the right
plan when the execution model changes, or must plans be re-tuned per
regime?

**Protocol.** Full 15-cell grid. In each cell: tune joint at P = 1; for
each P ∈ {4, 16, 64}, evaluate that serial plan under P against the
P-specific joint winner (both as 3-fresh-seed means, pipeline identical
except the chunked accumulation). A cell **transfers** if regret ≤ 1.2×
for every P.

**Decisive criterion.** "Plans transfer across execution regimes" if
≥ 12/15 cells transfer. **Author prior: MET**, with predicted failures in
stagnation-prone cells where chunking breaks sequential compensation
(Neumaier/Klein) and flips the A winner. Implementation note, fixed now:
the chunked pipeline reuses the library's reconstruction (representation +
quantization) unchanged and varies only the accumulation step; if that
requires extracting a small `pub` helper from
`pipeline_relative_error_with_levels`, the helper must be a pure refactor
(the parameterless entry points' outputs bit-unchanged, enforced by test).

## 3. Engineering complements (no research criteria)

- **E1** — adopt `scirust-bench-schema` in `scirust-tdi-bench` and
  demonstrate the pattern on one seeded criterion target; document the
  migration path for the remaining harnesses.
- **E2** — bind the `PlanCache` hardware key component to the unified
  `compute_capability` registry (or, if `BackendKind` remains the right
  key, add the consistency test that keeps the two abstractions from
  silently diverging, and document the decision).
- **E3** — promote the validated ablation-first rule (Addendum 3: gap
  threshold **plus the absolute-error-floor guard**) from example code
  into a documented public utility in `representation_graph`, with tests
  that reproduce the dose-response predictor's behavior.

## 4. Reporting rules

One results document (`ANEE_PHASE_D_RESULTS_2026-07-18.md`); every D-section
gets a verdict against its criterion exactly as written above (MET / NOT
MET / attack SUCCEEDS-FAILS / pays-dies-inconclusive); author priors are
restated next to outcomes, failures included; any post-hoc analysis is
labeled as such; all raw per-cell numbers ship as bench-schema JSONL in the
example outputs. Nothing in this file may be edited after the first
experiment runs — corrections, if needed, go in the results document as
labeled deviations.
