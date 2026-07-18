# ANEE Phase D — results

**Date:** 2026-07-18. **Pre-registration:**
[`ANEE_PHASE_D_PREREGISTRATION_2026-07-18.md`](ANEE_PHASE_D_PREREGISTRATION_2026-07-18.md)
(committed before any experiment code existed; every verdict below is judged
against those criteria, unmodified). **Experiment binaries:**
`scirust-core/examples/anee_phase_d_{cache,determinism_transfer,certificates}.rs`
— each prints the full per-cell numbers and a `scirust-bench-schema` JSONL
stream; all runs seeded and reproducible.

## 0. Verdict table

| # | Question | Pre-registered bar | Author prior | Outcome |
|---|---|---|---|---|
| D1 | Can colliding `DistributionSummary` keys serve a wrong plan? | ≥ 1 verified collision with regret ≥ 2.0× | attack succeeds | **SUCCEEDS** — P3 at **3.05×** (prior confirmed) |
| D2 | What does chunk-invariant determinism cost? | ρ ≤ 2.0 in ≥ 12/15 cells | MET | **MET 15/15**, max ρ **1.10** (prior confirmed) |
| D3 | How conservative are the round-trip certificates? | median slack ∈ [2, 100]; any observed > certified = soundness bug | in band | **CONFIRMED** — median **4.3×** [1.7–63.6], **0 violations**/60 |
| D3′ | How much pipeline error does the certificate not cover? | ≥ 10× at wide-range L = 8 | ≥ 10× | **CONFIRMED** — **6.0 × 10¹³×** |
| D4 | Does distribution-keyed caching pay at scale? | pays / dies bands (§2) | pays | **DIES** — kernel-only 0.81× vs distribution-keyed **31.66×** (prior falsified) |
| D5 | Is composition slack the product of hop slacks? | median ratio ∈ [0.75, 1.33] | ≈ 1.0 | **BELOW band** — median **0.13**: composition certifies ~8× *tighter* (prior falsified, favorably) |
| D6 | Do serial plans transfer across execution regimes? | ≥ 12/15 cells within 1.2× | MET, failures in stagnation | **MET 13/15** — but both failures in *benign*, not stagnation (verdict confirmed, mechanism prediction wrong) |

Three author priors failed visibly (D4's central prediction; D5's direction;
D6's failure-location mechanism) and are kept on the record, per the
program's convention.

## 1. D1 — the bucket-collision attack succeeds

All four constructed pairs verified as colliding at runtime — every one in
the `(log10_range_decades: 6, stagnation_risk: true)` bucket. Regrets of
serving x's jointly-searched plan to y (3-fresh-seed means, quantizer refit
on y's dev batch, L = 64):

| Pair | x → y | plans (x vs y) | regret |
|---|---|---|---|
| P1 | log-uniform-6dec → endpoint-bimodal-6dec | power+Naive vs anscombe+Naive | 0.57× |
| P2 | stagnation-80% → thin-small-tail-6dec | anscombe+Stochastic vs anscombe+Naive | 0.99× |
| P3 | benign+outlier → log-uniform-6dec | log1p+Naive vs power+Naive | **3.05×** |
| P4 | stagnation-15% → stagnation-60% | power+Stochastic vs anscombe+Naive | 1.47× |

**Attack SUCCEEDS** (bar: one pair ≥ 2.0×). The wrongly-served component in
P3 is the representation (R). Note P1's 0.57× — cross-serving can also
*help* by accident; coarse keys are wrong in both directions.

**Labeled deviations:** (i) the pre-registration's P1/P3 designs predicted
"risk-free" collisions; the runtime summaries came out `(6, true)` — the
collisions still verified and the criterion is untouched, but the author's
flag prediction was wrong. (ii) Fresh-seed scoring falls back to the ungated
identity default on a gate failure (counted; zero occurred).

## 2. D2 — determinism is essentially free here

The measured chunk-invariance matrix matched the design expectation exactly:
of the five committed accumulators, **only `PairwiseF32`** is bit-identical
under P ∈ {2, 4, 16} chunking (its fixed halving tree coincides with
power-of-two chunk boundaries at n = 8192; every sequentially-compensated
method diverges). Constrained joint search (A ∈ {PairwiseF32}) vs
unconstrained, 15 cells: **ρ ≤ 1.10 everywhere → MET 15/15** (bar 12/15).

Two cells came out ρ < 1 (benign L = 64: 0.58; benign L = 1024: 0.43) with
a different representation chosen — at the noise floor, the *smaller*
candidate set generalized better than the unconstrained search, which had
picked a dev/eval-noise winner. That is selection overfitting, the recurring
phenomenon of this phase (see §7).

## 3. D3 — certificates are modestly conservative, sound, and cover almost nothing of the pipeline

Within scope (round-trip error, 60 admissible (family, member) samples
including all two-hops): slack = certified/observed ∈ [1.7×, 63.6×],
**median 4.3×** — prior [2, 100] confirmed, on its tight end. **Zero
soundness violations**: no observed round-trip error exceeded its certified
bound anywhere. (One zero-observed sample excluded; identity reported apart.)

Coverage, kept separate on purpose: at wide-range L = 8 the joint winner's
pipeline error (8.0 × 10⁻²) exceeds its round-trip bound (1.3 × 10⁻¹⁵) by
**6.0 × 10¹³×**. The κ_rt certificate bounds representation round-trip only
— quantization and accumulation, which dominate the pipeline, are simply
outside its scope. Anyone reading `PlanSearchReport.certificate` as a
pipeline guarantee would be wrong by thirteen orders of magnitude; this
number is the honest label on that field.

## 4. D4 — the last unclosed novelty candidate dies

The pre-registered 240-batch drifting stream (three 80-batch blocks with
parameter drift + 20% seeded regime noise), three policies sharing one
search primitive:

| policy | mean held-out error | regret vs oracle | searches |
|---|---|---|---|
| oracle (re-search every batch) | 5.26 × 10⁻⁴ | 1.00× | 240 |
| kernel-only (one frozen plan) | 4.27 × 10⁻⁴ | **0.81×** | 1 |
| distribution-keyed (`PlanCache`) | 1.66 × 10⁻² | **31.66×** | 8 |

Both pre-registered kill conditions fire (kernel-only regret ≤ 1.1× the
distribution-keyed regret — by a factor of ~39). **The candidate DIES**, and
the author's prior ("pays") is falsified. Two mechanisms, both visible in
the per-batch records:

1. **D1's collision, occurring naturally.** The `(6, true)` bucket is first
   filled by an early wide-range regime-noise batch; its cached plan is then
   served to every stagnation-prone batch of block 3 (same bucket), where
   its accumulation choice stagnates — the 31.66× is concentrated exactly
   there. The attack of D1 is not adversarial exotica; the stream produces
   it on its own.
2. **Per-batch re-search overfits.** Kernel-only *beats the oracle*
   (0.81×): joint selection on a 2048/2048 split of each batch picks
   noise-floor winners that generalize worse than one robust frozen plan.
   Adaptivity has negative value on this stream even before keying;
   distribution keying then adds the collision failure on top.

ANEE §12.2's absence-of-prior-art observation (no surveyed autotuning cache
keys on data distribution) stands as a literature fact — but Phase D now
supplies the missing empirical half: **in the one implementation this
program built, keying on the committed 2-field distribution summary is
worse than not keying at all.** The candidate is closed, negative, by its
own pre-registered criteria. The standing falsifier reduces to historical
honesty; there is nothing left to monitor.

## 5. D5 — composition certifies tighter than compounding (bounded positive)

For 45 admissible (family, two-hop) samples: ratio =
slack(composed) / (slack(hop₁) × slack(hop₂ on hop₁'s image)) ∈
[0.08, 2.12], **median 0.13** — below the pre-registered [0.75, 1.33] band.
The author's prior (multiplicative compounding, ≈ 1.0) is **falsified in
the favorable direction**: certifying the composition directly is ~8×
tighter than multiplying per-hop certificates.

The structural reason is visible in the bound's shape: the composed
certificate `(κ_sup(b∘a)·B_ENC + B_DEC)·u` pays the decode budget **once**,
while the naive product `(κ_a·B_ENC + B_DEC)(κ_b·B_ENC + B_DEC)·u²`-style
compounding multiplies both hops' additive terms; and the endpoint-mapped
`κ_sup` of the composition is at most the product of independent sups.
Scoped consequence, stated within this repository's bounds only: **when a
two-hop representation is used, certify it as a composition
(`RepresentationChoice::Composed`), never by multiplying the hops'
individual bounds.** This does not solve certificate composition at field
scale; it is one concrete, measured instance where the exact-κ-product
formulation (Proposition ANEE-2) buys real tightness.

## 6. D6 — plans transfer across execution regimes; the failures are the floor again

15 cells × P ∈ {4, 16, 64} chunked reduction: **13/15 cells transfer**
(serial winner within 1.2× of the P-specific winner for every P) → bar MET.
But the author predicted failures in stagnation-prone cells (chunking
breaking sequential compensation); **that mechanism never materialized** —
every stagnation and wide-range cell transferred at 1.00–1.02×. The two
failures are benign L = 64 (1.67×) and benign L = 1024 (3.02×): noise-floor
cells where the serial "winner" was itself selection noise (D2 found the
same cells; Addendum 3 found the same cells). The mechanistic half of the
prior is recorded as wrong.

## 7. The cross-cutting finding: selection overfitting at the noise floor

One phenomenon recurs in every experiment that touched the benign family:
D2's ρ < 1 cells, D4's kernel-only-beats-oracle, D6's two non-transfers,
and Addendum 3's two false-PAYS before it. When the achievable error is at
the accumulation noise floor, *any* argmax over measured errors — joint
search, per-batch adaptation, P-specific re-tuning — selects noise, and
more searching makes results worse, not better. The practical rule already
shipped as the mandatory guard on `ablation_first_advice` (E3): **check the
absolute error floor before optimizing relative gaps.** Phase D upgrades
that from a caveat to the single most load-bearing lesson of the program.

## 8. Engineering complements delivered (E1–E3)

- **E1** — `scirust-tdi-bench`'s `tdi-holdout` now emits 36 CANR §9 records
  (7 models × 4 holdout metrics + 8 paired-bootstrap deltas — the schema's
  `ci` field exercised by a real harness for the first time; seeds are the
  true generator inputs: test-stream base and bootstrap seed). For the ~16
  criterion targets, `scirust_bench_schema::criterion_estimate_to_record`
  converts criterion's `estimates.json` after a run (fixture-tested); the
  crate docs carry the migration path, including why the adopter must
  supply the pinned data seed.
- **E2** — `representation_graph::current_hardware_key()` is the single
  sanctioned H-axis key source; decision documented (`BackendKind` stays
  the key type — `Copy + Eq + Hash`, allocation-free; the capability
  registry stays the reporting view) and pinned by an anti-divergence test
  binding the key's label to the registry's seeded `cpu-simd` entry.
- **E3** — the validated decision rule is now a public, documented,
  guard-mandatory utility: `ablation_first_advice` / `AblationAdvice` /
  `ABLATION_GAP_THRESHOLD`, with tests reproducing the dose-response
  predictor's behavior on both sides of the guard. The library refactor it
  and D6 required (`reconstruct_with_levels`) is bitwise-pinned to the old
  pipeline by test.

## 9. What Phase D changes in the program's closing state

The synthesis's §7 left three untested diagnostics, one unclosed candidate,
and two field-scale problems probed nowhere. Phase D leaves:

- the three diagnostics **run** (D1 attack confirmed; D2 constraint cheap;
  D3 conservatism quantified, bounds sound, coverage gap quantified);
- the unclosed candidate **closed, negative, empirically** (D4) — the
  program now ends with zero open novelty claims;
- the two field-scale problems probed **in-repo and bounded**, each
  returning one usable engineering fact (D5: certify compositions, don't
  compound; D6: serial plans survive execution-regime changes away from the
  noise floor) and no research program;
- one upgraded lesson (§7) already enforced in the shipped API.

Nothing in this phase reopens anything. The program remains **closed** —
now with the last candidate's grave dug by its own pre-registered
criterion, which is how this program buries things.
