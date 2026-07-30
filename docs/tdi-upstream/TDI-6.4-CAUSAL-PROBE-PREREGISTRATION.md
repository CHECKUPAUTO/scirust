# TDI-6.4 — Causal Probe: Does Recovery Depend on *Which* Node Is Perturbed?

## Preregistration

This document is the frozen preregistration for TDI-6.4. Once its SHA-256
manifest, the v64 evaluator, the reproduction script, the CI workflow and
the bounded tests are committed, this design is frozen under the Section 22
freeze rule: no scientific constant, definition, or criterion may change
without a new experiment identifier. Freezing the design does not authorize
a run; the real experiment begins only as the deliberate one-time human
action of Section 16. The authoring agent never invokes `--full`.

## 1. Experimental status, provenance, and the single changed factor

TDI-6.4 is the fourth and final experiment reserved under the roadmap's
Track B slot list (`docs/TDI-FORWARD-PROGRAM-ROADMAP.md` Section 2: "6.4 —
Causal probe. Move from prediction to intervention: use the existing exact
perturbation machinery to ask whether the overlap-associated structure is
causal for recovery, or merely predictive. The hardest and most valuable
frontier; needs its own careful design."). Every preregistration since
TDI-5.2 has carried the same placeholder limitation ("this does not establish
... causal intervention effects") and named "(TDI-6.4)" as where that
question would eventually be addressed (see TDI-6.3 preregistration
Section 1.1 for the full lineage account of the TDI-6.x reservation scheme).

TDI-6.4 is derived from the same minimal ancestor as TDI-6.3: **TDI-5.6**
(`tdi-independent-overlap-ablation-v56.rs`). It inherits TDI-5.6's candidate
generation, single base generator (widths 3 and 4), and exact contraction/
spectral descriptors **verbatim**. **The single changed factor is the
perturbation protocol itself**: every TDI-5.x/6.x experiment to date computes
`analyze_branching_recovery` with exactly **one fixed intervention** per
generated system — `Action::Flip { node: width - 1 }`, the same node in every
system, in every experiment, since TDI-5.2. TDI-6.4 instead computes
`analyze_branching_recovery` **once per possible intervention target**
(`Action::Flip { node: i }` for every `i` in `0..width`) on the **same**
generated system, and compares the resulting recovery trajectories against
each other. Nothing about candidate generation, acceptance/rejection, or the
exact descriptors changes; only *which, and how many, interventions are
analyzed per system* changes.

### 1.1 Why this is the causal question, and why it stayed unaddressed until now

`analyze_branching_recovery(system, reference_state, perturbation, action,
horizon)` (`tdi-core/src/branching_recovery.rs`) already computes an exact
**counterfactual comparison** for a single system: the future state
distribution if left alone (`reference_distributions`) versus the future
state distribution after one intervention (`perturbed_distributions`), at
every depth up to `horizon`. Their overlap (`distribution_overlap`, `= 1 −`
total variation distance) is therefore already, in the Neyman–Rubin sense, an
**exact potential-outcomes comparison of the same unit under treatment vs.
control** — not an approximation, and not something TDI-6.4 needs to invent.

What every prior experiment does with this machinery is hold the
intervention **fixed** (always node `width − 1`) and vary the **system**
across a population, asking whether the *early* (h=1,2) counterfactual
divergence predicts the *later* (h=3…8) one, beyond baseline descriptors —
a cross-sectional, associational question about systems, in which the
intervention itself is a constant, never a variable. This is why the
"causal effects" limitation has recurred, unaddressed, in every
preregistration since TDI-5.2: nothing has ever asked what happens when the
*intervention* is what varies, holding the *system* fixed — the actual
question a causal claim requires ("does *which* node you perturb change the
outcome, for the *same* system?"), as opposed to a predictive claim ("does
knowing this system's early counterfactual divergence predict its later
one?").

TDI-6.4 asks exactly that: for one fixed system, does the recovery
trajectory depend on which node is perturbed (Criterion 6.4A), and does the
early→late predictive relationship TDI-5.2 established for the one
historical perturbation choice generalize to every other possible
intervention target, or was it an artifact of that one fixed choice
(Criterion 6.4B)?

### 1.2 TDI-6.4 stays on the exact track (a discovered simplification, not an assumption)

The roadmap filed TDI-6.4 under "Track B — the non-exact frontier"
alongside 6.1 (needs an eigensolver), 6.2 (needs a nonlinear model
library) and 6.3 (needs `f64` logarithms/determinants for a closed-form
information measure) — all of which genuinely require abandoning bit-exact
rational arithmetic for some part of their core computation. TDI-6.4 does
not. `Action::Flip`, `TableSystem::successors`, and every function in
`branching_baseline.rs`/`branching_distribution.rs`/`branching_recovery.rs`
operate on `ExactRatio` (bit-exact rational arithmetic) throughout; running
`analyze_branching_recovery` once per node instead of once per system is
**more calls to the same exact function**, not a different, non-exact one.
The only floating-point step anywhere in this design is the same `U_h =
-log2(1 - O_h)` transform and ordinary descriptive summary statistics
(means, ranges, correlations) that every TDI-5.x exact experiment already
computes in `f64` on top of its exact generation — not a new non-exact
determinism discipline requiring declared tolerances or a relaxed
reproduction contract. **TDI-6.4 therefore reproduces byte-for-byte, exactly
like TDI-5.2 … 5.8, not tolerance-based like TDI-6.1/6.2/6.3/6.5.** It keeps
the TDI-6 identifier because the roadmap reserved that identifier for the
*causal* question regardless of arithmetic discipline (Section 1.1's
citation trail), not because it needs TDI-6's non-exact machinery — the same
kind of realized-design-differs-from-roadmap-sketch outcome already seen
with TDI-6.2 (whose eventual design, a degree-2 interaction ridge, differs
from the roadmap's speculative "gradient-boosted trees / kernel ridge / MLP"
sketch).

## 2. Research questions

Within the frozen candidate machinery and the frozen exact descriptors:

1. For a fixed generated system, how much does the recovery trajectory
   (hence `U_h` at every horizon) vary depending on which of the `width`
   possible nodes is perturbed (criterion **TDI-6.4A**, primary,
   descriptive)?
2. Does the early-overlap → late-overlap relationship TDI-5.2 established
   using one fixed perturbation node hold with comparable strength for
   *every* possible perturbation node, or only for the one historically used
   (criterion **TDI-6.4B**, descriptive)?
3. Is any systematic difference in effect size attributable to *which*
   node (e.g. first vs. last, by index) rather than being idiosyncratic per
   system (criterion **TDI-6.4C**, descriptive)?

TDI-6.4 does not re-test whether `{O_1,O_2}` improves on any baseline
(settled by 5.2 … 6.5); it fits no ridge model and runs no
Beneficial/Equivalent/Harmful/Inconclusive classifier. None of its criteria
is a pass/fail classification: a causal-heterogeneity measurement has no
natural "success" or "failure" outcome, and none is forced here.

## 3. Relationship to the frozen ancestors

**Inherited unchanged** from TDI-5.2 … 5.6 (frozen; bit-exact; not
re-derived): the exact candidate analysis and per-candidate exclusion
criteria; the single base generator (`build_system` over uniform non-empty
successor masks, widths 3 and 4); observation geometry and target geometry
`U_h = -log2(1 - O_h)`; the 13 structural/entropic baseline variables; the
two exact contraction descriptors δ, δ̄; the two exact spectral moments s₂,
s₃; the dense horizon grid `H = {3,4,5,6,7,8}`; the deterministic per-width
generation budgets; and the `tdi-core` exact primitives — including
`Action`, `analyze_branching_recovery`, `uniform_branching_state_distribution`
and `distribution_overlap`, all reused, not modified.

**New in TDI-6.4** (the only substantive addition): analyzing every
possible single-node `Flip` perturbation per system (Sections 5–6), instead
of only the historical `node = width − 1`, and the three descriptive
heterogeneity/transfer criteria built from comparing those per-node results
(Sections 13–15).

**Dropped relative to every predictive TDI-5.x/6.x experiment.** No feature
layouts, no ridge fitting, no target scalers used for prediction, no
MSE-based classifier, no train/holdout split (Section 4.4, mirroring
TDI-6.3's pooling rationale — this is a descriptive comparison across
intervention targets, not an out-of-sample predictive test).

## 4. Design notes and confirmatory integrity

### 4.1 Why single-node `Flip`, not `Clamp` or multi-node interventions

`Action` has two intervention variants: `Flip { node }` (state-dependent:
XORs whatever the bit currently is) and `Clamp { node, value }`
(state-independent: forces the bit to `value` regardless of its prior
state — the cleaner `do()`-style operator, but never constructed by any
evaluator to date). TDI-6.4 uses **`Flip` exclusively**, for direct
comparability with every prior experiment's perturbation mechanism (`Flip`
is what TDI-5.2 … 6.5 already use; changing to `Clamp` would confound "does
the intervention target matter" with "does the intervention *kind* matter,"
a second, genuinely separate question left for a future experiment,
Section 21). TDI-6.4 varies only the perturbed **node**, one bit at a time
(never two or more nodes simultaneously) — the minimal, single-factor
generalization of the existing protocol.

### 4.2 Exhaustive, not sampled, node coverage

Because the base generator's widths are only 3 and 4, every possible
single-node `Flip` is exhaustively enumerable (3 or 4 candidates per
system) at negligible extra cost — `analyze_branching_recovery`'s
per-horizon state space is bounded by `2^width ≤ 16`. TDI-6.4 therefore
computes **all** `width` per-system perturbations, not a sample: no
selection procedure, and hence no selection bias, is introduced.

### 4.3 Unconditional per-system heterogeneity, not a new predictive model

Criterion 6.4A reports, per system and per horizon, the exact range
(`max − min`) of `U_h` across the `width` node choices — a direct,
assumption-free measurement of how much the outcome depends on the
intervention target, requiring no model of any kind. Criterion 6.4B reuses
only a simple, symmetric association measure (Pearson correlation of
`(O_1,O_2)` against `U_6`, computed **separately for each node index**,
`f64`, matching the descriptive-statistics precedent already established
by every exact TDI-5.x experiment) — not a ridge fit, and not a
classifier: TDI-6.4 asks whether the *qualitative shape* of the established
early→late relationship transfers across intervention choice, not whether
it improves on a baseline (already settled).

### 4.4 Population re-use without a train/holdout split

Mirroring TDI-6.3 Section 4.5: TDI-5.6's population structure (three
blocks, each with training-w3/holdout-w3/training-w4/holdout-w4) is reused
**verbatim** for its seed reservations and generation budgets, but the
split served TDI-5.6's out-of-sample predictive-error testing, which
TDI-6.4 does not do. **All records generated for a block are pooled**
before computing that block's per-node analysis; there is no fitting/
prediction split.

### 4.5 The exact descriptors are generated but only descriptively cross-referenced

Every record still carries the 13 baseline variables and the exact
contraction/spectral descriptors δ, δ̄, s₂, s₃ (candidate generation is
inherited verbatim). TDI-6.4 reports, as Section 12 descriptive context
only, whether the per-system heterogeneity measured by Criterion 6.4A
correlates with these descriptors (a system with a larger δ̄ tends to show
more/less node-to-node variation in recovery) — reported, not used to
classify or filter anything.

## 5. Definitions

For a generated system of width `w` (3 or 4), reference state `T = 0`
(all-zero), and target horizon `h`: for each node index `i ∈ {0, ..., w-1}`,
let `perturbation_i = Action::Flip { node: i }` and

    outcome(i, h) = analyze_branching_recovery(system, T, perturbation_i, Action::Noop, h)
    O_i(h)        = outcome(i, h).final_overlap()
    U_i(h)        = -log2(1 - O_i(h))

(identical to every prior experiment's `U_h` formula, computed once per
node instead of once per system). The historical perturbation node
`i* = w - 1` reproduces exactly TDI-5.2 … 6.5's `O_1, O_2, U_h` values for
the same generated system (a direct consistency check, Section 19).

**Per-system, per-horizon heterogeneity range** (Criterion 6.4A):

    range(h) = max_i U_i(h) - min_i U_i(h)

**Per-node early→late association** (Criterion 6.4B): for a fixed node
index `i`, the Pearson correlation of `(O_i(1), O_i(2))` (as a two-column
predictor) against `U_i(6)` across the pooled population — reported as the
simple bivariate correlations `corr(O_i(1), U_i(6))` and `corr(O_i(2),
U_i(6))` (no multivariate fit; Section 4.3).

### 5.1 Full recovery at a non-historical node is a valid outcome, not a degeneracy — and must not silently corrupt an aggregate

The historical node `i*` already has a well-defined treatment for the
boundary case `O_{i*}(h) = 1` exactly (full recovery, `U_{i*}(h)` undefined):
the candidate is excluded at generation time
(`RejectionReason::TargetFullyRecovered`), inherited unchanged from TDI-5.6.
For every **other** node `i ≠ i*`, no such exclusion is possible — the
record has already been accepted based on `i*` alone — and full recovery at
some `(i, h)` is an entirely legitimate finding (that particular
perturbation's influence happened to vanish exactly by horizon `h` for that
system), not a computational defect. Because `U_i(h) = -log2(1 - O_i(h))
= -log2(0) = +∞` in that case, it must **not** be allowed to silently
enter any aggregate: an unguarded `range(h) = max_i U_i(h) - min_i U_i(h)`
would become `+∞` the instant any single node fully recovers, corrupting
every other (finite, informative) system's contribution to that horizon's
statistics, and an unguarded correlation would be poisoned identically.

**Rule:** for a given system and horizon `h`, if `U_i(h)` is undefined
(full recovery) for **any** `i ∈ {0, ..., w-1}`, that system is excluded
from Criterion 6.4A's `range(h)` statistic **at that horizon only** (other
horizons for the same system are unaffected), and the exclusion is counted
and reported explicitly (`full_recovery_exclusions(h)`, per block and
aggregate) — never silently dropped. Criterion 6.4B's per-node correlations
are computed only over the subset of the population where that specific
node's `U_i(6)` (and, for the `(O_i(1), O_i(2))` predictors, `O_i(1)`,
`O_i(2)` — always defined, since only the *joint* mutual-information-style
transform can diverge, not the raw overlap itself) is defined, with the
excluded count reported per node. This mirrors, at the per-node/per-horizon
level, the same discipline TDI-6.3's finiteness guard applied to the PID
components: a well-defined boundary case is tracked and reported, never
averaged in as if it were an ordinary finite value.

## 6. Computation method

Bit-exact rational arithmetic throughout candidate generation, the exact
descriptors, and every `analyze_branching_recovery`/`distribution_overlap`
call (Section 1.2). `U_h`, `range(h)`, and the Section 6.4B correlations are
computed in `f64`, single-threaded, fixed accumulation order — the same
regime every TDI-5.x experiment already uses for its own `f64` summary
statistics, not a new declared non-exact discipline.

## 7. Populations

Reused verbatim from TDI-5.6 Section 8: single generator, three fresh seed
blocks, no OOD populations. For each of the three seed blocks:

| Population | Width | Accepted records |
|---|---:|---:|
| training | 3 | 15,000 |
| holdout | 3 | 5,000 |
| training | 4 | 15,000 |
| holdout | 4 | 5,000 |

Accepted records per block: **40,000**; total: **120,000**. All four
populations of a block are **pooled** (Section 4.4). Generation budgets are
inherited unchanged from TDI-5.2 Section 7. Every accepted record is
analyzed at every node (3 or 4, by its own width) and every horizon of the
dense grid — no additional acceptance/rejection criteria beyond those
already inherited.

## 8. Independent seed blocks (fresh)

Three deterministic, pairwise-disjoint seed blocks (letters **V/W/X**,
continuing the series' letter sequence after TDI-6.3's S/T/U), disjoint
from every prior block (TDI-5.7 ≤ 2.53×10⁹; TDI-6.1 3.0–3.23×10⁹; TDI-6.2
4.0–4.23×10⁹; TDI-6.5 5.0–6.13×10⁹; TDI-5.8 7.0–7.81×10⁹; TDI-6.3
8.0–8.24×10⁹ approx.; TDI-6.4 starts at 9.0×10⁹):

    base(b) = 9_000_000_000 + b · 100_000_000   for block index b ∈ {0,1,2}

and the four populations start at `base + {0, 10, 20, 30} · 1_000_000`
(training-w3, holdout-w3, training-w4, holdout-w4), identical in structure
to TDI-6.3 Section 8. Explicitly the training-w3 bases: block 0 →
9,000,000,000; block 1 → 9,100,000,000; block 2 → 9,200,000,000. Twelve
total reservations. The evaluator verifies disjointness of all consumed
ranges at runtime.

## 9. Metrics

No MSE, MAE, R², Spearman, bias, or calibration (predictive-accuracy
metrics for a fitted model; TDI-6.4 fits no model). Its metrics are the
per-node `U_i(h)` values, the per-system-per-horizon `range(h)`, and the
per-node bivariate correlations of Section 5.

## 10. Focal horizons and grid

Inherited: the dense grid `H = {3,4,5,6,7,8}` and the focal horizon **U₆**
(primary). Criterion 6.4A reports across the full grid; Criterion 6.4B
uses U₆ as its late-horizon target, matching every prior experiment's
primary-horizon convention.

## 11. Deterministic bootstrap

For each block, and for the pooled aggregate across all three blocks (a
plain, non-stratified resample of pooled records — TDI-6.3 Section 11's
precedent and rationale apply identically here, since TDI-6.4's aggregate
is likewise a direct pooled-record estimate, not a paired-prediction
comparison), resample records with replacement (4,000 replicates,
inherited count) and recompute the mean `range(h)` and the Section 6.4B
correlations on each replicate, reporting the two-sided 95% percentile
interval. Bootstrap seeds are fresh, in the `0x5444_4936_3400_…`
(`TDI6`/`0x34`= ASCII `'4'` = ".4") range, verified disjoint from every
prior bootstrap seed used by any TDI-5.x/6.x evaluator (`.1`→`0x31`,
`.2`→`0x32`, `.3`→`0x33`, `.5`→`0x35`):

    block seed (block b)  : 0x5444_4936_3400_0000 + b + 1   (…0001/0002/0003)
    aggregate seed         : 0x5444_4936_3400_4700

## 12. Descriptor diagnostic (context only, no criterion)

For each block, report the pooled means of the four exact descriptors δ,
δ̄, s₂, s₃ (Section 4.5) — plain descriptive context, consumed by no TDI-6.4
criterion. Their correlation with the per-system `range(6)` is reported
separately, formally, as Criterion TDI-6.4C (Section 15) — not duplicated
here, to avoid the two sections making conflicting claims about whether
that correlation is itself a named criterion.

## 13. Criterion TDI-6.4A — per-system node-to-node heterogeneity (primary, descriptive)

For each block and the pooled aggregate, at every horizon of the dense
grid, report the distribution (median, IQR, min, max) across the
population of `range(h)` (Section 5), its 95% bootstrap interval, the
proportion of systems for which `range(h)` exceeds a purely descriptive
reference threshold (the population's own median `U_i(h)` scale, reported
for context, not as a pass/fail cut), and the count of systems excluded
from that horizon's statistic under Section 5.1's full-recovery rule
(`full_recovery_exclusions(h)`). TDI-6.4A is a preregistered **descriptive**
summary; it is not a pass/fail classification.

## 14. Criterion TDI-6.4B — transfer of the early→late relationship across intervention choice (descriptive)

For each node index `i` (0 to width−1, separately for width 3 and width
4), report `corr(O_i(1), U_i(6))` and `corr(O_i(2), U_i(6))` across the
pooled population (excluding, per Section 5.1, systems where `U_i(6)` is
undefined for that node — the excluded count is reported alongside each
correlation), with 95% bootstrap intervals. Report whether these
per-node correlations are **stable** (similar in sign and magnitude across
every node) or **shift** (materially different for the historical node
`i* = w-1` than for others), and if they shift, which node(s) differ and
by how much. Purely descriptive; no success/failure claim — a stable
pattern is evidence the early→late relationship is a general property of
recovery dynamics; a shift localized to `i*` is evidence it is specific to
the historically-used perturbation.

## 15. Criterion TDI-6.4C — descriptor correlates of heterogeneity (descriptive)

Report the simple correlation of per-system `range(6)` (Section 13) against
each of the four exact descriptors δ, δ̄, s₂, s₃ (Section 12), per block
and pooled aggregate, with 95% bootstrap intervals. Descriptive only; does
not condition or filter any other criterion.

## 16. Operational activation and full-run entrypoint contract

The evaluator will expose exactly three modes: `--termination-smoke`,
`--preflight`, `--full`. A bare invocation refuses to run. `--full` will
require the exact confirmation environment variable
`TDI64_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI64_FREEZE_RULE`. Without that exact
value, `--full` fails before any generation or computation. No TDI-6.4
commit, test, or CI run will supply the token; the authoring agent never
invokes `--full` with the real token.

## 17. Required raw output

git commit; compiler/Cargo versions; the evaluator, preregistration, and
scientific-manifest SHA-256; the full frozen ancestor chain; all frozen
constants; the seed-block definitions; per-block requested/accepted/
rejected/attempted counts; rejection counts by reason; final exclusive
seeds; per-node `U_i(h)` summary statistics at every horizon; the
Criterion 6.4A/B/C tables; the Section 12 descriptor-diagnostic table;
deterministic termination diagnostics; the historical-node consistency
check (Section 19).

## 18. Interpretation boundaries

A TDI-6.4 result characterizes how much a system's recovery trajectory
depends on *which single node* is perturbed, within the frozen exact
machinery, on the single base generator, using single-node `Flip`
interventions only. It does **not** establish: a causal effect of
multi-node or simultaneous interventions; anything about `Action::Clamp`
(state-independent) interventions, a genuinely different intervention
*kind* left for future work; a general theory of causal structure beyond
this system class; robustness across generator families (5.7/6.5) or
widths (5.8) — those are separate, already-settled questions; a
PID/information-decomposition account (TDI-6.3, separate and already
answered); or external empirical validity. The Criterion 6.4A/B/C
summaries may not be rewritten after observing the result.

## 19. Determinism and consistency check

Candidate generation, seed consumption, exclusions, and the exact
descriptor construction are deterministic functions of committed constants,
inherited unchanged from TDI-5.6. Every per-node `analyze_branching_recovery`
call, `range(h)` computation, correlation, and bootstrap resampling is a
deterministic function of the generated records and the frozen constants of
Sections 8 and 11. As a direct internal consistency check (not a
scientific criterion), the evaluator will assert that `U_{i*}(h)` (node
`i* = w-1`, the historical perturbation) computed here is bit-identical to
what TDI-5.6's own formula would produce on the same records — since both
are the same exact computation on the same generated systems, any
divergence would indicate a defect in this evaluator, not a scientific
finding.

## 20. Reproduction requirements

Byte-exact reproduction (Section 1.2 — TDI-6.4 is an exact-track
experiment despite its TDI-6 identifier). The reproduction script will
satisfy every requirement of TDI-5.2 Section 19 / TDI-5.8 Section 12
(refuse a dirty repository; verify all frozen hashes including the full
ancestor chain and TDI-6.4 itself; refuse an existing partial or complete
result; acquire an exclusive lock; compile offline in release mode; execute
the evaluator exactly once with `--full`; capture complete output; verify
all final criterion lines; write metadata and a completion marker; hash all
artifacts; make final artifacts read-only), plus: it must require the exact
confirmation variable before invoking the evaluator.

## 21. Deferred tracks

Multi-node or simultaneous interventions; `Action::Clamp` (state-independent
"hard" interventions) as a distinct intervention kind from `Flip`; a formal
causal-graph or do-calculus account beyond direct intervention-outcome
comparison; robustness across generator families or widths; information
decomposition (settled, TDI-6.3); and external empirical validity are all
out of scope for TDI-6.4, which is deliberately narrow: one intervention
kind (`Flip`), one node at a time, exhaustively enumerated, on the single
base generator.

## 22. Freeze rule

Once the SHA-256 manifests, the evaluator, the reproduction script, the CI
workflow, and the bounded tests are committed, this design is frozen:
scientific code must not change; constants and the byte-exact reproduction
contract must not change; the per-node intervention protocol and the
criteria must not change; no full run may begin before all frozen hashes
pass; any scientific-code defect discovered after freezing requires a new
experiment identifier — TDI-6.4 may not be silently patched. The result
classifications, once produced, are frozen as reported.
