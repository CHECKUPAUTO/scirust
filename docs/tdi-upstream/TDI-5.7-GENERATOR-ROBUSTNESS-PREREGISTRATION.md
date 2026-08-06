# TDI-5.7 — Generator Robustness: Does the Overlap Signal Survive a Family of Exact Generators?

## Preregistration

This document is the frozen preregistration for TDI-5.7. Once its SHA-256
manifest, the v57 evaluator, the reproduction script, the CI workflow and the
bounded tests are committed, this design is frozen under the Section 22 freeze
rule: no scientific constant, generator-family rule, seed block, feature
definition, baseline or criterion may change without a new experiment
identifier. Freezing the design does not authorize a run; the real experiment
may begin only as the deliberate one-time human action described in Section 16.
The authoring agent never invokes `--full`.

It supersedes the earlier draft roadmap entry; the design choices left open in
the draft are resolved here (family set F0–F3; three seed blocks per family;
transfer pair F0 → F1; a fourth descriptive criterion TDI-5.7D).

## 1. Experimental status and provenance

TDI-5.7 is a new confirmatory experiment derived from the completed and merged
TDI-5.6 result. It is **not** a continuation, patch, or reinterpretation of
TDI-5.1 … TDI-5.6, each of which remains frozen under its own identifier.

Every TDI-5.x result to date uses a **single generator**: `build_system`
assembles each candidate from per-state successor masks drawn by `splitmix64`
uniformly over all non-empty successor subsets (widths 3 and 4). Robustness to
*generator changes* is therefore the single most-repeated open limitation
(every prior Section 20). A skeptic can still argue that the overlap signal —
even after surviving the exact contraction descriptors (5.5) and the exact
spectral moments (5.6) — is an artifact of that one candidate distribution.

TDI-5.7 confronts this directly. It asks whether the **primary TDI-5.6
finding** — the marginal value of `{O_1, O_2}` beyond the full exact descriptor
set (baseline + Dobrushin δ, δ̄ + spectral moments s₂, s₃; i.e. the SKT-vs-SK
comparison) — **replicates across a family of structurally distinct exact
generators**, not just the inherited base generator.

Frozen ancestor identities (verified at runtime and in CI):

| Artifact | SHA-256 |
|---|---|
| TDI-5.6 evaluator (v56) | *(pinned by the frozen chain test, TDI-5.1 → TDI-5.6)* |
| TDI-5.6 preregistration | `59e3375b82d0bb7aad7be0591b9d1eac074d4b194678dfe0e06e73c8aac89807` |
| TDI-5.5 evaluator (v55) | `10df698d10f010b9f6c18e2a4d78042eb399d3812b8d69c2b4bb799de828b835` |
| TDI-5.5 preregistration | `37260b3349107659487e42e66c269ecad44efaf6131f8206bb28dfbcf83f9da1` |

The v57 evaluator and CI verify the **full frozen chain TDI-5.1 → TDI-5.6**
(every ancestor evaluator and preregistration hash) before any generation.

No full TDI-5.7 run may begin before all of the following are committed and
frozen: this preregistration; the final evaluator; the evaluator SHA-256
manifest; the scientific-code SHA-256 manifest; the deterministic reproduction
script; the dedicated CI workflow; bounded unit and termination tests.

## 2. Research questions

Within the frozen candidate machinery and the frozen exact descriptor set:

1. does the SKT-minus-SK marginal value of `{O_1, O_2}` classify **Beneficial
   in every generator family** at the focal horizons U₃ and U₆
   (**replication**, criterion TDI-5.7A)?
2. how **heterogeneous** is the effect size across families — is the aggregate
   relative-MSE reduction tightly clustered above the 2% margin, or does it
   swing widely / cross into Equivalence for some family (**heterogeneity**,
   criterion TDI-5.7B)?
3. does a model fitted on one family **transfer** to another at the same width
   (**cross-generator transfer**, criterion TDI-5.7C, descriptive)?
4. how much do the exact descriptors δ, δ̄, s₂, s₃ **themselves move** across
   families — i.e. how demanding is the SK baseline in each family
   (**descriptor drift**, criterion TDI-5.7D, descriptive)?

TDI-5.7 does **not** re-open the exact-descriptor questions (5.5/5.6), the
persistence confound (5.5B), nonlinear families, or the literal spectral gap
(deferred to TDI-6). It changes **only** the generator, holding every feature,
layout, model and criterion machine of 5.6 fixed.

## 3. Relationship to the frozen ancestors

**Inherited unchanged** from TDI-5.2 … 5.6 (frozen; not re-derived): the exact
candidate analysis, observation geometry, target geometry
`U_h = -log2(1 - O_h)`, the exact cardinality / mask machinery, the 13
structural/entropic baseline variables, the two early overlaps O₁, O₂, the two
exact contraction descriptors δ, δ̄, the two exact spectral moments s₂, s₃, the
layouts **CK / SK / SKT**, ridge `lambda = 1.0`, `build_system`, the paired +
stratified-aggregate bootstrap, the four-way ±2% classifier, and the
`tdi-core` exact primitives (unchanged).

**New in TDI-5.7** (the only substantive additions): a **family of exact
generators** — four deterministic successor-mask construction rules (Section 5)
— and the replication / heterogeneity / transfer / descriptor-drift criteria
(Sections 13–15) that compare the *same* SKT-vs-SK signal *across* those
generators.

## 4. Design notes and confirmatory integrity

### 4.1 Why change only the generator

Holding the entire 5.6 measurement apparatus fixed and varying only the
candidate distribution isolates one factor — the generator — so any change in
the SKT-vs-SK classification is attributable to the generator and nothing else.

### 4.2 Exact, deterministic, mask-based families

Every family member produces, for each of the `2^width` states, a `u64`
successor **mask** (one bit per successor state), assembled into a system by
the **unchanged** frozen `build_system`. Only the *rule that fills the mask*
differs. Every rule is a deterministic function of the candidate seed via the
inherited `splitmix64` chain, and every rule guarantees a **non-empty**
successor set for every state, so the one-step Noop kernel is always total (as
the frozen contraction / spectral descriptors require). No floating-point
candidate construction is introduced; TDI-5.7 stays bit-exact.

### 4.3 Fixed width; generator-only variation

The primary criteria hold width fixed (the base width-3 + width-4
in-distribution composition of 5.6) and vary the generator rule. Cross-width
generality is a separate question (roadmap TDI-5.8).

## 5. The generator family

Let `states = 2^width`. For each source state index `s` in `0..states`, the
rule advances the `splitmix64` chain (seeded by the candidate seed, exactly as
the inherited generator) and produces `mask[s]`. The four frozen rules:

### F0 — base *(inherited, unchanged)*

    draw d0 = next(chain);   mask[s] = d0 % (2^states − 1) + 1

a uniform draw over all non-empty successor subsets. This is the TDI-5.6
generator; F0 is the reference member and reproduces the 5.6 candidate
distribution exactly (under fresh seed blocks).

### F1 — sparse (low out-degree)

    draw r1 = next(chain);   out_degree d = 1 + (r1 % 2)          // d ∈ {1, 2}
    select d DISTINCT successor indices by rejection from the chain:
        repeat: draw r = next(chain); p = r % states;
                if bit p of mask[s] is unset, set it;
        until popcount(mask[s]) == d

Low branching → weaker mixing and larger spectral moments; stresses whether the
signal needs high branching.

### F2 — dense (high out-degree)

    draw r1 = next(chain);   excluded e = r1 % 2                  // e ∈ {0, 1}
    mask[s] = (2^states − 1)                                      // all states
    if e == 1: draw r = next(chain); clear bit (r % states) of mask[s]

Near-complete branching → fast mixing and a small spectral gap; stresses the
low-contraction regime. `mask[s]` always keeps ≥ `states − 1` bits, so it is
non-empty for `states ≥ 2` (guaranteed at width ≥ 1).

### F3 — local (Hamming ≤ 1 neighbourhood)

The neighbourhood of `s` is `N(s) = {s} ∪ {s XOR (1 << b) : b ∈ 0..width}`,
i.e. `s` itself and the `width` single-bit-flip neighbours (`width + 1`
candidate states, all valid width-`w` states).

    draw r = next(chain)
    for j in 0..(width + 1):   include neighbour N(s)[j] iff bit j of r is set
    if no neighbour was included: include N(s)[0] (= s itself)
    mask[s] = the bits of the included neighbours

Structured locality (a hypercube-local kernel with self-loops), a qualitatively
different topology and spectrum from the IID-mask families. The forced
inclusion of `s` on an empty draw guarantees non-emptiness.

All four rules are exact, deterministic, width-parametric, and produce a valid
total Noop kernel. The neighbour order for F3 is fixed as
`[s, s XOR 1, s XOR 2, s XOR 4, …, s XOR 2^{width−1}]` (index 0 = self, then
ascending bit position), so the rule is fully determined.

## 6. Feature layouts

Unchanged from TDI-5.6 Section 6: **CK** (baseline + δ + δ̄, 15), **SK**
(CK + s₂ + s₃, 17), **SKT** (SK + O₁ + O₂, 19). The confirmatory comparison is
**SKT vs SK** — the overlaps' marginal value beyond the full exact descriptor
set — computed independently *within each generator family*.

## 7. Populations

For **each** of the four generator families, generate the same in-distribution
populations as TDI-5.6, across **three** fresh, pairwise-disjoint seed blocks:

| Population | Width | Accepted records |
|---|---:|---:|
| training | 3 | 15,000 |
| holdout | 3 | 5,000 |
| training | 4 | 15,000 |
| holdout | 4 | 5,000 |

Accepted records per block: **40,000**; per family (3 blocks): **120,000**;
total (4 families): **480,000**. **No OOD populations.** Per-family models are
fitted on that family's combined width-3 + width-4 training populations and
every criterion is evaluated on that family's combined holdout populations.
Holdout records never affect fitting. The generation budgets are inherited
unchanged from TDI-5.2 Section 7.

## 8. Independent seed blocks (fresh)

Twelve deterministic, pairwise-disjoint seed blocks (three per family),
**disjoint from the TDI-5.6 blocks J/K/L and all earlier blocks**. For family
index `f ∈ {0,1,2,3}` (F0…F3) and block index `b ∈ {0,1,2}`, the block base is

    base(f, b) = 1_400_000_000 + f · 300_000_000 + b · 100_000_000

and the four populations start at `base + {0, 10, 20, 30} · 1_000_000`
(training-w3, holdout-w3, training-w4, holdout-w4). Explicitly:

| Family | Block bases (training-w3 seed) |
|---|---|
| F0 | 1,400,000,000 · 1,500,000,000 · 1,600,000,000 |
| F1 | 1,700,000,000 · 1,800,000,000 · 1,900,000,000 |
| F2 | 2,000,000,000 · 2,100,000,000 · 2,200,000,000 |
| F3 | 2,300,000,000 · 2,400,000,000 · 2,500,000,000 |

Blocks are spaced 100,000,000 apart and populations 10,000,000 apart; each
population consumes at most a few million seeds, so all 48 reservations are
pairwise disjoint and disjoint from every earlier block. The evaluator verifies
disjointness of all consumed ranges at runtime. Total seed reservations: **48**.

## 9. Deterministic bootstrap

The bootstrap engine, replicate count (4,000) and resampling discipline are
inherited unchanged from TDI-5.2 Section 10. TDI-5.7 uses fresh bootstrap
seeds, disjoint from every TDI-5.2 … 5.6 bootstrap seed, in the `0x5444493537…`
("TDI57") range:

    block seed (family f, block b) : 0x5444_4935_3700_0000 + (3·f + b) + 1
                                     (F0: …0001/0002/0003 … F3: …000A/000B/000C)
    family aggregate seed (family f): 0x5444_4935_3700_4700 + f
                                     (F0: …4700 … F3: …4703)

Each family's stratified-aggregate bootstrap runs over its own three blocks
with its own aggregate seed. For each confirmatory comparison, report the
two-sided 95% interval of the baseline-minus-challenger MSE difference and, for
equivalence classification, the two-sided 95% interval of the relative MSE
difference.

## 10. Metrics

For every family, block, population, horizon and layout, print the full metric
set of TDI-5.2 Section 9, plus, for every confirmatory comparison, the absolute
MSE difference, relative MSE reduction, absolute MAE difference, Spearman
difference, R² difference and absolute-bias difference.

## 11. Standardized-U primacy

Standardized U space is the primary confirmatory domain (TDI-5.2 Section 5).
Reconstructed-O-space quantities are secondary diagnostics only and cannot
determine any TDI-5.7 criterion.

## 12. Focal horizons and grid

Inherited from TDI-5.6: the dense grid `H = {3, 4, 5, 6, 7, 8}` and the focal
horizons **U₃** and **U₆**. The confirmatory criteria classify at the focal
horizons; per-family per-horizon reductions across the grid are reported.

## 13. Criterion TDI-5.7A — replication across generators (primary)

For **each** family Fᵢ, compute the SKT-vs-SK four-way classification (the exact
4-way logic of TDI-5.2 Section 13, symmetric 2% relative-MSE margin) at the
focal horizons **U₃** and **U₆** on that family's combined holdout. TDI-5.7A is
the preregistered conjunction:

- **replicated** iff the classification is **Beneficial at both U₃ and U₆ for
  all four families**;
- otherwise a **located non-replication** — the evaluator names each (family,
  horizon) whose classification is not Beneficial.

TDI-5.7A is a preregistered classification, not forced to any result. Full
replication would show the overlap signal is a property of exact branching
dynamics broadly; a located non-replication would bound TDI's generality to the
families where it holds.

## 14. Criterion TDI-5.7B — effect-size heterogeneity

Across the four families, report at each focal horizon the aggregate
relative-MSE reduction of SKT over SK: its **minimum, maximum and range**, and
whether **all four** exceed the 2% margin. A tight cluster well above 2% is
strong generality of effect size; a wide swing (even if all Beneficial) is a
preregistered caveat on transportability. Descriptive: it makes no pass/fail
claim beyond the "all four exceed 2%" flag.

## 15. Criteria TDI-5.7C and TDI-5.7D — transfer and descriptor drift (descriptive)

**TDI-5.7C — cross-generator transfer.** Fit the SK and SKT models on family
**F0**'s training population; evaluate the SKT-vs-SK comparison on family
**F1**'s combined holdout. Report the standardized-U R² of each layout and the
four-way classification. This distinguishes "the signal exists within each
generator" (5.7A) from "the *same fitted* signal transports across generators."

**TDI-5.7D — descriptor drift.** For each family, report the holdout means of
the four exact descriptors δ, δ̄, s₂, s₃ (and their across-family range). This
contextualizes how demanding the SK baseline is in each family — a family whose
kernels are near-uniform (δ, δ̄ ≈ 0) offers the exact descriptors little to
work with, making its 5.7A test effectively SKT-vs-baseline; a family with
strong contraction/spectral structure makes 5.7A a demanding test.

Both TDI-5.7C and TDI-5.7D are preregistered **descriptive** summaries; neither
makes a success/failure claim.

## 16. Operational activation and full-run entrypoint contract

The v57 evaluator exposes exactly three modes: `--termination-smoke`,
`--preflight`, `--full`. A bare invocation refuses to run. `--termination-smoke`
uses only bounded tiny data and produces no result artifacts. `--preflight`
performs no scientific generation: it verifies the full frozen configuration
(all 48 seed reservations, all expected counts, all bootstrap constants, all
four family rules present), verifies that the full pipeline is wired to
`--full`, prints all TDI-5.7 and ancestor identities and the exact real-run
command, and exits without a result.

`--full` requires the exact confirmation environment variable:

    TDI57_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI57_FREEZE_RULE

Without that exact value, `--full` fails before any generation, fitting or
bootstrap. The confirmation check is a pure function of the environment value,
unit-testable without starting the experiment. No TDI-5.7 commit, test or CI run
supplies the token. The full run is a deliberate, one-time human action; the
authoring agent never invokes `--full` with the real token.

## 17. Required raw output

Inherited from TDI-5.2 Section 17 with TDI-5.7 identities and **per-family**
reporting: git commit; compiler/Cargo versions; v57 evaluator SHA-256; TDI-5.7
preregistration SHA-256; TDI-5.7 scientific-manifest SHA-256; the frozen-ancestor
hashes; all frozen constants; the four family rules; the seed-block definitions;
requested/accepted/rejected/attempted counts per family; rejection counts by
reason; final exclusive seeds; generation budgets; target scalers; the CK, SK
and SKT model coefficients for every family and block; all metrics; all bootstrap
intervals; the per-family per-horizon SKT-vs-SK comparisons; the TDI-5.7A focal
classifications per family and the replication verdict; the TDI-5.7B
heterogeneity summary; the TDI-5.7C transfer classification; the TDI-5.7D
descriptor-drift table; deterministic termination diagnostics.

## 18. Determinism

Inherited from TDI-5.2 Section 18. Candidate generation (**including every
family's successor-mask rule and its `splitmix64` draw sequence**), seed
consumption, exclusions, preprocessing, contraction-descriptor and
spectral-moment construction, model fitting, bootstrap sampling, aggregation,
metric calculation, iteration order, scientific-value formatting and final
criteria are deterministic functions of committed constants. Wall-clock
timestamps are reproduction metadata only.

## 19. Reproduction requirements

The TDI-5.7 reproduction script must satisfy every requirement of TDI-5.2
Section 19 / TDI-5.6 Section 19 (refuse a dirty repository; verify all frozen
hashes including TDI-5.1 … 5.6 and TDI-5.7; refuse an existing partial or
complete result; acquire an exclusive lock; compile offline in release mode;
execute the evaluator exactly once with `--full`; capture complete output;
verify all final criterion lines; write metadata and a completion marker; hash
all artifacts; make final artifacts read-only), plus: it must require the exact
confirmation variable before invoking the evaluator, and must refuse to run over
an existing TDI-5.7 result.

## 20. Interpretation boundaries

A TDI-5.7 result establishes the (non)replication and effect-size stability of
the `{O_1,O_2}`-beyond-exact-descriptors signal **across the specific
preregistered generator family F0–F3 of Section 5**, at fixed width, within the
frozen exact machinery. It does **not** establish: robustness to generators
outside that family; cross-width invariance (TDI-5.8); control against non-exact
descriptors or model families (TDI-6); causal effects; or external validity. The
TDI-5.7A / B / C / D summaries may not be rewritten after observing the result.

## 21. Deferred non-exact track (TDI-6)

Unchanged from TDI-5.6 Section 21. The literal spectral gap / second eigenvalue,
the ε-mixing time, non-parametric model families, and a formal PID of O₁/O₂ are
deferred to the separate TDI-6 identifier with its own non-exact determinism
discipline. TDI-5.7 stays bit-exact and presupposes nothing about TDI-6.

## 22. Freeze rule

After the TDI-5.7 preregistration, v57 evaluator, manifests, reproduction script
and CI workflow are frozen: scientific code must not change; constants must not
change; the four generator-family rules, the seed blocks, the layouts and the
criteria must not change; no full run may begin before all frozen hashes pass
(TDI-5.1 … 5.6 and 5.7); any scientific-code defect discovered after freezing
requires a new experiment identifier — TDI-5.7 may not be silently patched.
