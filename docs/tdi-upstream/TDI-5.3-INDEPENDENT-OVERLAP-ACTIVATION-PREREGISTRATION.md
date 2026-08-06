# TDI-5.3 — Operational Activation of the Frozen TDI-5.2 Experiment

## Preregistration

## 1. Experimental status and provenance

TDI-5.3 is a new experiment identifier. It exists solely because the frozen
TDI-5.2 evaluator, by design, has no executable full-run entrypoint:
`run_full_experiment()` unconditionally returns an error, and no other path
in the frozen v52 binary can start the preregistered 165,000-record
generation. TDI-5.3 supplies that missing operational entrypoint without
touching the frozen TDI-5.2 artifacts in any way.

The following facts are load-bearing for this preregistration and must not
be reinterpreted:

1. No TDI-5.2 full result has been observed. The full 165,000-record
   generation has never been executed. No TDI-5.2A through TDI-5.2E verdict,
   and no TDI-5.2C classification, has ever been produced from real data.
2. No TDI-5.2 scientific artifact is modified by TDI-5.3. The frozen
   evaluator, preregistration, manifests, reproduction script and CI
   workflow listed below remain byte-for-byte unchanged.
3. TDI-5.3 does not reinterpret TDI-5.2 as having been executed. TDI-5.2
   remains, after TDI-5.3 is frozen, exactly as unexecuted as it was before.

Frozen TDI-5.2 identity:

    commit:
    ca8e37157563ecb099bfbd5fc2408dbdf6b40c6d

    preregistration SHA-256:
    f57a054bc95eb2e041434d6e2049509b0dce1a5397f9666d274b1bbac332be35

    evaluator SHA-256:
    2308607729659c7546a17530e69773f982d9a1cf41656ea7898e0123ca469ef7

    scientific-manifest SHA-256:
    7f8eeea2304ef14c1ab2fdb6835a6f5397be83209a80aaf08ccdabd54ccf3d61

Frozen TDI-5.2 files (must remain byte-for-byte unchanged by TDI-5.3 and
verified as such before every TDI-5.3 commit):

    docs/TDI-5.2-INDEPENDENT-OVERLAP-ABLATION-PREREGISTRATION.md
    docs/TDI-5.2-INDEPENDENT-OVERLAP-ABLATION-PREREGISTRATION.sha256
    docs/TDI-5.2-INDEPENDENT-OVERLAP-ABLATION-EVALUATOR.sha256
    docs/TDI-5.2-SCIENTIFIC-CODE.sha256
    tdi-bench/src/bin/tdi-independent-overlap-ablation-v52.rs
    scripts/reproduce-tdi5.2.sh
    .github/workflows/tdi52-ci.yml

For continuity, TDI-5.2's own frozen ancestor identity (TDI-5.1, as recorded
in TDI-5.2 Section 1) is reproduced here unchanged:

    commit:
    d9286eee25dd5b0735fa79fd64395e10ea4c93ac

    result SHA-256:
    f66666dba8a0a82a13aaf62e018a5500e57452db50eeff526b68b98f797a5895

No full TDI-5.3 scientific execution may begin before all of the following
are committed and frozen:

1. this preregistration;
2. the v53 evaluator;
3. the v53 evaluator SHA-256 manifest;
4. the TDI-5.3 scientific-code SHA-256 manifest;
5. the deterministic TDI-5.3 reproduction script;
6. the dedicated TDI-5.3 CI workflow;
7. bounded unit and termination tests for v53.

## 2. Relationship to TDI-5.2

TDI-5.3 inherits the complete scientific design of TDI-5.2 unchanged. The
only substantive difference between TDI-5.2 and TDI-5.3 is that TDI-5.3
provides a deliberate, explicit and reproducible operational route into the
already-implemented full pipeline, through an explicit full-run contract
(Section 5). No scientific threshold, seed, constant, metric, or criterion
is changed by TDI-5.3.

TDI-5.3 is therefore not a redesign and not a reinterpretation of TDI-5.2.
It is an activation revision: the v53 evaluator is derived mechanically from
the frozen v52 evaluator, preserving the numerical implementation, with only
the minimal changes required to identify the experiment as TDI-5.3, expose
the operational activation route, print TDI-5.3 identities, and reference
this preregistration and its manifests.

## 3. Inherited scientific design (unchanged from TDI-5.2)

Every element below is unchanged from its corresponding TDI-5.2 section and
is repeated here only for readability; TDI-5.2's text is the original,
frozen source, and no value below may ever diverge from it.

### 3.1 Frozen dynamical construction (TDI-5.2 Section 3)

Same generator, exact branching analysis, reference state, perturbation,
observation geometry, target geometry, scientific exclusions, and width-6
exact cardinality as TDI-5.1/TDI-5.2:

    observation horizon:      h_obs = 2
    target horizons:          H = {3, 4, 5, 6, 8}
    primary target:           U_6
    transformed target:       U_h = -log2(1 - O_h)
    width-6 successor space:  2^(2^6) = 2^64 = 18,446,744,073,709,551,616
    width-6 non-empty masks:  u64::MAX

Unsupported widths and arithmetic, structural or dynamic-analysis errors
must produce typed failures. They must never be converted into scientific
exclusions.

### 3.2 Predictors (TDI-5.2 Section 4)

Same 13 structural and entropic baseline variables. Same confirmatory
early-overlap predictors O_1 and O_2. Same confirmatory model layouts:

| Layout | Variables |
|---|---|
| B0 | baseline |
| B1 | baseline plus O_1 |
| B2 | baseline plus O_2 |
| B12 | baseline plus O_1 and O_2 |

Same exploratory-only layout:

| Layout | Variables |
|---|---|
| BD | baseline plus O_2 minus O_1 |

No confirmatory model may simultaneously contain O_1, O_2 and O_2 minus
O_1. BD cannot determine any confirmatory success criterion.

### 3.3 Model family and preprocessing (TDI-5.2 Section 5)

Same confirmatory model family: ridge regression with `lambda = 1.0`. Same
per-block preprocessing rules: width-3/width-4 training populations
combined per block; preprocessing and target standardization fitted only
on that block's training population; holdout and OOD records never affect
fitting; one model per target horizon and feature layout; deterministic
iteration and floating-point accumulation order; identical target scaler
per block and horizon across all layouts. Standardized U space remains the
primary confirmatory domain; reconstructed O-space metrics remain secondary
diagnostics.

### 3.4 Independent seed blocks (TDI-5.2 Section 6)

Same three deterministic, non-overlapping seed blocks, same 18 populations,
same accepted-record counts per population, same initial seeds:

| Population | Width | Accepted records |
|---|---:|---:|
| training | 3 | 15,000 |
| holdout | 3 | 5,000 |
| training | 4 | 15,000 |
| holdout | 4 | 5,000 |
| OOD principal | 5 | 10,000 |
| OOD extreme | 6 | 5,000 |

Accepted records per block: 55,000. Total accepted records: 165,000.

| Population | Block A seed | Block B seed | Block C seed |
|---|---:|---:|---:|
| training w3 | 160,000,000 | 260,000,000 | 360,000,000 |
| holdout w3 | 170,000,000 | 270,000,000 | 370,000,000 |
| training w4 | 180,000,000 | 280,000,000 | 380,000,000 |
| holdout w4 | 190,000,000 | 290,000,000 | 390,000,000 |
| OOD w5 | 200,000,000 | 300,000,000 | 400,000,000 |
| OOD w6 | 210,000,000 | 310,000,000 | 410,000,000 |

The evaluator must verify at runtime that all consumed seed ranges are
pairwise disjoint (same requirement as TDI-5.2 Section 6, last paragraph).

### 3.5 Generation budgets (TDI-5.2 Section 7)

Same deterministic generation limits:

| Width | Attempt multiplier | No-progress threshold |
|---:|---:|---:|
| 3 | 64 | 25,000 |
| 4 | 96 | 50,000 |
| 5 | 128 | 75,000 |
| 6 | 256 | 100,000 |

Same termination order (requested accepted count reached; attempt budget
exhausted; no-progress threshold reached; typed evaluator error), same
per-failure required print fields.

### 3.6 Scientific exclusions (TDI-5.2 Section 8)

Same six rejection categories only: exact recovery at observation horizon;
exact recovery at a target horizon; invalid observation-overlap geometry;
invalid target-overlap geometry; invalid transformed-target geometry;
non-finite predictor geometry after otherwise valid analysis. Evaluator
failures remain distinct from scientific exclusions and are never converted
into one.

### 3.7 Metrics (TDI-5.2 Section 9)

Same per-layout metrics (MSE, MAE, R-squared, Spearman correlation,
prediction bias, observed mean, predicted mean, calibration intercept,
calibration slope, lower/upper clipping fraction) and same per-comparison
metrics (absolute MSE difference, relative MSE reduction, absolute MAE
difference, Spearman difference, R-squared difference, absolute-bias
difference).

### 3.8 Deterministic bootstrap (TDI-5.2 Section 10)

Same 4,000 paired bootstrap replicates per block. Same bootstrap seeds:

    block A: 0x5444493532410001
    block B: 0x5444493532420002
    block C: 0x5444493532430003

Same stratified aggregate bootstrap seed:

    0x5444493532414747

Same aggregate resampling rules (seed-block membership preserved,
resampling within each block, deterministic block order retained, identical
resampled indices across compared models). Same reporting requirements: a
two-sided 95 percent interval of the baseline-minus-challenger MSE
difference for every confirmatory MSE comparison, and a two-sided 95
percent interval of the relative MSE difference for equivalence
classification.

## 4. TDI-5.3 criteria (reproduced under the new identifier)

TDI-5.3A, TDI-5.3B, TDI-5.3C, TDI-5.3D and TDI-5.3E are, respectively,
definitionally identical to TDI-5.2A, TDI-5.2B, TDI-5.2C, TDI-5.2D and
TDI-5.2E as defined in TDI-5.2 Sections 11 through 15. They are reproduced
under the TDI-5.3 identifier solely so that TDI-5.3's own required raw
output (Section 6) and final verdicts carry TDI-5.3 labels. No threshold,
margin, comparison, or classification rule differs in any way.

### TDI-5.3A — joint signal (identical to TDI-5.2A, Section 11)

Compare B12 against B0 on combined width-3 and width-4 holdout at U_6.
Succeeds only if: B12 has lower MSE in all three seed blocks; each block
bootstrap lower bound is above zero; median relative MSE reduction across
blocks is at least 15 percent; stratified aggregate relative MSE reduction
is at least 15 percent; aggregate bootstrap lower bound is above zero; B12
Spearman is greater than B0 Spearman in every block; aggregate absolute
bias is not worse by more than 0.02.

### TDI-5.3B — independent O2 signal (identical to TDI-5.2B, Section 12)

Compare B12 against B1 on combined width-3 and width-4 holdout at U_6.
Succeeds only if: B12 has lower MSE in all three seed blocks; each block
bootstrap lower bound is above zero; median relative MSE reduction across
blocks is at least 10 percent; aggregate relative MSE reduction is at least
10 percent; B12 Spearman is not lower than B1 Spearman in any block;
aggregate absolute bias is not worse by more than 0.02.

### TDI-5.3C — independent O1 contribution (identical to TDI-5.2C, Section 13)

Compare B12 against B2 on combined width-3 and width-4 holdout at U_6, using
a symmetric relative-MSE margin of 2 percent. Beneficial, Equivalent,
Harmful and Inconclusive are defined exactly as in TDI-5.2 Section 13.
TDI-5.3C is a preregistered classification and is not forced to produce a
positive result.

### TDI-5.3D — OOD transfer (identical to TDI-5.2D, Section 14)

Challenger B12, baseline B0. Width-5 and width-6 requirements are exactly
as in TDI-5.2 Section 14. TDI-5.3D succeeds only if both width-5 and
width-6 requirements succeed.

### TDI-5.3E — multi-horizon trajectory (identical to TDI-5.2E, Section 15)

Compare B12 against B0 on combined width-3 and width-4 holdout at U_3, U_4,
U_5 and U_8. Succeeds only under exactly the five conditions in TDI-5.2
Section 15.

### Exploratory analyses (identical to TDI-5.2, Section 16)

The same eight exploratory-only analyses apply, unchanged, and cannot
modify any TDI-5.3 confirmatory criterion.

## 5. Operational activation and full-run entrypoint contract

This is the only substantive addition TDI-5.3 makes. The v53 evaluator's
command-line interface exposes exactly three modes:

    --termination-smoke
    --preflight
    --full

No-argument invocation must not start the experiment: it must return a
clear usage or explicit-activation error, exactly like the frozen v52
guard, so that an accidental bare invocation of v53 is exactly as inert as
an accidental bare invocation of v52.

`--termination-smoke` uses only bounded tiny data, exercises the existing
smoke path inherited from v52, and creates no result artifacts.

`--preflight` performs no scientific population generation whatsoever. It
verifies the complete frozen configuration (all 18 seed reservations, all
expected counts, all bootstrap constants), verifies that the full pipeline
function is wired to the `--full` path, prints the v53 evaluator,
preregistration and scientific-manifest identities, prints the exact
command required for a real run, and exits successfully without producing
any result.

`--full` requires an exact confirmation environment variable:

    TDI53_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI53_FREEZE_RULE

Without that exact value, `--full` must fail before any generation,
fitting or bootstrap occurs. With that exact value, `--full` calls the real
full pipeline exactly once and passes the resulting report to the complete
required raw-output printer (Section 6); it returns success only after all
required output has been printed. The confirmation check is a cheap, pure
function of the environment value, independent of generation, so that its
rejection paths can be unit tested without ever starting the experiment.

The full run may occur only after all TDI-5.3 code, manifests, reproduction
script, CI workflow and tests are committed and frozen (see Section 10),
and only as a deliberate, one-time human action: setting the confirmation
environment variable and invoking the reproduction script is itself the
human confirmation token gating the experiment. No TDI-5.3 commit, test, or
CI run may supply that token.

## 6. Required raw output

Inherited unchanged from TDI-5.2 Section 17, with TDI-5.3 identities. The
v53 evaluator must print:

1. Git commit;
2. compiler and Cargo versions;
3. v53 evaluator SHA-256;
4. TDI-5.3 preregistration SHA-256;
5. TDI-5.3 scientific-manifest SHA-256;
6. all frozen constants;
7. all seed-block definitions;
8. requested, accepted, rejected and attempted counts;
9. rejection counts by reason;
10. final exclusive seeds;
11. generation budgets;
12. target scalers;
13. model coefficients;
14. all metrics;
15. all bootstrap intervals;
16. block-level criteria;
17. aggregate criteria;
18. final TDI-5.3A through TDI-5.3E verdicts;
19. TDI-5.3C classification;
20. deterministic termination diagnostics.

## 7. Determinism

Inherited unchanged from TDI-5.2 Section 18. The following remain
deterministic functions of committed constants: candidate generation; seed
consumption; exclusions; preprocessing; model fitting; bootstrap sampling;
aggregation; metric calculation; iteration order; scientific-value
formatting; final criteria. Wall-clock timestamps may be recorded only as
reproduction metadata and must not affect any scientific result.

## 8. Reproduction requirements

The TDI-5.3 reproduction script must satisfy every requirement of TDI-5.2
Section 19 (refuse a dirty repository; verify all frozen hashes; refuse an
existing partial or complete output; acquire an exclusive lock; compile
offline in release mode; execute the evaluator exactly once; capture
complete standard output and error output; verify all final criterion
lines; write metadata and a completion marker; hash all result artifacts;
make final artifacts read-only), plus two requirements specific to TDI-5.3's
operational activation:

1. it must require the exact confirmation environment variable before
   invoking the evaluator, and must invoke it with `--full` (never without
   arguments);
2. the experiment must be executed exactly once per invocation of the
   reproduction script; the script must refuse to run again over an
   existing (partial or complete) TDI-5.3 result.

## 9. Interpretation boundaries

Inherited unchanged from TDI-5.2 Section 20. A successful TDI-5.3 run would
establish replicated predictive information from early overlap observations
within the preregistered generator and linear ridge model family. It would
not establish universal validity across all dynamical systems, causal
intervention effects, nonlinear sufficiency, arbitrary-width calibration,
implementation-independent replication, or external empirical validity. The
independent contribution of O_1 (TDI-5.3C) may be classified as beneficial,
equivalent, harmful or inconclusive. No classification may be rewritten
after observing the full result.

## 10. Freeze rule

After the TDI-5.3 preregistration, v53 evaluator, manifests, reproduction
script and CI workflow are frozen:

1. scientific code must not change;
2. constants must not change;
3. seed blocks must not change;
4. criteria must not change;
5. no full run may begin before all frozen hashes pass (TDI-5.1, TDI-5.2,
   and TDI-5.3);
6. any scientific-code defect discovered after TDI-5.3 freezing requires a
   new experiment identifier — TDI-5.3 may not be silently patched, exactly
   as TDI-5.2 could not be silently patched by TDI-5.3 itself.
