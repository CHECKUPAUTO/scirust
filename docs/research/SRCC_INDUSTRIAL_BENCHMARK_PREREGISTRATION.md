# SRCC Industrial Benchmark — Preregistration

**Status: committed before any real-data evaluation.** This document freezes
the hypotheses, metrics, datasets, splits, baselines, hyperparameter policy,
exclusion rules and superiority criteria for phase 728 (real industrial
evaluation) of the SRCC Robust Structural Intelligence Program. Phase 728
results are valid only insofar as they follow this protocol; every deviation
must be reported in the evaluation report as a deviation, with its reason.

The harness implementing this protocol is `scirust-srcc-bench` (phase 727).
Nothing in this document is evidence; it is the contract under which evidence
will be produced.

---

## 1. Primary hypotheses

Each hypothesis is falsifiable, names its methods, and pre-commits to what
would count as a negative result. "SRCC" always refers to the deterministic
implementations in `scirust-srcc` at the evaluated commit.

- **H1 (scale invariance of decisions).** Under coordinate unit changes
  (`CoordinateScaleShift` with factors drawn from the preregistered grid
  {0.01, 0.1, 10, 100} applied to individual sensor columns), the
  scale-aware SRCC source clustering
  (`SrccSourceGeometrySpec::RobustDiagonal`, `minimum_scale = 0`, policy
  `Error` or `DropDimension`) preserves its cluster assignments and
  downstream projector decisions, while raw-Euclidean clustering does not.
  *Negative result if*: scale-aware assignments change under rescaling, or
  raw assignments are equally stable on the tested workloads.
- **H2 (robust regression under contamination).** On held-out clean test
  rows, at training contamination fractions {0.05, 0.1, 0.2} of the
  preregistered kinds (`CoherentAlternativeCluster`, `TargetFlip`-style
  gross target corruption), Huber IRLS and trimmed least squares achieve
  lower RMSE than ordinary least squares, with paired 95 % confidence
  intervals excluding zero across machines. *Negative result if*: intervals
  straddle zero or favor OLS. Median-of-means is evaluated under its
  documented block-majority assumption and reported either way.
- **H3 (trust identifiability).** Under `ViewConcentratedAttack` and
  `BurstAttack` within the declared per-group corruption bounds, trust-aware
  SRCC (`GroupContaminationBound`, `TemporalPersistence`) returns either the
  clean consensus or a typed unidentifiability — never the contaminant —
  while unweighted consensus elects the contaminant once it is a majority.
  *Negative result if*: a bounded policy accepts the contaminant, or the
  typed-refusal rate on clean data exceeds 5 %.
- **H4 (certified vs greedy clustering).** On per-view source sets from real
  trajectories, `CertifiedMedoid` finds partitions with equal or lower
  (count, medoid-cost) objective than `GreedyCompleteLink`, and resolves a
  nonzero fraction of the bridge-ambiguity rejections. Cost is compared only
  where greedy succeeds; certificates are reported (proven vs gapped).
  *Negative result if*: greedy matches the certified objective everywhere
  (certification adds proof but no partition improvement on these
  workloads) — this is explicitly an acceptable outcome worth publishing.
- **H5 (end-task transfer).** Improvements under H4, when present, translate
  into improvement on at least one preregistered end-task metric (H2's RMSE
  or the anomaly AUROC family) on at least one dataset. *Negative result
  if*: partition improvements do not move end-task metrics.

## 2. Primary metrics

- Regression: **RMSE on clean held-out test targets** (primary),
  parameter L2 error where ground truth exists.
- Anomaly detection: **AUROC** for score-producing detectors; balanced
  accuracy at the preregistered threshold rule (§7) for label-only methods.
- Stream detection: **detection delay** (steps) at a false-alarm budget of
  at most 1 pre-onset alarm per 100 in-control steps.
- Clustering: lexicographic **(cluster count, observed-medoid cost)**
  objective plus `proven_optimal` / `optimality_gap` from certificates;
  cluster recovery (ARI) where ground-truth states exist.
- Trust: contaminant-acceptance rate (must be 0 within bounds for H3),
  typed-refusal rate on clean data.

## 3. Secondary metrics

MAE, median absolute error, worst absolute error; precision, recall, F1,
false-alarm and missed-detection rates; stability ratio and Frobenius
projector distance (SRCC leave-one-out); explored/pruned nodes for the
certified solver; runtime and peak memory are recorded as
environment-dependent side channels, never inside hashed scientific content,
and never as primary evidence.

## 4. Datasets

Phase 728 selects at least: (1) one predictive-maintenance / run-to-failure
dataset; (2) one multivariate process-control / sensor-anomaly dataset —
from the candidate classes named in the program (turbofan degradation,
bearing vibration, pump/cavitation monitoring, multivariate process-control
trajectories, machine condition monitoring). Before use, each dataset must
have: verified redistribution terms and citation requirements; a recorded
SHA-256; a `DatasetManifest` with feature descriptors and units; documented
train/test semantics and label meaning. No dataset is committed to the
repository unless its license clearly permits it; fetch scripts verify
checksums and never run inside `cargo test`. Absent data ⇒ integration
tests skip, loudly.

**SRCC transport-view construction (fixed here to prevent post-hoc
choice):** for each machine/run, the 16-dimensional source vector is the
first 16 preregistered sensor channels (padded with zeros when fewer),
median-centered per trajectory; targets are the corresponding channels one
preregistered horizon step later; each machine/run is one view. The horizon
and channel list per dataset are fixed in the phase-728 configuration
*before* evaluation and hashed into `configuration_sha256`.

## 5. Split strategy

Grouped by physical unit (machine / run / trajectory) via
`SplitStrategy::GroupedHoldout` (train 0.6 / validation 0.2) for i.i.d.-unit
questions; `SplitStrategy::Temporal` within units for stream questions;
`SplitStrategy::LeaveOneGroupOut` for paired per-unit comparisons. Every
split manifest (seed, strategy, grouping key, dataset checksum) is published
with the results. No row of a unit may cross sides; no training row may
postdate an evaluation row within a temporal split.

## 6. Baselines

Regression: OLS, Huber IRLS (δ = 1.345), trimmed LS (h = 0.7),
median-of-means (5 blocks). Anomaly: Isolation Forest, LOF (declared
transductive), DBSCAN-noise (label-only), regularized Mahalanobis,
Hotelling T². Stream: CUSUM (k = 0.5, h = 5), EWMA (λ = 0.2, L = 2.7).
SRCC family: historical exact, source-clustered, scale-aware,
trust-aware; greedy vs certified clustering. RLS/QR-RLS are added for
streaming regression questions where applicable. Methods are never scored
on metrics their output shape does not support.

## 7. Hyperparameter selection

Fixed preregistered values above, or selection on the validation split
only, from preregistered grids (published in the phase-728 configuration
and hashed). The anomaly threshold rule for count metrics is the method's
median evaluation score (documented, untuned). Nested deterministic
cross-validation is permitted inside training groups. **No parameter is
ever selected on test results.** All candidate configurations and the
selection decision are recorded.

## 8. Exclusion rules

Rows are excluded only by preregistered schema validation (non-finite
values, missing mandatory channels) with counts published. Units are
excluded only for documented data-integrity reasons (checksum or schema
failure), never for unfavorable results. A method's run that errors is
reported as that typed error, not silently dropped (§9).

## 9. Failure reporting

Typed failures (non-convergence, singular covariance, degenerate scale,
ambiguity, unidentifiability, budget exhaustion) are results: they are
counted, tabulated per method × workload, and published alongside metric
rows. Timeouts and memory limits are reported with their budgets.
Negative and inconclusive outcomes are published with the same prominence
as positive ones.

## 10. Superiority criteria

The phrase "X outperforms Y" may be used in phase-728 conclusions only
when **all** of the following hold on a preregistered primary metric:

1. the paired per-unit improvement has a 95 % bootstrap confidence interval
   excluding zero (seeded, ≥ 2000 resamples);
2. the improvement is practically meaningful: ≥ 5 % relative on the primary
   metric (or ≥ 1 detection step for delay);
3. no disqualifying degradation: no safety-relevant metric (false-alarm
   rate, missed-detection rate) worsens by more than its preregistered
   budget (25 % relative) on the same workload;
4. the direction replicates on at least two meaningful splits or workloads.

Otherwise the conclusion is one of: "no significant difference",
"underperformed", or "inconclusive" — verbatim, with the interval shown.
Aggregate means without paired intervals are never sufficient.

## 11. Outputs

`scirust-bench-schema::BenchRecord` JSONL (kernel, dataset, method, seed,
metric, value, ci, cert), one row per measurement; manifests
(`DatasetManifest`, `SplitManifest`, `ContaminationManifest`) as JSON next
to the records; `RunMetadata` (git commit, dataset SHA, configuration SHA,
toolchain, feature flags) beside — never inside — hashed scientific
content. Deterministic content is separated from environment-dependent
runtime measurements.
