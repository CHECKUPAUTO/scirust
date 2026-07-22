# SRCC Industrial Evaluation Report (Phase 728)

Preregistered evaluation
(`docs/research/SRCC_INDUSTRIAL_BENCHMARK_PREREGISTRATION.md`) run on real
industrial data under the frozen configuration
`scirust-srcc-bench/configs/phase728.json`. This report states every
preregistered outcome — positive, negative, and inconclusive — with the same
prominence. **No claim exceeds what the preregistered superiority rule
permits.**

Reproduce with:

```
scripts/fetch_industrial_datasets.sh                 # verified download, no network in tests
cargo run -p scirust-srcc-bench --release --bin industrial-eval -- \
    --out results --git-commit "$(git rev-parse HEAD)"
```

```
SHA-256 (results/industrial_728.jsonl, deterministic scientific content):
d97f23f4c6069ff500579daee51babbcb2cf83a396251a51a66e1b6ff7ebd7a4
```

The run emits 1197 `BenchRecord` rows plus manifests. `run_metadata.json`
(git commit, toolchain, config hash) is environment identity and is excluded
from the determinism fingerprint by design; the JSONL and every scientific
manifest are byte-identical across runs.

## Datasets and protocol

| Workload | Family | Rows | Role |
|----------|--------|-----:|------|
| C-MAPSS FD001 | turbofan run-to-failure (PdM) | 20 631 | primary |
| C-MAPSS FD003 | turbofan run-to-failure (PdM), two fault modes | 24 720 | replication of FD001 |
| SECOM | semiconductor process/yield (real) | 1 567 | anomaly detection |

Provenance, licenses and checksums: `scirust-srcc-bench/DATASETS.md`. An
earlier draft used in-repo automotive OBD2 telemetry as a third workload; it
was removed as out-of-domain (consumer driving data, not industrial
machinery), and the SRCC replication requirement is met within the turbofan
domain by the FD001/FD003 pair.

Every fitted component (missing-value imputer, robust scalers, regression
coefficients, clustering radii, trust weights) is fit on training data only.
Trajectory decimation (stride 20, fixed a priori in the config) is applied
to the C-MAPSS SRCC and regression stages because consecutive engine cycles
are near-duplicate rows; it keeps per-view source sets and pooled training
sets tractable for certified branch-and-bound and 100-engine leave-one-out
refitting. It is a subsampling convenience, documented and never tuned on
outcomes.

## H1 — Scale invariance of source clustering (**supported, replicated**)

Under the preregistered per-channel unit rescalings (×0.01 and ×100), the
certified clustering assignment is compared to the un-rescaled baseline per
(geometry, radius, view).

| Workload | Geometry | Invariant assignments |
|----------|----------|-----------------------|
| FD001 | raw Euclidean | 15/48 (0.31) |
| FD001 | robust diagonal | **48/48 (1.00)** |
| FD003 | raw Euclidean | 13/48 (0.27) |
| FD003 | robust diagonal | **48/48 (1.00)** |

The scale-aware robust-diagonal geometry preserves every assignment under
the unit changes on both subsets; raw Euclidean preserves under a third or
fewer. The result replicates across FD001 and FD003. This is the one
preregistered hypothesis that meets its bar cleanly: scale-aware source
geometry delivers the decision-stability-under-unit-change property it
claims, on real turbofan sensors, in both fault-mode regimes.

## H4 — Certified vs. greedy clustering (**supported, modest**)

Per (geometry, radius, view) under the robust-diagonal geometry, 48
instances per subset:

- certified cluster count is **never worse** than the harness greedy
  first-fit reference (FD001: 2/48 strictly fewer clusters, 46/48 equal,
  0/48 worse; FD003 comparable);
- at equal cluster count, certified attains a strictly lower observed-medoid
  cost in 20/48 instances and equal cost in 26/48;
- all 48 instances are `proven_optimal` (the hybrid node budget of 2000
  sufficed — the certificate says so, it is not assumed).

Honest reading: on these decimated turbofan source sets the greedy pass is
*usually already count-optimal*, so certification most often adds proof
rather than a better partition — with a genuine minority (2/48 count, 20/48
cost) where the certified solver improves the objective. This is exactly the
"greedy matches everywhere is an acceptable outcome" branch the
preregistration named; the value here is the certificate, plus improvement
where it exists.

## H2 — Robust regression under contamination (**inconclusive; does not replicate**)

Paired leave-one-engine-out RMSE difference `OLS − robust` at 10 % coherent
training contamination, 95 % seeded bootstrap CI over 100 engines (0 units
dropped for fit failure on either subset):

| Workload | Comparison | Mean Δ | 95 % CI | Verdict |
|----------|-----------|-------:|---------|---------|
| FD001 | OLS − Huber | +0.586 | [−0.175, +1.355] | straddles 0 |
| FD001 | OLS − trimmed | −2.097 | [−4.406, +0.409] | straddles 0 (leans OLS) |
| FD003 | OLS − Huber | +1.520 | **[+0.369, +2.746]** | favours Huber |
| FD003 | OLS − trimmed | +0.007 | [−3.013, +3.147] | straddles 0 |

Huber improves on OLS with a CI excluding zero on **FD003 only**; the same
comparison straddles zero on FD001. The preregistered superiority rule
requires replication across both subsets, so the verdict is **no
significant difference / inconclusive** — Huber is not established as
superior. Trimmed least squares shows no advantage and slightly favours OLS
on FD001.

Median-of-means fails catastrophically on both subsets (RMSE ≈ 1.3×10⁵ on
FD001, 5.0×10⁵ on FD003 at 20 % contamination): pooled RUL is piecewise and
the coherent contamination touches enough seeded blocks that the
block-majority guarantee is void. This is the documented breakdown of the
method's assumption, printed rather than hidden.

## SECOM anomaly detection (**negative; exposed by the frozen test**)

Grid methods were selected on the validation split (best validation AUROC:
regularized Mahalanobis at ridge 0.001, val AUROC 0.648; LOF at k=20, val
AUROC 0.541) and then frozen for the test split:

| Detector | Test AUROC | Balanced accuracy |
|----------|-----------:|------------------:|
| regularized Mahalanobis | 0.469 | 0.486 |
| local outlier factor (k=20) | 0.441 | 0.455 |
| isolation forest | 0.424 | 0.455 |
| DBSCAN-noise | — (label-only) | 0.500 |
| Hotelling T² | — (degenerate fit) | — |

Every score-producing detector lands **at or below chance on the frozen test
split**, despite Mahalanobis reaching 0.648 on validation — a textbook
validation-to-test collapse that the frozen-test protocol exists to expose,
and that aggregate-only reporting would have hidden. DBSCAN flags every point
as noise (recall 1.0, balanced accuracy 0.5 — useless). Hotelling T² returns
a typed degenerate-fit failure (near-singular covariance across the imputed
sensor space). Honest verdict: **none of the evaluated unsupervised
detectors separates SECOM yield failures** in this feature space; SECOM is a
known-hard dataset and these results are consistent with that.

## SRCC stability and trust (**stable; trust comparison vacuous by construction**)

- **Leave-one-out stability** (FD001, 4 engines, 40 removal variants): mean
  projector Frobenius distance 0.0177, maximum 0.354, 38 stable dimensions —
  the historical exact pipeline is stable to single-sample removal on real
  data.
- **Trust** (one-view target-shift attack, Unweighted vs.
  GroupContaminationBound): projector Frobenius displacement **0.0 for both
  policies**. On continuous industrial sources every exact-source group is a
  singleton (no two views share a bit-identical source), so the adversarial
  margin has no group structure to act on and both policies are identical
  here. This measures — rather than assumes — that the trust mechanism is
  **not applicable** to continuous-source turbofan views; it is neither a
  success nor a failure of trust, and no trust benefit is claimed on this
  workload. The trust models remain validated on the synthetic
  identifiable-contamination battery of phase 725, where exact-source groups
  are populated by construction.

## Stream detection (**detects, with false alarms**)

Single-sensor CUSUM/EWMA on an FD003 engine's channel-11 trajectory with a
manifested contiguous burst (onset explicit):

| Chart | Detected | Delay (steps) | Pre-onset false alarms |
|-------|:--------:|--------------:|-----------------------:|
| CUSUM | yes | 1 | 9 |
| EWMA | yes | 0 | 16 |

Both charts detect the burst promptly but trip repeatedly before the onset —
the raw turbofan sensor is noisy enough that fixed `k`/`h` and `λ`/`L`
settings produce many pre-onset alarms. Detection is real; the false-alarm
budget is not met at these preregistered settings, and that is reported, not
tuned away.

## Overall

- **H1 (scale invariance):** supported and replicated — the strongest result.
- **H4 (certified clustering):** supported and modest — certificates always,
  objective improvement in a minority.
- **H2 (robust regression):** inconclusive — Huber beats OLS on FD003 but not
  FD001, so replication fails and no superiority is claimed; median-of-means
  breaks down as its assumption predicts.
- **SECOM anomaly:** negative — no evaluated detector beats chance on the
  frozen test; the validation-to-test collapse is the headline honest finding.
- **Trust:** not applicable to continuous-source views (measured null).
- **Streams:** detect the burst, miss the false-alarm budget.

No result in this report licenses the sentence "SRCC is superior." The one
clean win (H1) is a specific, testable property — decision invariance under
unit changes — not a blanket claim, and it is stated as exactly that.

## Limitations and deferred

- C-MAPSS regression pools RUL across engines with a linear model; RUL is
  piecewise and the pooled fit is a deliberately simple baseline, not a
  state-of-the-art RUL estimator. The point is the *comparison* under
  contamination, not absolute RUL accuracy.
- Decimation (stride 20) trades statistical power for tractability; the
  paired CIs are over 100 engines but each engine's fit uses ~10 cycles.
- SECOM used the top-of-file sensor columns after the train-fitted
  drop/impute policy; feature engineering (which the preregistration did not
  license mid-stream) might change the anomaly picture and is out of scope.
- Runtime and memory were not measured (declared side channels; no timings
  in hashed content).
