# ElasticTokenizer robust model selection

## Objective

Select the execution model used by ElasticTokenizer from measured evidence rather than a single median benchmark. The candidate set is the production semantic set: `Reference`, `TinyScan`, `Indexed`, and `Heap`. Layout experiments (for example CSR-SoA rule tables) must first enter one of those implementations behind exact semantic parity before they can influence selection.

## SciRust tools used

The selection pipeline deliberately composes existing SciRust capabilities instead of introducing an external benchmark/statistics stack:

1. `ElasticAutotuner` executes every compatible tokenizer kernel on the same complete calibration pieces and checks every output against `CanonicalBpeOracle`.
2. `ElasticProfileFitter` retains the existing six-class hardware-local routing model.
3. `scirust-stats::describe::{median, quantile}` supplies robust center and quantiles.
4. Tukey IQR fences reject scheduler/interrupt spikes before comparative statistics, following `scirust-core/examples/bench_ab_welch.rs`.
5. `scirust-stats::htest::t_test_two_sample(..., equal_var=false, Tail::TwoSided)` supplies Welch significance against the runner-up.
6. Existing raw-report fingerprints continue binding tokenizer, exact cases, hardware identity, and timing protocol for reproducible before/after comparison.

No semantic mismatch can be compensated by speed, p-value, p95, or any other metric.

## Measurement protocol

For every complete calibration case:

- compute the canonical token-id result once;
- warm every compatible kernel;
- for each measured round, rotate the first kernel and time all compatible kernels in interleaved order;
- retain integer nanoseconds and exact semantic-match status.

The rotating order prevents one model from always running first or last and spreads slow thermal/scheduler drift across models.

## Robust decision per piece length

For every semantically valid `(piece_len, kernel)` group:

1. reject the whole group if any sample produced a semantic mismatch;
2. compute Q1/Q3 and discard points outside `Q1 - 1.5 IQR .. Q3 + 1.5 IQR`;
3. compute clean median and p95;
4. rank first by median, then p95, then the stable kernel order only as a deterministic tie-break;
5. compare the winner and runner-up with Welch's unequal-variance two-sided t-test;
6. classify evidence as:
   - `Strong`: p < 0.01;
   - `Significant`: p < 0.05;
   - `Provisional`: median winner but statistically unresolved;
   - `Uncontested`: only one semantically valid model.

The report also records the median speedup `runner_up_median / winner_median` and number of Tukey outliers removed.

## Deployment rule

A production router profile remains hardware-local. Hosted x86_64 results are not evidence for Jetson AGX Thor. The same corpus fingerprint, tokenizer fingerprint, timing protocol, and candidate revision must be measured on each deployment class. A layout optimization is promoted only after semantic parity, robust hosted qualification, and target-hardware calibration when that hardware is a deployment target.

The selection layer does not invent a weighted score mixing latency, correctness, and confidence. Correctness is a hard gate; latency determines the candidate winner; tail latency and statistical confidence characterize whether that winner is sufficiently robust to promote.
