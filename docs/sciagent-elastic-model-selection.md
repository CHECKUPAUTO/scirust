# ElasticTokenizer robust model selection

## Objective

Select the execution model used by ElasticTokenizer from measured evidence rather than a single median benchmark. The candidate set is the production semantic set: `Reference`, `TinyScan`, `Indexed`, and `Heap`. Layout experiments (for example CSR-SoA rule tables) must first enter one of those implementations behind exact semantic parity before they can influence selection.

## SciRust tools used

The selection pipeline deliberately composes the relevant SciRust capabilities instead of introducing an external benchmark/statistics stack:

1. `ElasticAutotuner` executes every compatible tokenizer kernel on the same complete calibration pieces and checks every output against `CanonicalBpeOracle`.
2. `ElasticProfileFitter` synthesizes the six-class hardware-local router from semantics-safe Tukey-filtered costs, minimizing robust median sums first and p95 sums second.
3. `scirust-stats::describe::{mean, std_dev, median, quantile}` supplies center, dispersion, coefficient of variation and tail latency.
4. Tukey IQR fences reject scheduler/interrupt spikes before comparative statistics, following `scirust-core/examples/bench_ab_welch.rs`.
5. `scirust-stats::htest::t_test_two_sample(..., equal_var=false, Tail::TwoSided)` supplies Welch significance against the runner-up.
6. `scirust-metrology::allan_deviation` measures repeated single-case temporal stability at aggregation factors 1, 2 and 4. Allan metrics are never computed across mixed corpus cases.
7. Existing raw-report fingerprints bind tokenizer identity, exact calibration cases, hardware identity and timing protocol for reproducible before/after comparison.
8. Existing counterbalanced structural A/B workflows remain mandatory when changing one kernel implementation itself; model selection does not replace before/after promotion evidence.
9. The permanent workflow executes the same protocol independently on hosted x86_64 and Jetson AGX Thor ARM64. Results and fitted profiles are architecture-local.

No semantic mismatch can be compensated by speed, p-value, p95, dispersion, Allan stability or any other metric.

## Measurement protocol

For every complete calibration case:

- compute the canonical token-id result once;
- warm every compatible kernel;
- for each measured round, rotate the first kernel and time all compatible kernels in interleaved order;
- retain integer nanoseconds and exact semantic-match status.

The rotating order prevents one model from always running first or last and spreads slow thermal/scheduler drift across models.

The main selection campaign uses multiple real corpus cases per length. A second campaign uses exactly one case per selected length with 63 measured repetitions; only this single-case campaign is eligible for Allan analysis.

## Robust decision per piece length

For every semantically valid `(piece_len, kernel)` group:

1. reject the whole group if any sample produced a semantic mismatch;
2. compute Q1/Q3 and discard points outside `Q1 - 1.5 IQR .. Q3 + 1.5 IQR`;
3. compute mean, standard deviation, coefficient of variation, median and p95;
4. rank first by median, then p95, then coefficient of variation, then the stable kernel order only as a deterministic final tie-break;
5. compare the winner and runner-up with Welch's unequal-variance two-sided t-test;
6. classify evidence as:
   - `Strong`: p < 0.01;
   - `Significant`: p < 0.05;
   - `Provisional`: measured winner but statistically unresolved;
   - `Uncontested`: only one semantically valid model.

The report also records the median speedup `runner_up_median / winner_median`, number of Tukey outliers removed and, for valid single-case stability reports, Allan deviations and Allan/median ratio.

## Six-class production profile

The production profile is no longer fitted from unfiltered timing groups. Its dynamic program uses the same semantics gate and Tukey filtering as model selection. Segment cost is lexicographic:

1. sum of robust medians across the segment;
2. sum of robust p95 values only if median sums tie.

This deliberately avoids an arbitrary weighted score. Thresholds remain deterministic midpoints between adjacent measured probe lengths selected by the optimal six-segment path.

## Deployment rule

A production router profile remains hardware-local. Hosted x86_64 results are not evidence for Jetson AGX Thor. The same corpus fingerprint, tokenizer fingerprint, timing protocol, and candidate revision are measured independently on each deployment class. A layout optimization is promoted only after semantic parity, robust qualification, target-hardware calibration when applicable, and its own counterbalanced before/after A/B evidence.

Direct measured evidence is preferred to fitting an unrelated predictive model. SciRust interpolation, symbolic, evolutionary and ML crates are therefore not inserted into the decision path merely to increase tool count: they would add assumptions without improving the evidence for this discrete four-kernel routing problem.
