# SciRust Elastic Cache Policy Discovery

This isolated research crate replaces the competing Elastic-Cache project's
single fixed threshold

```text
refresh iff mean_attention_cosine < gamma
```

with a deterministic, interpretable policy discovered by SciRust.

The learned risk uses eight runtime signals:

```text
drift, worsening trend, head disagreement, cache age,
untracked attention mass, layer depth, drift×age, refresh cost
```

SciRust's seeded CMA-ES fits the risk weights. A deterministic validation sweep
then calibrates the hard deployment threshold to an explicit stale-cache loss
budget. On held-out trajectories, the tool compares the result with the best of
2,001 fixed gamma values under the learned policy's measured quality loss.

## Run

From the repository root:

```bash
cargo run --release \
  --manifest-path experiments/elastic-cache-policy/Cargo.toml -- \
  --seed 20260804 \
  --steps 600 \
  --max-quality-loss 0.05
```

The default run uses a deterministic nonlinear synthetic oracle. It validates
only the discovery machinery; it is not evidence of a gain on LLaDA or Dream.

Real trace:

```bash
cargo run --release \
  --manifest-path experiments/elastic-cache-policy/Cargo.toml -- \
  --trace artifacts/elastic-cache/trace.csv \
  --seed 20260804 \
  --steps 1200 \
  --max-quality-loss 0.01
```

Add `--symbolic` to run `scirust-symreg` and produce a Pareto front of compact
symbolic surrogates for stale-loss-per-refresh-cost.

## Required trace

```text
trajectory_id,step,layer_id,similarity,similarity_delta,head_variance,cache_age,attention_mass,layer_fraction,refresh_cost,stale_loss
```

Rows must be produced by an offline counterfactual dual run of the same decode
state: one reuse path and one forced-refresh path. Split is by whole trajectory
ID, never by individual row.

## Tests

```bash
cargo test --manifest-path experiments/elastic-cache-policy/Cargo.toml
cargo test --release --manifest-path experiments/elastic-cache-policy/Cargo.toml -- --ignored
```
