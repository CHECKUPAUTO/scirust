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
calibrates the hard deployment threshold with a safety reserve. On held-out
trajectories, the tool compares the frozen result with the best of 2,001 fixed
`gamma` values under the same pre-registered stale-cache loss budget.

## Measured synthetic result

The deterministic run `seed=20260804`, 600 CMA-ES steps, 400 independent
trajectories and a final quality-loss budget of `0.05` produced:

| Held-out test | Quality loss | Compute fraction | Refresh rate |
|---|---:|---:|---:|
| SciRust multi-signal policy | 0.04748920 | 0.18439180 | 0.19843750 |
| Best fixed gamma (`0.865`) | 0.04966883 | 0.49985299 | 0.50039062 |

Both policies satisfy the same quality constraint. The SciRust policy uses
`63.110794%` less normalized refresh compute and strictly Pareto-dominates the
best fixed threshold on this oracle. The exact coefficients and provenance are
versioned in `results/synthetic_seed_20260804.json`.

This result is **synthetic only**. It validates the discovery method; it does
not prove a gain on LLaDA, Dream, or another real diffusion LLM.

## Run

From the repository root:

```bash
cargo run --release \
  --manifest-path experiments/elastic-cache-policy/Cargo.toml -- \
  --seed 20260804 \
  --steps 600 \
  --max-quality-loss 0.05
```

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
