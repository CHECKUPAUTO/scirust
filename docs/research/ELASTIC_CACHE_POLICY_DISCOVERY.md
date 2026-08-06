# SciRust Elastic Cache Policy Discovery

Status: experimental research tool  
Scope: replace the fixed cosine threshold used by the attached Elastic-Cache diffusion-LLM project with a deterministic, interpretable policy discovered by SciRust.

## 1. Problem

The competing implementation refreshes downstream layers when

\[
\bar c_{t,\ell}<\gamma,
\]

where \(\bar c_{t,\ell}\) is the mean cosine similarity between old and current attention distributions of a small tracked-token set. Its scripts use fixed values such as `gamma=0.9` or `gamma=0.98`.

This decision discards information already available at runtime:

- whether similarity is falling rapidly;
- disagreement between attention heads;
- cache age;
- attention mass represented by the tracked tokens;
- layer depth and therefore refresh cost;
- interactions such as high drift on an old cache.

A fixed threshold is optimal only if similarity is a sufficient statistic and the cost of refreshing is constant. Neither assumption is established by the competing project.

## 2. SciRust formulation

The deployment policy is deliberately small:

\[
R(x)=w_1d+w_2\dot d_-+w_3\sqrt{v_h}+w_4a+w_5(1-m)+w_6\ell+w_7da+w_8C_{\mathrm{refresh}},
\]

with:

- \(d=1-\operatorname{cos}\): current attention drift;
- \(\dot d_-=\max(-\Delta\operatorname{cos},0)\): worsening trend;
- \(\sqrt{v_h}\): normalized standard deviation across heads;
- \(a\): normalized cache age;
- \(m\): normalized tracked-token attention mass;
- \(\ell\): normalized layer index;
- \(da\): drift-age interaction;
- \(C_{\mathrm{refresh}}\): measured or normalized recomputation cost.

The cache is refreshed iff

\[
R(x)\ge 0.
\]

The policy is therefore auditable, branch-cheap, and directly translatable to Rust, C++, CUDA, or Python.

## 3. Counterfactual training target

A real trace must be collected in an **offline dual-run oracle**. For the same decode state, execute:

1. the reuse path;
2. the forced-refresh path.

Define the stale-cache loss, for example, as

\[
L_{\text{stale}}
=\alpha\bigl(1-\cos(o_{\text{reuse}},o_{\text{refresh}})\bigr)
+\beta\,D_{KL}(p_{\text{refresh}}\Vert p_{\text{reuse}}),
\]

then normalize it to `[0,1]`. This oracle is used only during research; deployed inference never computes both paths.

The normalized refresh cost is

\[
C_{\text{refresh}}\approx\frac{L-\ell-1}{L},
\]

or, preferably, the measured layer-time fraction on the target hardware.

## 4. Optimization

For a policy with refresh probability \(p_i=\sigma(R(x_i)/T)\), the current implementation uses seeded SciRust CMA-ES to minimize

\[
J_\varepsilon
=\frac{\sum_i p_iC_i}{\sum_i C_i}
+\rho\left[\max\left(0,
\frac{\sum_i(1-p_i)L_i}{\sum_i L_i}-\varepsilon
\right)\right]^2
+\eta\lVert w\rVert_2^2,
\]

where \(\varepsilon\) is the pre-registered stale-loss budget. The tool then:

1. fits smooth risk weights on complete training trajectories;
2. calibrates the hard deployment threshold on validation trajectories;
3. compares the frozen policy on held-out trajectories with a complete sweep of 2,001 fixed \(\gamma\) values under the same measured quality bound;
4. optionally invokes `scirust-symreg` to discover a compact symbolic surrogate for stale-loss-per-refresh-cost.

Repeating the experiment over several \(\varepsilon\) values yields a quality/compute Pareto front; the first implementation runs one explicit budget per invocation.

This avoids the unfair comparison “learned policy versus gamma 0.9”. The baseline is the **best possible fixed gamma under the same measured quality budget**.

## 5. Trace format

Strict CSV header:

```text
trajectory_id,step,layer_id,similarity,similarity_delta,head_variance,cache_age,attention_mass,layer_fraction,refresh_cost,stale_loss
```

All values except `similarity_delta` must be finite and normalized to `[0,1]`. `similarity_delta` must lie in `[-1,1]`.

Example:

```text
trajectory_id,step,layer_id,similarity,similarity_delta,head_variance,cache_age,attention_mass,layer_fraction,refresh_cost,stale_loss
42,17,8,0.934,-0.021,0.083,0.625,0.712,0.286,0.749,0.137
```

## 6. Running the SciRust discovery binary

Synthetic deterministic smoke test:

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
  --trace artifacts/elastic-cache/train.csv \
  --seed 20260804 \
  --steps 1200 \
  --max-quality-loss 0.01 \
  --symbolic
```

The no-trace mode is only a deterministic integration check. It deliberately uses a nonlinear multivariate oracle on which similarity alone is insufficient. It is **not evidence of a gain on LLaDA or Dream**.

## 7. Required evaluation protocol

Split by complete prompts or generation trajectories, never by individual rows:

- training: fit CMA-ES coefficients and symbolic surrogate;
- validation: choose the quality budget and one candidate;
- test: freeze the policy and compare against the gamma sweep.

Report:

- task accuracy / pass@1;
- output or logit divergence;
- full refresh-equivalent compute fraction;
- wall-clock latency and throughput;
- GPU peak memory;
- refresh frequency by layer;
- false-reuse loss and unnecessary-refresh cost;
- repeated-run determinism;
- confidence intervals across prompts.

A learned policy is better only if it Pareto-dominates the best gamma baseline on the held-out test set, or improves one objective under a pre-registered bound on the other.

## 8. Integration with ElasticKvCache

The discovered risk should become one signal in ElasticKvCache, not an isolated replacement:

\[
\text{decision}=f(\sigma_E,\lambda,I,\text{age},\text{drift},\text{cost},\text{codec},\text{state}).
\]

Possible actions are richer than the competitor's binary reuse/refresh decision:

```text
KEEP → DROP_CORRECTION → WARM → REFRESH → COLD → REHYDRATE
```

The first experiment intentionally restricts the action space to `REUSE` versus `REFRESH` so the learned policy can be compared directly and fairly with fixed gamma. Multi-action ElasticKvCache discovery comes after this baseline is validated.

## 9. Reproducibility guarantees

- seeded SciRust CMA-ES;
- deterministic synthetic trace generator;
- total-order floating-point sorting;
- deterministic tie-breaking by row index;
- strict finite/range validation;
- no hidden timestamps or random system entropy.

The discovered coefficients must be versioned with:

- SciRust commit;
- trace manifest and hashes;
- model/checkpoint hash;
- dataset split hashes;
- hardware profile;
- optimizer seed and step count;
- selected quality budget.

## 10. Source competitor snapshot

The initial audit used the user-supplied archive `Elastic-Cache-main.zip` with
SHA-256 `1ba7593c48b9aea82df7369c5424ea5602b57168c8cc4e5f68d8ab9476dbc4e4`.
The archive itself is not vendored. Only the abstract decision rule and the
required measurement schema are retained in SciRust.
