# 08 · SciRust Integration

> [← Extension API & Plugins](./07-extension-api-and-plugins.md) · [Workspace & Crates →](./09-workspace-and-crates.md)

SciRust is SDE's **default** backend — the batteries that make the engine useful
on day one — but it enters through the *same* plugin seam as any other backend
([07 §5](./07-extension-api-and-plugins.md#5-three-plugin-mechanisms-one-registry-three-transports)).
`sde-scirust` is one crate of adapters: it implements SDE's stage traits by
calling SciRust crates, and it depends only on `sde-registry`, never the reverse.
Rip it out and SDE still compiles; the pipeline just has no default algorithms.

Rust below is illustrative sketch (see [03 preamble](./03-object-model.md)).

---

## 1. The uniform adapter contract

The capability survey of SciRust surfaced one lucky, load-bearing fact: **almost
every relevant crate is already deterministic and already seed-aware.** `stats`,
`symreg`, `automl`, `nas` carry explicit `u64` seeds; `gp`, `solvers`, `units`,
`symbolic` are seedless-deterministic; `scirust-gpu` guarantees a bit-exact
`CpuBackend` oracle while `scirust-cuda` openly declares it is *not* bit-exact
(bf16, ~5e-2). That means one small adapter shape maps onto all of them:

```rust
/// Every sde-scirust adapter is this thin: thread the seed in, capture the
/// determinism level out, wrap the result in an SDE Object with provenance.
struct ScirustAdapter<F> {
    inner: F,                    // the SciRust call
    level: DeterminismLevel,     // declared once, matches bench-schema D0..D3
}
```

The adapter's whole job is **impedance matching**: SDE speaks `Object<B>` +
`seed` + determinism level; SciRust speaks plain function calls. The seed comes
from the stage context (mandatory — [06 §2](./06-provenance-and-reproducibility.md#2-the-environment-digest--the-briefs-traceability-list-itemized)),
and the level is declared to match `scirust-bench-schema::Certificate.determinism`
so SDE and SciRust share one determinism vocabulary end to end.

---

## 2. Stage → crate map (with exact API surfaces)

Each row is a stage trait from [07 §2](./07-extension-api-and-plugins.md#2-the-stage-trait-extension-api)
and the SciRust crate + entry point that `sde-scirust` calls to satisfy it.

| SDE stage | SciRust crate | Concrete entry point | Notes |
|---|---|---|---|
| **HypothesisGenerator** (equation discovery) | `scirust-symreg` | `discover(data, inputs, seeds, pop, gens, …) -> Vec<(usize, f64, Expr)>` | Returns a **precision/complexity Pareto front** — a *set* of hypotheses with a built-in Occam axis. Explicit `seeds: &[u64]`. |
| **HypothesisGenerator** (model search) | `scirust-automl` | `AutoML::new(cfg).fit(x, y, is_clf) -> AutoMLReport` | Pipeline + hyperparameter search; `AutoMLConfig.seed`; models via `trait Model`. |
| **HypothesisGenerator** (architecture search) | `scirust-nas` | `NasSearch::{random_architecture(seed), evolve}` | Evolutionary NAS over `LayerSpec`. |
| **HypothesisGenerator** (LLM-assisted) | `scirust-sciagent` | `SciAgentInference` / `Generator`, `temperature = 0.0` greedy | Deterministic decode path; outputs enter the DAG as ordinary, criticizable hypotheses (Invariant VI). |
| **Model representation** | `scirust-symbolic` · `scirust-gp` · `scirust-automl` | `Expr` · `GaussianProcess` · `AutoMLReport` model | Each satisfies SDE's `Model` trait ([03 §3](./03-object-model.md#3-models-and-how-a-domain-stays-generic)) via associated types. |
| **Predictor** (symbolic) | `scirust-symbolic` | `eval`, `diff`, `simplify` on `Expr` | Exact (L3); derivatives via built-in autodiff for sensitivity. |
| **Predictor** (forward models) | `scirust-solvers` | `ode` (RK4, Dormand-Prince 5(4)), `nonlinear` (Newton/Broyden), `quadrature` | `Tolerance{abs, rel, max_iter}` → L2 + certificate. |
| **Predictor** (with uncertainty) | `scirust-gp` | `GaussianProcess::fit(...).predict(x*) -> (mean, var)` | Predictive **variance** is the uncertainty SDE needs for EIG; L3 (pure Cholesky, `forbid(unsafe)`). |
| **Executor** (simulator) | `scirust-solvers` · `scirust-signal` · `scirust-sim` | ODE integrate · DSP pipeline · sim harness | The in-process L2/L3 executor kinds of [04 §5](./04-workflow-engine.md#5-the-effect-boundary-executors). |
| **EvidenceExtractor** | `scirust-signal` | `fft`, `psd`, `spectral_entropy`, `rms/kurtosis`, `wavelet_denoise`, CFAR/DoA | Reduces raw `Observation` time-series to hypothesis-relevant features. |
| **EvidenceExtractor** (summaries) | `scirust-stats` | `describe` (`mean/variance/quantile`), `robust` (`median_absolute_deviation`, `trimmed_mean`) | Robust reductions for noisy L0 observations. |
| **StatisticalEvaluator** (likelihood/tests) | `scirust-stats` | `Distribution` trait + `t_test_two_sample`, `one_way_anova`, `chi_square_gof`, `ks_test_one_sample` | Distributions give the likelihood term for Bayes updating; tests give frequentist goodness-of-fit. Seeded `SplitMix64`. |
| **StatisticalEvaluator** (paired model compare) | `scirust-automl` | `paired_t_test` | For head-to-head model comparison with CV folds. |
| **HypothesisRanker** | `scirust-stats` + `sde-infotheory` | marginal likelihood → Bayes factors → posterior | Ranking is Bayes-factor ordering ([05 §1](./05-information-theory.md#1-the-bayesian-core)). |
| **InfoGain / Planner** | `scirust-gp` · `scirust-automl` · `scirust-stats` | see §3 | The information-theoretic core. |
| **TheoryReviser / contradiction** | `scirust-symbolic` · `scirust-units` | `prove_equal(a, b) -> bool` · `Dimension` / `Quantity` checked arithmetic | Symbolic + **dimensional** contradiction detection (§5). |
| **Prior-art / literature retrieval** | `scirust-retrieval` | `SemanticRetriever`, `Bm25Index`, `HybridRetriever`, `reciprocal_rank_fusion` | Deterministic, auditable retrieval to link `Evidence` to prior findings and to seed hypotheses from `docs/kb/literature`. |
| **Provenance / signing** | `scirust-provenance` (`scirust-license::hashsig`) | `sign_artifact`, `verify_artifact -> Verdict`, `MerkleSigner` | [06 §4](./06-provenance-and-reproducibility.md#4-signing--tamper-evidence). |
| **Determinism backbone** | `scirust-gpu` · `scirust-cuda` | `CpuBackend` (L3 oracle) · `DeterministicValidator` · cuda bf16 (L2) | The engine can *validate a backend against the CPU oracle* and record the verdict. |

---

## 3. The planner adapter (the interesting one)

The planner's continuous design search is **Bayesian optimization over the
utility surface**, and SDE does not reinvent it — it adapts what SciRust ships:

- **Closed-form EIG for GP surrogates.** `scirust-gp` exposes `predict → (mean,
  var)` and `log_marginal_likelihood()`. For a GP model, the EIG about
  parameters at a design `ξ` is an analytic function of the predictive variance
  — no Monte Carlo needed. This is the cheap, L3 fast path of
  [05 §4](./05-information-theory.md#4-estimators-eig-is-hard-be-honest-about-it).
- **BO machinery already exists.** `scirust-automl` ships `bayesian_optimize`
  and `expected_improvement` over an internal GP. `sde-planner` reuses this loop
  to maximize `U(ξ) = EIG(ξ)/cost(ξ)` over a continuous design box, treating the
  utility surface as the objective. (Note: `automl`'s GP is *internal and
  distinct* from `scirust-gp` — the adapter picks one explicitly and records
  which, so the choice is reproducible.)
- **Nested Monte-Carlo where needed.** For discrete hypothesis discrimination
  with non-Gaussian likelihoods, the adapter falls back to seeded sampling via
  `scirust-stats`' `SplitMix64`, and — per [05 §4](./05-information-theory.md#4-estimators-eig-is-hard-be-honest-about-it)
  — attaches the estimator's own standard error so the planner never treats a
  noisy EIG as exact.

```rust
// illustrative: EIG for a GP surrogate is closed-form in predictive variance
impl InfoGain for GpEigAdapter {
    fn eig(&self, d: &DesignCandidate, b: &BeliefState, _t: Target) -> Estimate {
        let (_mean, var) = self.gp.predict(&d.x);          // scirust-gp
        Estimate { value: 0.5 * (1.0 + var / self.noise).ln() / LN_2,  // bits
                   se: 0.0, level: DeterminismLevel::L3 }   // analytic ⇒ exact
    }
}
```

---

## 4. Domain map — which SciRust crates power which field

The brief insists SDE must not be physics-only. Because a `Domain` is one small
trait ([07 §3](./07-extension-api-and-plugins.md#3-the-domain-contract--one-small-trait-to-enter-a-field)),
each field in the brief becomes a `Domain` impl that leans on the SciRust crates
already in this workspace:

| Domain (from the brief) | Primary SciRust crates | What the `Domain` gets cheaply |
|---|---|---|
| **Mathematics** | `scirust-symbolic`, `scirust-solvers`, `scirust-special`, `scirust-fractional` | equation hypotheses, symbolic prediction, exact evaluation |
| **Optimization** | `scirust-solvers` (`optimize`: BFGS, Nelder-Mead, SPG), `scirust-automl` (`HyperOptimizer`) | designs = optimizer configs; observables = objective values |
| **Signal processing** | `scirust-signal` | evidence extraction, spectral hypotheses, radar/vibration domains |
| **AI / ML** | `scirust-learning`, `scirust-automl`, `scirust-nas`, `scirust-autodiff`, `scirust-som`, `scirust-sciagent` | model-search hypotheses; datasets/metrics as designs/observables |
| **Finance** | `scirust-trader`, `scirust-forecast`, `scirust-seasonal`, `scirust-sequential`, `scirust-finmigrate` | strategy hypotheses; backtests as record/replay executors |
| **Biology** | `scirust-biomed` | dose-response / assay models; L0 instrument observations |
| **Chemistry** | `scirust-thermo`, `scirust-units`, `scirust-fluids` | reaction/thermo models; dimensional safety via `units` |
| **Quantum simulation** | `scirust-tn` (tensor networks); quantum track on the roadmap (`docs/research/SCIRUST_QUANTUM_ROADMAP`) | state hypotheses; simulator executors |
| **Engineering** | `scirust-control`, `scirust-fatigue`, `scirust-machining`, `scirust-electrotech`, `scirust-civil`, `scirust-hvac`, `scirust-reliability`, `scirust-tolerance` | plant models; tolerance/reliability observables |
| **Robotics** | `scirust-robotics`, `scirust-nav`, `scirust-estimation`, `scirust-control`, `scirust-signal` | dynamics hypotheses; sensor-rig L0 executors; state estimation as evidence |

Cross-cutting crates every domain reuses: `scirust-units` (dimensional
correctness of predictions vs. observables), `scirust-stats` (likelihoods),
`scirust-metrology` (measurement uncertainty), `scirust-gpu` (the L3 oracle).

---

## 5. Contradiction detection, concretely

Two SciRust facilities give SDE contradiction detection almost for free, at two
different levels:

1. **Structural (pre-data).** `scirust-units` `Quantity::try_add` /
   `Dimension` arithmetic returns a checked error when dimensions don't match. A
   `Prediction` whose dimensions disagree with the target `Observable` is a
   **structural contradiction caught before any experiment runs** — the cheapest
   possible refutation.
2. **Symbolic (model-level).** `scirust-symbolic::prove_equal(a, b)` (a
   deterministic 200-point agreement check over `Expr`) flags when two accepted
   laws cannot both hold, or when a hypothesis contradicts an established
   invariant. `sde-theory` records the result as a `Contradiction` object.

Both are deterministic, so a detected contradiction is itself a reproducible,
citable node — not a runtime surprise.

---

## 6. Determinism, honestly, at the backend seam

The determinism taxonomy ([01 §6](./01-vision-and-philosophy.md#6-determinism-honestly-the-taxonomy))
lands precisely on SciRust's own guarantees, which is why the two fit:

- **`scirust-gpu::CpuBackend` is the permanent L3 oracle.** SDE treats it as the
  bit-exact reference and can validate any other backend against it with
  `DeterministicValidator`, recording the verdict as an object. "Is this GPU
  trustworthy for this study?" becomes an empirical, logged decision.
- **`scirust-cuda` is honestly L2.** It declares bf16 inputs / fp32 accumulate /
  ~5e-2 relative tolerance and is *not* bit-identical. The adapter tags its
  outputs L2 with that tolerance as the `Certificate` — so a study that used the
  GPU path says so, and its reproducibility contract reflects it.
- **Seeded stochastic crates are L1.** `symreg`, `automl`, `nas`, and
  `stats` samplers reproduce *in distribution* given their recorded `u64` seed —
  the L1 guarantee, with the seed captured in `ReproMeta`.

---

## 7. Other backends enter identically

Nothing above is SciRust-privileged. A `sde-python` adapter (NumPy/PyTorch over
MCP), an HPC executor (a Slurm job), or a lab-robot executor implements the same
stage traits and registers the same `PluginDescriptor`. The only difference is
the declared determinism level (often L0/L1 for external/physical backends) and
the transport (MCP/wire instead of in-process). SciRust is the *default*, not
the *definition* — which is the backend-agnostic requirement, satisfied
structurally.

---

> [← Extension API & Plugins](./07-extension-api-and-plugins.md) · [Workspace & Crates →](./09-workspace-and-crates.md)
