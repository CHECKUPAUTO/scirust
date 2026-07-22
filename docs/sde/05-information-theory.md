# 05 · Information Theory & Planning

> [← Workflow Engine](./04-workflow-engine.md) · [Provenance & Reproducibility →](./06-provenance-and-reproducibility.md)

The brief names one *major* objective: **maximise scientific information.** This
chapter specifies the subsystem (`sde-infotheory`) that estimates expected
information gain, experiment utility, uncertainty reduction, and hypothesis
discrimination — and the planner (`sde-planner`) that turns those estimates into
"run this experiment next."

Rust below is illustrative sketch (see [03 preamble](./03-object-model.md)).

---

## 1. The Bayesian core

Everything here rests on one commitment: **belief is a probability distribution,
and evidence updates it by Bayes' rule.** Two coupled uncertainties are tracked:

- **Which hypothesis is right** — a distribution `P(H)` over a (possibly
  countable) hypothesis set, `H ∈ {H₁…H_k}`.
- **What the parameters are** — for each hypothesis, a posterior `P(θ | H, D)`
  over its parameter space, given data `D`.

A `Confidence` object ([03 §2](./03-object-model.md#2-the-object-catalog)) is a
serialized snapshot of this state: the posterior over hypotheses, the per-model
parameter posteriors (or sufficient summaries), and the derived quantities
below. It is produced by `sde-statistics` and consumed by `sde-ranking` and
`sde-planner`.

```rust
pub struct BeliefState {
    pub hypotheses: Vec<(ObjectId, f64)>,      // posterior P(H_i | D), Σ = 1
    pub params: Vec<ParamPosterior>,           // P(θ | H_i, D) per hypothesis
    pub log_evidence: Vec<f64>,                 // log marginal likelihood per H_i
}
```

Model comparison is then standard and fully explicit:

- **Marginal likelihood** `P(D | H_i) = ∫ P(D | θ, H_i) P(θ | H_i) dθ` — the
  evidence for a hypothesis, computed by `sde-statistics`.
- **Bayes factor** `B_ij = P(D|H_i) / P(D|H_j)` ranks competing models; its log
  is the natural "how much did the data favor i over j" in nats/bits.
- **Posterior over hypotheses** `P(H_i|D) ∝ P(D|H_i) P(H_i)`.

Every one of these is a number derived from objects in the graph — no opaque
score (Invariant VI). The likelihood model itself is a named, versioned object,
so "under what assumptions" is always answerable.

---

## 2. Information gain (the core quantity)

The information a dataset `D` carries about the unknown `Θ` (a hypothesis
identity, a parameter, or both) is the reduction in Shannon entropy:

```
IG(D) = H(Θ) − H(Θ | D)        # bits of uncertainty removed by having seen D
```

But when *planning*, we have not run the experiment yet — we do not know `D`. So
we take the expectation over what the experiment *might* yield, under a design
`ξ`. This is the **Expected Information Gain**, the central object of Bayesian
Optimal Experimental Design (BOED):

```
EIG(ξ) = 𝔼_{D ~ P(D | ξ)} [ H(Θ) − H(Θ | D, ξ) ]
       = I(Θ ; D | ξ)          # the mutual information between the unknown
                               # and the (future) observation, under design ξ
```

`EIG(ξ)` is the number the planner maximises. Intuitively: **the best next
experiment is the one whose outcome we can least predict given what we already
believe** — because an outcome we can already predict teaches us nothing.

Two specializations matter, and `sde-infotheory` computes both:

| Target `Θ` | EIG measures | Use |
|---|---|---|
| **Hypothesis identity** `H` | expected reduction in entropy of `P(H)` → **hypothesis discrimination** | choose the experiment that best separates competing models |
| **Parameters** `θ` | expected shrinkage of `P(θ\|H)` → **uncertainty reduction / refinement** | choose the experiment that best pins down a model you already believe |

```rust
pub trait InfoGain {
    /// Expected information gain (bits) of a candidate design about a target.
    fn eig(&self, design: &DesignCandidate, belief: &BeliefState, target: Target)
        -> Estimate;    // returns a value AND its own uncertainty (see §4)
}
```

---

## 3. Experiment utility = information per unit cost

Raw EIG ignores that experiments cost money, time, samples, and risk. The
planner optimizes **utility**, which folds in the `Experiment` cost model:

```
U(ξ) = EIG(ξ) / cost(ξ)          # default: bits per unit cost
```

Utility is a **pluggable policy**, not a hardcoded ratio — because "most
informative" is domain-relative:

| Policy | `U(ξ)` | When |
|---|---|---|
| `eig_per_cost` (default) | EIG(ξ) / cost(ξ) | resource-bounded labs |
| `eig_budgeted` | EIG(ξ) s.t. cost(ξ) ≤ B | fixed per-experiment budget |
| `knowledge_gradient` | expected improvement in the *decision* the study feeds | decision-driven studies |
| `min_max_regret` | worst-case discrimination over an adversarial prior | safety-critical / robust design |
| `custom` | any `UtilityPolicy` plugin | domain-specific value of information |

The planner selects `ξ* = argmax_ξ U(ξ)` over the candidate design space, and —
because every quantity is recorded — can *defend* the choice: "the assay buys
2.3 bits at 1/3 the cost of RNA-seq, so U is 6.9× higher." That sentence, with
its numbers, becomes a `Plan` object in the graph.

---

## 4. Estimators (EIG is hard; be honest about it)

EIG is a nested expectation (an outer expectation over data, each term an inner
posterior entropy) and is **expensive and biased** if estimated naively. This is
the single hardest piece of numerical work in SDE, so the subsystem offers a
ladder of estimators with declared cost and bias, and **every EIG estimate
carries its own uncertainty** (`Estimate { value, se, level }`) — the planner
never treats a noisy EIG as exact.

| Estimator | Cost | Bias | Backed by |
|---|---|---|---|
| **Closed-form** (linear-Gaussian, GP) | cheap | none | `scirust-gp`: posterior variance is analytic, so EIG for a GP surrogate is a closed form of predictive variance |
| **Nested Monte Carlo** | high | O(1/M) bias, O(1/N) variance | `scirust-stats` samplers (seeded `SplitMix64`) |
| **Variational bounds** (e.g. likelihood-ratio / contrastive lower bounds) | medium | lower-bounds EIG (safe for maximization) | `sde-infotheory` + backend models |
| **Laplace / moment approximations** | cheap | approximation error, certified where possible | `scirust-solvers` (Hessians via `scirust-autodiff`) |
| **Surrogate-accelerated** | amortized | surrogate error | fit a `scirust-gp` over `ξ ↦ EIG(ξ)` and optimize the cheap surrogate |

The surrogate-accelerated path is worth spelling out because it reuses
machinery this workspace already ships: `scirust-automl` contains
`bayesian_optimize` / `expected_improvement` over an internal GP, and
`scirust-gp` gives closed-form predictive variance. The planner's continuous
design search *is* Bayesian optimization over the utility surface — SDE does not
reinvent it, it adapts it (see [08](./08-scirust-integration.md#3-the-planner-adapter)).

---

## 5. Uncertainty reduction & contradiction as information events

- **Uncertainty reduction** is reported directly: prior entropy `H(Θ)` minus
  realized posterior entropy `H(Θ|D)` after an experiment resolves. A study's
  "progress" is a monotone-ish decreasing entropy curve, itself a series of
  `Confidence` objects — you can *watch* a question get answered.
- **Contradiction detection** (a brief requirement) has an information-theoretic
  reading: a hypothesis whose posterior mass collapses below a threshold given
  evidence is *excluded*; a pair of accepted theories that make incompatible
  predictions on the same design is a *conflict*. `sde-theory` records both as
  explicit `Contradiction` objects. Symbolic contradictions (two laws that
  cannot both hold) are checkable with `scirust-symbolic::prove_equal` and with
  `scirust-units` dimensional analysis (a prediction whose dimensions don't
  match the observable is a *structural* contradiction caught before any data).
- **Discrimination** between two live hypotheses is the expected log-Bayes-factor
  under a design — literally "how many bits will this experiment move us toward
  one model over the other," which is what a good discriminating experiment
  maximizes.

---

## 6. The planner interface

The planner ties it together: given the current belief, a candidate design
space, a utility policy, and a budget, recommend the next experiment (or signal
that information is exhausted).

```rust
pub trait Planner {
    fn recommend(
        &self,
        belief: &BeliefState,
        designs: &DesignSpace,     // discrete set, or continuous box for BO
        utility: &dyn UtilityPolicy,
        budget: &Budget,
    ) -> Plan;   // ranked designs + their EIG/cost/utility estimates, or STOP
}
```

The returned `Plan` is an object: a ranked list of candidate experiments, each
annotated with its EIG estimate (and *that* estimate's uncertainty), its cost,
and its utility — plus the stopping verdict. It feeds straight back into
`sde-experiment` ([04 §4](./04-workflow-engine.md#4-the-iteration-controller--stopping-rules)).

**Myopic by default, non-myopic by extension.** The default planner is
one-step-greedy (maximize immediate EIG/cost), which is cheap and often optimal
enough. A `SequentialPlanner` plugin can look ahead (finite-horizon BOED /
active learning), trading compute for better long-run experiment sequences —
this is flagged as a research direction in [10](./10-roadmap-risks-future.md#future-research-directions),
not a v1 promise.

---

## 7. Worked micro-example (why the number matters)

Suppose three hypotheses about a growth law remain, with posterior
`P(H) = [0.5, 0.3, 0.2]` — entropy `H(H) ≈ 1.49` bits. Two experiments are on
the table:

- **Design A** (cheap) is predicted to produce nearly the same observable under
  all three hypotheses. Its outcome is highly predictable → `EIG_A ≈ 0.05` bits.
- **Design B** (2× the cost) is predicted to split `H₁` from `{H₂,H₃}` cleanly.
  Its outcome is much less predictable → `EIG_B ≈ 0.9` bits.

Utility: `U_A = 0.05/1 = 0.05`, `U_B = 0.9/2 = 0.45`. The planner recommends
**B**, and the graph now contains the sentence "B was chosen because it buys
0.9 bits vs 0.05, and 0.45 utility vs 0.05" — reproducibly, citably, and
defensibly. If instead `EIG_B` had come back at `0.02 ± 0.03` bits, the planner
would report that *no* experiment clears the `eig < ε` floor and the study
**stops** — information is exhausted. That honest "we can't learn more here" is
a first-class output, not a silent loop-forever.

---

> [← Workflow Engine](./04-workflow-engine.md) · [Provenance & Reproducibility →](./06-provenance-and-reproducibility.md)
