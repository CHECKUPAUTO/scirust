# ADR 0008: A stochastic capability can be asked for many realisations

## Status

Accepted and implemented (Phase 3B-3). Supersedes two consequences recorded
in [ADR 0007](0007-seeded-stochastic-capabilities.md) — see *What this changes
in 0007* below.

## Context

ADR 0007 made the seed real and added the first stochastic capability, and
closed by naming what it had not done:

> **What this does not address.** Ensembles. A single seeded path answers
> "what does one realisation look like"; many questions about a stochastic
> model ("what is the distribution of the first passage time") need many paths
> and a result schema that can hold them.

That is the gap this closes. It was worth closing now rather than later for a
reason that has nothing to do with stochastic models: an ensemble changes the
shape of a `RunResult`, and every adapter constructs one. The cost of the
change grows with the catalogue.

The scientific case is simpler. `dX = θ·(μ − X)·dt + σ·dW` produces a sample.
One sample cannot answer where the process sits on average, how wide the
spread is, or whether the sample mean converges where the theory says. A run
that shows one path and calls it the model's behaviour is showing an anecdote.

## Decision

### 1. `experiment.replicates`, derived from the same one seed

```toml
[experiment]
seed = 42
replicates = 256
```

The replicates' seeds are **derived** from `experiment.seed`, so an ensemble
is reproducible from the same single number a lone run is. Nothing else is
added to the scenario.

### 2. The seed derivation is scrambled, for a stated reason

This is the part that is easy to get wrong quietly.

`SplitMix64` keeps a 64-bit state that advances by a fixed odd constant
`γ = 0x9e3779b97f4a7c15` on every draw, and `SplitMix64::new` takes the seed
**as** the initial state. Every seed therefore lands on the *same* cycle of
length 2⁶⁴, at a different offset. Two replicates whose seeds differ by
`m·γ (mod 2⁶⁴)` produce the same noise stream shifted by `m` draws — for small
`m`, two "independent" realisations sharing most of their randomness. That
would bias every across-replicate statistic while looking entirely healthy.

Counting seeds (`base`, `base+1`, …) does not actually trigger this: for a
small difference `d` to equal `m·γ (mod 2⁶⁴)` requires an enormous `m`. But
that makes the scheme's safety an arithmetic accident rather than a property
anyone stated. Passing the base seed through SplitMix64's output mixer instead
makes the pairwise offsets uniform over the cycle, so the probability that any
of `n` replicates overlaps another within `L` draws is about `n²·L/2⁶⁴` — and
it is that for a reason. This is what SplitMix64 was published to do.

`ensemble::tests::no_two_replicate_streams_overlap_within_any_plausible_path`
recovers `m` for every pair by inverting `γ` modulo 2⁶⁴ and asserts none is
within any path this crate could integrate. A companion test characterises
the counting scheme, so the claim that it "would have been safe by accident"
is checked rather than asserted.

**Replicate 0 keeps the base seed unchanged.** That buys two things: a
one-replicate run is bit-for-bit the run this capability already did — so the
outputs quoted in ADR 0007 and in the tutorial still stand — and
`replicate_seeds(base, 4)` is a prefix of `replicate_seeds(base, 64)`, so
enlarging an ensemble adds realisations rather than replacing the ones already
reported.

### 3. The summary is accumulated in one pass

4 096 replicates of a 40 001-sample path is 1.3 GiB of `f64` held in order to
produce two vectors of 40 001. `EnsembleAccumulator` folds each realisation in
as it is drawn, costing `O(samples)` regardless of the replicate count. A
large ensemble is a question of patience, not of memory.

The update is **Welford's**, not `E[x²] − E[x]²`. The shortcut loses
catastrophically when the variance is small next to the mean — which is
exactly an OU ensemble at stationarity, sitting at `μ` with a spread that is
routinely orders of magnitude smaller. For three values of variance 1 on a
mean of 1e9 it returns 0, because the ulp of 1e18 is 128. It can also return a
negative variance. Welford's form costs one extra subtraction and cannot.

### 4. Individual realisations are retained in bounded number, and it is said

The summary is what the run is for and costs one path's worth of memory. Eight
individual realisations are kept so a reader can see that the realisations
genuinely differ and the band is not decoration.

Dropping the other 248 is a real loss of information, so it is *reported*:
both `replicates` and `retained_members` are metrics, the CLI prints
`256 independent realisations, 8 kept in the result (248 not stored)`, and the
chart's text alternative says how many it is drawing. A truncation nobody is
told about reads as "here is the ensemble".

### 5. The check an ensemble exists to make possible

Along one realisation the samples are autocorrelated, so ADR 0007's check has
to divide the sample size by the integrated autocorrelation time: the shipped
tutorial's 39 001 samples are worth about 98 independent ones.

Across replicates they are independent by construction, so `n` realisations
carry `n` realisations' worth of information and the standard error is
`σ/√(2θ)/√n` with **no correction at all**. `ensemble_moments` compares the
across-replicate mean and variance at the final time against the exact
stationary law on that basis. The variance tolerance is derived too, from the
`√(2/(n−1))` relative spread of a sample variance.

The check declines rather than fails when the run is too short for the
initial condition to have been forgotten — measuring the transient and
blaming the process for it is not a verification. A negative test displaces
the ensemble's mean and asserts the check catches it, so "passed" means
something.

`ensemble_derived_from_seed` re-derives the replicates' seeds from the base,
checks them against the ones the run consumed, and re-runs one realisation
from its *derived* seed bit for bit. ADR 0007's principle applies to the
second link in the chain as much as the first: a claim that one number
regenerates the whole ensemble is worth what any other claim is.

### 6. `Series.role`, so a consumer need not infer from the id

An ensemble result carries several kinds of curve at once. A chart that drew a
mean over 256 realisations and one of those realisations identically would
present a summary statistic and a single noisy sample as the same kind of
evidence — a misleading picture, not merely an ugly one.

`SeriesRole` is defaulted, so results written before the field read back as
`Trajectory`, which is what they were. It also gave the two pre-existing
comparison lines (the OU long-run mean, logistic growth's closed-form
solution) somewhere honest to say they were not computed by the solver.

The interfaces act on it: the mean gets the heaviest stroke, realisations and
band edges are drawn faintly in one shared grey — eight colours would read as
eight different quantities rather than eight draws of one — and references are
dashed. The decision lives in `chart.rs` with the geometry so it is host
tested; only the SVG is in the component.

### 7. A capability with nothing to draw refuses

`resolve_replicates` is called by **every** adapter, not only the stochastic
ones, and refuses `replicates > 1` for any capability whose `DeterminismClass`
does not draw a sample. A spring-mass-damper asked for 500 realisations would
return the same trajectory 500 times; ignoring the field wastes the time
silently, and honouring it presents 500 identical curves as a distribution.

The rule lives on `DeterminismClass::draws_a_sample`, so the mapping is stated
once. `NonDeterministic` is excluded for the opposite reason to the
deterministic classes: its realisations do differ, but with no seed to derive
them from, the ensemble could not be obtained again.

A registry-driven test walks `all_adapters()` and asserts each one answers
according to its class, so an eleventh capability that forgets fails in CI
rather than silently ignoring the field.

## What this changes in ADR 0007

Two of that ADR's consequences no longer hold, both under its
"No progress reporting" heading:

* **Progress.** 0007 said the capability "reports no progress rather than an
  invented fraction", because `ou_path` cannot be chunked. That is true of one
  realisation and false of an ensemble, whose atom is the realisation. The
  honest unit is therefore the realisation: a one-replicate run has one
  indivisible unit of work and reports a single step from nothing to done — an
  accurate description of a job with one part, not an invented fraction.

  This also fixed a live bug. OU declares `fixed_step: true`, from which the
  app service derives `supports_progress: true`, so the desktop had been
  drawing a determinate progress bar that never moved. The test that should
  have caught it asserted the adapter emitted no progress through a proxy —
  `!any(fixed_step && id == "rk4")` — that passed trivially because the
  solver's id is `exact_gaussian_transition`. Descriptor and adapter now
  agree, and the replacement tests assert the emissions directly.

* **Cancellation.** 0007 said cancellation is "pre-execution only". It is
  now honoured between realisations, which is the first thing in this
  capability long enough to be worth interrupting.

Nothing about the seed, the refusal to default it, or the single-path
statistical check changes.

## Consequences

**The catalogue's second determinism class is now doing work.** ADR 0007 made
`InherentlyStochasticRecordedSeed` inhabited; this makes the distinction
between the classes *load-bearing*, because it decides whether a scenario is
allowed to ask for an ensemble at all.

**`RunResult` gained a vocabulary rather than a container.** The alternative
design was a dedicated `EnsembleSummary` structure hanging off the result.
Roles on the existing `Series` were chosen instead: the mean, the band and the
members are all curves against the same axis, and the existing validator,
chart, downsampler and store already handle curves. What was missing was not a
place to put them but a way to say what they are.

**The result validator gained structural ensemble rules**, deliberately not
statistical ones: one mean, a band that brackets it, one axis. Whether the
mean is in the *right* place is a question for a capability's own verification
checks, which know what distribution is being sampled.

**The result grew, but not by as much as it looks.** An ensemble carries about
six times the series of a single run (a mean, two band edges and eight
realisations, against one path and a reference). Measured: the shipped
ensemble at 256 replicates over 60 s serialises to 2.02 MB — *smaller* than
the 400 s single-path tutorial's 2.33 MB, because payload is
`samples × series` and the ensemble is shorter. The IPC's 64 MiB frame limit
is therefore not newly at risk; ensembles bring the existing bound roughly six
times closer for a given run length, and exceeding it still needs a run long
enough that a single-path result would be approaching it too. That failure
is a named `MessageTooLarge`, not corruption.

**What this still does not address.** First-passage times, and anything else
that is a distribution over a scalar rather than over a curve. The ensemble
here summarises `n` realisations *pointwise in time*, which answers "where is
the process at each moment and how spread out". A histogram of when each
realisation first crossed a level is a different shape of result, and it needs
a kind of axis this schema does not have — one whose coordinates are bins
rather than time. That is the next thing an ensemble makes worth wanting.
