# ADR 0007: A stochastic capability must carry its seed

## Status

Accepted and implemented (Phase 3B-2).

## Context

The scenario schema has had an `experiment.seed` field since Phase 1. Every
tutorial in `docs/studio/tutorials/` sets it. Until this phase, **nothing read
it** — a search for `.seed` across every Studio crate found exactly one hit, in
a schema unit test asserting the field parses.

That was harmless while the catalogue held only deterministic capabilities: a
spring-mass-damper's trajectory is a function of its parameters, and a seed
would have changed nothing. But it left the product making a promise it did
not keep. A scenario file that accepts a seed tells its author the seed
controls something. And `DeterminismClass` has had a variant reading
`InherentlyStochasticRecordedSeed` — literally "only the seed is recorded" —
against a `RunProvenance` that recorded no seed at all.

Adding the first stochastic capability forces the question, because for such a
capability the seed is not a detail. `dX = θ·(μ − X)·dt + σ·dW` produces one
sample from a distribution. Two runs with different seeds give different
answers and **neither is wrong**. Without the seed, a stored result is a
picture of something that happened once and cannot be obtained again — which
fails the same test this project applies everywhere else: a result is only
evidence if someone else can get it.

## Decision

### 1. `RunProvenance` records the seed the computation *consumed*

```rust
pub struct RunProvenance {
    …
    #[serde(default)]
    pub seed: Option<u64>,
}
```

`None` for every capability whose result does not depend on a seed. Not "the
seed that was in the scenario" — recording a number there would imply it
mattered, and it did not. The scenario source is stored verbatim beside every
result, so nothing is lost.

### 2. Added to schema v2, not as v3

The field is optional with a default, and no type in the runtime or the store
uses `serde(deny_unknown_fields)`. So a v2 file written before the field reads
back as `None`, and a v2 reader built before the field ignores it. A version
bump would have bought a second compatibility path and no additional
guarantee. `src-tauri/tests/bridge_contract.rs` pins both directions.

### 3. A stochastic capability is *refused* without a seed

`validate_support::resolve_seed` returns `SRST-VAL-0095` when
`experiment.seed` is absent, and the Ornstein–Uhlenbeck adapter calls it
during validation.

The alternative — pick a seed silently — was rejected in every form it could
take:

* **From the clock or the OS.** The run becomes unreproducible, which is the
  problem the seed exists to solve.
* **From a fixed constant.** Every run returns the same sample while appearing
  to have drawn one. That is worse than unreproducible: it is misleading.
* **Warn and continue.** A warning about a result that cannot be reproduced is
  a warning about a result that should not have been produced.

Refusing costs the user one line in their scenario and buys the guarantee the
whole store is built on.

### 4. Verification is statistical, and says so

There is no reference trajectory. What is pinned is the *distribution*: the
stationary law of the OU process is exactly `N(μ, σ²/(2θ))`, so the check
compares the sample moments of the path's tail against it.

Two details make that a real check rather than a gesture:

* **The transient is discarded.** Five relaxation times `1/θ`, by which point
  the initial condition's influence has decayed by a factor of 148.
* **The tolerance accounts for autocorrelation.** Successive samples of an OU
  path are correlated, so `n` samples do not carry `n` samples of information;
  the integrated autocorrelation time is `1/θ`, giving an effective sample
  size of about `n·dt·θ/2`. Using the raw `n` would produce a band so tight
  that correct runs fail — and a check that fails on correct runs gets
  deleted. The shipped tutorial draws 39 001 samples worth about 98
  independent ones.

A test constructs a path whose true mean is displaced from the declared `μ`
and asserts the check catches it, so "passed" means something.

### 5. The run demonstrates its own reproducibility

`reproducible_from_seed` re-derives the first 1 000 samples from the recorded
seed and compares them bit for bit. A provenance field saying "seed 42" is a
claim; re-deriving the sample from it inside the same run is evidence.

## Consequences

**The seed field is now real.** It is required where it matters, ignored where
it does not, recorded when consumed, displayed by the CLI next to the
determinism class, and shown in the desktop's provenance panel — where a
`None` deliberately renders nothing rather than a zero.

**`InherentlyStochasticRecordedSeed` is now inhabited.** The determinism
vocabulary had five variants and used one. The interface's existing handling
of that field was never exercised against a capability that was not
bit-identical from its parameters; now it is.

**No progress reporting, by the same rule as Robertson.** `ou_path` seeds its
generator once and samples the whole path in a single call. It cannot be
chunked: each call re-seeds, so splitting the span would produce a different
and differently-distributed sample. Cancellation is therefore pre-execution
only, and the capability reports no progress rather than an invented fraction.
This is the second capability in that column, which is worth noting — the
interface's indeterminate path is no longer a special case built for one
model.

**What this does not address.** Ensembles. A single seeded path answers "what
does one realisation look like"; many questions about a stochastic model
("what is the distribution of the first passage time") need many paths and a
result schema that can hold them. That is a larger change to the result model
than a seed field, and it is not attempted here.
