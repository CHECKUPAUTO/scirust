# ADR 0011 — A histogram is not a field with one row

Status: **accepted**, implemented in Phase 3B-7.

## Context

`REPOSITORY_AUDIT.md` §19 listed, among the outstanding items, "a result axis
whose coordinates are bins (ADR 0008's first-passage distributions — a
histogram is not a field with one row)". That parenthesis was the whole
argument and it was never written out. This is it.

Two capabilities forced the issue at once:

- `sim.stochastic.mm1_queue` runs `n` independent discrete-event simulations.
  Each reports four aggregate statistics and no trajectory. The *result* is
  the distribution of those statistics across realisations.
- `sim.stochastic.geometric_brownian_motion` produces paths, but the thing a
  reader needs to see is the right skew of the terminal value — which a mean
  and a symmetric band are precisely unable to convey.

## The shape problem

Every existing result member is aligned **one-to-one** with an [`Axis`]:
`n` coordinates, `n` values. That is true of `Series` by construction and of
`Field` in both directions.

A histogram is not. Its `n` bins are delimited by `n + 1` edges. The
off-by-one is not bookkeeping — it is the difference between "the value at
this point" and "how much fell between these two points".

### Why a single-row `Field` was the tempting wrong answer

It requires no new type, no new validator rules, and no new rendering path:
one row, `n` columns, done.

It is wrong because a field's rows and columns are both **point-sampled**. A
one-row field with `n` values needs an axis carrying `n` coordinates, and a
reader — human or code — then has to guess whether those coordinates are bin
centres, left edges, right edges, or something else. There is no place to put
the answer, because the type has no place for an extra coordinate.

Every downstream consumer would then have to agree on a convention that is
nowhere stated. That is the same failure mode ADR 0006 was written about: a
result that carries `t_start`, `t_end` and a count and expects the reader to
reconstruct the coordinates. It was wrong for time and it is wrong for bins.

## Decision

`RunResult` gains `distributions: Vec<Distribution>`, `#[serde(default)]`ed so
every stored result decodes unchanged.

```rust
pub struct Distribution {
    pub id: String,
    pub display_name: String,
    pub unit: String,          // of the binned quantity, not of the counts
    pub edges: Vec<f64>,       // counts.len() + 1 of them, strictly increasing
    pub counts: Vec<u64>,
    pub underflow: u64,
    pub overflow: u64,
}
```

Three decisions inside that are worth stating.

**It references no axis.** A distribution is self-describing. Binding it to a
shared `Axis` would reintroduce exactly the alignment it does not have, and
would let two distributions with different binnings claim the same axis.

**Under- and overflow are counted, never clamped.** Folding an out-of-range
sample into the end bin turns a badly chosen range into a plausible-looking
histogram with a spike at its own edge. Both interfaces print the counts
whenever they are non-zero. A non-finite sample counts as overflow, because a
run that produced one has a problem the picture must not absorb.

**`estimated_mean` is named as an estimate.** The individual samples are gone
by the time a distribution exists, so a mean recovered from bin centres is
accurate to within half a bin. A capability that needs the exact mean computes
it from the samples and records a `Metric`; the M/M/1 queue does both, and a
test asserts they agree to within half a bin width.

## What `validate_result` checks, and what it does not

Four structural defects: a duplicate id, an edge count that is not one more
than the bin count, edges that are not finite and strictly increasing, and no
bins at all.

**Only** structural ones. Whether the counts are a plausible sample of
anything is the capability's own business — that is what its verification
checks are for. `validate_result` is the last gate before a result reaches a
consumer, and its job is that nothing downstream can be misled about the
*shape* of what it received. A histogram with the wrong number of edges is a
picture drawn against the wrong bins; a histogram of a surprising sample is a
scientific question, and the validator has no standing to answer it.

## Consequences

Both interfaces render them, and both make the same two choices for the same
reason:

- **Bars, not shading.** A distribution is one-dimensional, so a *length* is
  available and exact where a shade would make the reader estimate a count
  from a colour. The field renderer shades because a field is
  two-dimensional and has no spare dimension for length.
- **The interval is labelled, not the centre.** Printing a centre makes the
  reader infer the width — the same off-by-one that made this its own type.

The CLI prints a `DISTRIBUTION` block per histogram; the desktop renders one
`Histograms` panel beside the chart and the field map, and says in words when
every sample fell outside the binned range rather than drawing empty bars.

### The related item this does *not* close

ADR 0008 mentioned first-passage-time distributions as the motivating case.
Those are still not implemented — no capability computes one. What has been
built is the *representation*, driven by two capabilities that needed it now.
Building the representation for a capability that does not exist is the thing
this project declines to do, so the first-passage capability will arrive with
its own oracle or not at all.
