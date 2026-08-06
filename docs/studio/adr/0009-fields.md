# ADR 0009: A result can hold a field, not only curves

## Status

Accepted and implemented (Phase 3B-4).

Corrects a claim repeated in `REPOSITORY_AUDIT.md` §13–§16.

## Context

`thermal`'s `HeatRod1d` had been on the "still catalogue-only" list since the
catalogue existed, with the same note each time:

> Result schema v2 can already express it — `axes` is a vector and every
> series names its axis — but a line chart is the wrong presentation and the
> interface has no other.

The first half of that is **wrong**, and reading it again while writing the
adapter is what showed it. `axes` being a vector means a result may have more
than one axis; it does not mean a single quantity may span two. A `Series`
holds `values: Vec<f64>` aligned **one-to-one** with the axis it names. The
most it can carry is a slice of a field: one node's history, or one instant's
profile.

So `u(x, t)` could be expressed as `n` series and lose the fact they are one
quantity, or as `m` series and lose it the other way. Neither is the field.
The blocker was never only presentation.

## Decision

### 1. `Field` is its own type

```rust
pub struct Field {
    pub id: String,
    pub display_name: String,
    pub unit: String,
    pub row_axis_id: String,
    pub column_axis_id: String,
    pub columns: usize,
    pub values: Vec<f64>,   // row-major
}
```

`RunResult::fields` is defaulted, so every result written before this reads
back with none — which is what those results had. Nothing else in the model
changes: a capability whose outputs are curves is untouched.

### 2. The shape is stored redundantly, and checked

`columns` duplicates the column axis's length. That is deliberate. A consumer
holding only the field — a downsampler, a renderer — knows its shape without
being handed the axes too, and `validate_result` checks the two agree.

The checking matters more here than anywhere else in the result model,
because **a shape error is the one field defect that leaves every number
finite and plausible**. A row-major buffer read with the wrong stride is still
a buffer of doubles; it renders as a convincing picture of nothing. So the
shape is verified twice — against the declared count and against both axes —
rather than trusted from either.

### 3. Reduction keeps extremes, in two dimensions

Neither a terminal nor a browser can draw 2 448 × 40 cells. The existing rule
for series is *reduction never hides a peak*, implemented as min/max bucketing.

A heat-map cell is one colour, so it cannot show both a minimum and a maximum.
The two-dimensional form of the same rule is therefore: **each cell keeps the
source sample furthest from the field's mean.** An average is precisely the
operation that makes a spike disappear, and a heat map that smooths away the
hot spot is a picture of a different experiment. A test plants a single hot
cell in an otherwise flat field sixteen times larger than the budget and
asserts it survives.

The colour scale is taken from the *source*, not from the reduction, so it
still describes the data where the reduction did not land on the extreme.

### 4. The heat map's colours are ordered

Blue → light → red, not a rainbow. A rainbow has no perceptual ordering, so a
reader cannot tell which of two colours is the larger value without consulting
the legend — which is the one thing a heat map exists to make unnecessary.

### 5. `GridView`/`GridWire`, not `FieldView`/`FieldWire`

Those names were already taken across the bridge, by the description of one
`model.*` parameter a scenario may set — a *form* field. Two unrelated
meanings of the word met, and the newcomer yielded: renaming the older one
would touch the catalogue panel, the bridge contract and every capability
view, for a naming preference.

## What the capability itself decides

**Its checks come from facts the model states in closed form.** The
semi-discrete system's steady state is *exactly* the linear profile between
the boundary temperatures, and its slowest sine mode decays at exactly
`λ₁ = (2α/dx²)(1 − cos(π/(n+1)))`. The run measures its own decay over the
tail and compares — on the shipped tutorial, ratio 1.000.

The third check, the discrete maximum principle, exists because the first two
can both pass on a wrong stencil that still relaxes to something plausible.
Diffusion with no source creates no new extremes; a violation means the
stencil made heat that was not there.

**The stability limit is a validation error, not a discovery.** Explicit RK4
on this stencil diverges above `h ≈ 0.7·dx²/α`. Above it the run does not get
less accurate — it produces `NaN` in a few hundred steps. So the scenario is
refused, with the limit and a usable step in the message. Reporting
"non-finite value" afterwards would tell the user what happened instead of
what to do.

**Two series are plotted against position**, not time — the first in the
catalogue. That broke `every_adapter_emits_exact_axis_coordinates`, which
asserted every series is on the time axis. The assertion was generalised to
"every series names an axis this result has, and matches its length", which is
the property that was always meant, plus a new one that at least one series is
against time.

## Consequences

**The catalogue reaches 11 capabilities across 8 of 16 module families**, and
the presentation gap that kept `thermal` out is closed for every future field
capability, not just this one.

**`REPOSITORY_AUDIT.md`'s claim is corrected** rather than quietly dropped.
It had been repeated across four updates; a note that survives that long
without being checked is worth recording as an error, because the next
uninspected claim is the interesting one.

**What this does not address.** A field over *three* axes, and a field on a
non-rectangular mesh. Both are real (a 2-D plate over time; anything with an
unstructured grid), and neither is a small extension of a row-major buffer
with two axis ids. Also unaddressed, still: a result axis whose coordinates
are **bins** rather than a continuum, which ADR 0008 named for first-passage
distributions and which a field does not provide — a histogram is not a
field with one row.
