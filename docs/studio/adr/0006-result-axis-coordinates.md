# ADR 0006: Result schema v2 — axes carry their coordinates

## Status

Accepted and implemented (Phase 3A-1).

## Context

Result schema v1 (`docs/studio/adr/0002-structured-run-results.md`) described
each axis with an id, a display name and a unit — but not its values. A
consumer holding a v1 result knows a series has 6 285 points and that the run
went from `t = 0` to `t = 100 s`, and has no way to learn *when* each sample
was taken.

There is exactly one way to draw such a result: assume the samples are evenly
spaced. For the four fixed-step capabilities that assumption happens to hold.
For `sim.chemistry.robertson` it is simply false — the adaptive Rosenbrock-W
solver chooses its own steps, and over the shipped tutorial they span a
**26 391×** range, from `1e-6 s` to `2.6e-2 s`. A chart drawn against a
regenerated linear axis is not a slightly-imprecise picture of that run; it
is a picture of a different experiment.

This was survivable while the only consumer was a CLI printing point counts.
It stops being survivable the moment anything plots the data, which is
exactly what Phase 3A does.

## Decision

Bump `RESULT_SCHEMA_VERSION` to 2 and change two types:

```rust
pub struct Axis {
    pub id: String,
    pub display_name: String,
    pub unit: String,
    pub monotonicity: AxisMonotonicity,
    pub values: Vec<f64>,          // the integrator's own coordinates
}

pub struct Series {
    pub id: String,
    pub display_name: String,
    pub unit: String,
    pub axis_id: String,           // which axis these belong to
    pub values: Vec<f64>,
}
```

`Axis::values` are passed through from the trajectory unmodified. They are
never regenerated from a start, an end and a count — the operation that was
implicitly happening in every consumer, and that this version exists to make
impossible.

`Series::axis_id` is required rather than optional. A consumer must never
have to guess which axis a series belongs to, and the moment a capability
emits two axes, guessing would be wrong.

### Declared monotonicity

`AxisMonotonicity` is declared by the capability that produced the axis (the
adapter is that capability's own code) and enforced by the validator. A
forward integration's time axis is `StrictlyIncreasing`; an axis that merely
enumerates samples claims nothing. Putting the claim in the data means a
reader can check it, rather than a chart discovering mid-render that its
x-values double back.

### A full consistency validator

`assert_finite` becomes `validate_result`, which returns **every** defect
rather than the first:

| Defect | Why a consumer would otherwise be misled |
|---|---|
| non-finite axis / series / metric value | serializes as JSON `null` |
| series names a missing axis | an unplottable series |
| duplicate axis or series id | ambiguous lookup |
| axis/series length mismatch | points drawn against the wrong coordinates |
| empty axis with dependent data | a series with nowhere to go |
| declared-increasing axis that goes backwards | a chart that folds over itself |
| `summary.t_start`/`t_end` disagreeing with the axis | two answers to one question |

The summary comparison uses **exact equality**, deliberately. Both numbers
are read out of the same stored trajectory, so any difference at all means
one of them was recomputed rather than carried through — which is precisely
the class of bug this schema version exists to prevent. The adapters were
changed to take `t_start` from `traj.t.first()` rather than from the
requested start time so the property actually holds.

## Consequences

- Results are larger: an axis is one more `Vec<f64>` alongside the series
  that reference it. For the shipped tutorials this is a few percent, and it
  buys the difference between a plot and a guess.
- Every adapter had to change, and a test now asserts that each one emits
  coordinates, binds every series to them, and agrees with its own summary.
  A separate test asserts that Robertson's recorded steps vary by at least an
  order of magnitude — a regression that resampled onto a uniform grid would
  pass everything else and fail that one.
- Stored v1 runs are unaffected and stay readable; see §6 of the phase brief
  and the compatibility behaviour documented in
  `docs/studio/STORAGE_LAYOUT.md`. Nothing is migrated in place: an immutable
  store that rewrote its own files would invalidate every hash it had
  recorded.
- `scirust runs show` now distinguishes the two versions in its output rather
  than printing series that look interchangeable.
