# ADR 0010 — Not every capability integrates in time

Status: **accepted**, implemented in Phase 3B-6.

## Context

Eighteen capabilities in, every single one integrated a system of ODEs
forward in time. The result model reflected that so completely that the
assumption had never been written down anywhere:

- `RunResult::time_axis()` looked for an axis with the hardcoded id `"t"`.
- `RunSummary` had `t_start`, `t_end` and `steps`, and `validate_result`
  compared them against whatever `time_axis()` returned.
- The registry-driven test `every_adapter_emits_exact_axis_coordinates`
  asserted every capability had a `t` axis, that it was strictly increasing,
  and that at least one series was plotted against it.
- `scirust-cli` printed `t final` with the label hardcoded.
- The desktop's `RunView` builder read `time_axis()` first and fell back to
  `axes.first()` — the only place that had already noticed the assumption
  might not hold, and it dealt with it by guessing.

`scirust_sim::apd` breaks all of that. An avalanche photodiode's receiver
analysis is *algebraic*: given a gain, closed forms give the signal current,
the multiplied shot noise, the thermal floor and the signal-to-noise ratio.
The module has no `impl System` at all, because there is nothing to
integrate. What makes it worth having as a capability is the trade-off it
captures — the SNR rises with avalanche gain and then collapses, so there is
an optimum — and seeing that means sweeping the *gain*, not stepping time.

## Decision

**A capability declares what its independent variable is, and the runtime
holds it to that declaration.**

Concretely:

1. `CapabilityDescriptor` gains `domain: RunDomain`, with two variants:
   - `RunDomain::Time` — the result carries a `t` axis and the summary
     describes it.
   - `RunDomain::ParameterSweep` — the result carries the swept parameter's
     axis and **no** `t` axis.
2. `RunSummary` gains `axis_id: String`, naming the axis its bounds describe.
   It is `#[serde(default)]`ed to `"t"`, so every result already in a store
   decodes unchanged and means exactly what it always meant.
3. `RunResult::summary_axis()` resolves it. `validate_result` compares the
   summary's bounds against *that* axis, with the same exact equality it
   always used.
4. The registry-driven test demands a `t` axis of exactly the capabilities
   that declared one — and demands its **absence** from the ones that did
   not.

`solver.start`, `solver.end` and `solver.step` are reused for the sweep's
bounds and increment. They are in the swept parameter's units, and the
capability's `SolverDescriptor` says so in its own summary text.

## Alternatives considered

**Give the sweep a `t` axis anyway, holding gains.** Cheapest by far: no
schema change, no registry change, no test change. Rejected because
`summary.t_start` would then hold an avalanche gain, and `t` means time
everywhere else in this codebase. Every downstream reader — the CLI's `t
final` line, the desktop chart's axis label, anything reading a stored result
in a year — would be told a run went from 1 second to 120 seconds. The saving
is one afternoon; the cost is a result format that lies.

**Weaken the test to "every result has at least one axis."** Also cheap, and
it was tempting because it needs no new vocabulary. Rejected for a reason
that is easy to miss: the assertion's *value* is in what it forbids. Softened
to "some axis", it would no longer catch a time-integrating capability that
stopped emitting `t` — which is precisely the regression it was written for.
Declaring the domain keeps the strong assertion for the eighteen capabilities
that should be held to it and adds an equally strong one (no `t` axis) for the
one that should not.

**Infer the domain from the emitted result rather than declaring it.** The
test could simply read `summary.axis_id` and check consistency. Rejected
because then the capability's own output is the only authority on what the
capability is, and a bug that emitted the wrong axis would be
self-consistent. A declaration in the catalogue is also something a *reader*
can use: a user browsing capabilities can now tell, before running anything,
whether they will get a trajectory or a curve.

**Rename `RunSummary::t_start`/`t_end` to `axis_start`/`axis_end`.** Honest,
and it was the first instinct. Rejected as a change whose blast radius —
every stored-result fixture, the store's v1/v2 compatibility tests, the IPC
payloads, the app service, the Dioxus views — is out of proportion to what it
buys over a doc comment and an `axis_id` beside them. If a third domain ever
arrives the rename becomes worth doing; one sweep is not enough evidence.

**A separate `SweepResult` type.** Rejected quickly. Everything else about a
sweep is identical to a run: axes with real coordinates, series bound to them
by id, metrics, warnings, verification checks, provenance. A parallel type
would duplicate `validate_result`, the store, the IPC codec and both
interfaces to change one field's meaning.

## Consequences

The first capability with no time in it also turned out to be the first with
**no tolerances**, which is not a coincidence — an algebraic model admits
exact statements that an integrated one does not.

`sim.optoelectronics.avalanche_photodiode` verifies two things:

- **the optimum is where the closed form says.** Maximising
  `SNR = A*M²/(C*M²*F(M) + T)` means minimising `C*F(M) + T/M²`, whose
  stationary condition is `C*k*M³ + C*(1−k)*M = 2T`. The left side is
  strictly increasing in `M` for every `k` in `[0, 1]`, so it has exactly one
  positive root and bisection finds it with no derivative and no initial
  guess. The SNR at that root must then be at least the SNR at *every* swept
  gain — which is what a maximum is, so the only slack allowed is round-off.
- **the SNR curve turns exactly once.** The same monotonicity that gives one
  root gives one turn. A model with the right optimum but a spurious second
  bump passes the first check and fails this one.

A sweep whose range does not bracket the root reports both as
`NotApplicable` with a warning, rather than reporting its endpoint as the
answer. That distinction is the reason the check locates the optimum
analytically rather than by taking the sweep's `argmax`: an `argmax` always
returns something, and an `argmax` at an endpoint looks exactly like an
`argmax` at a peak.

### What did not change

The desktop application. Its `RunView` builder already read the result's own
axis rather than assuming a time axis, and the chart already labels whatever
axis it is given. A sweep charts correctly with no change to the shell, the
bridge, the chart or the interface — the same property every previous
capability addition has had, and for the same reason: the interface is driven
from the registry.
