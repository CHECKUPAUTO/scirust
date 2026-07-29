# ADR 0012 — Two things deliberately not built, and why

Status: **accepted**, Phase 3B-7.

The `CAPABILITY_MATRIX.md` and `REPOSITORY_AUDIT.md` outstanding lists carried
two items that read like unfinished work and are not. Leaving them on a list
implies somebody will get to them; that is worse than saying they were
considered and declined, because it hides a decision behind an omission.

Both were reopened deliberately, with the intent of finishing them, and both
were closed on evidence gathered while doing so.

---

## 1. `scirust-sim::envs` will not become a Studio capability

`envs` provides `CartPole` and `GridWorld`, which implement `Environment`
(`reset` / `step(action)`) rather than `System` (`y' = f(t, y)`).

### Why the obvious adaptation was attempted and abandoned

An `Environment` needs an **agent**. The scenario schema has nowhere to put
one — `model.*` fields are `ValueWithUnit`, so a policy cannot be a parameter.
The workable idea was to encode the policy as `solver.id`, on the reasoning
that for an environment the policy *is* what drives the run. That is
syntactically fine and it makes `solver` mean "numerical method" for eighteen
capabilities and "controller" for two, which every future reader then has to
hold.

That alone would be a cost worth paying for a good enough capability. The
deciding question was what such a capability could actually **verify**.

### What is and is not verifiable here

Reading `CartPole`'s implementation settles it:

- the force is **bang-bang ±10 N** with no null action, so the system is
  always actuated and there is no conservative limit to check energy against;
- the integrator is **explicit Euler**, which conserves nothing and has no
  closed form;
- there is no analytic solution, no invariant, and no asymptote.

What remains checkable is the *harness*: that `done` is reported exactly when
`|x| > 2.4` or `|θ| > 12°`, that the return equals the episode length because
the reward is 1 per step, that a seed replays. Every one of those tests the
adapter and the bookkeeping. None tests the physics.

That is the disqualifying fact. This catalogue's premise is that a capability
carries an oracle its own model states — a closed form, an invariant, a fixed
point, an eigenvalue. A capability whose checks verify that the counters were
counted correctly would be the first one that does not, and it would sit in
the same list as a run that matches a Bateman solution to 1.2e-12.

`GridWorld` is the better of the two and fails differently: its exact oracle
(shortest path length equals the Manhattan distance when there are no walls)
is a combinatorial fact about a graph. Adapting it would put a pathfinding
exercise in a catalogue of physical models.

### The decision

`envs` stays catalogue-only. Its own crate tests it; Studio does not claim it.
If a reinforcement-learning surface is wanted later it should be its own
thing, with its own vocabulary for policies and its own idea of what a
verified result means — not a `solver.id` that quietly means something else
for two entries.

---

## 2. The result model will not grow a three-axis field

The outstanding list said "fields over three axes or on unstructured meshes".

### The investigation, which found something

Grepping for a producer turned one up: `scirust-itd` has `field3::Field3`, a
dense row-major 3-D scalar field, and `region3` builds on it. So this was
**not** speculative, which is what the first draft of this ADR assumed. That
assumption is recorded here rather than quietly corrected, because "no model
produces one" would have been a wrong reason for a right conclusion.

What the crate actually does with `Field3` is the deciding detail:

- `region3` uses it for **6-connected component labelling** of a 3-D mask,
  region records, mask-overlap metrics (IoU, Dice) and persistence-gated
  topology events;
- the crate's **simulation driver** (`simulate`) returns a `SimulationResult`
  of per-interval **time series** — intensity rate, heterogeneity,
  localization, roughness, sign mixing, temporal deformation — plus scalar
  indices. Its fields are 2-D.

So the 3-D structure is an *analysis* input, not a simulation output over
three continuous axes. A Studio capability wrapping `scirust-itd::simulate`
would produce a time axis, six series and eight metrics — which the existing
schema expresses exactly, with no change at all.

### The decision

No three-axis `Field`, and no unstructured-mesh support, until a capability
produces one. The existing `Field` remains two-axis.

This is the same rule the schema has followed since v1 — the CURRENT_SCHEMA_VERSION
comment says a fake migration with nothing on the other end is worse than no
migration — and the same rule ADR 0011 followed in declining to build a
first-passage histogram before a capability computes one.

It is worth being precise about what would change the answer: not "somebody
might want it", but a model in this repository whose *output* is a quantity
varying over three axes. `scirust-itd::simulate` is the nearest thing and it
is not that.

### What this leaves

Adapting `scirust-itd` as a capability is a real and open piece of work — it
would be the first capability from outside `scirust-sim`, and the audit's
standing rule that other crates need a real API review before adoption
applies. It is listed as future work in `CAPABILITY_MATRIX.md`. What is closed
here is only the schema question it was attached to.
