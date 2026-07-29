# SciRust Studio runtime contract

This document describes the actual, implemented contract between
`scirust-cli` (and, later, a worker process or desktop application) and a
capability's execution. Every type and function named here exists in the
repository at the paths given — this is not a proposal.

## Pipeline

```text
.scirust.toml text
      |  scirust_studio_schema::parse_toml
      v
  Scenario  ------------------------------------------+
      |  scirust_studio_schema::validate(&scenario,    |  generic: schema
      |    Some(&known_capability_ids))                |  version, units,
      v                                                |  ranges, string
  Vec<SchemaError>  (empty = passed)                   |  lengths, ...
      |
      v
  scirust_studio_runtime::find_adapter(&scenario.capability.id)
      |  -> Option<Box<dyn CapabilityAdapter>>
      v
  adapter.validate(&scenario)                          |  capability-specific:
      |  -> Result<ValidatedScenario, ValidationReport> |  missing/unknown field,
      v                                                 |  wrong dimension,
  adapter.execute(&validated, &control, &mut sink)      |  cardinality, range,
      |  -> Result<RunResult, ExecutionError>           |  unsupported solver
      v
  RunResult  (schema_version, capability_id, summary, axes,
              series, metrics, warnings, verifications, provenance)
      |
      v
  scirust-cli: print_result_text(&result)  or  result.to_json_pretty()
```

`scirust-cli/src/studio.rs` implements exactly this pipeline for `scirust
run`, and the read-only half (`build_registry().to_text()`/`.to_json()`)
for `scirust catalog`. It does not import `scirust_sim` — every capability
is reached only through `CapabilityAdapter`.

## The `CapabilityAdapter` trait

```rust
pub trait CapabilityAdapter: Send + Sync {
    fn descriptor(&self) -> &'static CapabilityDescriptor;
    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport>;
    fn execute(&self, scenario: &ValidatedScenario, control: &ExecutionControl, sink: &mut dyn EventSink)
        -> Result<RunResult, ExecutionError>;
}
```

(`scirust-studio-runtime/src/adapter.rs`)

- **`validate`** must not execute anything. It is called after generic
  schema validation has already passed, and should assume the scenario
  parses and its units resolve — its job is capability-specific meaning:
  is this field one I recognise, does its dimension match, is it in range,
  does the requested solver exist and carry what it needs. Every
  implementation in this repository builds its error list from the shared
  helpers in `scirust-studio-runtime/src/validate_support.rs`
  (`resolve_model_scalar`, `resolve_state_vector`,
  `check_unknown_model_fields`, `check_unknown_state_fields`,
  `resolve_solver`, `check_sum_constraint`), called with the adapter's own
  `FieldDescriptor`/`SolverDescriptor` tables — not five independent
  re-implementations of the same logic.
- **`execute`** receives a `ValidatedScenario` (constructible only by a
  successful `validate()`), an `ExecutionControl`, and an `&mut dyn
  EventSink` that receives
  `RunEvent::Started`/`Progress`/`Warning`/`Completed`/`Cancelled`/`Failed`.
  It returns a fully-populated `RunResult` or an `ExecutionError`, never a
  partially-built one.
- Every adapter calls `scirust_studio_runtime::assert_finite(&result)`
  immediately before returning `Ok(result)`, converting any non-finite
  derived value into `ExecutionError::Internal` instead of a silently
  "successful" result containing JSON `null`.

## Cancellation and progress

Fixed-step capabilities check `ExecutionControl` after **every accepted
step** and report genuine progress, via
`scirust_sim::simulate_observed`/`simulate_second_order_observed` (see
`docs/studio/adr/0003-worker-process-and-ipc.md`). `simulate` and
`simulate_second_order` are wrappers around those same functions, so
observing a run cannot change the numbers a completed run produces —
`scirust-sim`'s `observed_rk4_run_is_bit_identical_to_the_unobserved_one`
test asserts exactly that.

`RunEvent::Progress` carries the fraction of the scenario's **time span**
already integrated, taken from the `t` the solver reports. It is not a
timer and not an estimate, and it is emitted at most 20 times per run.

Not every capability can do this, and the table below says which:

| Capability | Cancellation | Progress |
|---|---|---|
| `sim.mechanics.spring_mass_damper` | every step | yes |
| `sim.epidemiology.sir` | every step | yes |
| `sim.orbital.two_body` | every step (both solvers) | yes |
| `sim.electrical.rlc` | every step | yes |
| `sim.chemistry.robertson` | **before execution only** | **none** |
| `sim.ecology.lotka_volterra` | every step | yes |
| `sim.ecology.logistic_growth` | every step | yes |
| `sim.mechanics.pendulum` | every step | yes |
| `sim.mechanics.double_pendulum` | every step | yes (the main run only) |
| `sim.stochastic.ornstein_uhlenbeck` | **between realisations** | **per realisation** |

Robertson integrates through `scirust_sim::stiff_bridge`'s adaptive
Rosenbrock-W solver, which exposes no per-step callback. It is given no
fabricated progress fraction. Mid-run cancellation for it is covered at the
process level instead — the worker can be killed — which is one of the
reasons the worker exists.

The Ornstein-Uhlenbeck process is the one row whose unit of progress is not
the time step. `ou_path` samples a whole realisation in a single call and
cannot be chunked — each call re-seeds, so splitting the span would produce a
different and differently-distributed sample. Its atom is therefore the
*realisation*: an ensemble of `n` reports `n` units and can be cancelled
between them, and a one-replicate run has a single indivisible unit and
reports one step from nothing to done. That last case is not a fabricated
fraction; it is an accurate description of a job with one part. See
`docs/studio/adr/0008-ensembles.md`, which supersedes ADR 0007 on this point.

Every adapter, including Robertson, honours an already-cancelled control
before starting; `every_adapter_honours_a_pre_cancelled_control` asserts
this for all of them.

## Implemented capabilities (Phase 2A)

| Capability id | Source model | Solvers | Verification checks |
|---|---|---|---|
| `sim.mechanics.spring_mass_damper` | `scirust_sim::mechanics::SpringMassDamper` | `rk4` | `energy_drift` |
| `sim.epidemiology.sir` | `scirust_sim::epidemiology::Sir` | `rk4` | `population_conservation`, `non_negative_compartments` |
| `sim.orbital.two_body` | `scirust_sim::orbital::TwoBody` | `symplectic_euler`, `rk4` | `energy_drift`, `angular_momentum_drift`, `finite_trajectory` |
| `sim.electrical.rlc` | `scirust_sim::electrical::SeriesRlc` | `rk4` | `finite_solution`, `damping_regime`, `energy_non_increasing` |
| `sim.chemistry.robertson` | `scirust_sim::chemistry::Robertson` (via `scirust_sim::stiff_bridge::simulate_rosenbrock`, feature `stiff`) | `stiff_rosenbrock_w` | `mass_conservation`, `non_negative_concentrations`, `solver_completion` |
| `sim.ecology.lotka_volterra` | `scirust_sim::ecology::LotkaVolterra` | `rk4` | `first_integral_drift`, `populations_stay_positive` |
| `sim.ecology.logistic_growth` | `scirust_sim::ecology::LogisticGrowth` | `rk4` | `analytic_error`, `stays_below_capacity` |
| `sim.mechanics.pendulum` | `scirust_sim::mechanics::Pendulum` | `rk4` | `energy_drift`, `amplitude_bounded` |
| `sim.mechanics.double_pendulum` | `scirust_sim::mechanics::DoublePendulum` | `rk4` | `energy_drift`, `sensitive_dependence` |
| `sim.stochastic.ornstein_uhlenbeck` | `scirust_sim::stochastic::ou_path` | `exact_gaussian_transition` | `stationary_moments`, `reproducible_from_seed`, `ensemble_moments`*, `ensemble_derived_from_seed`* |

Every row is a real, tested adapter with a shipped, executed tutorial
scenario under `docs/studio/tutorials/`. Checks marked `*` appear only when
the scenario asks for more than one realisation. See
`docs/studio/CAPABILITY_MATRIX.md` for how these relate to the rest of
`scirust-sim` and the wider workspace.

A capability may not emit a verification its catalogue entry does not
declare; `no_adapter_emits_a_verification_its_catalogue_entry_does_not_declare`
runs every adapter's own tutorial and checks it. The converse is deliberately
not asserted, because a check that applies only to some scenarios would
otherwise have to emit `NotApplicable` filler to satisfy it.

## Error codes

`scirust-studio-command::ErrorCode` formats as `SRST-<FAMILY>-<NNNN>`.
Validation codes currently in use:

- `SRST-VAL-0001`..`0012`: generic scenario schema errors
  (`scirust-studio-schema/src/error.rs`) — parse errors, schema version,
  unknown units, non-finite values, end-before-start, non-positive step,
  unsupported precision/backend, unknown capability, oversized strings,
  too many outputs, zero sample interval.
- `SRST-VAL-0090`..`0094`: generic *capability*-level validation errors
  (`scirust-studio-runtime/src/validate_support.rs`) — unknown field,
  unsupported solver, missing step, missing tolerance, sum-constraint
  violation.
- `SRST-VAL-0100`..`0109`: `sim.mechanics.spring_mass_damper` field errors.
- `SRST-VAL-0110`..`0119`: `sim.epidemiology.sir` field errors.
- `SRST-VAL-0120`..`0129`: `sim.orbital.two_body` field errors.
- `SRST-VAL-0130`..`0139`: `sim.electrical.rlc` field errors.
- `SRST-VAL-0140`..`0149`: `sim.chemistry.robertson` field errors.
- `SRST-VAL-0150`..`0159`: `sim.ecology.lotka_volterra` field errors.
- `SRST-VAL-0160`..`0169`: `sim.ecology.logistic_growth` field errors.
- `SRST-VAL-0170`..`0179`: `sim.mechanics.pendulum` field errors.
- `SRST-VAL-0180`..`0189`: `sim.mechanics.double_pendulum` field errors.
- `SRST-VAL-0190`..`0199`: `sim.stochastic.ornstein_uhlenbeck` field errors.
- `SRST-VAL-0095`: a stochastic capability was given no `experiment.seed`
  (see `docs/studio/adr/0007-seeded-stochastic-capabilities.md`).
- `SRST-VAL-0096`: `experiment.replicates` above 1 on a capability that draws
  no sample, so every realisation would be the same curve.
- `SRST-VAL-0097`: `experiment.replicates` is zero, or above the limit of
  4 096 (see `docs/studio/adr/0008-ensembles.md`).

Each capability's exact field-to-code mapping is in that capability's
adapter module (the `FieldDescriptor.error_code` on each `const`). A new
capability should claim the next unused ten-number block and record it
here.

## CLI exit codes

`scirust run` maps outcomes to exit codes per the original Studio brief's
table:

| Code | Meaning | Source |
|---|---|---|
| 0 | success | — |
| 2 | usage error | missing argument, unreadable file, unknown `--format` |
| 3 | validation error | schema validation, capability validation, or `ExecutionError::InvalidModelState` |
| 5 | numerical failure | `ExecutionError::Numerical` (integrator blow-up or step underflow) |
| 6 | cancelled | `ExecutionError::Cancelled` |
| 7 | internal failure | unregistered adapter for a validated capability id (a bug), `ExecutionError::Internal`, JSON serialization failure |

## Out-of-process execution

The same pipeline runs inside `scirust-studio-worker`, which speaks
`scirust-studio-ipc` over stdin/stdout — see
`docs/studio/IPC_PROTOCOL.md` and
`docs/studio/adr/0003-worker-process-and-ipc.md`. The worker reaches
capabilities through the identical `find_adapter`/`validate`/`execute` path
this document describes; it is not a second execution route. The test
`a_run_through_the_worker_matches_an_in_process_run` runs a scenario in a
real spawned worker process and asserts the returned series, metrics, and
verifications equal an in-process run of the same scenario exactly.

`scirust-cli` still executes in-process: spawning a worker per CLI
invocation would add latency a command-line user gains nothing from. The
worker exists for the desktop shell, and is tested directly.

## Result schema

Results use **schema version 2**: every `Axis` carries its coordinates and
every `Series` names the axis it belongs to, so nothing infers sample
spacing. Runs stored under v1 remain readable and verifiable but are never
given a reconstructed time axis — see
`docs/studio/adr/0006-result-axis-coordinates.md` and
`docs/studio/STORAGE_LAYOUT.md`.

`validate_result` replaces `assert_finite` and checks the whole result for
internal consistency, reporting every defect rather than the first.

## Application orchestration

`scirust-studio-app-service` supervises the worker, owns the job lifecycle
and selects the run store, with no dependency on any GUI toolkit. See
`docs/studio/APP_SERVICE.md`.

## Run storage

`scirust run --store <dir>` (or `SCIRUST_STUDIO_STORE`) records the run
immutably — scenario, result, and a hashed provenance manifest — and
`scirust runs list|show|verify|discard` inspects the store. See
`docs/studio/STORAGE_LAYOUT.md` and
`docs/studio/adr/0004-immutable-run-storage.md`. Storage is opt-in; there is
no default location.

## Output formats

`scirust catalog`, `scirust run`, and `scirust runs list|show` accept
`--format text|json` (default `text`). `text` is meant for a human at a terminal; `json` is
meant for tests, scripts, and — eventually — a desktop client, and is a
direct serialization of `CapabilityRegistry`'s entries or a `RunResult`
with stable field names (see the two ADRs for why the field types are
shaped the way they are).

## Seeds and determinism

Until Phase 3B-2 every capability was
`DeterminismClass::StrictSameBinarySameTarget`: the result was a function of
the parameters alone, and `experiment.seed` was read by nothing.

`sim.stochastic.ornstein_uhlenbeck` is
`InherentlyStochasticRecordedSeed`. For such a capability:

- **`experiment.seed` is required.** Validation fails with `SRST-VAL-0095`
  without one. A single sample from a distribution is not evidence unless
  someone else can obtain the same sample.
- **The seed is recorded** in `RunProvenance::seed`, which is `None` for every
  capability that did not consume one — recording a scenario's seed against a
  result that ignored it would imply it mattered.
- **The verification is statistical.** No individual trajectory can be right
  or wrong; the distribution can. See ADR 0007 for how the tolerance is
  derived rather than chosen.
- **The run re-derives itself** from the recorded seed and compares bit for
  bit, so the reproducibility claim is demonstrated and not just asserted.

Adding another stochastic capability means calling
`validate_support::resolve_seed` in `validate`, passing the seed to the model,
and setting `RunProvenance::seed`. Skipping any of the three produces a result
nobody can reproduce, which is why the first two are enforced by the type
system and the third by the bridge contract test.

## Ensembles

`experiment.replicates` asks a stochastic capability for `n` independent
realisations instead of one. Absent, or `1`, means what every scenario meant
before the field existed.

- **The replicates' seeds are derived** from `experiment.seed` by
  `ensemble::replicate_seeds`, so an ensemble is reproducible from the same
  one number a lone run is. Replicate 0 keeps the base seed, which is what
  makes a one-replicate run bit-identical to the run the capability did before
  ensembles, and a small ensemble a prefix of a large one.
- **Every adapter calls `validate_support::resolve_replicates`**, not only the
  stochastic ones. A capability whose `DeterminismClass::draws_a_sample` is
  false refuses `replicates > 1` with `SRST-VAL-0096`: `n` copies of one
  trajectory is not a distribution.
  `every_capability_answers_for_replicates_according_to_its_class` walks the
  registry, so a new adapter that forgets fails in CI.
- **The summary is accumulated in one pass** by `ensemble::EnsembleAccumulator`,
  using Welford's update. Memory is `O(samples)` regardless of the replicate
  count, and a small spread on a large mean survives — which the
  `E[x²] − E[x]²` shortcut does not.
- **Retention is bounded and reported.** At most
  `ensemble::MAX_RETAINED_MEMBERS` individual realisations are kept; both the
  count drawn and the count kept are metrics, and both interfaces show them.
- **`Series::role` says what each curve is** — `EnsembleMean`,
  `EnsembleBandLower`/`Upper`, `EnsembleMember`, plus `Reference` for a line
  the solver did not compute. `validate_result` enforces the structural rules
  (one mean, a band that brackets it, one shared axis) but not statistical
  ones, which belong to the capability's own checks.

See `docs/studio/adr/0008-ensembles.md`, in particular for why the seed
derivation is scrambled rather than counted — SplitMix64 places every seed on
one shared cycle, so two replicates whose seeds differ by a multiple of its
increment would draw overlapping noise.
