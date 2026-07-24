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
  successful `validate()`), an `ExecutionControl` (checked for
  cancellation before the integration call — see
  `docs/studio/adr/0002-structured-run-results.md` for why finer-grained
  cancellation is Phase 2B), and an `&mut dyn EventSink` that receives
  `RunEvent::Started`/`Warning`/`Completed`/`Cancelled`/`Failed`. It
  returns a fully-populated `RunResult` or an `ExecutionError`, never a
  partially-built one.
- Every adapter calls `scirust_studio_runtime::assert_finite(&result)`
  immediately before returning `Ok(result)`, converting any non-finite
  derived value into `ExecutionError::Internal` instead of a silently
  "successful" result containing JSON `null`.

## Implemented capabilities (Phase 2A)

| Capability id | Source model | Solvers | Verification checks |
|---|---|---|---|
| `sim.mechanics.spring_mass_damper` | `scirust_sim::mechanics::SpringMassDamper` | `rk4` | `energy_drift` |
| `sim.epidemiology.sir` | `scirust_sim::epidemiology::Sir` | `rk4` | `population_conservation`, `non_negative_compartments` |
| `sim.orbital.two_body` | `scirust_sim::orbital::TwoBody` | `symplectic_euler`, `rk4` | `energy_drift`, `angular_momentum_drift`, `finite_trajectory` |
| `sim.electrical.rlc` | `scirust_sim::electrical::SeriesRlc` | `rk4` | `finite_solution`, `damping_regime`, `energy_non_increasing` |
| `sim.chemistry.robertson` | `scirust_sim::chemistry::Robertson` (via `scirust_sim::stiff_bridge::simulate_rosenbrock`, feature `stiff`) | `stiff_rosenbrock_w` | `mass_conservation`, `non_negative_concentrations`, `solver_completion` |

Every row is a real, tested adapter with a shipped, executed tutorial
scenario under `docs/studio/tutorials/`. See `docs/studio/CAPABILITY_MATRIX.md`
for how these five relate to the rest of `scirust-sim` and the wider
workspace.

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

## Output formats

Both `scirust catalog` and `scirust run` accept `--format text|json`
(default `text`). `text` is meant for a human at a terminal; `json` is
meant for tests, scripts, and — eventually — a desktop client, and is a
direct serialization of `CapabilityRegistry`'s entries or a `RunResult`
with stable field names (see the two ADRs for why the field types are
shaped the way they are).
