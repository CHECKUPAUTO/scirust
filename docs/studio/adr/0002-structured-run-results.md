# ADR 0002: Structured run results and the CapabilityAdapter contract

## Status

Accepted. Implemented in `scirust-studio-runtime` (Phase 2A).

## Context

Phase 1's `scirust run` printed formatted text directly from inside the
command handler, using `println!` calls interleaved with the actual
`scirust_sim::simulate` call. That is fine for one capability and one
output format, but it does not survive contact with:

- a second output format (`--format json`, needed now for tests and for
  any future GUI/API consumer);
- a worker process (Phase 2B) that must send *something* back to a
  supervisor over IPC, and that something cannot be pre-rendered ANSI text;
- a desktop chart, which needs named series and units, not a formatted
  table.

The brief's own instruction is explicit: "Keep terminal formatting outside
the runtime" and "structured events are emitted." Phase 2A implements both.

## Decision

### The `CapabilityAdapter` trait

```rust
pub trait CapabilityAdapter: Send + Sync {
    fn descriptor(&self) -> &'static CapabilityDescriptor;
    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport>;
    fn execute(&self, scenario: &ValidatedScenario, control: &ExecutionControl, sink: &mut dyn EventSink)
        -> Result<RunResult, ExecutionError>;
}
```

- `validate` runs *after* generic schema validation
  (`scirust_studio_schema::validate`) has already passed. It performs
  capability-specific checks — missing/unknown fields, wrong physical
  dimension, wrong cardinality, out-of-range values, unsupported solver,
  missing step/tolerance — via the shared helpers in
  `scirust-studio-runtime/src/validate_support.rs`, which every adapter
  calls with its own `FieldDescriptor` table rather than re-implementing
  the same logic five times.
- `ValidatedScenario` can only be constructed by a successful `validate()`
  call (its constructor is `pub(crate)`). "This scenario passed validation"
  is therefore a fact the type system carries into `execute()`, not a
  convention.
- `execute` takes an `ExecutionControl` (a cheap, cloneable cancellation
  flag) and an `&mut dyn EventSink` (receiving `RunEvent::Started` /
  `Warning` / `Completed` / `Cancelled` / `Failed`). Phase 2A's adapters
  call `scirust_sim::simulate`/`simulate_rosenbrock` etc. as a single
  blocking call with no progress callback from the integrator itself, so
  cancellation is checked once before the call starts, and no adapter emits
  a fake fractional `Progress` event — there is nothing genuine to report
  between `Started` and `Completed`. Fine-grained mid-run cancellation
  (chunking the integration, or a real killable worker process) is Phase
  2B's job; the trait's shape does not need to change for that — only what
  calls `is_cancelled()`, and how often.

### The `RunResult` model

`RunResult` is the brief's own sketch, implemented close to verbatim:
`schema_version`, `capability_id`, `summary`, `axes`, `series`, `metrics`,
`warnings`, `verifications`, `provenance`. Two decisions worth recording:

1. **No `NaN`/infinite value may reach a "successful" `RunResult`.**
   `scirust_sim`'s own blow-ups are already caught by `SimError::NonFinite`
   from the integrator (mapped to `ExecutionError::Numerical`), but a
   *derived* computation — an energy-drift ratio, a damping-ratio
   classification — could still individually divide by zero or take a
   log of a negative number even when the underlying trajectory is fully
   finite. `result::assert_finite` scans every `Series` value and every
   `Metric::Scalar` and is called by every adapter immediately before
   returning `Ok(result)`; a violation becomes `ExecutionError::Internal`
   instead of silently serializing as JSON `null` (which is what
   `serde_json` does with `NaN`/infinity by default — exactly the silent
   failure mode the brief warns against).
2. **`RunProvenance` is deliberately minimal.** It records the capability
   id, its `DeterminismClass`, the adapter crate/version
   (`env!("CARGO_PKG_VERSION")` of `scirust-studio-runtime` itself — the
   one version this crate can stamp robustly without a build script), and
   real wall-clock start/end timestamps plus measured elapsed duration. It
   does **not** record a scenario hash, a result hash, a Git commit, or a
   CPU/OS fingerprint — those belong to the run-manifest/provenance system
   that Phase 2B's storage layer will build, and fabricating placeholder
   values for them now would be exactly the "claim a property that has not
   been implemented and verified" the brief prohibits.

### Where JSON serialization lives

`RunResult::to_json_pretty()` lives in `scirust-studio-runtime`, mirroring
`scirust_studio_registry::CapabilityRegistry::to_json()` — the crate that
owns a type also owns its serialization, so `scirust-cli` (and later a
worker/desktop client) needs no direct `serde_json` dependency merely to
expose `--format json`.

## Consequences

- `scirust-cli` no longer imports `scirust_sim` at all (verify with
  `cargo tree -p scirust-cli | grep scirust-sim`, which returns nothing);
  every capability is reached through `find_adapter`/`build_registry`.
- Adding a capability's *worker-driven* execution path later (Phase 2B)
  means implementing a new `EventSink` that forwards over IPC and driving
  the same `CapabilityAdapter::execute` — no change to the trait or the
  result model.
- `RunResult` round-trips through JSON (`serde` `Serialize` **and**
  `Deserialize`, unlike the registry's static tables — see ADR 0001) so
  tests, and eventually a desktop client, can parse a result back into a
  typed value instead of re-scraping text.
