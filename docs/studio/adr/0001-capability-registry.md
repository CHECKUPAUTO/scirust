# ADR 0001: Capability registry and descriptor design

## Status

Accepted. Implemented in `scirust-studio-registry` (Phase 2A).

## Context

Phase 1 (`docs/studio/adr/0000-scope-and-sequencing.md`) shipped a single
hard-coded path: `scirust-cli` knew about exactly one capability
(`sim.mechanics.spring_mass_damper`), with its parameters, units, and solver
choice baked directly into `scirust-cli/src/studio.rs`. That does not scale
to five capabilities, let alone the full `scirust-sim` model set, and it
gives no machine-readable answer to "what does this capability accept,
in what units, with what solver" — a question both a `catalog --format json`
consumer and (later) a desktop form need answered without re-deriving it
from adapter source.

## Decision

Introduce `scirust-studio-registry` as a small, dependency-light crate that
owns exactly one thing: typed, `&'static` capability descriptors and a
registry that catalogues them.

- `CapabilityDescriptor` and its component types (`FieldDescriptor`,
  `SolverDescriptor`, `OutputDescriptor`, `VerificationCheckDescriptor`)
  are plain data — no behaviour, no execution. Each capability's adapter
  crate (`scirust-studio-runtime`) declares one `pub static DESCRIPTOR:
  CapabilityDescriptor` per capability, as a compile-time table, the same
  zero-allocation pattern `scirust-cli`'s own `Command` tables already use.
- `FieldDescriptor.dimension` is a real `scirust_units::Dimension`, not a
  string — so "does this field's unit match what the field expects" is a
  checked equality, not a convention. `FieldDescriptor.error_code` ties each
  field to one stable `SRST-VAL-*` code (see ADR 0002 and
  `scirust-studio-runtime/src/validate_support.rs`'s code-block allocation:
  generic codes at 90-99, then one ten-number block per capability).
- `CapabilityRegistry` keeps its entries sorted by `CapabilityId` at all
  times (on every `register()` call), so `iter()`, `to_text()`, and
  `to_json()` are deterministic regardless of registration order — required
  by the brief's "return capabilities in deterministic order," and simpler
  to guarantee by construction than to re-derive at every call site.
- **The registry cannot construct a descriptor from a bare id or name.**
  The only descriptors that ever reach a `CapabilityRegistry` are the
  `&'static CapabilityDescriptor` values returned by
  `CapabilityAdapter::descriptor()` in `scirust-studio-runtime`, and every
  `CapabilityAdapter` implementation in that crate is a real, tested
  adapter over a real `scirust-sim` model. "Never advertise a capability
  without a tested executable adapter" is therefore a structural property —
  there is no code path that registers a descriptor without the adapter
  that backs it — rather than a rule someone has to remember to follow.

### What is deliberately *not* here

`CapabilityDescriptor` derives `serde::Serialize` but not `Deserialize`:
every field type in this crate is built from `&'static str`/`&'static [T]`
slices, and a deserializer cannot produce a `'static`-lifetime borrow from
its own, non-`'static`, input buffer. This is not a workaround; it is the
correct reflection of what these types are for (compile-time tables, read
out to JSON, never parsed back into this exact shape). Runtime types that
*do* need to round-trip through JSON (`scirust_studio_runtime::RunResult`
and its `RunProvenance`) use owned `String`s instead of `&'static str` for
exactly this reason — see ADR 0002.

## Consequences

- Adding a sixth capability means adding one descriptor and one adapter
  module in `scirust-studio-runtime`; nothing in `scirust-studio-registry`
  or `scirust-cli` changes.
- `scirust catalog --format json` is a real, stable, tested artifact
  (`CapabilityRegistry::to_json`), not a hand-written summary that can
  drift from the adapters it describes.
- The registry crate has no dependency on `scirust-sim`, `scirust-stiff`,
  or any specific model crate — only on `scirust-units` (for `Dimension`)
  and `scirust-studio-command` (for `ErrorCode`). It compiles fast and
  changes rarely, which matters once a desktop client depends on it too.
