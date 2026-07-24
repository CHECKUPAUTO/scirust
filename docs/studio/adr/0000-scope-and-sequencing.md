# ADR 0000: Scope reality check and phase sequencing for SciRust Studio

## Status

Proposed — recorded now so the scope decision is explicit and reviewable,
rather than silently assumed. See `docs/studio/REPOSITORY_AUDIT.md` for the
evidence this decision is based on.

## Context

The SciRust Studio brief (47 sections) specifies a full commercial Windows
desktop product: a Tauri/Dioxus GUI, a supervised worker-process architecture,
a versioned IPC protocol, a scenario schema, a unified command registry, a
bilingual (EN/FR) accessibility-audited offline help system, a Windows
installer (NSIS + MSI) with Authenticode signing and a signed auto-updater,
threat modeling, fuzzing, property tests, performance benchmarks, an SBOM, and
a tag-triggered release pipeline — with an explicit instruction not to stop
after planning and to implement through the full definition of done.

Phase 0 audit findings that bear directly on sequencing:

1. The repository is a 134-member monorepo already in production-grade shape
   for its existing scope (deep learning, symbolic math, numerics, and 15+
   regulated-industry verticals), with mature CI (fmt/clippy/miri/fuzz/SBOM/
   determinism gates already exist and should be extended, not duplicated).
2. `scirust-sim` (16 real, oracle-tested, dependency-free domain models) has
   zero presence in `scirust-cli`. This is the single most concrete, most
   valuable, most tractable gap identified.
3. No Tauri/Dioxus/WebView2/installer/signing infrastructure exists anywhere
   in the tree today. All of it is net-new.
4. This session runs in a Linux container. Every cross-platform Rust crate
   (schema, command registry, adapters, worker, CLI) can be written, built,
   and tested here. A Windows GUI, MSI/NSIS installer, Authenticode signature,
   or Tauri updater artifact cannot be built or executed here — only authored
   here and then verified on a `windows-latest` CI runner or a real Windows
   machine.
5. The base repository is PolyForm Noncommercial 1.0.0; the person directing
   this work is the copyright holder (matching email in `LICENSING.md`), so
   commercial framing is self-authorized but should be stated accurately in
   any installer/EULA text rather than implied.

Building an entire commercial product end-to-end in one pass is not an honest
claim regardless of effort spent — the brief's own anti-fabrication principles
(no stubs, no fabricated coverage, no claiming "tested" without evidence) apply
to the *process* of building Studio, not only to Studio's own UI.

## Decision

Sequence work in the order the brief itself lays out (Phase 0 → 1 → 2 → …),
but treat each phase as a real, shippable, independently-tested increment
committed to this branch, rather than attempting all phases before any commit.
Concretely, for the immediate next increment (Phase 1):

- Add new, cross-platform, dependency-light library crates to the **existing
  root workspace** (not a separate desktop workspace yet — that split is only
  justified once a GUI dependency tree actually needs it, per the brief's own
  §7 condition): a versioned scenario schema crate, a typed command descriptor
  registry, and an error catalogue.
- Wire `scirust-cli` to gain real `scirust-sim` commands (e.g. `catalog`,
  `run`) through that registry, without breaking any of the ~55 existing
  dispatched commands or their tests.
- Defer the Tauri/Dioxus desktop shell, the worker-process/IPC layer, the
  Windows installer/signing/updater pipeline, and the bilingual help system to
  later, explicitly-scoped increments — each of those is independently a
  multi-week-or-more workstream, and the GUI/installer ones cannot be verified
  from this container, so starting them without confirming direction risks
  producing exactly the kind of untested, unverifiable "implementation" the
  brief prohibits.

## Consequences

- Every increment actually merged is real and tested where this environment
  allows testing; nothing is presented as done that wasn't run.
- The desktop shell and Windows packaging remain fully intended, not
  abandoned — they need a concrete "yes, do the GUI/installer next, here is
  how Windows-side verification will happen" decision before code is written
  for them, since that decision changes crate layout (`apps/scirust-studio/`,
  possible separate workspace) and CI (new `windows-latest`-based workflows).
- This ADR will be superseded or updated once that direction is confirmed.

## Update (first Phase 1 increment landed)

`scirust-studio-command` (command descriptors, a registry, and the
`SRST-VAL-*` error catalogue) and `scirust-studio-schema` (the versioned
`.scirust.toml` scenario schema, a small explicit unit table over
`scirust-units`, and validation) were added to the root workspace, with real
unit tests (34 tests + 1 doctest). `scirust-cli` gained two new commands,
`catalog` and `run`, additive to its existing ~55 commands (all of which keep
passing, including the two tests — `dispatch_reaches_each_group` and
`help_lists_every_dispatched_command` — that exist specifically to catch a
command being wired into dispatch without being documented). `run` executes a
real capability end to end: `sim.mechanics.spring_mass_damper`, backed by the
actual `scirust_sim::mechanics::SpringMassDamper` model and
`scirust_sim::simulate` (RK4), not a mock. The shipped tutorial scenario
(`docs/studio/tutorials/spring_mass_damper.scirust.toml`) is the exact file
executed by `scirust-cli`'s own test suite via `include_str!`, so the example
a user is told to run is the example that is tested — running it prints a
measured relative energy drift around `7e-15` for the undamped case, which is
a real oracle check (RK4 approximately conserves energy; a bug that broke the
integrator would show up as a much larger number here, not just a passing
test).

Deliberately not done in this increment: refactoring any of the ~55 existing
`scirust-cli` commands onto the new registry (Phase 4, once the registry has
proven itself further), wiring up any other `scirust-sim` model (Phase 3),
and anything Tauri/Dioxus/Windows-installer-shaped (still pending the
direction decision above).

## Update (direction confirmed; Phase 2A landed)

The person directing this project confirmed the direction explicitly:
**build the shared headless execution core first** (capability registry,
structured result contract, representative multi-domain adapters, then the
cross-platform runtime/worker/storage layer), **then** the Tauri/Dioxus
desktop shell on top of that real core — explicitly *not* by migrating the
~55 legacy commands first, and *not* by building a GUI that calls
`scirust-cli` or duplicates execution logic. The full seven-step sequence
they specified: (1) capability registry + result contract, (2)
representative adapters, (3) cross-platform runtime/worker/storage, (4)
first desktop vertical slice, (5) remaining adapters + help, (6) legacy CLI
migration, (7) Windows installer/signing/updater.

Phase 2A implements step (1) and a deliberately-chosen slice of step (2):
`scirust-studio-registry` (capability descriptors) and
`scirust-studio-runtime` (the `CapabilityAdapter` contract, the structured
`RunResult` model, and five real adapters — the original spring-mass-damper
plus SIR, two-body orbital, series RLC, and stiff Robertson kinetics, chosen
specifically to force the architecture to handle vector-valued state,
multiple solvers per capability, and a genuinely different stiff solver
family). See `docs/studio/adr/0001-capability-registry.md`,
`docs/studio/adr/0002-structured-run-results.md`,
`docs/studio/RUNTIME_CONTRACT.md`, and `docs/studio/CAPABILITY_MATRIX.md`.
`scirust-cli` now reaches every capability through the registry/adapter
pair — it no longer depends on `scirust-sim` at all — but the ~55 legacy
commands are still untouched, per the explicit instruction not to migrate
them yet.

Still not started: the worker process, bounded IPC, job lifecycle,
cancellation beyond a pre-execution check, run storage/manifest/provenance
(step 3); anything Tauri/Dioxus (step 4); the remaining 11 `scirust-sim`
model families (step 5); legacy CLI migration (step 6); and Windows
installer/signing/updater (step 7).
