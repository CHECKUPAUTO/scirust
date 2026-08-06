# ADR 0004: Immutable, content-hashed run storage

## Status

Accepted and implemented (Phase 2B).

## Context

`scirust_studio_runtime::RunProvenance` records what produced a result and
when, and Phase 2A's own doc comment said what it deliberately left out: "a
content-addressed run manifest, a scenario hash, a result hash, and a full
hardware/OS fingerprint belong to the run-storage system (Phase 2B), which
does not exist yet."

A scientific tool's claim to reproducibility rests on being able to answer
three questions about a result someone is looking at months later:

1. What exactly was run? (Not "a spring-mass-damper scenario" — *which
   bytes*.)
2. Is this result still the one that run produced, or has it been edited?
3. On what did it run?

None of those can be answered by keeping results in memory and printing
them.

## Decision

One run is one directory:

```text
<root>/runs/<run_id>/
    manifest.json           provenance, hashes, status, environment
    scenario.scirust.toml   the exact input, byte for byte
    result.json             the RunResult (completed runs only)
```

### Immutable

`scirust-studio-store` exposes no update, overwrite, or re-save operation.
That is necessary but not sufficient — a file on disk can be edited by
anything — so the manifest records the SHA-256 of the scenario and result
bytes, and `RunStore::verify` recomputes them. A run modified behind the
store's back is *detected*, and the error names the file that no longer
matches. `scirust runs verify` exposes this, and exits non-zero when any run
fails.

Hashing the stored bytes rather than a re-serialization of the parsed value
is deliberate: it detects any change to the file, including ones that parse
to an equal value.

### Atomic

A run is assembled under `runs/.partial-<run_id>/` and published by a single
`std::fs::rename` into `runs/<run_id>/`. A reader therefore never observes a
half-written run — either the directory is there and complete, or it is not
there at all. Directory rename is atomic on every platform this targets.

### Interruption is evidence, not corruption

If the process dies between `begin` and finalization, the `.partial-`
directory remains, and `RunStore::interrupted` reports it *along with the
scenario that was being attempted* — which is why the scenario is written
immediately at `begin` rather than at the end.

Leftovers are never silently cleaned up. A directory that shouldn't be there
is the only evidence a run was attempted at all, and throwing it away to
keep a listing tidy destroys exactly the information someone debugging a
crash needs. `scirust runs list` surfaces interrupted runs alongside real
ones; `runs discard` removes one, and refuses anything that is not a
`.partial-` directory so it can never delete a finalized run.

This is also why `scirust run --store` opens the store *before* executing:
recording the attempt separately from the outcome is the whole mechanism.

### Failed and cancelled runs are recorded too

A run that failed is still a run someone will want to ask about. Its
manifest records the failure message; `load_result` for it returns
`StoreError::NoResult` naming the status, rather than an empty result the
caller cannot distinguish from a real one.

### What the environment fingerprint is, and is not

`EnvironmentFingerprint` records OS, architecture, family, and pointer
width — `std::env::consts`, i.e. the compilation target.

It is **not** a hardware fingerprint. There is no CPU model, microcode
revision, or core count, because nothing in this workspace can obtain those
portably, and a field that is accurate on Linux and invented on Windows is
worse than an absent one. What is recorded is enough to answer the question
determinism actually turns on — "was this the same target?" — which is what
`DeterminismClass::StrictSameBinarySameTarget` is asserting.

### Run ids

`<UTC timestamp>-<first 16 hex chars of the scenario hash>`, with a numeric
suffix if that name is taken. Time-sortable so a directory listing is
chronological; content-tied so two runs of identical input are visibly
related. Uniqueness comes from the collision check, not from either
component — the same scenario really can be launched twice in one second.

### No default location

Storage happens only when asked: `--store <dir>`, or the
`SCIRUST_STUDIO_STORE` environment variable. There is no built-in path under
the user's home directory. Writing run history somewhere unrequested is an
application-level decision, and this build has no user-facing setting to
turn it back off — so it does not make it.

## Consequences

- A storage failure after a successful computation is reported but does not
  change the exit code. The run did succeed; saying otherwise would be as
  wrong as swallowing the problem silently.
- Nothing prunes the store. A long-lived store grows without bound, and no
  retention policy is implemented — that is a real limitation, recorded here
  rather than left for a user to discover. `runs list`/`runs discard` are
  the only management surface today.
- The store is not concurrency-safe across processes beyond what the atomic
  rename provides. Two processes writing to one store will not corrupt each
  other's runs (each assembles in its own `.partial-` directory and
  publishes atomically), but nothing coordinates them, and `runs discard`
  racing a writer is not defined. Single-writer use is what is tested.
- `scirust-studio-store` does not depend on the worker or the IPC layer. It
  stores `RunResult`s, wherever they came from — so the desktop shell can
  use the same store for worker-produced results without a second format.
