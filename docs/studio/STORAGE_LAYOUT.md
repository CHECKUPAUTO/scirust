# SciRust Studio run storage layout

What `scirust-studio-store` writes to disk. Everything here is implemented;
see `docs/studio/adr/0004-immutable-run-storage.md` for the reasoning.

## Layout

```text
<root>/
  runs/
    20260728T215109Z-ef820514594aa333/     a finalized run
      manifest.json
      scenario.scirust.toml
      result.json                          completed runs only
    .partial-20260728T215230Z-a1b2c3d4/    an interrupted run
      scenario.scirust.toml
```

A directory whose name starts with `.partial-` is a run whose writing
process never finished. It is reported by `RunStore::interrupted` and by
`scirust runs list`, and is never silently removed.

## Run ids

```text
<YYYYMMDD>T<HHMMSS>Z-<first 16 hex chars of the scenario SHA-256>[-<n>]
```

Time-sortable, so listing a directory lists runs chronologically.
Content-tied, so two runs of byte-identical input are visibly related. The
`-<n>` suffix is added only if the name is already taken — uniqueness comes
from that check, not from the timestamp.

## `manifest.json`

```json
{
  "manifest_schema_version": 1,
  "run_id": "20260728T215109Z-ef820514594aa333",
  "capability_id": "sim.orbital.two_body",
  "scenario_sha256": "ef820514594aa333da0acad902e5016542bbe2e9e998fe8af71bb29597f03f56",
  "result_sha256": "509a692f6f6066a27cfc42e6f5df5cf92cba00feeced4767f878adeeb5e35ff6",
  "status": "completed",
  "started_at_rfc3339": "2026-07-28T21:51:09.762091223+00:00",
  "finished_at_rfc3339": "2026-07-28T21:51:09.839571754+00:00",
  "environment": {
    "os": "linux",
    "arch": "x86_64",
    "family": "unix",
    "pointer_width": 64
  },
  "result_schema_version": 1,
  "store_version": "0.1.0"
}
```

`status` is flattened, so a failed run carries its reason alongside it:

```json
{ "status": "failed", "message": "numerical failure: state became non-finite at t = 0.003" }
```

`result_sha256` is absent for cancelled and failed runs, which have no
result.

Three schema versions appear because three things version independently:
the manifest (storage), the result contract
(`scirust_studio_runtime::RESULT_SCHEMA_VERSION`), and the scenario schema
(recorded inside `scenario.scirust.toml` itself).

## Hashes

SHA-256 over the **stored bytes**, lowercase hex — not over a
re-serialization of the parsed value, so any edit is detected including one
that would parse to an equal value.

`RunStore::verify` recomputes both and reports which file no longer matches:

```console
$ scirust runs verify --store ./store
  [FAILED] 20260728T215109Z-ef820514594aa333: run `...` has been modified:
    result.json hashes to f704bc50..., but its manifest records 509a692f...
error: 1 of 1 runs no longer match their recorded hashes
```

Exit code 7. A clean store exits 0.

## Writing a run

1. `begin` creates `.partial-<run_id>/` and writes `scenario.scirust.toml`
   immediately — so an interrupted run still records what was attempted.
2. The run executes.
3. `complete`/`cancel`/`fail` writes `result.json` (completed only) and
   `manifest.json`, then renames `.partial-<run_id>/` to `<run_id>/`.

Step 3's rename is the single step that publishes the run. Everything before
it writes into a directory no reader looks in, so a reader never sees a
partial run.

## Using it

Storage happens only when asked — `--store <dir>`, or the
`SCIRUST_STUDIO_STORE` environment variable. There is no default path.

```console
$ scirust run docs/studio/tutorials/two_body_orbit.scirust.toml --store ./store
recorded as run 20260728T215109Z-ef820514594aa333
...

$ scirust runs list --store ./store
RUNS
  20260728T215109Z-ef820514594aa333  sim.orbital.two_body             completed

$ scirust runs show 20260728T215109Z-ef820514594aa333 --store ./store
$ scirust runs verify --store ./store
$ scirust runs discard <interrupted-run-id> --store ./store
```

`runs list` and `runs show` also accept `--format json`. `runs discard`
refuses anything that is not an interrupted run, so it cannot delete a
finalized one.

## Result schema versions

`manifest.json` records `result_schema_version`, and `RunStore::load_result`
returns a `LoadedRunResult` tagged with the version found **in the result
file itself** (not the manifest's copy — the file is what is being decoded).

| Version | Axis coordinates | Readable | Verifiable |
|---|---|---|---|
| 1 | absent | yes | yes |
| 2 | stored exactly | yes | yes |

A v1 result's metrics, warnings, verification checks, provenance and series
*values* are all still available. What is not available is when each sample
was taken, and none is invented: `LoadedRunResult::x_axis_meaning()` returns
`SampleIndexOnly` for v1, so a consumer is made to acknowledge the
difference. `load_result_v2` refuses a v1 run outright for callers that
genuinely need coordinates.

Nothing is migrated in place. An immutable store that rewrote its own files
would invalidate every hash it had recorded — so a v1 run stays a v1 run, and
new runs are written as v2 beside it. Hashes are over the stored bytes and
are therefore version-agnostic; `tests/v1_compatibility.rs` pins all of this
against real v1 fixtures produced by the shipped Phase 2B code.

## Limitations, stated rather than discovered

- **Nothing prunes the store.** It grows without bound; no retention policy
  is implemented.
- **Single-writer.** Two processes sharing a store will not corrupt each
  other's runs — each assembles in its own `.partial-` directory and
  publishes atomically — but nothing coordinates them, and `discard` racing
  a writer is undefined. Single-writer use is what is tested.
- **The environment fingerprint is the compilation target, not the
  hardware.** No CPU model, microcode revision, or core count: nothing here
  can obtain those portably, and an invented field is worse than an absent
  one.
