# SciRust Studio application service

`scirust-studio-app-service` is the layer between the Studio core and
whatever is displaying it. Everything named here is implemented; see
`docs/studio/adr/0003-worker-process-and-ipc.md` for the worker it drives.

It has **no dependency on any GUI toolkit** — a test asserts that against the
manifest. The desktop shell is one caller, the backend smoke-test binary is
another, the integration tests are a third.

## What it owns

| Concern | Notes |
|---|---|
| Worker lifecycle | locate, spawn, handshake, read, restart |
| Job lifecycle | one active job, snapshots, terminal classification |
| Event fan-out | bounded buffer, drops reported not hidden |
| Run store | selection, listing, loading, verification |
| Validation | routed through the existing stages, reshaped for an editor |
| Tutorials | from the real registry |
| Shutdown | cancels an active run rather than orphaning a worker |

## What it does not own

No capability adapters, no integrators, no validation *rules*, no IPC
framing, no store verification, no CLI formatting. Each of those already has
a crate; duplicating one here would create a second definition that drifts.

## Policies, and why

### One worker, one active job

Adapters are CPU-bound. A second concurrent run finishes neither sooner, and
two progress bars competing for the same cores is worse than one honest one.
The worker's own FIFO queue still exists but the application never uses it —
a second `start_run` returns `AppServiceError::Busy` naming the active job,
so the UI can disable the button rather than silently queueing work.

### An interrupted job is never re-run

If the worker exits while a job is running, that job becomes
`JobState::Interrupted` and stays that way. Re-executing someone's
calculation without being asked produces a result they did not request and
cannot distinguish from the one they lost.

The store entry is deliberately *not* finalized: dropping the `PendingRun`
leaves its `.partial-` directory in place, which is exactly the evidence
`scirust-studio-store` was designed to preserve. `interrupted_runs()`
reports them.

`start_worker()` will start a fresh worker on request — so the application
stays usable — but only new jobs run on it.

### Cancellation means cancelled

| Capability | Mechanism | Reported as |
|---|---|---|
| the four fixed-step ones | cooperative, checked between steps | `Cancelled` |
| `sim.chemistry.robertson` | worker process terminated | `Cancelled` |

The adaptive stiff solver exposes no per-step callback, so terminating the
process is the only thing that actually stops it. The job is still
classified as **cancelled**, because that is what the user asked for; calling
it a failure would blame them for their own decision. `JobSnapshot`
distinguishes the two cases up front through `supports_progress`, so the UI
picks the right control before the first event rather than switching styles
mid-run.

A cooperative cancellation leaves the worker alive and immediately reusable;
a terminating one does not, and the application must restart it.

### Storage opens before execution

`start_run` calls `RunStore::begin` *before* the worker is asked to do
anything. A run killed halfway therefore leaves a record of what was
attempted. This is the whole reason the store separates "begin" from
"finalize".

### Bounded events, with visible loss

`EventBus` holds a fixed number of events and discards the **oldest** when
full — the newest are what a user is looking at. `drain()` returns a
`dropped` count so the interface can say it fell behind rather than showing
a silent gap.

## Job states

```text
Queued ──► Running { fraction, t } ──┐
       └─► RunningIndeterminate ─────┤
                                     ├─► Completed { run_id }
              Cancelling ────────────┼─► Cancelled
                                     ├─► FailedNumerical / FailedValidation / FailedInternal
                                     └─► Interrupted { detail }
```

`Cancelled` and `Interrupted` are separate states and never collapse into
one another: one is the user's decision, the other is the engine dying.

`JobState::fraction()` returns `None` for an indeterminate run rather than
`0.0`, so a caller must handle the absence instead of drawing a bar that
looks stuck.

## Storage location

`AppServiceConfig::platform_default()` uses:

- **Windows** `%APPDATA%\Memorithm\SciRust Studio`
- **macOS** `~/Library/Application Support/Memorithm/SciRust Studio`
- **Linux** `$XDG_DATA_HOME/Memorithm/SciRust Studio` (or `~/.local/share/…`)

with `settings/`, `runs/`, `logs/` and `recovery/` beneath it. Not the
installation directory (read-only, shared, replaced by an installer) and not
the user's Documents folder (theirs, not the application's).

Unlike the CLI — which keeps storage opt-in because it has no settings UI to
turn it back off — the desktop application chooses a default, and must tell
the user where it is.

## Validation

`validate_source` runs the existing stages in order and stops after the
generic one if it found anything: running capability validation against a
scenario whose units are already known to be wrong produces a second wave of
errors about the same mistake.

Each `Problem` carries a stable `SRST-…` code, a title, an explanation, a
suggested action where one exists, the field name where the underlying error
names one, and a source location where it can be determined. Locations are
best-effort and **absent rather than guessed** — an editor that jumps to a
fabricated line is worse than one that does not jump.

## Errors

`AppServiceError` carries a stable `AppErrorCode`, a title, a recoverability
flag and a suggested action. That is what makes "something went wrong"
impossible to write at the UI layer: there is always something specific to
say.

## Testing

`tests/worker_supervision.rs` runs against the **real** compiled worker,
covering: bootstrap and handshake, a missing worker, a completed run
committed and verified, genuine fixed-step progress, Robertson's
indeterminate state, cooperative cancellation, a cancelled run stored as
cancelled, the single-active-job rule, an unexpected worker exit producing
`Interrupted` with no rerun and preserved partial evidence, restart for a
future job only, validation refusing before the worker is involved, bounded
event overflow, tutorials and catalogue from the real registry, shutdown
with and without an active run, and tamper reporting.

The worker is binary-only, so it cannot be a path dependency. The tests
locate it beside the test executable and fail with an explicit instruction if
it is absent:

```bash
cargo build -p scirust-studio-worker
cargo test  -p scirust-studio-app-service
```

## Not implemented here

Window management, rendering, localization and keyboard handling belong to
the shell. Worker *pooling*, pause/resume, and remote execution are not
implemented at all and are not stubbed.
