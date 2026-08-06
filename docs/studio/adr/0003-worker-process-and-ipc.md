# ADR 0003: An out-of-process worker and the protocol it speaks

## Status

Accepted and implemented (Phase 2B). Supersedes nothing; extends
`docs/studio/adr/0001-capability-registry.md` and
`docs/studio/adr/0002-structured-run-results.md`.

## Context

Phase 2A produced a real execution core: a capability registry, a
`CapabilityAdapter` contract, a structured `RunResult`, and five tested
adapters, all driven in-process by `scirust-cli`. Two properties that a
desktop application needs were explicitly *not* provided, and were recorded
as missing rather than faked:

1. **Cancellation was pre-execution only.** Every adapter checked
   `ExecutionControl::is_cancelled()` once, before handing the entire time
   span to `scirust-sim` as a single blocking call. A run that had started
   could not be stopped.
2. **Progress was absent.** `RunEvent` had no `Progress` variant, because
   there was no genuine intermediate signal to report and inventing a
   fraction from a timer would have been a fabricated measurement.

Both limitations have the same root cause: the integration loop lives inside
`scirust-sim` and offered no way in. A desktop shell that ran capabilities
in its own process would inherit both, plus a third problem — a capability
that aborts the process would take the user interface and any unsaved work
with it.

## Decision

Two changes, in the two places that can honestly make them.

### 1. A per-step observer in `scirust-sim`

`scirust_sim::simulate_observed` and `simulate_second_order_observed` take a
closure called after every accepted step, which returns `StepAction::Continue`
or `StepAction::Stop`. The pre-existing `simulate` and
`simulate_second_order` are now thin wrappers that pass an observer which
never stops.

This is the *only* placement that preserves the numerics. Chunking the span
from outside and calling `simulate` repeatedly was considered and rejected:
the loop computes `dt = h.min(t_end - t)` and advances `t = t + h`, so a
chunk boundary that does not land exactly where accumulation would put it
produces a shortened step, and the run silently stops being the run the
unchunked call would have produced. Re-implementing RK4 inside Studio was
rejected for the obvious reason — a second copy of an integrator is a second
copy to keep correct.

Because the wrapper and the observed function are literally the same code,
observing cannot change a completed run's output. `scirust-sim`'s
`observed_rk4_run_is_bit_identical_to_the_unobserved_one` and
`observed_symplectic_run_is_bit_identical_to_the_unobserved_one` assert
exactly that, and
`stopping_an_rk4_run_truncates_it_without_changing_the_kept_values` asserts
the complementary property: stopping early truncates the trajectory and
leaves every value that *was* produced identical.

`scirust-studio-runtime`'s `execute_support` builds on this to give every
fixed-step adapter real mid-run cancellation and real progress. Progress is
capped at 20 events per run — enough to drive a progress bar, few enough
that reporting never costs more than the work it reports on.

**What this does not cover, stated plainly.** The Robertson adapter
integrates with `scirust-stiff`'s adaptive Rosenbrock-W solver through
`scirust_sim::stiff_bridge`, which exposes no per-step callback. That
capability therefore still has pre-execution-only cooperative cancellation
and emits no progress events. It is not given a fabricated fraction, and
`RUNTIME_CONTRACT.md` says so per capability rather than implying uniform
behaviour.

### 2. A separate worker process

`scirust-studio-worker` is a real binary that speaks `scirust-studio-ipc`
over stdin/stdout. It exists for what in-process execution cannot give:

- **A crash is survivable.** An aborting capability takes down the worker,
  not the application.
- **Cancellation always works.** Where cooperative cancellation does not
  reach — the adaptive stiff path above — a supervisor can kill the process.
  That is the guarantee that makes "cancel" a promise rather than a hope.
- **The interface never blocks.** Long runs happen elsewhere.

Design points worth recording:

- **One writer.** The stdin reader thread and every job thread funnel into a
  single channel; only the main loop writes to stdout. Message ordering is
  therefore deterministic and no two writes can interleave mid-line. The
  alternative — a mutex around stdout — permits correct output but not
  *predictable* output.
- **One job at a time, FIFO queue.** Adapters are CPU-bound; running several
  concurrently on the same cores finishes none of them sooner and makes
  progress reporting meaningless. A queued job can still be cancelled before
  it starts, and gets exactly one terminal message either way.
- **Exactly one terminal message per request.** The job thread's outcome is
  authoritative; the runtime's own `Started`/`Completed`/`Cancelled`/`Failed`
  events are deliberately *not* forwarded, so a client can never observe a
  completion before the result belonging to it.
- **The client's disappearance stops the work.** Closing the client's pipe
  cancels the running job rather than computing a result nobody will read.

### The protocol

`scirust-studio-ipc` is transport-free: messages plus framing, with nothing
about pipes or process spawning, so the worker can speak it over stdio and a
test can speak it over an in-memory buffer. Three properties, each tested:

- **Versioned** — `PROTOCOL_VERSION` is exchanged in the opening handshake.
  The check is exact-match on purpose. A range check would be a
  compatibility promise with no tested compatibility behind it; there is
  exactly one version in existence.
- **Bounded** — `read_message` refuses anything past a caller-supplied limit
  *before* decoding it, so a peer that never sends a newline cannot drive
  the other side into unbounded allocation.
- **Correlated** — every worker message carries the client-assigned
  `RequestId`, so replies are never matched by arrival order.

Framing is newline-delimited JSON, which is safe because `serde_json`
escapes control characters inside strings — asserted by a test that encodes
a payload full of newlines and checks the result occupies exactly one line,
rather than left as a comment.

## Consequences

- A bug found by building this: `serde_json`'s default float parser is not
  bit-exact, so a `RunResult` crossing the process boundary came back with
  values one ULP away from the ones computed. The worker's
  `a_run_through_the_worker_matches_an_in_process_run` test caught it. Every
  Studio crate that reads floats back from JSON now enables the
  `float_roundtrip` feature, matching what `scirust-bench-schema` and
  `scirust-causal` already required for the same reason, and dedicated
  bit-exactness tests pin it in place.
- `scirust-sim` gained public API (`StepAction`, `ObservedRun`,
  `simulate_observed`, `simulate_second_order_observed`). This is additive:
  no existing signature or behaviour changed, and the crate's 134 tests pass
  unchanged.
- The worker is not yet used by `scirust-cli`'s `run`, which still executes
  in-process. Wiring the CLI through the worker would add process-spawn
  latency to every invocation for no benefit a command-line user can
  observe; the worker exists for the desktop shell, and it is tested
  directly rather than through a CLI flag that only tests would exercise.
  Doing this the other way round — adding a `--worker` flag and calling that
  "integration" — would have produced a code path with no real user.
- Supervision (restart policy, health checks, spawn/respawn) is **not** in
  this phase. The worker is a well-behaved process with a tested lifecycle;
  deciding *when* to restart one belongs with the application that owns its
  window, which does not exist yet.
