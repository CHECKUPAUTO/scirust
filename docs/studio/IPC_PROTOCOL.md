# SciRust Studio worker protocol

The wire protocol between a Studio client and `scirust-studio-worker`. Every
type named here exists in `scirust-studio-ipc`; this describes what is
implemented, not what is planned. See
`docs/studio/adr/0003-worker-process-and-ipc.md` for why it is shaped this
way.

- **Version:** `PROTOCOL_VERSION = 1`
- **Framing:** newline-delimited JSON (NDJSON), one message per line
- **Default size limit:** 64 MiB per message (`DEFAULT_MAX_MESSAGE_BYTES`)
- **Transport:** the worker uses stdin/stdout; the crate itself works on any
  `BufRead`/`Write`

## Session shape

```text
client                              worker
  |  hello {protocol_version}          |
  |----------------------------------->|
  |          ready {protocol_version,   |
  |                 worker_version}     |
  |<-----------------------------------|
  |  run {request_id, scenario_toml}    |
  |----------------------------------->|
  |          progress {request_id, ...} |   0..20 times
  |<-----------------------------------|
  |          warning {request_id, ...}  |   0..n times
  |<-----------------------------------|
  |          completed {request_id,     |   exactly one terminal
  |                     result}         |   message per request
  |<-----------------------------------|
  |  shutdown                           |
  |----------------------------------->|
  |          shutting_down              |
  |<-----------------------------------|
```

`hello` must be the first message. Anything else first — or a version
mismatch — gets `protocol_error` and the worker exits.

## Client messages

| `type` | Fields | Meaning |
|---|---|---|
| `hello` | `protocol_version` | Open the session. Must be first. |
| `catalog` | `request_id` | Ask for the capability catalogue. |
| `validate` | `request_id`, `scenario_toml` | Validate without running. |
| `run` | `request_id`, `scenario_toml` | Validate and run. |
| `cancel` | `request_id` | Cancel an in-flight or queued request. |
| `shutdown` | — | Exit once the current request settles. |

`request_id` is a bare number, assigned by the client. The worker never
invents one.

## Worker messages

| `type` | Fields | Meaning |
|---|---|---|
| `ready` | `protocol_version`, `worker_version` | Handshake accepted. |
| `catalog` | `request_id`, `catalog_json` | The registry's own JSON, verbatim. |
| `validated` | `request_id`, `capability_id` | Terminal: validation passed. |
| `progress` | `request_id`, `fraction`, `t` | Genuine progress (see below). |
| `warning` | `request_id`, `warning` | A non-fatal `RunWarning`. |
| `completed` | `request_id`, `result` | Terminal: a `RunResult`. |
| `cancelled` | `request_id` | Terminal: stopped before finishing. |
| `failed` | `request_id`, `kind`, `message` | Terminal: see kinds below. |
| `protocol_error` | `message` | The channel is unusable; the worker exits. |
| `shutting_down` | — | Exiting in response to `shutdown`. |

### Terminal messages

Every `catalog`, `validate`, and `run` request receives **exactly one** of
`catalog` / `validated` / `completed` / `cancelled` / `failed`. This holds
for requests cancelled while queued, and for queued requests discarded by a
`shutdown` — a client waiting on a request is never left waiting forever.

`cancel` for an unknown or already-finished request is silently ignored, so
a client racing a completion is not punished for it.

### Failure kinds

| `kind` | Meaning | CLI exit code |
|---|---|---|
| `validation` | Parse, schema, or capability validation failed | 3 |
| `numerical` | Integration failed numerically | 5 |
| `internal` | A bug in the worker or an adapter | 7 |

These match `docs/studio/RUNTIME_CONTRACT.md`'s exit-code table;
`FailureKind::exit_code` is the single source and is asserted against those
numbers in a test.

### Progress

`fraction` is the proportion of the scenario's **time span** that has been
integrated, computed from the `t` the solver reports after an accepted step.
It is not a timer, not an estimate, and not a spinner.

At most 20 progress events are emitted per run. Capabilities whose solver
cannot report intermediate steps emit **none** rather than an invented one —
today that is `sim.chemistry.robertson`, whose adaptive stiff solver exposes
no per-step callback. A client must therefore treat progress as optional and
not use its absence as evidence a run is stuck.

## Framing rules

- One message per line, terminated by `\n`. A trailing `\r` is tolerated.
- Blank lines are skipped.
- A line longer than the limit is rejected **before** decoding, with
  `MessageTooLarge`.
- End of stream between messages is a clean shutdown; end of stream
  *part-way through* a line is `Truncated` — the peer died mid-write.
- Encoded messages never contain a raw newline: `serde_json` escapes control
  characters inside strings. A test asserts this by encoding a payload full
  of newlines and checking it occupies exactly one line.

## Floating point

Results carry computed `f64` values, so encoding and decoding must be
bit-exact. `serde_json`'s default float parser is **not** — it can return a
value one ULP away from the one serialized. Every Studio crate that reads
floats back from JSON therefore enables `serde_json`'s `float_roundtrip`
feature, and both `scirust-studio-ipc` and `scirust-studio-store` have tests
comparing `f64::to_bits` across a round trip so the feature cannot be
dropped silently.

This was not a theoretical concern: it was found by the worker test that
compares a result computed in the worker process against the same scenario
run in-process, which failed with 1-ULP differences until the feature was
enabled.

## Concurrency

One job runs at a time; further `run` requests queue in arrival order. See
ADR 0003 for why. Cancelling a queued job removes it from the queue and
replies `cancelled` without it ever having started.

## What is not here

- No supervision: spawning, restart policy, and health checks belong to the
  application that owns the worker, which does not exist yet.
- No streaming of partial results — a run returns one `RunResult` at the
  end. A 250 000-step run is a few tens of megabytes of JSON, comfortably
  inside the size limit; chunked result streaming is only worth building
  when something needs it.
- No authentication or encryption. The transport is a pipe to a child
  process this build spawned; adding a handshake secret would protect
  against nothing that can reach that pipe.
