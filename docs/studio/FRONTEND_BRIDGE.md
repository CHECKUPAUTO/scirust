# The frontend bridge

The WebAssembly interface reaches the native shell through one trait and
seventeen commands. This document is the contract: what crosses, in what
shape, and what keeps the two sides from drifting apart.

## The trait

```rust
pub trait StudioBackend {
    async fn bootstrap(&self)        -> Result<BootstrapWire, FrontendError>;
    async fn catalog(&self)          -> Result<Vec<CapabilityWire>, FrontendError>;
    async fn tutorials(&self)        -> Result<Vec<CapabilityWire>, FrontendError>;
    async fn load_tutorial(&self, capability_id: &str) -> Result<ScenarioWire, FrontendError>;
    async fn validate_scenario(&self, source: &str)    -> Result<ValidationWire, FrontendError>;
    async fn start_run(&self, source: &str)            -> Result<JobWire, FrontendError>;
    async fn cancel_run(&self, job_id: &str)           -> Result<(), FrontendError>;
    async fn job_snapshot(&self, job_id: &str)         -> Result<JobWire, FrontendError>;
    async fn active_job(&self)       -> Result<Option<JobWire>, FrontendError>;
    async fn list_runs(&self)        -> Result<Vec<StoredRunWire>, FrontendError>;
    async fn load_run(&self, run_id: &str)   -> Result<RunWire, FrontendError>;
    async fn verify_run(&self, run_id: &str) -> Result<IntegrityWire, FrontendError>;
    async fn poll_events(&self)      -> Result<EventBatchWire, FrontendError>;
    async fn store_path(&self)       -> Result<String, FrontendError>;
    async fn restart_worker(&self)   -> Result<BootstrapWire, FrontendError>;
    async fn worker_diagnostics(&self) -> Result<Vec<String>, FrontendError>;
    async fn frontend_ready(&self)   -> Result<BootstrapWire, FrontendError>;
}
```

This is the interface's **entire** connection to the outside world. There is
no `fetch`, no WebSocket, no `localStorage` cache of scientific data and no
shell access — the window is granted none of those (`DESKTOP_SECURITY.md`),
and this trait is what it has instead.

Note what is *not* here: nothing takes a path, a program name, a URL or an
environment variable. Adding a capability to the interface means adding a
method here and a typed command there; there is no general-purpose escape
hatch to reach for instead.

## The commands

| Command | Arguments | Returns |
|---|---|---|
| `studio_bootstrap` | — | `BootstrapView` |
| `studio_catalog` | — | `Vec<CapabilityView>` |
| `studio_tutorials` | — | `Vec<CapabilityView>` |
| `studio_load_tutorial` | `capabilityId` | `ScenarioView` |
| `studio_validate_scenario` | `source` | `ValidationOutcome` |
| `studio_start_run` | `source` | `JobSnapshot` |
| `studio_cancel_run` | `jobId` | `()` |
| `studio_job_snapshot` | `jobId` | `JobSnapshot` |
| `studio_active_job` | — | `Option<JobSnapshot>` |
| `studio_list_runs` | — | `Vec<StoredRunView>` |
| `studio_load_run` | `runId` | `RunView` |
| `studio_verify_run` | `runId` | `VerificationReportView` |
| `studio_poll_events` | — | `EventBatch` |
| `studio_store_path` | — | `String` |
| `studio_restart_worker` | — | `BootstrapView` |
| `studio_worker_diagnostics` | — | `Vec<String>` |
| `frontend_ready` | — | `BootstrapView` |

Argument names are camelCase, which is what Tauri 2 maps onto a command's
snake_case parameters.

Every command returns `Result<T, ErrorView>`, and an `ErrorView` always
carries a code, a title, a full explanation, whether the user can carry on,
and a suggested action when there is a concrete one. There is deliberately no
path by which the interface ends up with only "something went wrong."

## How the call is made

`TauriBridge` looks up `window.__TAURI__.core.invoke` **by reflection at call
time** rather than binding it as a `#[wasm_bindgen]` extern import.

That is a deliberate trade. The import form is more idiomatic; the reflection
form means a page opened outside the application gets a readable
`BRIDGE_UNAVAILABLE` error naming exactly the property that was missing,
instead of an uncatchable JavaScript `TypeError` during module instantiation
that leaves a blank window and nothing in the interface to read.

Answers cross as JSON text: `JSON.stringify` on the JavaScript side,
`serde_json` with `float_roundtrip` on the Rust side. Both directions of that
conversion are exact for `f64` — `ryu` emits the shortest text that
round-trips, and the parser reads it back bit for bit — so a coordinate that
reaches the chart is the coordinate the integrator produced. The one shape
JSON cannot represent, a non-finite value, cannot arise: results are validated
to contain none before they are ever stored
(`scirust_studio_runtime::validate_result`).

## The error taxonomy

`FrontendError` distinguishes four origins, because they send a user to four
different places:

| Code | Means | Recoverable |
|---|---|---|
| *(from the shell)* | The service refused: `Busy`, `Validation`, `WorkerUnavailable`, … | As the shell says |
| `BRIDGE_UNAVAILABLE` | There is no Tauri host — the page is open somewhere it cannot work | No |
| `BRIDGE_DECODE_FAILED` | The shell sent something this build cannot read: a version mismatch between interface and shell | Yes |
| `BRIDGE_REJECTED` | The call failed with something that is not an error view (a panic in a command) | Yes |

Keeping the decode failure distinct matters: telling a user "the simulation
failed" when the real problem is a mismatched build would send them looking in
entirely the wrong place.

A validation failure carries its structured problems through, so a rejected
scenario lands in the editor's problem list with source locations — not only
in a modal the user has to dismiss before they can act on it.

## Why the types are declared twice

The frontend compiles to `wasm32-unknown-unknown` and cannot link `tauri`, the
registry, the store or the worker supervisor. So `views.rs` (shell) and
`backend/wire.rs` (interface) are separate declarations of the same shapes,
and nothing in the type system stops one from being renamed while the other
is not.

`apps/scirust-studio/src-tauri/tests/bridge_contract.rs` is what stops it. Each
test takes a **real** value from the shell's own code paths, serialises it
exactly as the IPC does, and deserialises it into the type the interface
actually uses. Twelve tests, covering:

* every capability in the real registry, field by field, asserting the
  crossing carries content and not merely that it parses;
* every `JobState` variant with its internal `state` tag — including that
  **cancelled and interrupted do not collapse into each other**, because the
  interface presents them differently and a user acts on them differently;
* a real validation failure, with its problems, fields, source locations and
  stage tags;
* a `RunView` whose coordinates and series values are asserted to cross
  **bit for bit**, using values chosen to be awkward (`1/3`,
  `f64::MIN_POSITIVE`, `1.2345678901234567e-9`);
* every `AppEvent` variant, including the externally-tagged `WorkerExitClass`;
* the bootstrap view, the stored-run summary and the error view.

A renamed field or a changed enum tag fails there, at build time, rather than
showing up later as a mysteriously empty panel.

## The mock backend

`MockBackend` exists so the interface's logic can be tested and its layout
previewed without a native build. Three properties make it safe:

1. **It is not a simulation.** It integrates nothing. Every number is a
   hard-coded literal — nine unevenly-spaced coordinates and nine values
   chosen to make a chart draw. Writing even a simple integrator here would
   create exactly the thing this project must not have: two answers to one
   scientific question.
2. **It announces itself.** Its capability is named "(MOCK DATA)", its series
   is "Displacement (MOCK)", its adapter crate is `mock`, and its verification
   status is `not_applicable` with the explanation "Mock data is not verified
   and never can be." Tested.
3. **It cannot ship.** A release build with the feature enabled does not
   compile, and when it is active the interface shows a permanent banner. The
   CI job asserts the guard by trying the build and failing if it succeeds.

Its job lifecycle is a pure function of the poll count — queued, 25 %, 50 %,
75 %, completed — so a test drives the whole run lifecycle deterministically
rather than waiting on a timer.

## Adding a command

1. Add the method to `StudioBackend` and implement it in `bridge.rs` and
   `mock.rs`.
2. Add the `#[tauri::command]` to `src-tauri/src/commands.rs` and register it
   in `tauri::generate_handler!`.
3. Add the view type to `views.rs` and the wire type to `backend/wire.rs`.
4. Add a case to `bridge_contract.rs`.
5. If it takes an identifier, run it through `check_id`; if it takes source,
   through `check_source`.

If step 5 does not apply because the new command takes a path, a program name
or a URL — stop. That is the command this design exists to prevent.
