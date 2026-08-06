# ADR 0005: The desktop application is Tauri 2 + Dioxus Web, not a second implementation

## Status

Accepted and implemented (Phase 3A-2).

## Context

Everything scientific in Studio already exists and is tested: a capability
registry, five adapters over `scirust-sim`, a chunked runtime with real
progress and cancellation, an out-of-process worker with a versioned IPC
protocol, an immutable content-hashed run store, and — since ADR 0006 —
results that carry their own axis coordinates.

None of it has a window. The CLI is the only way in, and a CLI cannot show
someone a trajectory.

The decision is therefore not "how do we build a scientific application" but
"how do we put a window in front of one that already exists, without
acquiring a second one." Every failure mode worth avoiding here is a
variation on the same theme:

* A frontend that re-implements a model "just for the preview" — now there
  are two answers to one question, and the interface's is untested.
* A frontend that reaches the filesystem or spawns processes directly — now
  the security surface of the application is the security surface of a
  webview.
* A frontend that regenerates an axis because it did not receive one — now
  the picture is of a different experiment (ADR 0006).
* A frontend whose logic can only be exercised by clicking — now the
  lifecycle a user experiences is the least-tested part of the system.

## Decision

Four choices, each made against a named alternative.

### 1. Tauri 2 as the native shell

The shell owns the window, the sidecar worker and a narrow set of typed
commands. It contains no scientific code: every command delegates to
`scirust-studio-app-service` and returns a view type.

* **Not Electron.** Electron ships a Chromium and a Node.js runtime with the
  application. Node at application runtime is a scripting host inside the
  trust boundary, and the bundle is an order of magnitude larger for a
  program whose actual work is native Rust.
* **Not a local HTTP server.** A production server on a loopback port is a
  network service any local process can reach, and it makes the application's
  privilege boundary a port number.

Tauri uses the operating system's WebView (WebKitGTK on Linux, WKWebView on
macOS, WebView2 on Windows) and communicates over an in-process IPC channel
with a per-window capability manifest.

### 2. Dioxus 0.7 **Web**, compiled to WebAssembly

The interface is a Rust crate compiled to `wasm32-unknown-unknown` and
bundled as static assets.

* **Not Dioxus Desktop.** Dioxus Desktop is itself a webview shell. Using it
  under Tauri would mean two shells; using it instead of Tauri would mean
  giving up Tauri's capability model, sidecar management and bundler.
* **Not Dioxus Native/Blitz.** It is not ready to carry a product interface,
  and choosing it would mean betting the phase on a renderer rather than on
  the science.
* **Not React or Vue.** A JavaScript frontend cannot import the Rust types it
  is displaying, so the wire contract becomes a hand-maintained parallel
  definition — and it puts an npm dependency tree inside a scientific
  application's supply chain.

Writing the interface in Rust is what makes the next decision possible.

### 3. The interface's logic is pure Rust, tested on the host

Dioxus, `wasm-bindgen` and `web-sys` are declared under
`[target.'cfg(target_arch = "wasm32")'.dependencies]`. Everything else — the
`(state, message) -> state` reducers, the chart geometry, the action registry,
the string table, the wire decoding — compiles and tests on the host with no
browser and no WebAssembly toolchain.

The result is that the run lifecycle a user actually experiences (queued →
running → cancelling → cancelled, and the separate interrupted path) is unit
tested, and `cargo clippy --target wasm32-unknown-unknown --all-targets`
type-checks the components against that tested logic.

### 4. One typed bridge, and a mock that cannot ship

The frontend's entire connection to the outside world is the `StudioBackend`
trait — one method per Tauri command. There is no `fetch`, no WebSocket, no
generic file or process access, because the window is granted none
(`DESKTOP_SECURITY.md`).

A deterministic `MockBackend` exists for frontend tests and for previewing
layout with `dx serve`. It integrates nothing: every number it returns is a
hard-coded literal, labelled `MOCK` wherever it appears. It is behind a
feature flag, **a release build that enables it does not compile**, and when
it is active the interface shows a permanent banner. A scientific application
that can silently display invented numbers is worse than one that refuses to
start.

## Consequences

**The desktop cannot drift from the science.** The catalogue, the validation
messages, the progress, the results and the verification all come from the
same crates the CLI uses. Adding a capability adds it to both.

**Two workspaces.** `apps/scirust-studio` is excluded from the root
workspace, so the GUI toolchain cannot affect the root MSRV gate, the root
dependency graph, or the time of a `cargo check --workspace` that has nothing
to do with the desktop. The cost is a second `Cargo.lock` and a second CI
workflow (`.github/workflows/studio-desktop.yml`).

**A three-step build.** The worker must be built and staged as a sidecar, the
frontend must be built to WebAssembly and staged into `dist/`, and only then
can the bundle be produced. Both staging steps are Rust tools with tests that
refuse to stage something that is not what it claims to be, because the
failure they prevent — an application that builds cleanly and then cannot
find its own worker, or opens a blank window — is invisible to `cargo build`.
See `DESKTOP_BUILD.md`.

**The frontend and the shell can drift apart in principle.** They declare the
wire types separately, because the frontend compiles to WebAssembly and
cannot link Tauri, the registry or the store. `src-tauri/tests/bridge_contract.rs`
is the answer: it serialises real values from the shell's own code paths and
decodes them into the frontend's types, asserting among other things that
coordinates cross bit for bit and that *cancelled* and *interrupted* do not
collapse into each other.

**Not decided here.** Code signing, an updater, licensing enforcement and
installer publication are out of scope for this phase; the Windows artifact
is explicitly an unsigned preview (`WINDOWS_DESKTOP_ACCEPTANCE.md`). Opening
and saving scenario files needs a native file picker the shell does not yet
expose — the actions exist and are shown disabled with that reason, rather
than hidden or wired to nothing.
