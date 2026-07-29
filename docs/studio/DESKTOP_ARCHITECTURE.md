# SciRust Studio — desktop architecture

What runs, where it runs, and why the boundaries are where they are. The
decision record is `docs/studio/adr/0005-tauri-dioxus-desktop-architecture.md`;
this document describes the thing that was built.

## The four processes and one module

```
┌─────────────────────────────────────────────────────────────────────┐
│ scirust-studio  (native, Rust, Tauri 2.11.5)                        │
│                                                                     │
│   src-tauri/  window · sidecar supervision · 17 typed commands      │
│      │  no scientific code: every command delegates                 │
│      │                                                              │
│      ├── scirust-studio-app-service   bootstrap, jobs, events       │
│      │      ├── scirust-studio-registry    the capability catalogue │
│      │      ├── scirust-studio-runtime     adapters + result schema │
│      │      ├── scirust-studio-schema      scenario validation      │
│      │      └── scirust-studio-store       immutable run storage    │
│      │                                                              │
│      ▼ spawns and supervises                                        │
│   ┌───────────────────────────────────────────┐                     │
│   │ scirust-studio-worker  (sidecar process)  │                     │
│   │   length-prefixed JSON over stdio         │                     │
│   │   one job at a time · cancellable         │                     │
│   └───────────────────────────────────────────┘                     │
│                                                                     │
│   ▼ creates                                                         │
│   ┌───────────────────────────────────────────┐                     │
│   │ OS WebView (WebKitGTK / WKWebView /        │                    │
│   │             WebView2)                      │                    │
│   │   loads bundled static assets              │                    │
│   │   ┌──────────────────────────────────────┐ │                    │
│   │   │ scirust-studio-ui (WebAssembly)      │ │                    │
│   │   │   Dioxus 0.7.9 Web                   │ │                    │
│   │   │   one way out: StudioBackend         │ │                    │
│   │   └──────────────────────────────────────┘ │                    │
│   └───────────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────┘
```

There is no HTTP server, no `localhost` port, no Node.js and no bundled
browser. The WebView loads local files over Tauri's custom protocol and talks
to the shell over an in-process IPC channel.

## Why the worker is a separate process

Unchanged from Phase 2B (`docs/studio/adr/0003-worker-process-and-ipc.md`),
and the desktop is the reason it was built that way:

* A numerical failure in an adapter cannot take the window down with it.
* An adaptive stiff solve with no per-step callback can still be *stopped* —
  by terminating the process. Cooperative cancellation covers the fixed-step
  capabilities; process termination covers the rest. Both are reported to the
  user as **cancelled**, because that is what the user asked for.
* The interface stays responsive while a long integration runs, without the
  UI thread and the integrator sharing an address space.

The worker is bundled as a Tauri **sidecar**: `externalBin` in
`tauri.conf.json`, staged under the `<name>-<target-triple>[.exe]` name Tauri
resolves at runtime. The frontend has no permission to spawn it and no way to
name it — the shell owns its whole lifecycle.

## The three layers, and what each may not do

| Layer | May | May not |
|---|---|---|
| `scirust-studio-ui` (WASM) | Render, hold interface state, call the 17 commands | Touch the filesystem, spawn anything, make network requests, contain scientific logic |
| `src-tauri` (native shell) | Own the window, supervise the sidecar, expose typed commands, run native file operations on its own behalf | Contain scientific logic, expose a generic file/process/network command |
| `scirust-studio-app-service` and below | Everything scientific | Know that a GUI exists |

The third row is what makes the CLI and the desktop the same application. Both
call `scirust-studio-app-service`; neither contains a second implementation of
anything.

## The interface's own structure

Everything that can be decided without a DOM lives in a module that compiles
on the host:

| Module | What it holds | Tested on the host |
|---|---|---|
| `model` | `AppModel`, `Msg`, `update()` — pure `(state, message) -> state` | The whole run lifecycle, including late progress arriving after a cancellation |
| `chart` | Coordinate mapping, min/max bucket reduction, SVG point generation, the accessible summary | That every-Nth decimation loses a spike the bucketing keeps |
| `actions` | One registry driving the palette, the shortcuts and the console | That the console refuses `sh`, `bash`, `cmd`, `powershell`, `rm -rf /`, `eval` |
| `i18n` | The English/French string table | That no key is left untranslated outside an explicit allowlist |
| `backend::wire` | The types that cross the bridge | That the shell's JSON decodes, tags and all |

Only `ui` needs a browser, and it is compiled for `wasm32` alone. The
practical effect: `cargo test -p scirust-studio-ui` runs in about a second and
covers the application's rules; `cargo clippy --target wasm32-unknown-unknown
--all-targets` type-checks the components against them.

## The run lifecycle, end to end

1. **Bootstrap.** The frontend calls `frontend_ready`, which both tells the
   shell the WebView really loaded (used by `--smoke-test-window`) and returns
   the bootstrap view: worker state, store path, capability count, and any
   runs whose writer never finished.
2. **Catalogue.** `studio_catalog` returns the real registry. Nothing is
   hard-coded in the interface.
3. **Tutorial.** `studio_load_tutorial` returns the shipped, tested scenario
   for a capability.
4. **Validation.** `studio_validate_scenario` runs the same three stages the
   CLI does — parse, schema, capability — and returns structured problems with
   source locations, not a formatted blob.
5. **Run.** `studio_start_run` validates and starts a job. One run at a time;
   a second is refused with `Busy` and a suggested action.
6. **Progress.** The interface polls `studio_job_snapshot`. A capability whose
   solver reports genuine fractions gets a progress bar; one that cannot
   (`sim.chemistry.robertson`) gets indeterminate activity and **never** an
   invented percentage. The distinction is in the type: `RunDisplay::Determinate`
   versus `RunDisplay::Indeterminate`.
7. **Cancellation.** `studio_cancel_run` moves the job to *cancelling*. Late
   progress arriving afterwards does not undo it — there is a test for exactly
   that.
8. **Settlement.** Completed, cancelled, failed (numerical / validation /
   internal) or **interrupted**. Interrupted is not a synonym for cancelled: it
   means the engine died underneath the user, and it also marks the engine
   stopped so Run is disabled and Restart engine becomes available.
9. **Result.** `studio_load_run` returns the stored run *and* its
   store-integrity check as a separate field. "The bytes are unchanged" and
   "the physics checks passed" are different claims and the interface does not
   merge them.
10. **Chart.** Plotted against the coordinates the integrator produced. A v1
    result has none, so it is plotted against sample ordinals and labelled as
    such in a notice above the chart.

## Charting, and what it refuses to invent

* **Coordinates come from the result.** For schema v2 that is the stored axis.
  For v1 the shell returns `XAxisKind::SampleIndex` with the label
  `Sample index`, and the interface prints a notice saying the spacing shown
  is not the spacing the solver used.
* **Reduction never hides a peak.** Series longer than 2 000 points are
  reduced by min/max bucketing, which keeps every bucket's extremes. Plain
  every-Nth decimation is rejected, and a test constructs a spike that
  decimation drops and bucketing keeps.
* **The caption states what was reduced.** "showing 2 000 of 10 002 points".
* **There is always a text alternative.** An accessible summary plus a table
  of the plotted coordinates, so the chart can be read by someone who cannot
  see it and checked by a reviewer against numbers.

## Events

The shell keeps a bounded event buffer and the interface drains it by polling
(`studio_poll_events`, every 120 ms). Push would let a slow interface build an
unbounded backlog inside the native process; the buffer instead drops the
oldest and reports how many, and the interface logs that it fell behind.

## Where things live on disk

`%APPDATA%\Memorithm\SciRust Studio` on Windows,
`~/Library/Application Support/Memorithm/SciRust Studio` on macOS,
`$XDG_DATA_HOME/Memorithm/SciRust Studio` (or `~/.local/share/...`) on Linux —
with `settings`, `runs`, `logs` and `recovery` beneath. The store layout
itself is unchanged from `docs/studio/STORAGE_LAYOUT.md`, so a run recorded by
the CLI is readable by the desktop and the reverse.

## What this phase deliberately does not include

* The other eleven simulation adapters (Phase 3B).
* Migration of the legacy CLI commands.
* Opening and saving scenario files — needs a native file picker; the actions
  are shown disabled with that reason.
* Code signing, an updater, licensing enforcement, installer publication.
