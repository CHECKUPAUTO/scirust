# SciRust Studio — desktop security model

The window renders untrusted-by-construction code: a WebAssembly module in an
operating-system WebView. This document states exactly what that window can
reach, why each grant exists, and what is deliberately absent.

Nothing here is aspirational. Every claim below is asserted by a test in
`apps/scirust-studio/src-tauri/tests/security_audit.rs`, which reads the real
`tauri.conf.json` and the real capability manifest rather than a copy.

## The trust boundary

```
   ┌──────────── untrusted ────────────┐   ┌───────── trusted ──────────┐
   │  WebView + WebAssembly interface  │──▶│  native shell (Rust)       │
   │  no filesystem                    │IPC│  17 typed commands         │
   │  no process spawn                 │   │  supervises the sidecar    │
   │  no network                       │   │  owns every path it opens  │
   └───────────────────────────────────┘   └────────────────────────────┘
```

The arrow is the only crossing. It carries JSON, in one direction at a time,
through named commands with typed parameters.

## What the window is granted

`src-tauri/capabilities/main-window.json`, in full:

| Permission | Why it is here |
|---|---|
| `core:default` | Tauri's baseline: the IPC channel itself, and the ability to render |
| `core:app:allow-version` | The status bar shows the application version |
| `core:window:allow-set-title` | The title reflects the open scenario |
| `core:window:allow-minimize` | Window controls |
| `core:window:allow-close` | Window controls, with a confirmation while a run is active |
| `core:event:allow-listen` / `allow-unlisten` | Tauri's own event plumbing |

Every entry is `core:*`. No plugin permission is granted, because no plugin is
used.

## What the window is not granted, and would need to be

These are the permissions whose absence is the security model. The audit test
asserts that none of them appears:

* `shell:allow-execute`, `shell:allow-spawn` — **the frontend cannot start a
  process.** Not even the worker: the shell spawns and supervises the sidecar
  itself, and the frontend has no command that names a binary.
* `fs:*` — no path-taking read or write of any kind.
* `http:*` — no outbound request. The application does not phone home, check
  for updates, or fetch anything.
* `dialog:*` — **not granted, although the shell now links the plugin.**
  `studio_open_scenario` and `studio_save_scenario` show a native picker from
  Rust and exchange the file's *contents* with the frontend. Granting the
  webview `dialog:*` would let it open its own picker and keep the resulting
  path, which is the one thing those two commands exist to prevent: "the file
  the user just picked" is not a smaller permission than "any file" once the
  frontend holds the location. `tests/security_audit.rs` asserts the absence,
  and the assertion was checked by granting it and watching the test fail.
* `tauri-plugin-fs` appears in `Cargo.lock` as a dependency of the dialog
  plugin. The crate is linked; its permissions are not granted, which is what
  the `fs:*` line above is asserting.
* `process:*`, `os:*`, `store:*` — absent.

There is also no *general-purpose* command hiding behind an innocuous name.
The 19 commands are enumerated in `FRONTEND_BRIDGE.md`; none of them takes a
path, a program name, a URL or an environment variable.

## Input the shell does not trust

Identifiers and scenario source arrive from the least-trusted input the
process handles, so both are checked before use:

* **Identifiers** (`run_id`, `job_id`, `capability_id`): non-empty, at most
  128 characters, ASCII alphanumeric plus `-`, `_`, `.`. Anything that could
  be mistaken for a path — `../etc/passwd`, `a/b`, `a\b`, a NUL byte,
  `run;rm -rf /` — is refused with a structured error before it reaches the
  store. Tested.
* **Scenario source**: at most 1 MiB. A scenario is a short configuration
  file; the bound exists so a runaway frontend cannot hand the process an
  arbitrarily large allocation.

## Content Security Policy

```
default-src 'self';
script-src 'self' 'wasm-unsafe-eval';
style-src 'self';
img-src 'self' data:;
font-src 'self';
connect-src 'self' ipc: http://ipc.localhost;
object-src 'none';
frame-src 'none';
frame-ancestors 'none';
base-uri 'self';
form-action 'none'
```

Three points worth stating plainly:

* **`'wasm-unsafe-eval'`, not `'unsafe-eval'`.** The first permits
  instantiating a WebAssembly module and nothing else. The second would permit
  arbitrary JavaScript evaluation. The interface is a WebAssembly module, so
  the first is exactly what it needs and the second would be a strictly larger
  grant for no benefit.
* **No `'unsafe-inline'` anywhere.** `index.html` carries no inline script and
  no inline style, and the stylesheet is a bundled file. The rule was not
  relaxed to make an error disappear.
* **`connect-src` is the IPC channel and nothing else.** `ipc:` and
  `http://ipc.localhost` are how Tauri's own transport addresses itself on the
  various platforms. No external origin appears anywhere in the policy, the
  HTML, the CSS or the interface source — a CDN font or an analytics tag would
  be blocked at runtime, so shipping a reference to one would mean shipping a
  page that only works on the machine that wrote it.

The policy is not disabled (`dangerousDisableAssetCspModification: false`),
prototype freezing is on, and the asset protocol is disabled with an empty
scope.

## The sidecar

* It is the worker **this repository builds**. `tools/prepare-sidecar` copies
  it from the local `target/` directory and reports its SHA-256; it has no
  download path at all, and a missing input is a hard failure naming the
  `cargo build` command that produces it.
* It is never resolved through `PATH`. `WorkerLaunchConfig` carries an
  explicit list of candidate paths, so a binary called `scirust-studio-worker`
  planted earlier in a user's `PATH` cannot be picked up.
* Built binaries are not committed (`apps/scirust-studio/.gitignore`). A
  checked-in executable is a supply-chain question nobody wants, and this one
  is reproduced by one command.

## The activity console is not a shell

The console at the bottom of the window accepts the action registry's console
words — `run`, `validate`, `cancel`, `catalogue`, `runs`, `help` — and nothing
else. There is no path from a typed string to a process. A test asserts that
`sh`, `bash`, `cmd`, `powershell`, `rm -rf /` and `eval` all resolve to no
action.

## The mock backend cannot ship

A build with `--features mock-backend` and debug assertions off does not
compile; the CI job asserts this by trying it and failing if it succeeds. When
the mock *is* active, the interface shows a permanent banner reading "This
build serves fabricated data and must not be used for any scientific purpose."

This is a scientific-integrity control rather than a conventional security
one, but it belongs in the same list: the worst thing this application could
do to a user is show them a number that came from nowhere.

## What is out of scope in this phase

* **Code signing.** The Windows artifact is an unsigned preview and says so;
  see `WINDOWS_DESKTOP_ACCEPTANCE.md`.
* **An updater.** None is configured. An update mechanism is a remote code
  path into a signed bundle and needs signing to exist first.
* **Licensing enforcement.**
* **Hardened-runtime / notarization on macOS**, sandbox profiles, and
  Windows AppContainer.

None of these are refused on principle; they need infrastructure this
repository does not yet hold, and claiming them without it would be worse than
their absence.
