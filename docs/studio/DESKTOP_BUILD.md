# Building the SciRust Studio desktop application

Three artifacts have to exist before a bundle can be produced: the worker, the
WebAssembly interface, and the shell. Building them in the wrong order
produces an application that compiles and then cannot find its own worker, or
opens a blank window — failures `cargo build` cannot see, which is why both
staging steps are tools that verify what they stage.

## Prerequisites

| Tool | Version | Why exactly this one |
|---|---|---|
| Rust | stable (nightly for `cargo fmt`) | The root `rustfmt.toml` uses unstable options |
| `wasm32-unknown-unknown` target | — | `rustup target add wasm32-unknown-unknown` |
| Dioxus CLI (`dx`) | **0.7.9** | Must match the `dioxus` crate version; the CLI writes the bootstrap that loads the compiled module |
| Tauri CLI (`cargo-tauri`) | **2.11.4** | The newest CLI compatible with the pinned `tauri` 2.11.5 — see below |

```bash
rustup target add wasm32-unknown-unknown
cargo binstall -y dioxus-cli@0.7.9 tauri-cli@2.11.4
# or, from source:
cargo install dioxus-cli --locked --version 0.7.9
cargo install tauri-cli  --locked --version 2.11.4
```

Two traps in those two lines, both of which cost a red CI job:

* **The Tauri CLI is a different crate from the `tauri` library, versioned
  independently.** The crate is `tauri-cli` (the *binary* it installs is
  `cargo-tauri`, which is not a crate at all), and its latest release is
  2.11.4 while the library is 2.11.5. Assuming they move together produces
  `could not find tauri-cli in registry crates-io with version =2.11.5`.
* **`-y` is `--no-confirm`.** `cargo binstall -y --no-confirm …` is rejected
  for repeating an argument — which, behind a `||` fallback, silently turns
  a thirty-second prebuilt install into a several-minute source build.

### Linux system dependencies

Tauri 2 links against WebKitGTK 4.1:

```bash
sudo apt-get install --no-install-recommends -y \
    libwebkit2gtk-4.1-dev libgtk-3-dev libayatana-appindicator3-dev \
    librsvg2-dev patchelf
```

macOS and Windows ship their WebView (WKWebView and WebView2); on Windows 10
before 21H2, WebView2 may need the Evergreen runtime installed.

## The build, in order

```bash
# 1. The worker. This is the sidecar; it is never downloaded.
cargo build --release -p scirust-studio-worker

# 2. Stage it under the target-triple name Tauri resolves at runtime.
cd apps/scirust-studio
cargo run -p prepare-sidecar -- --profile release

# 3. The interface, compiled to WebAssembly.
dx build --release --platform web

# 4. Stage the bundle into dist/, which is what tauri.conf.json bundles.
cargo run -p stage-frontend -- --profile release

# 5. The application. The config is discovered from src-tauri/; do not pass
#    it with --config, which *merges* a patch over the discovered config and
#    would apply the whole file twice.
cargo tauri build
```

Step 4 also runs automatically as the `beforeBuildCommand` in step 5, so a
`cargo tauri build` with neither a frontend build nor a staged bundle fails
loudly — naming `dx build` — rather than bundling whatever happens to be in
`dist/`.

### What the staging tools refuse

`prepare-sidecar`:

* refuses a missing worker, with the `cargo build` command that produces it;
* puts `.exe` **after** the target triple
  (`scirust-studio-worker-x86_64-pc-windows-msvc.exe`), which is the naming
  rule most easily got wrong and the one Tauri actually resolves;
* makes the copy executable on Unix, and reports its size and SHA-256;
* never downloads anything and never invokes a shell string.

`stage-frontend`, whose contract is *"`dist/` holds a real bundle when this
exits zero"* rather than *"a copy just happened"*:

* searches for the built bundle rather than hard-coding a path, because the
  Dioxus CLI's output layout is its own business and has changed between
  versions;
* accepts an **already-staged** `dist/` when there is no local build to copy,
  after verifying it — on a packaging machine the bundle arrives as a CI
  artifact unpacked straight into `dist/` and there is no `target/dx` at all.
  A fresh build always wins over a staged one, so a rebuild is never silently
  ignored;
* refuses to stage a directory with no `index.html` (nothing to load) or no
  `.wasm` module (nothing to run), reporting which half is missing;
* empties `dist/` first, so a file the new build no longer produces cannot
  survive into the next bundle;
* reports the file count, the total WebAssembly size and the index's SHA-256.

Both are ordinary Rust binaries with unit tests, and both run identically on
Linux, macOS and Windows.

## Running without building a bundle

```bash
# The application, from the workspace.
cargo run -p scirust-studio-desktop --release

# The scientific path, end to end, with no window at all.
cargo run -p scirust-studio-desktop --release -- --smoke-test-backend

# Prove the WebView really loaded the bundled interface. Exits non-zero if
# the frontend does not call `frontend_ready` within 60 seconds.
cargo run -p scirust-studio-desktop --release -- --smoke-test-window
```

`--smoke-test-backend` prints JSON on both success and failure. A successful
run looks like:

```json
{
  "ok": true,
  "worker_version": "0.1.0",
  "capabilities_available": 5,
  "result_schema_version": 2,
  "axis_points": 10002,
  "store_integrity": "verified",
  "scientific_checks": [
    { "id": "energy_drift", "status": "passed", "measured": 6.99e-15 }
  ]
}
```

## Previewing the interface without a native build

```bash
cd apps/scirust-studio
dx serve --features mock-backend
```

This serves the deterministic mock backend, which integrates nothing — every
number is a hard-coded literal. The window carries a permanent banner saying
so, and a release build with this feature **does not compile**:

```bash
cargo check -p scirust-studio-ui --release --features mock-backend
# error: the `mock-backend` feature is enabled in a build with debug
#        assertions off. …
```

## The checks

```bash
cd apps/scirust-studio

# Interface logic — reducers, chart geometry, actions, i18n, wire decoding.
# Fast: the GUI stack is gated on cfg(target_arch = "wasm32").
cargo test -p scirust-studio-ui
cargo test -p scirust-studio-ui --features mock-backend

# The components, type-checked against that logic. Both feature settings:
# `ActiveBackend` is a cfg-resolved type alias, so the mock build binds every
# component to a different backend type and is a different compilation.
cargo clippy -p scirust-studio-ui --all-targets \
    --target wasm32-unknown-unknown -- -D warnings
cargo clippy -p scirust-studio-ui --all-targets \
    --target wasm32-unknown-unknown --features mock-backend -- -D warnings

# The shell: commands, views, the security audit, the bridge contract.
cargo test -p scirust-studio-desktop

# The build tooling.
cargo test -p prepare-sidecar -p stage-frontend

# Formatting, against the repository's rustfmt.toml.
cargo +nightly fmt --all -- --check
```

`.github/workflows/studio-desktop.yml` runs all of these, plus the native
shell on Ubuntu, Windows and macOS, plus the unsigned Windows preview.

## Why two workspaces

`apps/scirust-studio` is a separate Cargo workspace, listed under the root
workspace's `exclude`. The GUI toolchain therefore cannot affect the root MSRV
gate, the root dependency graph, or the time of a `cargo check --workspace`
that has nothing to do with the desktop. The scientific crates are consumed by
path, so there is exactly one copy of them.

The cost is a second `Cargo.lock` and a second CI workflow. Both are
committed and both are pinned with `=` versions, because a desktop
application's toolchain is part of the artifact a user installs.

## Troubleshooting

**`cargo tauri build` fails at `stage-frontend`.** There is neither a local
Dioxus build nor an already-staged bundle in `dist/`. Run
`dx build --release --platform web` first.

**The application opens but the window is blank.** `dist/` has an
`index.html` with no WebAssembly module, or the module failed to instantiate.
Check the WebView console; if the error mentions the content security policy,
the bundle contains an inline script or an external reference, neither of
which is allowed (`DESKTOP_SECURITY.md`).

**The application starts but Run is disabled and the status bar says the
engine is stopped.** The sidecar is missing or was staged under the wrong
name. Re-run `prepare-sidecar` and check the name matches
`scirust-studio-worker-<triple>[.exe]` exactly.

**`--smoke-test-backend` reports `ok: false`.** The JSON names the stage that
failed. This exercises the real worker, the real adapters and the real store,
so a failure here is a genuine problem with the scientific path on this
platform, not a packaging one.
