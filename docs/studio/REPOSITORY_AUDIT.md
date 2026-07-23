# SciRust Studio — Phase 0 Repository Audit

**Generated:** 2026-07-23
**Branch:** `claude/scirust-studio-windows-euprx0`
**Commit audited:** `9e94060671953dbbcfcee4a8c946c094e3b0bd3a`
**Method:** direct inspection of source files, `Cargo.toml`, CI workflow definitions,
and license files in this checkout. Claims below are scoped to what was actually
read; crates not opened this pass are marked as such rather than inferred from
their names.

This document is intentionally honest about scale: the SciRust Studio brief
describes a full commercial Windows desktop product (Tauri/Dioxus shell, worker
process, IPC protocol, installer, code signing, updater, bilingual accessible
help system, threat model, fuzzing, benchmarks, release pipeline). That is a
multi-person-year scope under any reasonable estimate. This audit exists so that
scope is chosen deliberately against real facts, not against the assumptions in
the brief.

## 1. Toolchain and MSRV

- `Cargo.toml` declares `rust-version = "1.89"` (the floor checked by CI's `msrv`
  job via `cargo +1.89.0 check --workspace --all-targets --locked`).
- `rust-toolchain.toml` pins the **development** toolchain to
  `nightly-2026-07-02` (components: `rustfmt`, `clippy`, `llvm-tools-preview`).
  Nightly is required for optional, off-by-default features elsewhere in the
  workspace (e.g. `portable-simd`), not for the core libraries.
- Installed in this container: `rustc 1.98.0-nightly (4c9d2bfe4 2026-07-01)`,
  `cargo 1.98.0-nightly`.
- **Implication for Studio:** new Studio crates should target stable 1.89+ and
  must not silently acquire a nightly-only dependency. The desktop workspace (if
  split out per §7 of the brief) needs its own toolchain story since Tauri/Dioxus
  tooling has its own MSRV expectations independent of this repo's nightly pin.

## 2. Workspace shape

- Root `Cargo.toml` `[workspace] members` lists **134 path members** (crates +
  a handful of `examples/*` and the `scirust-som/crates/*` sub-workspace-in-place).
- `[workspace] exclude` currently lists 6 entries kept out of the default build:
  `examples/simd_views_demo`, `examples/benchmarks`, `fuzz` (its own nightly
  libfuzzer workspace), `scirust-burn-bridge` (needs the heavy external `burn`
  crate), `scirust-hypermemory` (mandates a nightly-only `portable_simd`
  feature), and `sos` (the "Scientific Operating System" — its own Cargo
  workspace entirely, documented under `docs/sos/`).
- This is **not** a small or emerging library. It is a large, actively
  maintained monorepo already covering deep learning, symbolic math, ODE/stiff
  solvers, tensor networks, radar/optronics signal processing, a dozen
  regulated-industry verticals (grid, biomed, maritime, fab, agtech, fatigue,
  tolerancing, functional safety, SIS), relativity/fractional-calculus research,
  a licensing/entitlement crate, a provenance/anti-leak signing crate, an MCP
  server, and more. `README.md` alone is ~55 KB; `CHANGELOG.md` is ~360 KB.

## 3. Licensing — read before treating this as a generic "commercial app" task

- `LICENSE.md`: **PolyForm Noncommercial License 1.0.0**. `LICENSING.md` states
  plainly: *"Commercial use is not granted by the PolyForm Noncommercial
  terms. A separate commercial agreement may be obtained from the copyright
  holder: Tarek Zekriti, zekrititarek@gmail.com."*
- The copyright holder's email matches this session's user
  (`zekrititarek@gmail.com`), so the person directing this work appears to be
  the licensor themselves — they have standing to authorize commercial use of
  their own copyright. This is **not** treated as a blocker, but it is recorded
  here because "build a commercial desktop application" is a materially
  different instruction from "build a desktop application" given the
  repository's default license, and because any release artifact, EULA, or
  installer text SciRust Studio ships should say what it actually is (a
  commercial product built by the copyright holder) rather than silently
  implying the underlying PolyForm-Noncommercial code is being relicensed to
  end users.
- `LICENSING.md` also documents an **existing entitlement/licensing mechanism**
  (`scirust-license`, referenced from `scirust-provenance` via `hashsig`) that
  gates "high-value capabilities (e.g. the GPU acceleration module)" behind a
  signed, offline-verifiable license file, node-locked optionally, no
  phone-home. **Studio must respect this existing gate** rather than route
  around it — e.g. a capability card for a GPU-gated feature must reflect
  "requires a license" rather than silently degrading or silently unlocking it.

## 4. Platform / Windows reality check

- Grepping every workflow in `.github/workflows/` for `windows` returns exactly
  one hit: the `platform-check` job in `ci.yml` runs
  `cargo +stable check --workspace --all-targets --locked` on a
  `windows-latest` **and** `macos-latest` matrix. That is a compile check only
  — no GUI build, no installer, no packaging, no Windows-specific test
  execution exists anywhere in this repository today.
- There is **no Tauri, no Dioxus, no WebView2 integration, and no `apps/`
  directory** anywhere in the tree (`grep -ri "tauri|dioxus"` across the repo:
  zero matches).
- **This session runs in a Linux container.** I can write, and `cargo check`,
  every cross-platform Rust crate here. I cannot build a Windows `.exe`/MSI/NSIS
  installer, cannot launch or screenshot a WebView2 window, and cannot run the
  PowerShell installer-smoke-test scripts the brief specifies. Those steps are
  only verifiable on a real Windows host or via a `windows-latest` GitHub
  Actions runner — which I can configure but not execute interactively from
  here. Any completion report must not claim a Windows build was "tested"
  unless CI (or a real Windows machine) actually ran it.

## 5. `scirust-cli` — actual command surface (read from source, not inferred)

`scirust-cli/src/lib.rs` is a **hand-rolled, string-matched dispatcher** (no
`clap`, no structured `ArgumentDescriptor`, no generated help — `dispatch()` is
a single `match` on `args.first()`). It is a thin wrapper: "adds no new compute,
only a command surface." Actual dispatched commands (verified against the
`match` arms and the `dispatch_reaches_each_group` test, 55 total):

- **Learning/optimization:** `quickstart`, `som train`, `evo`, `cmaes`
- **Symbolic math:** `diff`, `simplify`, `eval`, `solve`, `prove`, `gradient`,
  `to-rust`, `regress`, `symreg`, `trig`, `patterns`
- **Logic:** `sat`
- **Numerical solvers:** `pinn`, `integrate`, `root`, `minimize`, `optimize`,
  `linsolve`, `lstsq`, `det`, `cholesky`, `qr`, `cg`, `inverse`,
  `solve-system`, `polyroots`, `ode`, `fem-heat`
- **Tensor networks:** `tt`, `quantum`
- **NLP/sequence models:** `bpe`, `lm`, `deltanet`, `mamba`, `retnet`, `gla`,
  `hgrn`, `rwkv`
- **Code analysis:** `analyze` (delegates to `scirust_som_cli::run`)
- **SciAgent SLM:** `sciagent ask|chat|explain|generate|info|attest|quantize`
- **Inference integrity:** `verify` (delegates to `scirust_runtime::proofcli`),
  `certify`, `conformal`, `calibrate`, `guard`, `attest`
- **Compression:** `gptq`, `awq`, `bitnet`, `kvcache`
- **Meta:** `help`, `version`, `info`
- **Trading:** `trader run|predict|audit|info`

The help text additionally **advertises** 8 "pattern detection" crates
(`scirust-vision`, `scirust-audio`, `scirust-graph`, `scirust-sequential`,
`scirust-multivariate`, `scirust-unsupervised`, `scirust-seasonal`,
`scirust-nlp-advanced`) and 6 "algorithm creation" crates (`scirust-automl`,
`scirust-synthesis`, `scirust-algogen`, `scirust-codetrans`, `scirust-rl-algo`,
`scirust-scaffold`) as informational lines with **no `args` and no dispatch
arm** — they are mentioned, not runnable, from `scirust-cli` today.

Other CLI-shaped entry points exist **outside** `scirust-cli` and are not
unified with it: `scirust-provenance/src/bin/prov.rs` (a separate `prov`
binary for artifact signing), and (per `README.md`, not independently verified
this pass) a dedicated `scirust-industrial` CLI and an MCP server in
`scirust-mcp` exposing many of the vertical-specific tools (including
`scirust-sim` scenarios) as MCP tools rather than CLI commands.

**Confirmed gap:** `scirust-sim` has **zero** presence in `scirust-cli` — no
`run`, `sim`, or `catalog` command exists. Its only current exposure (per
`README.md`) is through `scirust-mcp` tools (`sim_epidemic`,
`sim_battery_discharge`, `sim_grid_stability`, `sim_hvac_zone`,
`sim_pharmacokinetics_oral`, `sim_stiff_robertson`). This is the single
clearest, most real, best-scoped gap the Studio brief's "CLI ↔ sim" language
refers to.

## 6. `scirust-sim` — actual model surface (read from `src/lib.rs`)

`#![forbid(unsafe_code)]`, `#![deny(missing_docs)]`, zero dependencies. Public
modules, each oracle-tested per its module doc comment:

`apd`, `battery`, `chemistry`, `ecology`, `electrical`, `epidemiology`, `grid`,
`hvac`, `laser`, `mechanics`, `orbital`, `pharmacokinetics`, `photodiode`,
`rigid_body`, `thermal` — plus the engine itself (`engine`: `System` /
`SecondOrderSystem` traits, `simulate`/`simulate_adaptive`/
`simulate_second_order`), the interaction layer (`env`, `envs`: `CartPole`,
`GridWorld`), and the seeded RNG (`rng::SplitMix64`).

Two integrations are feature-gated rather than always-on:
`stiff_bridge` (feature `stiff`, bridges to `scirust-stiff`'s Backward
Euler/Rosenbrock-W for the stiff Robertson kinetics) and `rl_bridge` (feature
`rl`, adapts `Environment` to `scirust_learning::rl::Env`).

This crate is a strong, clean candidate for direct integration exactly as the
brief hopes — it's dependency-free, deterministic-by-construction (explicit
seeds, no ambient randomness), and every model already documents its own
oracle. Building typed Studio adapters over it is real, valuable, and
tractable; it does not require touching the numerics.

## 7. Other Studio-relevant crates actually opened this pass

- **`scirust-units`**: `Dimension` (7 SI base-dimension integer exponents) +
  `Quantity` (f64 magnitude tagged with a `Dimension`), checked
  (`Result`-returning) arithmetic that rejects mixed-dimension operations
  instead of panicking. Directly usable for Studio's unit/dimension
  validation (§13 of the brief) — no adapter needed, just a dependency edge.
- **`scirust-provenance`**: **not** what the brief assumes. Its actual purpose
  (per its own doc comment) is offline Lamport/Merkle signing of
  transpiler-emitted source artifacts for **leak attribution** — "a
  provenance / leak-attribution tool, not an anti-clone shield" — with an
  explicit warning that it does not protect against reimplementation. It has
  no notion of a run manifest, a determinism class, or reproducibility
  metadata for a simulation run. **Studio's run-manifest/provenance model
  (brief §17) must be built new**; it should not attempt to repurpose this
  crate, though it may reuse the SHA-256/hash-chaining *pattern* used here and
  in the predictive-maintenance and OT-integrity crates.
- **`scirust-license`**: entitlement/license-file gating for high-value
  features (see §3 above). Relevant as a boundary Studio must not bypass.

Crates referenced by `scirust-sim`'s feature-gated bridges
(`scirust-stiff`, `scirust-learning::rl`) were not independently opened this
pass — they are catalogued, not yet API-audited. The remaining ~120 workspace
members (industrial verticals, radar/optronics, relativity research, the SOM
sub-workspace, tensor-network stack, GPU/CUDA backends, `scirust-mcp`,
`scirust-industrial`, etc.) were **not** opened this pass beyond what
`README.md` advertises. Per the brief's own rule ("do not infer a crate's
capability solely from its name"), none of them should be marked "operational"
in the capability matrix until someone actually reads their public API and
tests — most should start out, and likely remain for a long time, in
"catalogued, no tested Studio adapter."

## 8. Existing CI gates (`.github/workflows/ci.yml`, 23 jobs)

`fmt`, `clippy`, `epsilon-audit`, `cobol-corpus`, `finmigrate-compiler`,
`build-test`, `portable-simd`, `hypermemory`, `transformer-inference`,
`nightly-simd`, `build-test-stable`, `msrv`, `platform-check` (Windows/macOS
compile-check only, see §4), `opt-in-features`, `cross-check-aarch64`, `deny`
(cargo-deny), `miri`, `fuzz`, `determinism`, `gpu-wgpu`, `gpu-cuda-fallback`,
`sbom`, `coverage`. Separate workflows exist for `native-arm64.yml`,
`release.yml`, and `sos-ci.yml`.

**Implication:** `cargo-deny`, fuzzing infrastructure, an SBOM job, and a
determinism-check job **already exist** at the root-workspace level. Studio's
CI (brief §36) should extend/reuse these conventions (same pinned-SHA action
style, same `deny.toml` license allowlist) rather than re-inventing a parallel
supply-chain security setup. `deny.toml` already documents the two accepted
RUSTSEC advisories and the permissive-license allowlist for third-party deps
(the workspace's own crates are `publish = false` and PolyForm-licensed, so
license scanning only applies to dependencies).

`cargo check --workspace --all-targets --locked` was run in this session
(nightly-2026-07-02, the pinned dev toolchain) and **passed**: `Finished
\`dev\` profile [unoptimized + debuginfo] target(s) in 1m 23s`, exit code 0,
across all 134 workspace members. That is the one gate actually executed and
confirmed green this pass. `cargo test --workspace`, `cargo fmt --check`,
`cargo clippy -D warnings`, `cargo audit`, and `cargo deny check` were **not**
run in this pass (a full test run across 134 members — including deep-learning
training loops, fuzz-adjacent code, and Miri-gated tests — is a substantially
longer operation); they should be run and their real output captured before
anyone claims the full baseline is green, per the brief's own anti-fabrication
rule.

## 9. Summary: what the Studio brief assumes vs. what is actually here

| Brief assumption | Reality found |
|---|---|
| Unified `scirust-cli` fronting most crates | `scirust-cli` fronts ~55 commands across learning/symbolic/numeric/NLP/trading; most of the other ~120 crates (industrial verticals, radar, GPU, tensor networks, `scirust-mcp`, `scirust-industrial`) are separate binaries/MCP tools/libraries, not part of it |
| `scirust-sim` reachable from the CLI | Not reachable at all from `scirust-cli`; only via `scirust-mcp` tools |
| Reusable "provenance" facility for run manifests | `scirust-provenance` exists but solves a different problem (leak attribution signing); a run-manifest/determinism model is net new work |
| Rust MSRV ~1.89 stable dev loop | MSRV floor is 1.89 (CI-checked), but the pinned dev toolchain is nightly (needed elsewhere, not by `scirust-sim`/`scirust-cli`) |
| A repo roughly scoped to "a scientific computing library" | A 134-member monorepo already spanning deep learning, symbolic math, 15+ regulated-industry verticals, radar/optronics DSP, relativity research, an MCP server, a licensing/entitlement system, and a leak-attribution provenance system |
| Generic "commercial desktop application" | Base repository is PolyForm-Noncommercial; the person directing this work is the copyright holder, so this is self-authorized, but installer/EULA text must say so accurately |
| Windows build/installer "built and tested" | No Windows GUI, installer, or signing infrastructure exists yet; this session (Linux container) can author it but cannot itself build or test a Windows binary — only CI on a `windows-latest` runner, or a real Windows machine, can |

## 10. First-pass crate integration classification

**Directly integrable now (real API inspected, dependency-light, deterministic-by-construction):**
`scirust-sim` (16 domain modules), `scirust-units`.

**Need a real adapter, not yet built:** ODE/stiff integration surfaced through
`scirust-sim`'s bridges; a new run-manifest/provenance/determinism-classification
layer (brief §17) — cannot reuse `scirust-provenance` as-is; a command-registry
layer that both `scirust-cli`'s existing dispatcher and any future desktop
shell can share without duplicating the ~55 existing command implementations.

**Should stay catalogue-only for an initial release, pending explicit review:**
the remaining ~120 workspace members, in particular anything touching OT/ICS
device discovery (`scirust-mcp`, `scirust-discovery`), live trading
(`scirust-trader` with its `live` feature), GPU/CUDA backends (already
license-gated for a reason), and the relativity/TDI research crates whose own
docs describe them as experimental and non-empirically-validated. None of
these have been API-audited this pass, and several are explicitly
security- or safety-sensitive.

## 11. Recommendation

Proceed with Phase 1 of the brief (shared scenario schema + typed command
registry, built as new cross-platform library crates, exercised through
`scirust-cli` and automated tests in this Linux container) as the first real,
testable, non-wasted increment — it is required by every later phase
regardless of which GUI framework or installer technology is ultimately used,
and it is the one piece of the brief's Phase 0–4 arc that is fully verifiable
here. Desktop shell (Tauri/Dioxus), Windows packaging, code signing, and the
bilingual accessibility-audited help system are real, large, separable
workstreams that should be sequenced explicitly with the person directing this
project rather than assumed, given (a) their size relative to a single session
and (b) this session's inability to build or test Windows GUI artifacts
directly.
