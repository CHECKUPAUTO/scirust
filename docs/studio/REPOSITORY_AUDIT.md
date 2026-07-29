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

## 12. Update — Phase 2A (capability registry, adapter runtime, four more models)

Phase 1's single hard-coded capability path has been replaced with a real
capability registry (`scirust-studio-registry`) and adapter runtime
(`scirust-studio-runtime`), and four more `scirust-sim` models —
`sim.epidemiology.sir`, `sim.orbital.two_body`, `sim.electrical.rlc`, and
the stiff `sim.chemistry.robertson` — were adapted alongside the original
`sim.mechanics.spring_mass_damper`, chosen specifically to force the
architecture to support vector-valued state, multiple solvers per
capability, and a genuinely different (adaptive, linearly-implicit) solver
family. `scirust-cli` no longer depends on `scirust-sim` at all — every
capability is reached through `CapabilityAdapter`. Full detail:
`docs/studio/adr/0001-capability-registry.md`,
`docs/studio/adr/0002-structured-run-results.md`,
`docs/studio/RUNTIME_CONTRACT.md`, and `docs/studio/CAPABILITY_MATRIX.md`
(which supersedes the crate-integration classification in §10 above for
`scirust-sim` specifically — §10 remains accurate for the other ~120
workspace members, which Phase 2A did not touch).

The remaining 11 `scirust-sim` model families (pendulum/projectile/double
pendulum, SEIR, ecology, the two non-stiff chemistry models, thermal, RC/Van
der Pol, stochastic, pharmacokinetics, rigid-body, battery, HVAC, grid,
laser, photodiode, APD) are unchanged: real and tested in `scirust-sim`'s
own suite, not yet wired to a Studio scenario. Adapting them is Phase 3.

## 13. Update — Phase 3A (desktop shell, WebAssembly interface, typed bridge)

§4's Windows reality check said this repository contained "no Tauri, no
Dioxus, no WebView2 integration, and no `apps/` directory". That is no longer
true, and the parts of §4 that remain true are worth restating precisely,
because they bound what this phase may claim.

**What now exists.** `apps/scirust-studio` is a second Cargo workspace
(excluded from the root one) holding four crates:

| Crate | What it is |
|---|---|
| `scirust-studio-desktop` (`src-tauri`) | The Tauri 2.11.5 native shell: window, sidecar supervision, 17 typed commands, no scientific code |
| `scirust-studio-ui` | The Dioxus 0.7.9 **Web** interface, compiled to `wasm32-unknown-unknown` |
| `prepare-sidecar` | Stages the locally-built worker under the target-triple name Tauri resolves |
| `stage-frontend` | Stages the WebAssembly bundle into the directory Tauri bundles, refusing anything that is not one |

Alongside them, `scirust-studio-app-service` was added to the **root**
workspace: the application-facing layer (bootstrap, worker supervision, job
lifecycle, bounded events) that both the shell and any future host calls. It
is the reason the shell contains no scientific logic — every command
delegates to it.

Documentation: `docs/studio/adr/0005-tauri-dioxus-desktop-architecture.md`,
`DESKTOP_ARCHITECTURE.md`, `DESKTOP_SECURITY.md`, `DESKTOP_BUILD.md`,
`FRONTEND_BRIDGE.md`, `WINDOWS_DESKTOP_ACCEPTANCE.md`.

**What §4's constraint still means.** This session still runs in a Linux
container. The following were verified here, on Linux, with commands that
exited zero:

- the interface's logic (reducers, chart geometry, action registry, string
  table, wire decoding) — host tests;
- every Dioxus component — `cargo clippy --target wasm32-unknown-unknown
  --all-targets -D warnings`;
- the Tauri shell, its command surface and its security audit — host tests;
- the bridge contract between the shell's view types and the interface's wire
  types — host tests over real values;
- the scientific path end to end through the real worker, adapters and store
  — `--smoke-test-backend`.

The following were **not** run here and must not be reported as tested until
CI or a real Windows host runs them:

- `dx build --platform web` (the Dioxus CLI is not installed in this
  container, so no WebAssembly bundle was produced);
- `cargo tauri build` and the NSIS installer;
- launching a WebView window on any platform;
- `scripts/studio/test-desktop-artifact.ps1`.

`.github/workflows/studio-desktop.yml` runs all four, on
`ubuntu-latest`/`windows-latest`/`macos-latest`, and publishes the unsigned
Windows preview. Its results — not this document — are the evidence that the
packaged application works.

**What §4's licensing and platform notes still govern.** Nothing in this phase
changed the licensing position (§3) or added a Windows-specific dependency
beyond WebView2, which ships with Windows 11 and recent Windows 10. Code
signing, an updater, licensing enforcement and installer publication remain
out of scope and unimplemented.

**Capability coverage did not change.** Still 5 of 16 `scirust-sim` model
families, exactly as in §12. The desktop reads the same registry, so its
coverage is the adapter coverage; see `CAPABILITY_MATRIX.md` for the per-
capability desktop columns. Adapting the remaining eleven is Phase 3B and
needs no desktop work.

## 14. Update — Phase 3B-1 (four more adapters)

§13 closed by saying capability coverage was unchanged at 5 of 16 module
families and that adapting more "needs no desktop work, because the catalogue,
the run view and the chart are all driven from the registry". This phase tests
that claim by doing it.

Four capabilities were added — `sim.ecology.lotka_volterra`,
`sim.ecology.logistic_growth`, `sim.mechanics.pendulum` and
`sim.mechanics.double_pendulum` — bringing the catalogue to **9 capabilities
across 6 of 16 module families**. Nothing in `apps/scirust-studio` changed:
not the shell, not the interface, not the bridge, not the chart. The claim
held.

**What each was chosen to exercise**, beyond raising a count:

- Logistic growth is the first **one-component state** in the catalogue, and
  the first capability verified against a **closed-form solution** at every
  recorded point rather than against a conservation law.
- Lotka-Volterra has an **exact first integral** — a sharper oracle than
  periodicity, because a trajectory can look periodic and still have drifted
  off its orbit.
- The pendulum solves the nonlinear equation and reports the period measured
  from its own trajectory beside the small-angle formula, which at the
  tutorial's 90-degree release is wrong by 18%.
- The double pendulum is **deterministic but not predictable**, and measures
  that: it integrates a perturbed twin and reports the separation, with a
  warning naming the time after which the angles stop being predictions.

**Two supporting changes**, both deliberately minimal. The unit table gained
`rad` and `rad/s`; both are SI-coherent with a conversion factor of exactly 1,
and a test now asserts that *no* symbol in the table carries a hidden
conversion — the moment one needs a real factor (mg, litre, hour) that is a
separate change with its own tests. The registry gained an `Ecology` category.

**A measurement bug this phase found and fixed.** The double pendulum's energy
check was first written to report drift relative to the *initial* energy, as
the other mechanics adapters do. That is wrong for this model: potential
energy is measured from the pivot, so a pendulum released from horizontal
starts at exactly zero energy, and an absolute drift of two picojoules was
reported as a relative drift of `1.8e3`. The check now normalises by the
system's gravitational energy scale, `(m1+m2)*g*l1 + m2*g*l2`, which is
strictly positive for any valid model. A test pins the reason.

**Still catalogue-only**: `thermal`, `stochastic`, `pharmacokinetics`,
`rigid_body`, `battery`, `hvac`, `grid`, `laser`, `photodiode`, `apd`, `envs`,
plus SEIR, the two non-stiff chemistry models, the projectile and Van der Pol.
Several of the remaining ones need work beyond an adapter: `thermal`'s
`HeatRod1d` is a spatially discretised field whose natural presentation is not
a line chart, and `stochastic` is the first model whose determinism class
would not be `StrictSameBinarySameTarget`. Both are worth doing deliberately
rather than folding into a tranche of lumped-parameter models.

## 15. Update — Phase 3B-2 (the seed becomes real)

§14 listed `stochastic` among the families still needing work "beyond an
adapter", on the grounds that it would be the first model whose determinism
class was not `StrictSameBinarySameTarget`. That turned out to understate it:
adapting it surfaced a promise the product had been making and not keeping.

`experiment.seed` has been in the scenario schema since Phase 1 and is set by
every shipped tutorial. Before this phase, a search for `.seed` across every
Studio crate returned exactly one hit — a schema unit test asserting the field
parses. Nothing read it. Meanwhile `DeterminismClass` had carried a variant
named `InherentlyStochasticRecordedSeed`, describing a recording that did not
happen.

Both are now true statements. `RunProvenance` records the seed the computation
consumed (`None` when it consumed none), a stochastic capability is refused
without one, and the CLI and desktop display it. The reasoning, including the
three ways of picking a seed silently that were rejected, is in ADR 0007.

`sim.stochastic.ornstein_uhlenbeck` is the capability that forced it: 10
capabilities across 7 of 16 module families.

**Still catalogue-only**: `thermal`, `pharmacokinetics`, `rigid_body`,
`battery`, `hvac`, `grid`, `laser`, `photodiode`, `apd`, `envs`, plus SEIR,
the two non-stiff chemistry models, the projectile, Van der Pol, GBM and the
M/M/1 queue.

Of those, two still need design work rather than another adapter, and the list
has changed since §14:

- `thermal`'s `HeatRod1d` is a spatially discretised field. Result schema v2
  can already express it — `axes` is a vector and every series names its axis
  — but a line chart is the wrong presentation and the interface has no other.
- `stochastic`'s remaining models raise the **ensemble** question. A seeded
  single path answers "what does one realisation look like". Questions like
  "what is the distribution of the first passage time" need many paths and a
  result model that can hold them, which is a larger change than a seed field.
  ADR 0007 records this as explicitly not attempted. *(Addressed in §16.)*

## 16. Update — Phase 3B-3 (ensembles)

Closes the gap §15 ended on, and the one ADR 0007 named as not attempted.

`experiment.replicates` asks a stochastic capability for many independent
realisations, seeded by derivation from the scenario's single seed. The result
carries the across-replicate mean, a two-sigma spread band and a bounded
number of individual paths; `Series.role` distinguishes them, so no consumer
has to infer from an id which curve is a summary over 256 realisations and
which is one of them.

The catalogue is unchanged at **10 capabilities across 7 of 16 module
families**. This phase added no adapters — it changed what an existing one can
be asked.

**What the phase actually turned on.** Three things worth recording because
none of them was the obvious version:

- **The seed derivation is scrambled, and the reason is specific to this
  codebase's generator.** SplitMix64 takes its seed *as* its state and
  advances by a fixed increment, so every seed sits on one shared 2^64 cycle.
  Two replicates whose seeds differ by a multiple of that increment draw the
  same noise offset by a few steps — independent-looking realisations sharing
  their randomness. Counting seeds happens not to trigger it; scrambling makes
  the safety a stated property rather than an accident. A test inverts the
  increment modulo 2^64 to recover the separation of every pair.
- **The across-replicate check is materially stronger than the single-path
  one**, and the run now prints both side by side: the shipped tutorial's
  39 001 correlated samples are worth about 98 independent ones, while 256
  realisations are worth 256.
- **A pre-existing bug surfaced.** The Ornstein-Uhlenbeck capability declared
  `fixed_step: true`, from which the app service derives
  `supports_progress: true`, while the adapter emitted no progress — so the
  desktop drew a determinate bar that never moved. The test that should have
  caught it checked a proxy (`!any(fixed_step && id == "rk4")`) that passed
  trivially. Fixed by taking the *realisation* as the unit of progress, which
  is honest for both an ensemble and a single run.

**Still catalogue-only**: unchanged from §15, minus the ensemble question.
`thermal`'s `HeatRod1d` remains the one entry that needs interface design
rather than another adapter — a spatially discretised field is expressible in
schema v2 but a line chart is the wrong presentation for it.

**What ensembles now make worth wanting**: a result whose axis is *bins*
rather than time. The ensemble here summarises realisations pointwise in time,
which answers "where is the process at each moment". A histogram of first
passage times is a distribution over a scalar, and needs an axis kind the
schema does not have. ADR 0008 records it as the next thing, and as not
attempted.
