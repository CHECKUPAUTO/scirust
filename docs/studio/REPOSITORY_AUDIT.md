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

## 17. Update — two promises nothing kept

Not a phase; a correction. Both items were found by asking the question ADR
0007 asked about `experiment.seed` — *does anything actually read this?* — of
the fields around it.

**`backend.precision` was accepted and ignored.** The schema validated
`"f32"`, every descriptor declared `supported_precisions:
&[PrecisionKind::F64]`, and **nothing compared the two**. A scenario asking
for single precision passed validation and was computed in double, recording
a stated precision it did not have.

That is worse than a field nobody reads. An ignored `seed` produced a result
that was merely unreproducible; an ignored `precision` produces a result that
looks like the user got what they asked for. And `f32` is not a smaller
`f64` — someone selecting it is usually asking about conditioning or matching
another implementation's arithmetic, so answering in `f64` answers a different
question and says nothing about having done so.

Closed by `validate_support::resolve_precision`, called by every adapter, with
`SRST-VAL-0098`. `PrecisionKind` gained an `F32` variant that no capability
declares — a vocabulary entry, not a claim, so a descriptor can say "not here
yet" instead of the schema saying "yes" on its behalf.
`resolve_backend_kind`/`SRST-VAL-0099` closes the identical hole in
`backend.kind` before it can matter, which will be the first time a capability
is not CPU-only.

**`commands.rs` documented a feature that does not exist.** Its module comment
read *"File dialogs are native, run here, and hand back only the selected
file's contents"* — a description of the intended design, written as though it
had been built. There is no dialog command; `dialog:*` is not granted; the
only scenario sources are the compiled-in tutorials and what the user types.

`DESKTOP_SECURITY.md` and `DESKTOP_ARCHITECTURE.md` were both accurate about
this, which is the interesting part: the inaccuracy survived because it was in
the one place a reader would take most literally — the source. Corrected to
state the absence, and to record the constraint the picker must satisfy when
it is built, since that constraint is the reason the comment was written in
the first place: a `read_file(path)` reintroduced as "the file the user just
picked" is still `read_file(path)`.

**Two registry-driven tests** walk `all_adapters()` and assert each honours
the precision and backend it declares, following the descriptor rather than
today's answer — so a capability that grows an `f32` path is covered without
editing them.

**Still outstanding**, unchanged: the native file picker itself; `thermal`'s
`HeatRod1d`, which needs a presentation the interface does not have; a
result axis whose coordinates are bins; code signing and everything that
depends on it; and the remaining unadapted `scirust-sim` families.

## 18. Update — the native file picker

`studio_open_scenario` and `studio_save_scenario` show a real dialog. The
desktop can open a user's own scenario for the first time; until now the only
sources were the tutorials compiled into the binary and whatever was typed.

**The design constraint is the interesting part**, and it is the one §17's
corrected comment recorded before the feature existed: the dialog runs in the
shell and the frontend exchanges **contents**, never a path. It supplies no
destination and receives no location it could name again, so it cannot re-read
a file the user picked once, and cannot reach a file the user never picked.
`dialog:*` and `fs:*` stay ungranted for exactly that reason — they would let
the webview go around the two commands and hold the path itself. A command
taking a path would be `read_file(path)` with a friendlier name.

The audit test now forbids `dialog:` alongside `shell:`, `fs:` and the rest,
and that assertion was checked by granting the permission and watching the
test fail rather than by trusting it.

**Two things were removed rather than left inert.** `Unavailable::NotWired`
and `actions::is_wired` existed to say "this build has no command behind the
action" — the honest answer while the picker was missing, and dead the moment
it landed. An enum variant nothing can produce is a state the interface claims
exists and cannot reach, which is the same category of untruth as §17's two
items. The test that pinned the disabled behaviour was inverted, not deleted:
it is the only record that those actions were once refused for a reason other
than application state.

`Model::source_path` became `source_name` for the same reason. The frontend
holds a file *name*, for a title bar; leaving the field called `path` would
invite the first reader who needs a location to assume one is there.

**A note on how the wasm gate paid for itself.** `bridge.rs` is
`cfg(target_arch = "wasm32")`, so the host build never compiles it. The new
wire type's missing import was invisible to every host check and to
`cargo test`, and was caught by the wasm32 clippy step added two changes
earlier — the step that existed because a feature-gated file broke master the
same way.

**Still outstanding**: `thermal`'s `HeatRod1d` and the heat-map presentation
it needs; a result axis whose coordinates are bins; code signing and the
updater that depends on it; and the remaining unadapted `scirust-sim`
families.

## 19. Update — fields, and a claim this document repeated four times

`sim.thermal.heat_rod_1d` is adapted. **11 capabilities across 8 of 16 module
families.**

**The correction first.** §13 through §16 each carried the same note about
`thermal`'s `HeatRod1d`:

> Result schema v2 can already express it — `axes` is a vector and every
> series names its axis — but a line chart is the wrong presentation and the
> interface has no other.

The first half is wrong. `axes` being a vector means a result may have several
axes; it does not mean one quantity may span two. A `Series` is aligned
one-to-one with the axis it names, so the most it holds is a slice of a field:
one node's history, or one instant's profile. `u(x, t)` could be n series or m
series, and neither is the field.

That claim survived four updates without being checked, which is worth
recording rather than quietly fixing — it was believed because it sounded
right and nobody had tried it. `RunResult` now carries `fields`, and ADR 0009
records the design.

**Everything else this capability turned on:**

- Its checks are facts the model states in closed form: the steady state is
  *exactly* the linear profile, and the slowest mode decays at exactly
  `(2a/dx^2)(1 - cos(pi/(n+1)))`. The run measures its own decay and matches
  to ratio 1.000. The third check, the discrete maximum principle, exists
  because the first two can both pass on a wrong stencil that still relaxes to
  something plausible.
- The RK4 stability limit is a **validation error**, not a discovery. Above
  `0.7*dx^2/alpha` the run does not degrade, it produces NaN — so it is
  refused with the limit and a usable step in the message.
- Reduction to a drawable size keeps, per cell, the sample **furthest from the
  mean** rather than the cell's average. This is the two-dimensional form of
  the chart's existing "reduction never hides a peak"; a test plants one hot
  cell in a flat field sixteen times over budget and asserts it survives.
- `every_adapter_emits_exact_axis_coordinates` asserted every series is on the
  time axis. Two of this capability's series are profiles against *position*,
  so the assertion was generalised to the property that was always meant
  rather than special-cased.

**Still outstanding**: a result axis whose coordinates are bins (ADR 0008's
first-passage distributions — a histogram is not a field with one row); fields
over three axes or on unstructured meshes; code signing and the updater that
depends on it; and the remaining unadapted families — `pharmacokinetics`,
`rigid_body`, `battery`, `hvac`, `grid`, `laser`, `photodiode`, `apd`, `envs`,
plus SEIR, the two non-stiff chemistry models, the projectile, Van der Pol,
GBM and the M/M/1 queue.

## 20. Update — 16 of 16, and the first capability with no time in it

Eight more adapters. `scirust-sim` now has **no model module that Studio
cannot execute**: 16 of 16 module families, 19 capabilities. What remains
unadapted are model *families inside* adapted modules — SEIR, the two
non-stiff chemistry models, the projectile, Van der Pol, GBM, the M/M/1
queue, and pharmacokinetics' IV-bolus and two-compartment variants — plus
`envs`, which implements `Environment` rather than `System` and is a
different shape of thing entirely.

The number is the least interesting part. What the eight capabilities forced
is below.

### Tolerances stopped being chosen

`sim.optoelectronics.photodiode` was written with `LEVEL_TOLERANCE = 1e-3`
and it failed. The run was correct; the measurement was `6.738e-3`, which is
`exp(-5)` — the residual a run spanning five time constants necessarily
leaves behind. The fix was not a bigger number, it was the realisation that
the number was derivable all along:

```rust
let residual = (-span / tau).exp();
let threshold = residual * LEVEL_MARGIN;
```

Every capability since has followed that rule, and each one made it sharper:

- **`sim.thermal.hvac_zone`** — a 2R2C network's slow time constant is an
  exact root of a quadratic, 46.2 h, not the naive `R*C` sum's 57.8 h. Using
  the naive value would have been a 25 % over-estimate of how much of the
  transient the run had left, i.e. a silently loosened threshold.
- **`sim.energy.battery_thevenin`** — with two first-order states the
  threshold takes the residual of the *slower*, and a test pins that down.
  Holding the run to `R1*C1` = 20 s when `R_th*C_th` = 200 s governs would
  have failed a perfectly correct integration.
- **`sim.power.swing_equation`** — the period tolerance is the *computed*
  leading nonlinear correction, `a²·(1/16 + 5·tan²δ*/48)`, from Lindstedt.
  Measured lengthening `2.408e-4` against a predicted `2.431e-4`: 1 %
  agreement with a perturbation-theory coefficient. And the energy threshold
  is `h·ω_n`, which falls out of symplectic Euler's modified Hamiltonian with
  the swing amplitude cancelling entirely — measured `1.485e-3` against a
  derived `1.475e-3`.

That last one is worth stating plainly: the correction is used to **size the
tolerance**, not to shift the prediction. Correcting the prediction would
make the check depend on the coefficient being exactly right; sizing the
tolerance with it means the check survives the coefficient being wrong by a
factor of two and still fails a frequency error the nonlinearity cannot
explain. A separate test asserts the swing comes out *longer* than the
small-signal limit — a direction, which no tolerance can fake.

### A capability can be checked against its own Jacobian

`sim.optoelectronics.semiconductor_laser`'s second check does not compare
against a closed-form trajectory. It linearises the rate equations about the
clamped operating point, reads `ω_n²` and `−2γ` off the Jacobian's
determinant and trace, and compares the run's own measured ringing period
against `2π/√(ω_n² − γ²)`.

Numerical integration and linear stability analysis are different routes to
the same equations. A model with the right fixed point and the wrong
curvature there passes the operating-point check and fails this one, which is
the entire reason for having two.

The damping correction is not cosmetic: the damped period is 0.57 % longer
than the undamped `1/f_r` usually quoted, against a 0.1 % tolerance, so a
version of the check that compared against `1/f_r` would fail. A test asserts
the two periods stay far enough apart for the check to be able to tell them
apart — because if they ever converge, the check has silently stopped testing
the correction.

### A measurement was biased, and it said so by pointing the wrong way

The period measurement extracted from the laser adapter into `measure.rs`
averaged over crossings in **both** directions, treating each consecutive
pair as a half-period. That is true only for an oscillation symmetric about
the level it is measured against.

A finite-amplitude swing in an asymmetric potential acquires a second
harmonic and a DC offset, so its crossings alternate short-long — and an
*odd* number of them carries the full bias while an even number cancels it.
The generator rotor measured `4.4e-4` **faster** than its small-signal limit,
which is the wrong side of a result that only ever goes one way. That is what
caught it: not a failing tolerance but a sign.

Counting upward crossings only removes the failure mode rather than making it
rarer, since crossings in one direction are a full period apart whatever the
waveform looks like. A test builds an offset sine, shows the fix is exact,
and shows the old average is 6 % wrong over an odd count and right over an
even one — which is what made it intermittent instead of obvious.

### Not every capability integrates in time

`scirust_sim::apd` has no `impl System`. An avalanche photodiode's receiver
analysis is algebraic, so
`sim.optoelectronics.avalanche_photodiode` sweeps the *gain* and its result
carries a `gain` axis and no `t`.

That is declared rather than inferred: `CapabilityDescriptor::domain`, and
`RunSummary::axis_id` naming which axis the summary's bounds describe
(defaulting to `"t"`, so every stored result decodes unchanged). The
registry-driven test now demands a time axis of exactly the capabilities that
promised one and demands its absence from the one that did not — instead of
being weakened to "some axis" for everybody, which would have let a
time-integrating capability quietly stop emitting `t`. ADR 0010 records the
alternatives, including the cheap one that was rejected.

It is also the first capability whose checks have **no tolerance at all**,
which is not a coincidence — an algebraic model admits exact statements an
integrated one does not. Both of its checks are theorems: the SNR at the
analytic optimum must beat the SNR at every swept gain, and the curve must
turn exactly once. The optimum comes from bisecting a stationary condition
whose left side is provably monotone, so there is no derivative, no initial
guess, and no chance of converging to the wrong root.

A sweep that does not bracket the optimum reports both checks as
`NotApplicable` and warns. That distinction is why the check locates the
optimum analytically instead of taking the sweep's `argmax`: an `argmax`
always returns something, and an `argmax` at an endpoint looks exactly like
an `argmax` at a peak.

### Abstaining became a normal thing to do

Three of the eight capabilities have configurations in which their oracle
does not apply, and all three say so rather than failing, passing vacuously,
or refusing to run:

- the laser with `β > 0` or below threshold — the closed forms are the
  `β → 0` above-threshold limit, and the model does not state the `β > 0`
  operating point. Re-deriving it in the adapter would put a second copy of
  the physics there and leave the verification checking the adapter against
  itself.
- the swing equation with damping (the transient energy is genuinely
  dissipated) or with `P_m > P_max` (loss of synchronism — no equilibrium
  exists to swing about, which is the failure mode the model is *for*).
- the APD sweep that misses its optimum.

In each case the run is valid and the *oracle* is what does not apply. Every
one carries a `RunWarning` naming the reason, and every one has a test that
also asserts the run still did the right thing physically — the damped rotor
really settles, the runaway rotor really runs away.

### Two smaller things

The unit table gained `pu` (per-unit), dimensionless with a factor of exactly
one, for the same reason it carries `rad`: a per-unit power written
`unit = "1"` reads as a mistake. `PrecisionKind::F32` remains a vocabulary
entry no capability declares.

Error codes now run to `SRST-VAL-0285`, and the ten-number-per-capability
blocks stopped being ten wide at `0250`. A capability with twelve fields
needs twelve codes; stretching one to fit a round number would mean two
fields sharing a code, which is worse than an untidy table.

**Still outstanding at the time of writing** — every item here is revisited
in §21, which closes or reduces all of them: a result axis whose coordinates
are bins (ADR 0008's first-passage distributions — a histogram is not a field
with one row); fields over three axes or on unstructured meshes; code signing
and the updater that depends on it; and the model families listed at the top
of this section, none of which is now the last of its module.

## 21. Update — the three outstanding lists, closed

§20 ended with three groups of outstanding items: unadapted model families,
limits of the result model, and desktop code signing. Each was reopened with
the intent of finishing it. Two were finished; the third was finished as far
as it can be without a certificate, and the remainder is named rather than
implied. Two items inside the first two groups were **closed by decision**,
which is a different thing from being deferred and is recorded as such in ADR
0012.

### Model families: 28 capabilities, and one decided non-goal

Nine more adapters: SEIR, consecutive reactions, the reversible reaction, the
RC circuit, the projectile, Van der Pol, the two-compartment IV bolus, the
M/M/1 queue and geometric Brownian motion. Every `scirust-sim` model family
that implements `System` now has one.

`envs` does not, and will not. It implements `Environment` — `reset` /
`step(action)` — so it needs an agent the scenario schema cannot express. The
workable encoding was `solver.id` as the policy name, which makes `solver`
mean "numerical method" for twenty-six capabilities and "controller" for two.
That cost might have been worth paying; what settled it was reading
`CartPole`: bang-bang ±10 N with no null action, explicit Euler, no invariant
and no closed form. Nothing about its *physics* is checkable. What remains —
that `done` fires exactly at the bounds, that the return equals the episode
length — tests the harness, and a capability whose checks verify the
bookkeeping would be the first in this catalogue that does not carry an oracle
its own model states.

### The result model: distributions, and a corrected assumption

`RunResult` gains `distributions`. A histogram's `n` bins need `n + 1` edges,
so it fits neither a `Series` nor a `Field`, both of which are aligned
one-to-one with an axis. Storing one as a single-row field was the tempting
wrong answer and ADR 0011 says why: a field's rows are point-sampled, so a
reader would have to guess whether the coordinates were centres or edges, and
the type has nowhere to put the answer.

The other item, "fields over three axes", was drafted as speculative — no
model produces one — and that draft was **wrong**. Grepping for a producer
found `scirust-itd::field3::Field3`, a dense 3-D scalar field with an
oracle-validated test suite. The assumption is recorded in ADR 0012 rather
than quietly corrected, because a right conclusion reached from a wrong
premise is still a thing to notice.

The conclusion held for a different reason. `scirust-itd` uses `Field3` for
6-connected component labelling and region topology — an *analysis input* —
while its simulation driver returns per-interval time series and 2-D fields,
which the schema already expresses exactly. So the schema stays two-axis until
a capability's **output** varies over three.

Adapting `scirust-itd` itself is real open work, and the first capability that
would come from outside `scirust-sim`.

### Desktop: a pipeline that refuses to be half-signed

The signing material is a purchase, a private key and a URL somebody has to
own. None of it can live here, and no amount of code changes that. What could
be built was everything else, and it was.

`release-config` decides the signing posture from the environment and
**refuses to build** when that decision is partial. Each partial state
produces a build that completes, an artifact that runs, and a security
property that silently is not there: a public key with no private key checks
for updates it can never verify; a private key with no public key ships signed
artifacts nothing verifies; a certificate with no timestamp URL produces a
signature that stops working the day it expires. Fourteen tests, none of which
need a certificate.

The updater *plugin* is deliberately not registered. It would put outbound
network into a webview whose entire posture is that it holds no general
network, filesystem or process capability, and it would be a grant that could
not be exercised end-to-end here — there is no key to sign with and no
endpoint to talk to. An untested capability grant is the kind this project
refuses everywhere else. `docs/studio/DESKTOP_SIGNING.md` names every variable
a maintainer must supply and the order the work has to happen in.

### One defect found by a test written this pass

Replacing the `fixed_step` proxy for "reports progress" with a declared
`reports_progress`, and adding a registry-driven test that runs every
capability's tutorial and compares declaration against emission, immediately
found `sim.electrical.van_der_pol` reporting progress that reached one hundred
percent and started again — it integrates two trajectories and each reported
its own span. `double_pendulum` had the same shape and had worked around it
with a `NullEventSink`, which trades a resetting bar for one that finishes
halfway through the work.

`sink::SubRangeSink` fixes the class. The proxy it replaced had already caused
one live defect in the other direction (the Ornstein-Uhlenbeck determinate bar
that never moved), so this is the second time inferring that property from an
adjacent one has been wrong — which is the argument for declaring it.

**Still outstanding**, and now genuinely so: a first-passage-time capability
(ADR 0008 named the distribution; the representation now exists, the
capability does not); adapting `scirust-itd`; three-axis or unstructured
fields, gated on a capability that produces one; and the signing credentials
themselves.
