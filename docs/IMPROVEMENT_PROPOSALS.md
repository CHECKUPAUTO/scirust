# SciRust — Improvement Proposals

Objective: evolve SciRust from a large experimental research workspace into a
coherent, discoverable, industrial-grade scientific-computing framework without
discarding the deterministic, pure-Rust architecture that already exists.

Audit snapshot for this document: 154 Cargo workspace packages, 2,893 `.rs`
files outside `external/`, 61 unique CLI commands, 137 workspace packages with
publishing explicitly disabled, and an existing multi-gate acceptance protocol.
These numbers should be regenerated rather than copied forward when the
workspace changes.

---

## 1. Governance and distribution

- **Clarify the licensing strategy instead of making unsupported adoption
  claims.** The repository currently uses PolyForm Noncommercial 1.0.0 and
  documents a commercial licensing path. If broad third-party commercial
  adoption is a project goal, decide explicitly which components, if any,
  should also have an OSS license and which remain commercial/noncommercial.
  Do not attach an invented percentage to the impact of the current license.
- **Create a selective crates.io publication plan.** The workspace currently
  contains 154 packages: 137 explicitly disable publishing and 17 do not. The
  proposed core packages `scirust-core`, `scirust-solvers`,
  `scirust-symbolic`, `scirust-simd`, and `scirust-signal` are all currently
  blocked from publishing. Define a supported public subset, verify its
  dependency closure and package metadata, then gate it with `cargo package`
  or `cargo publish --dry-run` before enabling publication.
- **Complete the contributor-facing governance surface.** `CONTRIBUTING.md`
  already defines PR scope, verification, provenance, security, and licensing
  expectations. Add a `CODE_OF_CONDUCT.md` and GitHub issue templates, and
  point contributors to `scripts/test-protocol.sh` as the canonical local
  acceptance command.

## 2. Public API and compatibility

- **Stabilize and complete the existing root facade.** The root `scirust` crate
  already re-exports `core`, `learning`, `rsi`, `simd`, `solvers`, and
  `symbolic`, provides `scirust::prelude::*`, and exposes the optional
  `tensor_canonical` pipeline with CPU, WGPU, and CUDA adapters. Treat this as
  the supported entry point: document stability tiers, reduce duplicate public
  paths, and add carefully chosen high-level modules only where they improve
  discoverability.
- **Publish an explicit compatibility policy.** The root package is currently
  `0.14.0` with `rust-version = "1.89"`. Define what SemVer compatibility means
  before 1.0, how MSRV changes are announced, and how migrations are documented
  for releases that intentionally change public APIs.
- **Unify the public error surface without erasing domain errors.** SciRust
  already has many typed error enums and several crates use `thiserror`. Add a
  facade-level error/conversion strategy for cross-crate workflows instead of
  replacing specialized errors or introducing another error crate everywhere.
- **Generate a workspace feature matrix.** Document root and important crate
  features, backend implications, default state, hardware requirements, and
  determinism guarantees. Prefer generating or testing the matrix from Cargo
  metadata so it cannot silently drift.

## 3. Documentation and discoverability

- **Keep the 61-command reference mechanically synchronized.** `docs/REFERENCE.md`
  now covers the 61 unique CLI commands. Add a CI check that compares the
  documented command names against `scirust-cli::ALL_COMMANDS` or equivalent
  generated output so additions and removals cannot drift.
- **Publish a searchable documentation site.** Workspace API documentation is
  already validated with `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --workspace`,
  but no mdBook, MkDocs, Docusaurus, or GitHub Pages configuration was found.
  Publish the existing English guides plus generated rustdoc behind one stable
  landing page.
- **Add task-oriented tutorials.** Build a small set of runnable end-to-end
  walkthroughs around existing capabilities such as ODE solving, signal
  denoising, tensor/autodiff workflows, PINNs, and deterministic inference.
  Tutorials should execute in CI or reuse tested examples rather than duplicate
  unverified snippets.
- **Measure rustdoc example coverage before setting a blanket target.** The
  workspace already enforces warning-free rustdoc. Identify high-use public APIs
  without examples or doctests and improve those first instead of claiming that
  every public function currently lacks documentation.

## 4. Quality and performance regression control

- **Treat `scripts/test-protocol.sh` as the canonical local acceptance command.** It already runs the required formatting, clippy, build, test, SIMD, determinism, rustdoc, aarch64, and cargo-deny gates, with optional GPU/stable/example gates and a retained evidence bundle. Keep this script mechanically aligned with CI instead of creating a duplicate `ci-gates.sh`.
- **Expand property-based coverage selectively.** `proptest` is already present across the workspace. Measure coverage by numeric subsystem and add properties where boundary behaviour remains weak: NaN/Inf handling, denormals, empty shapes, indexing, broadcasting, serialization round-trips, and algebraic invariants.
- **Expand the existing fuzzing programme.** The repository already has `fuzz/` targets for ONNX JSON, QSR1, safetensors, N-D safetensors, tape backward, and tensor N-D operations. Prioritize parsers and public tensor/shape APIs not yet covered, and keep crash artifacts reproducible.
- **Turn the existing benchmark schema into regression gates.** Criterion benchmarks already cover I/O, neural-network operations, SIMD, and scalar-vs-SIMD comparisons. `scirust-bench-schema` already converts Criterion estimates into seeded `BenchRecord`s with confidence intervals, certificates, and optional preregistration evidence. Add explicit per-kernel regression budgets and CI comparison policies rather than introducing another benchmark stack.


## 5. Ecosystem, interoperability, and adoption

- **Make the existing WASM story discoverable.** `scirust-studio` already contains `wasm-bindgen` dependencies, `wasm32` conditional code, and a documented `wasm32-unknown-unknown` UI path. Document and test this existing capability before creating another WASM demo.
- **Keep interoperability additive and Rust-first.** No PyO3 bridge was found in SciRust itself. If non-Rust consumers become a priority, prefer stable process/protocol boundaries or the existing WASM path before introducing a native Python FFI layer.
- **Publish a project-level `ROADMAP.md`.** `docs/roadmaps/memory-discovery-roadmap.md` already exists, but it is a specialized branch/subsystem roadmap. Add a short public English roadmap for the whole project that links to specialized roadmaps rather than replacing them.
- **Surface determinism evidence as a product feature.** The repository already has cross-process determinism gates, portable-f32 evidence, and measured O1 runs. Create a concise "Determinism in SciRust" guide that links claims to reproducible commands, commits, datasets, and retained evidence.

## 6. Industrial readiness

- **Make the certification story operational.** Consolidate the existing conformal prediction, IBP/CROWN verification, execution attestation, and evidence-bundle mechanisms into a practical regulated-workflow guide. The guide should distinguish demonstrated guarantees from aspirational compliance claims and link every claim to reproducible evidence.
- **Strengthen the existing SBOM release chain.** SciRust already generates an aggregated deterministic CycloneDX SBOM for the whole workspace, validates completeness, uploads it in CI, and publishes it with a SHA-256 checksum in releases. The next step is signed provenance/attestation for the released SBOM and binaries rather than creating a second SBOM system.
- **Publish an embedded deployment guide.** Consolidate the existing embedded, aarch64/Jetson, feature-selection, determinism, and hardware-evidence material into one deployment guide covering supported targets, no_std boundaries where applicable, memory/size constraints, and reproducibility expectations.
- **Extend `scirust-frame` instead of creating a duplicate data crate.** `scirust-frame` already provides a deterministic typed dataframe with RFC-4180 CSV read/write. Add formats that are still missing and materially useful, such as Parquet and NPY, behind explicit features and with round-trip tests.

---

## Priorities

P0: selective crates.io publication plan; stabilize the existing public facade and compatibility policy; publish a searchable documentation site; add a project-level `ROADMAP.md`; keep `scripts/test-protocol.sh` aligned with CI.

P1: facade-level error strategy; feature matrix; benchmark regression budgets; broader property/fuzz coverage; signed SBOM/binary provenance; embedded deployment guide; Parquet/NPY support in `scirust-frame`.

P2: broader interoperability only where demand justifies it, while preserving SciRust's Rust-first architecture.
