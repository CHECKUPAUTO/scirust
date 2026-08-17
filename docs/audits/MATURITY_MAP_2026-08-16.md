# SciRust maturity map — 2026-08-16

Audit base: `master` at `4dbc92811a67471bcc07fa7e23cd1887d75934fb`.

This map deliberately avoids a single "complete/incomplete" flag. In SciRust,
several surfaces are absent by design, some are experimental but functional,
some are compile-only on hosted CI, and some are explicitly planned. Treating
all four states as "missing" would make the audit less accurate.

## Status vocabulary

| Status | Meaning in this audit |
|---|---|
| **Operational / gated** | Implemented and exercised by the repository's automated contract for the stated path. |
| **Experimental / functional** | Real implementation exists and is tested on documented paths, but API/readiness is explicitly unstable or research-grade. |
| **Incomplete / contract gap** | Public API/docs promise or imply something the implementation does not currently provide, or a required validation path is missing. |
| **Planned** | Repository documentation explicitly says the capability is not yet built/delivered. |
| **Not exposed by decision** | Capability exists but an integration surface deliberately declines to expose it, with an architectural reason recorded. |
| **Compile-only on this CI leg** | The code is type-/link-checked in the named environment but runtime hardware behavior is not proved by that leg. |

## Platform-level map

| Surface | Verified status | Evidence / qualification |
|---|---|---|
| Root workspace | **Experimental / functional** | README explicitly calls SciRust experimental and warns APIs/crate boundaries may change. Audited workspace has 154 members. |
| Tensor/autodiff core | **Experimental / functional** | README maturity: "Research implementation". `scirust-core` also documents two parallel tensor/autograd stacks that do not interoperate. |
| CPU paths | **Operational / gated** for documented paths | Stable/nightly workspace jobs, SIMD feature jobs, deterministic rayon-vs-serial fingerprint, Windows/macOS checks and aarch64 checks are present. |
| WGPU canonical path | **Experimental / functional** | README labels WGPU experimental. CI executes the real WGPU path against CPU oracles on Mesa lavapipe with `SCIRUST_REQUIRE_WGPU=1`. |
| CUDA canonical path | **Experimental / functional**, hardware-qualified separately | Hosted CI compiles/lints and validates no-runtime fallback only; source comments explicitly say this proves no CUDA execution. Native/self-hosted hardware gates carry the execution claim. |
| SciAgent / RSI | **Experimental / functional** | README labels both experimental. PR #1218 itself states physical Thor execution is required before the I250 fourth-token diagnosis is complete. |
| Scientific/domain crates as a whole | **Mixed** | README explicitly says maturity/scope varies by crate; no workspace-wide production-readiness claim is justified. |
| Industrial/regulated domains | **Research/educational only** | README explicitly disclaims certification/validated medical, financial or industrial-control status. |

## `scirust-core` public orphan modules

`scirust-core/src/lib.rs` explicitly records these modules as having zero
workspace consumers outside their own tests (audit note dated 2026-07). Their
absence of consumers is therefore source evidence, not inferred from naming.

| Module | Verified status | Audit result / action |
|---|---|---|
| `lazy` | **Experimental + public error-contract gap** | Missing/wrong-shape dynamic feeds panic at public boundaries. PR #1224 adds fallible `Plan::try_execute*` methods without duplicating the executor. Further `LazyGraph`/`LazyTensor` fallible migration remains possible. |
| `amp` | **Experimental + duplicated abstraction** | Source says it overlaps `autodiff::mixed_precision` and should converge. Rustdoc example called nonexistent methods because it was ignored. PR #1225 makes the example executable and corrects the contract description. |
| `dp` | **Experimental + privacy guarantee bug** | Zero/non-positive noise could silently disable noise while `dp_sgd_gradient` returned a value documented as privatized. Issue #1227; PR #1228 adds validation/fallible entry points and numerical clipping hardening. |
| `pruning` | **Incomplete public capability** | Module/enum advertises structured row pruning (`StructuredRows`) but no row-pruning implementation exists. Issue #1229 also records dimension/sparsity/rewind validation gaps. |
| `logging` | **Experimental + format contract mismatch** | Public `TensorBoard` mode writes textual `EVENT|...` records, not a TensorBoard event/protobuf file; only CSV is tested. Issue #1226 requires either real TensorBoard output or honest renaming/deprecation. |

## Unified CLI and Studio

| Surface | Verified status | Audit result / action |
|---|---|---|
| Core `scirust` dispatcher | **Operational / gated** for implemented commands | Broad command tests exist and invalid command/usage behavior is tested. |
| Library capability entries in `scirust help` | **UX contract gap** | 14 library-only crate names are intentionally non-dispatchable but rendered alongside commands. `docs/REFERENCE.md` explains the distinction; interactive help does not make it sufficiently explicit. Issue #1231. |
| Global exit-code reference | **Documentation gap** | Reference summarizes 0/1/2, while the same unified binary's Studio path really returns 3/5/6/7. Issue #1231. |
| Studio runtime registry | **Operational / gated for 28 adapters** | `all_adapters()` contains 28 concrete adapters and is the single wiring point for registry + dispatch. |
| `scirust-sim` `System` model integration | **Operational / gated** | Current capability matrix/source say every model family implementing `System` has a Studio adapter: 16/16 module families, 28 capabilities. |
| `scirust-sim::envs` | **Not exposed by decision** | Implements `Environment`, not `System`; Studio docs record that an agent is required and the scenario schema cannot express one. This is closed by architecture decision, not pending work. |
| Studio capability-count prose/help | **Documentation drift** | Source registry has 28 adapters; CLI help still says `currently: 5`, and current matrix prose still contains a stale `Desktop-exposed: 19`. Issue #1232. |
| Capability-matrix generation | **Planned tooling** | Matrix explicitly says automatic regeneration from `scirust catalog --format json` plus source scan is future work. The stale counts demonstrate the value of implementing it. |

## Documentation assurance

| Surface | Verified status | Audit result / action |
|---|---|---|
| Main workspace rustdoc | **Contract not CI-gated** | `docs/REFERENCE.md` calls rustdoc exhaustive/authoritative, but the main release-gate list/CI does not provide a workspace `cargo doc` job with rustdoc warnings denied. Issue #1230. |
| `scirust-hypermemory` rustdoc | **Operational / gated** | Its separate nightly CI job already runs `cargo doc --no-deps` with `RUSTDOCFLAGS=-D warnings`, providing a model for the main workspace. |
| Executable examples | **Mixed** | Normal examples/doctests exist, but the stale AMP example showed that `ignore` can hide API drift. Audit recommendation: ignore only examples that truly cannot execute in CI. |

## Relativity platform (explicit roadmap states)

The nonlocal-relativity roadmap is unusually explicit and should be preserved
as a model for maturity labelling elsewhere.

| Layer | Verified documented status |
|---|---|
| Geometry Core (Layer 1) | **Partially delivered**; a substantial validated subset is enumerated with exact/numerical oracles. |
| Covariant Gravity Workbench (Layer 2) | **Partial delivered slices**; linearized gravity, PPN gamma/beta, EH variation, ADM kinematics are named as delivered slices. |
| Numerical Relativity (Layer 3) | **Opening / partial**; ADM/BSSN and periodic 1D/2D/3D substrate delivered, while strong fields, punctures, radiative boundaries and constraint damping are named next work. |
| Gravitational Memory Lab (Layer 4) | **Experimental / phenomenological** only. |
| Astrophysical Inference (Layer 5) | **Planned**. |
| Relativistic Navigation (Layer 6) | **Planned**. |

The roadmap explicitly says "planned" means not started or only an experimental,
clearly-labelled prototype, and says the platform currently makes no
"empirically validated physics" claims.

## CI/readiness observations

### Strong existing gates

- exact MSRV Rust 1.89.0 workspace check;
- stable and pinned-nightly workspace build/tests;
- Windows/macOS stable checks;
- aarch64 cross-check plus QEMU execution for selected portable proofs;
- WGPU real execution on lavapipe;
- CUDA no-runtime fallback/compile/lint on hosted runners, with hardware execution separated honestly;
- cargo-deny across default and selected opt-in dependency graphs;
- targeted Miri, bounded fuzz smoke runs, SBOM generation and deterministic fingerprints.

### Missing/weak assurance discovered in this pass

1. No equivalent main-workspace rustdoc release gate despite rustdoc being called authoritative (#1230).
2. Hand-maintained capability counts have already drifted from the registry (#1232).
3. Several experimental public modules still expose panic/assert-based caller validation despite `scirust-core::error` documenting a `Result<T, SciRustError>` migration policy (#1224, #1229).
4. Security/privacy claims need configuration validation at the claim boundary; DP zero-noise behavior demonstrated why (#1227/#1228).

## Recommended priority order

1. **P0/P1 guarantee correctness** — merge a validated DP fix (#1228); continue reviewing security/privacy/certification-labelled modules for configurations that silently disable the claimed mechanism.
2. **P1 public error boundaries** — finish the `lazy` fallible migration (#1224), then apply the same policy to orphan APIs before promotion.
3. **P1 contract truthfulness** — resolve the TensorBoard format mismatch (#1226) and the missing `StructuredRows` implementation (#1229).
4. **P1 documentation assurance** — add the workspace rustdoc gate (#1230) and remove/convert ignored public examples that can be doctested.
5. **P2 generated metadata** — derive Studio capability counts/matrices from the registry (#1232); avoid hard-coded counts in CLI help.
6. **P2 UX consistency** — distinguish runnable commands from library capabilities and make the exit-code contract exhaustive (#1231).
7. **P2 governance** — for every public zero-consumer module, record one disposition: `promote`, `merge`, `incubate`, or `deprecate/remove`, with an evidence gate required to move to `promote`.
