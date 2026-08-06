# GitHub presentation and information audit — 2026-07-31

## Scope

Audit of the public repository presentation at commit
`e0428610dd224e23e09db8eac0f8295a9f16aa7f`: root layout, README, information
files, Cargo manifests, documentation links, workflows, and public project
positioning.

This audit does not certify numerical correctness, security, legal compliance,
or production readiness. Initial static inspection was followed by validation
on an NVIDIA Jetson AGX Thor using the pinned Rust toolchain. Historical test
totals were not treated as current results.

## Executive findings

| Severity | Finding | Evidence | Resolution in this change |
|---|---|---|---|
| High | The README mixed a landing page, feature inventory, research report, and CLI reference in 392 lines. Several individual bullets ran for thousands of characters. | Previous `README.md`, especially “Validated capabilities” | Replaced with a 205-line scoped overview and links to detailed documents. |
| High | Current-looking test totals were dated 2026-06-19 while the audited revision was 2026-07-31. | Previous README claimed “1718 passing workspace tests”; other reports contain different totals. | Removed global totals from the landing page; results must now point to dated evidence and exact commands. |
| High | Absolute or comparative claims were not supportable from repository inspection alone. | “only self-contained DL framework” and broad determinism wording in the previous README | Removed exclusivity claims and scoped determinism to specific paths, targets, features, and evidence. |
| High | Experimental industrial, medical, safety, and hardware-oriented code could be read as product capability. | Previous capability catalogue and “Stable/New” status table | Added explicit research-only and non-certified boundaries; replaced binary status labels with maturity descriptions. |
| Medium | More than 50 dated reports occupied the repository root. | `RADAR_OPTRONICS_*`, `SIMULATION_ENVIRONMENTS_*`, audits, and project reports | Moved to indexed directories under `docs/` with history preserved. |
| Medium | The README described optional FFI and native dependencies inconsistently. | Previous opening claimed “No C++, no Python” while optional TLS, BLAS, WGPU, CUDA, OS, and exported C ABI paths exist elsewhere in the repository. | Replaced with a feature-scoped Rust-first statement and explicit exceptions. |
| Medium | The README linked a hardware performance figure that hosted CI does not reproduce. | Historical Jetson Thor figure in the previous GPU bullet | Removed from the landing page; historical measurements remain in dated reports. |
| Medium | Contributor expectations were undocumented at the repository root. | No root `CONTRIBUTING.md` | Added contribution, evidence, provenance, and verification requirements. |
| Low | Two license files can confuse readers. | `LICENSE` is a notice; `LICENSE.md` contains the complete terms. | README now links directly to the complete terms and explains commercial licensing separately. |

## Information-file review

### README

The revised README states what the repository is, what it is not, which areas
are experimental, how optional hardware backends are activated, how to run the
actual CLI, and where authoritative detail lives. It avoids changing numerical
claims into marketing claims.

### Security policy

`SECURITY.md` contains a private reporting route and documents important FFI,
unsafe-code, SBOM, and tamper-evidence boundaries. Its moved audit link was
updated. Some dependency assertions can only be reconfirmed with `cargo tree`
in an environment containing the pinned Rust toolchain.

### Licensing

`LICENSE.md` contains the PolyForm Noncommercial License 1.0.0 terms. `LICENSE`
is a short notice pointing to those terms, and `LICENSING.md` describes a
separate commercial path. The project must not be described simply as
“open-source”; “source-available under a noncommercial license” is accurate.

### Changelog and historical reports

`CHANGELOG.md` is unusually large and written primarily in French. It remains
useful as a development ledger but is not a conventional release-only
changelog. This change preserves it rather than rewriting historical records.
Dated reports were archived under `docs/` and explicitly marked as
point-in-time evidence.

## Verification performed

Validation on an NVIDIA Jetson AGX Thor completed successfully:

- `cargo metadata --no-deps --locked`: 148 workspace crates;
- `cargo fmt --all -- --check`;
- `cargo clippy --workspace --all-targets --locked -- -D warnings`;
- `cargo test --workspace --locked`;
- `git diff --check`;
- local-link validation after all document moves.

GitHub Actions run `30694591414` completed successfully after retrying one
transient LLVM coverage-profile merge failure. All 24 checks passed, including
stable, nightly, MSRV 1.89, Windows, macOS, aarch64, Miri, WGPU, CUDA compile
checks, dependency audits, fuzz smoke tests, SBOM generation, and coverage.
