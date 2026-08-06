# CCOS Core — Supply-Chain Security

## Controls in place

| Control | Where |
|---|---|
| `cargo deny` (advisories, licenses, bans, sources) | `deny.toml` + audit workflow |
| SBOM generation (CycloneDX) | audit workflow artifact |
| Secret detection (gitleaks) | audit workflow |
| Pinned toolchain | `rust-toolchain.toml`, MSRV 1.89 CI job |
| Locked dependency graph | `Cargo.lock` + `--locked` everywhere |
| Vendored/git deps policy | git CLI fetch for pinned public deps; single tag-pinned git dep (`octasoma v0.4.0`) |
| Build-time key hygiene | `build.rs`: no key file → `none` profile; test keys forbidden in release |
| Fuzzing of parser boundaries | `fuzz/fuzz_targets/` (license token, revocation list, mcp json, persistence blob, base64url) |
| Reproducible-build comparison | security workflow (two clean builds, SHA-256) |
| Author/identity policy | `scripts/check-author-policy.sh` + git hooks |

## Third-party notices

See THIRD_PARTY_NOTICES.md. Licenses are enforced by `deny.toml`
(license allowlist; `LicenseRef-TarekZekriti-Dual` for first-party code).

## Dependency admission rule

A new dependency must: solve a real problem, be maintained, have an acceptable
license, not duplicate in-tree capability, and not weaken determinism or the
offline-test posture. Any optional dependency that cannot promise bit-exact
replay is feature-gated and documented in DETERMINISM.md.

## Historical note

The historical repositories contained AI-authored commits; the migration
normalized all authorship to the sole human owner (migration report 03).
This is a provenance statement, not a contribution claim.
