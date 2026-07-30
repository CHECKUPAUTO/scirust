# Security Reporting

## Supported Versions

Security fixes are applied to the current `main` branch and the latest tagged
release. Older releases should be upgraded before requesting backports.

## Reporting A Vulnerability

Report sensitive vulnerabilities privately through the repository's GitHub
Security Advisory interface or the private maintainer contact published on the
repository profile. Do not open a public issue containing exploit details,
credentials, license tokens, machine identifiers, customer data, or a working
sandbox escape.

Include the affected commit and feature set, host architecture and kernel,
reproduction steps, impact, and whether generated code, network access, FFI,
licensing, or persisted data is involved. Remove secrets and source content
that are not needed to reproduce the defect.

Maintainers will acknowledge receipt, reproduce and classify the report,
coordinate remediation and tests, and arrange disclosure after affected users
have an update. Publication timing depends on exploitability and deployment
impact; reporters are asked to avoid disclosure while a coordinated fix is in
progress.

## Security Scope

In scope are the deterministic core, optional premium boundaries, license and
revocation verification, egress policy, MCP parsers, persistence, FFI, DGM and
Forge sandboxing, CI/release provenance, and privacy controls.

The generated-code sandbox assumes an uncompromised Linux kernel, Bubblewrap,
and pinned Rust toolchain and must run as an unprivileged account. It does not
claim semantic correctness of generated patches. Independent cryptographic,
legal, penetration, SIMD/FFI, production key-ceremony, and real hardware
reviews remain separate assurance activities.
