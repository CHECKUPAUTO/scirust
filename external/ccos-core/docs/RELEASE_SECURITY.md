# Release Security

Releases are built from a pinned lockfile and Rust toolchain, with cargo-deny, SBOM, provenance metadata, and SHA-256 artifact checksums. Production license verification keys are injected through `CCOS_LICENSE_PUBLIC_KEYS_FILE`; private signing keys never belong in this repository. Record commit, target, features, compiler, and lockfile hash for every artifact.
