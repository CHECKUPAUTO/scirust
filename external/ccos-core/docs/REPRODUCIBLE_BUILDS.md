# CCOS Core — Reproducible Builds

## Controls

- `Cargo.lock` committed; CI builds with `--locked`;
- `rust-toolchain.toml` pins the toolchain (MSRV 1.89 gate in CI);
- all dependencies pinned; **no moving-branch git dependency** (the single
  git dependency, `octasoma`, is pinned to tag `v0.4.0` and the lockfile
  records the resolved commit);
- no network downloads during build beyond the locked crate graph;
- no local paths embedded in release artifacts (checked at release time);
- build-time key baking (`build.rs`) is deterministic: no key file → the
  `none` profile; test keys are forbidden in release builds.

## Verification

The `Periodic Security Validation` workflow builds the `ccos` binary twice
from clean and compares SHA-256:

```bash
cargo build --release --locked && sha256sum target/release/ccos > a.sha
cargo clean && cargo build --release --locked && sha256sum target/release/ccos > b.sha
diff a.sha b.sha
```

**If hashes differ, reproducibility is not claimed** — the workflow fails and
the difference is documented (§46).

## Current status

Two clean builds on the migration host (rustc 1.96.0, Debian 13, x86_64):
bit-identical. Platform-to-platform reproducibility is tracked in
migration report 15 (pending).
