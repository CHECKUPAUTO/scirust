# Contributing to SciRust

SciRust is an experimental research workspace. Contributions should be small,
reproducible, and explicit about the guarantee they add.

## Before opening a pull request

1. Create a focused branch from the current `master` branch.
2. Add or update tests for behavioral changes.
3. Document the supported scope and known limitations.
4. Do not add benchmark, accuracy, determinism, safety, or standards-compliance
   claims without a reproducible command and retained evidence.
5. Keep optional native and hardware dependencies behind explicit features.

Run the same baseline checks used by the main CI workflow:

```bash
cargo +nightly-2026-07-02 fmt --all -- --check
cargo +nightly-2026-07-02 clippy --workspace --all-targets --locked -- -D warnings
cargo +nightly-2026-07-02 build --workspace --all-targets --locked
cargo +nightly-2026-07-02 test --workspace --locked
```

Feature-specific or hardware-specific changes must also run the relevant jobs
documented in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) and, where
applicable, [`.github/workflows/native-arm64.yml`](.github/workflows/native-arm64.yml).

## Pull-request description

Include:

- the problem and intended scope;
- the implementation boundary;
- exact verification commands and results;
- new dependencies or feature changes;
- compatibility, determinism, safety, and performance limitations;
- documentation updated by the change.

Generated code, copied algorithms, and datasets must include their provenance
and compatible license before review.

## Security

Do not disclose exploitable vulnerabilities in a public issue. Follow the
private reporting instructions in [`SECURITY.md`](SECURITY.md).

## License

By contributing, you agree that your contribution is distributed under the
repository's [PolyForm Noncommercial License 1.0.0](LICENSE.md) and may be
offered under the commercial licensing terms described in
[`LICENSING.md`](LICENSING.md).
