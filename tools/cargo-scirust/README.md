# cargo-scirust

`cargo-scirust` is SciRust's lightweight repository-aware Cargo plugin. It stays in its own workspace under `tools/`, so installing the developer tool does not compile the full SciRust dependency graph.

## Install

```bash
cargo install --locked --path tools/cargo-scirust --force
cargo scirust help
```

The tool keeps SciRust's Rust 1.89 MSRV and adds no dependency beyond `serde`/`serde_json`.

## `affected`

Maps Git changes to owning crates with `cargo metadata`, then follows reverse local dependencies transitively.

```bash
cargo scirust affected
cargo scirust affected --base origin/master --head HEAD --json
cargo scirust affected --names-only
cargo scirust affected --direct-only
```

The default base is the Git merge-base with the PR/base branch when available. Root build-contract changes remain conservative and select the whole workspace. `--fail-if-empty` is useful in CI when an empty selection is itself an error.

## `check`

Runs repository gates only for the affected closure.

```bash
cargo scirust check
cargo scirust check --full
cargo scirust check --all-features
cargo scirust check --dry-run
```

Cargo.lock is enforced by default for Clippy, tests and the Rust 1.89 check. `--unlocked` is an explicit escape hatch. `--all` deliberately selects the whole workspace.

## `parity`

Compares two commands after normalizing CRLF/LF. Both commands must succeed by default; two identical failures are not a parity pass.

```bash
cargo scirust parity \
  --left "cargo run -q -p some-crate --example cpu" \
  --right "cargo run -q -p some-crate --example wgpu" \
  --repeat 5

cargo scirust parity --left "..." --right "..." --json
```

On mismatch the tool reports deterministic FNV-1a diagnostic fingerprints and the first differing normalized byte/line. `--allow-failure` exists only for tests that intentionally compare failure surfaces.

## `determinism`

Repeats a direct command and requires successful, exact normalized output on every run.

```bash
cargo scirust determinism --repeat 5 -- cargo test -q -p scirust-core some_test -- --exact
cargo scirust determinism --repeat 10 --json -- ./target/release/my-harness
```

Each run receives `SCIRUST_DETERMINISM_RUN=1..N`. `--ignore-stderr` and `--allow-failure` are explicit opt-outs for specialized harnesses.

## `cost`

The default mode scans Rust source for explicit copy/allocation/materialization/GPU-transfer/host-sync indicators. This remains a source heuristic and is never presented as measured performance.

```bash
cargo scirust cost -p scirust-gpu
cargo scirust cost --path scirust-learning/src --json
```

Measured mode adds real wall-clock samples for a command, with warm-up plus min/median/mean/max reporting:

```bash
cargo scirust cost --no-static --warmup 2 --measure 9 -- ./target/release/my-bench
cargo scirust cost -p scirust-gpu --measure 7 --inherit-io -- cargo test -q -p scirust-gpu my_test -- --exact
```

This measures process wall time; it does not claim allocation counts or hardware-counter profiling.

## `features`

Lists features or executes the complete bounded baseline/single/pair matrix.

```bash
cargo scirust features scirust-gpu
cargo scirust features scirust-gpu --cover pairwise --max 256
cargo scirust features scirust-gpu --cover pairwise --max 256 --execute
```

Execution no longer stops at the first failing combination. The final report separates pair-specific incompatibilities (both singles pass, the pair fails) from intrinsic single-feature failures. The command fails when any case fails unless `--allow-incompatible` was deliberately requested. `--json` emits the full matrix.

## `bench`

Runs locked `cargo bench` only for affected, selected, or all workspace crates.

```bash
cargo scirust bench
cargo scirust bench -p scirust-gpu -- --bench my_bench
cargo scirust bench -p scirust-gpu --repeat 5
```

When repeated, `cargo-scirust` also reports outer process wall-time min/median/mean/max. Criterion or crate-specific benchmark output remains the authoritative inner benchmark measurement.

## `calibrate`

The primary mode now drives SciAgent's real semantics-gated ElasticTokenizer autotuner:

```bash
cargo scirust calibrate \
  --tokenizer data/tokenizer.json \
  --input scirust-core/src \
  --input scirust-sciagent/src \
  --output ~/.local/share/scirust/elastic-profile.json \
  --recursive \
  --device local
```

It runs `tokenizer-autotune` in release mode by default. That harness measures Reference/TinyScan/Indexed/Heap kernels, checks every measured result against canonical BPE token ids, rejects semantic mismatches, fits exactly six contiguous S/M/L/XL/XXL/XXXL execution classes, and persists a hardware-local profile. `--debug` is available only when a debug-build calibration is intentionally desired.

A distribution-only compatibility mode remains available:

```bash
cargo scirust calibrate --pieces decoded-pieces.txt
cargo scirust calibrate --lengths piece-byte-lengths.txt --json
```

This mode does **not** select kernels. It requires at least six distinct positive observed lengths and derives five strictly increasing midpoint boundaries; it cannot silently emit duplicate/unreachable execution classes.

## Design constraints

- pure Rust orchestration; no FFI;
- SciRust MSRV 1.89;
- deterministic ordering/reporting where practical;
- Cargo.lock enforced by default for executable gates;
- no hidden full-workspace build merely to discover affected crates;
- root build-contract changes remain conservative;
- parity/determinism require successful execution unless explicitly overridden;
- static source heuristics are never mislabeled as runtime measurements;
- ElasticTokenizer routing may change execution strategy, never canonical BPE token identity.
