# cargo-scirust

`cargo-scirust` is SciRust's lightweight repository-aware Cargo plugin. It is deliberately a standalone Cargo workspace under `tools/` so installing the developer tool does not compile the full SciRust dependency graph.

## Install

From the SciRust repository:

```bash
cargo install --locked --path tools/cargo-scirust --force
```

Cargo then discovers the binary automatically:

```bash
cargo scirust help
```

The tool keeps SciRust's MSRV contract (`rust-version = 1.89`) and uses only `serde`/`serde_json` in addition to the standard library.

## Commands

### `affected`

Maps Git changes to workspace packages with `cargo metadata`, then follows reverse local path dependencies transitively:

```bash
cargo scirust affected
cargo scirust affected --base origin/master --head HEAD
cargo scirust affected --base origin/master --head HEAD --json
```

Without `--base`, the tool first uses the Git merge-base with `GITHUB_BASE_REF` when available, then tries `origin/master`, `master`, `origin/main`, and `main`; only if none can provide a merge-base does it fall back to `HEAD^` (or `HEAD` for a one-commit repository). Without `--head`, tracked working-tree changes plus untracked files are included.

Changes to root `Cargo.toml`, `Cargo.lock`, `rust-toolchain*`, or `.cargo/config*` conservatively affect the entire workspace.

### `check`

Runs only the package gates required by the affected dependency closure:

```bash
cargo scirust check --base origin/master --head HEAD
cargo scirust check --dry-run
cargo scirust check --full
```

Default gates are workspace `fmt`, affected-package `clippy -D warnings`, and affected-package tests. `--full` additionally runs `cargo +1.89.0 check --all-targets` for those packages. Use `--all` to select every workspace package deliberately.

### `parity`

Runs two commands from the SciRust root and compares exit code, stdout, and stderr exactly (CRLF/LF is normalized):

```bash
cargo scirust parity \
  --left "cargo run -q -p some-crate --example cpu" \
  --right "cargo run -q -p some-crate --example wgpu"
```

Use `--ignore-stderr` only when diagnostics are intentionally backend-specific. Output fingerprints use deterministic FNV-1a-64 and are diagnostic fingerprints, not cryptographic hashes.

### `determinism`

Repeats a command and requires exact exit code/stdout/stderr:

```bash
cargo scirust determinism --repeat 5 -- cargo test -q -p scirust-core some_test -- --exact
```

Each run receives `SCIRUST_DETERMINISM_RUN=1..N`, allowing a harness to perturb scheduling or seeds deliberately while the command surface remains deterministic.

### `cost`

Scans Rust source for explicit source-level indicators of copies, allocations, GPU readbacks/uploads, materialization, and host synchronization:

```bash
cargo scirust cost -p scirust-gpu
cargo scirust cost --path scirust-learning/src --json
```

This command is intentionally honest: its output is a **static heuristic**, not measured performance and not an allocation/GPU profiler. It identifies review targets; benchmarks/profilers establish actual cost.

### `features`

Lists package features or creates a bounded no-default baseline/single/pair matrix:

```bash
cargo scirust features scirust-gpu
cargo scirust features scirust-gpu --cover pairwise --max 128
cargo scirust features scirust-gpu --cover pairwise --max 128 --execute
```

The `--max` guard prevents accidentally exploding CI time on crates with many features.

### `bench`

Runs `cargo bench` only for affected packages, all packages, or one selected package:

```bash
cargo scirust bench --base origin/master --head HEAD
cargo scirust bench -p scirust-gpu -- --bench my_bench
cargo scirust bench --dry-run
```

### `calibrate`

Derives six deterministic ElasticTokenizer piece-size classes from observed piece byte lengths, without changing token identities or BPE merge ranks:

```bash
cargo scirust calibrate --pieces decoded-pieces.txt
cargo scirust calibrate --lengths piece-byte-lengths.txt --json
```

`--pieces` expects one decoded token piece per line. `--lengths` expects one positive byte length per line. Five equal-frequency nearest-rank cuts create `S`, `M`, `L`, `XL`, `XXL`, and `XXXL` buckets. This is the repository-level calibration primitive; direct export from the ElasticTokenizer can be wired to the same input contract without changing the BPE model.

## Design constraints

- pure Rust tool implementation;
- no FFI;
- SciRust MSRV 1.89;
- deterministic ordering and output where practical;
- no hidden full-workspace build merely to discover affected packages;
- conservative behavior when a root build contract changes;
- performance claims are never inferred from static source heuristics.
