# SciRust Option B — Custom rustc Driver Architecture

## Overview

This directory (`scirust-rustc-driver/`) contains a **custom Rust compiler driver**
that wraps `rustc` and injects SciRust transformations directly into the MIR
(Mid-level IR) pipeline.

This is the bridge between the proc-macro prototype (Option C) and a true
compiler-level scientific computing extension for Rust.

---

## Why MIR?

| Stage | Representation | Pros | Cons |
|-------|---------------|------|------|
| AST | High-level trees | Easy to parse | Lost type info |
| HIR | Typed trees | Borrow-check info | Still high-level |
| **MIR** | **Control-flow graph** | **SSA, explicit control flow, type-stable** | **Complex to manipulate** |
| LLVM-IR | Target-independent | Mature optimisations | Too late for AD semantics |

MIR is the sweet spot:
- **After type-checking** → we know every local's type.
- **After borrow-check** → we don't fight the borrow checker.
- **Before LLVM** → we can inject `Dual`, SIMD vectors, or GPU kernels natively.

---

## File Layout

```
scirust-rustc-driver/
├── Cargo.toml                    # Depends on rustc internal crates
├── rust-toolchain.toml           # Pins nightly + rustc-dev
├── .cargo/config.toml            # Build flags + RUSTC_SRC env
├── src/
│   ├── main.rs                   # Entry point: wraps rustc_driver
│   └── passes/
│       ├── mod.rs                # MirPass trait + PassManager
│       ├── autodiff.rs           # Dual-number MIR rewrite
│       ├── simd.rs               # Loop vectorisation
│       └── gpu.rs                # Kernel extraction
```

---

## Pipeline

```
Source (.rs)
    │
    ▼
Parse ──> HIR ──> TypeCheck ──> MIR (built)
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │  SciRustPassManager    │
                         │  ┌──────────────────┐   │
                         │  │ AutodiffPass   │   │
                         │  │ SimdPass       │   │
                         │  │ GpuPass        │   │
                         │  └──────────────────┘   │
                         └────────────────────────┘
                                      │
                                      ▼
                                Optimised MIR
                                      │
                                      ▼
                                LLVM-IR ──> Machine Code
```

---

## Passes

### 1. AutodiffPass (`passes/autodiff.rs`)

**Input**: MIR body of a function annotated with `#[autodiff]`.
**Output**: A new function `foo_grad` with MIR rewritten to use `Dual`.

**Algorithm**:
1. Clone the original `Body`.
2. Iterate `local_decls` — promote every `f64` local to `scirust_autodiff::Dual`.
3. Iterate every `BasicBlock`:
   - Replace `Rvalue::BinaryOp(Add, ...)` with a call to `Dual::add`.
   - Replace `Rvalue::BinaryOp(Mul, ...)` with `Dual::mul`.
   - Replace method calls (`powi`, `sin`, `exp`...) with `Dual` equivalents.
4. Before `TerminatorKind::Return`, insert a call to `Dual::grad(ret)`.
5. Emit the new `foo_grad` function with signature `fn foo_grad(...) -> (f64, ...) `.

**Key MIR types**:
- `rustc_middle::mir::Body`
- `rustc_middle::mir::LocalDecl` (type in `.ty`)
- `rustc_middle::mir::Rvalue::BinaryOp`
- `rustc_middle::mir::TerminatorKind::Return`

### 2. SimdPass (`passes/simd.rs`)

**Input**: MIR body with scalar loops over arrays/slices.
**Output**: Vectorised loop using `std::simd::Simd` or `core::arch` intrinsics.

**Algorithm**:
1. Detect loop headers via backward `Goto` terminators.
2. Identify induction variables (counters incremented by a constant).
3. Check if the loop body contains only "vectorisable" operations (Add, Mul, etc.).
4. Create new SIMD locals (e.g., `Simd<f64, 4>`).
5. Replace scalar operations inside the loop with SIMD equivalents.
6. Insert a scalar tail block for the remainder (`len % 4`).
7. Wrap the function with `#[target_feature(enable = "avx2")]`.

**Key MIR analysis**:
- Detect `Goto { target: loop_header }` as a loop back-edge.
- Use `rustc_mir_dataflow` for loop-invariant analysis.

### 3. GpuPass (`passes/gpu.rs`)

**Input**: MIR body with slice element-wise operations.
**Output**: Extracted GPU kernel + dispatch call.

**Algorithm**:
1. Find loops that mutate `slice[i]` element-wise.
2. Extract the loop body into a standalone `kernel` function.
3. Compile the kernel:
   - **SPIR-V** path: use `rust-gpu` to compile MIR/Rust -> SPIR-V.
   - **PTX** path: use `cust` + inline PTX strings.
4. Replace the original loop with:
   ```rust
   scirust_gpu::dispatch::gpu_or_cpu(slice, |chunk| kernel(chunk));
   ```

---

## Prerequisites

You need a **nightly Rust toolchain** with the `rustc-dev` component:

```bash
# Via the provided setup script
chmod +x setup-rustc-dev.sh
./setup-rustc-dev.sh

# Or manually:
rustup toolchain install nightly
rustup component add rustc-dev llvm-tools-preview rust-src --toolchain nightly
rustup default nightly
```

## Building

```bash
cd scirust-rustc-driver
# The Cargo.toml references rustc internal crates via RUSTC_SRC env
cargo build --release
```

## Usage

```bash
# Instead of rustc, use the SciRust driver
./target/release/scirust-rustc-driver --edition 2024 my_sci_file.rs

# Or set it as the compiler for a Cargo project
RUSTC=/path/to/scirust-rustc-driver cargo build
```

---

## From Option C to Option B

| Feature | Option C (Proc-macros) | Option B (MIR driver) |
|---------|------------------------|----------------------|
| Autodiff | `#[autodiff]` rewrites AST -> Dual | Pass rewrites MIR -> Dual |
| SIMD | `#[simd]` generates target_feature wrappers | Pass vectorises loops in MIR |
| GPU | `#[gpu]` wraps rayon dispatch | Pass extracts kernels -> SPIR-V/PTX |
| Compilation | Standard cargo | Custom `scirustc` driver |
| Integration | Library crate | True compiler extension |

The proc-macro crates (`scirust-macros`, `scirust-simd-macros`, `scirust-gpu-macros`)
remain useful as a **fallback** for stable Rust users. The MIR driver is the
high-performance path for nightly users.

---

## References

- [rustc-dev-guide — MIR](https://rustc-dev-guide.rust-lang.org/mir/index.html)
- [rustc-dev-guide — Queries](https://rustc-dev-guide.rust-lang.org/query.html)
- [rustc-dev-guide — The Compiler Driver](https://rustc-dev-guide.rust-lang.org/rustc-driver.html)
- [rust-gpu](https://github.com/EmbarkStudios/rust-gpu) — Rust -> SPIR-V
- [cust](https://github.com/Rust-GPU/cust) — CUDA bindings for Rust
