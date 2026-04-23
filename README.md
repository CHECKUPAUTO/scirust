# SciRust — Scientific Computing Extensions for Rust

A prototype transpiler that extends Rust with native scientific computing primitives:
**exact automatic differentiation**, **SIMD auto-vectorization**, and **GPU/parallel dispatch**.

---

## Architecture

```
scirust/
├── scirust-autodiff/          # Dual-number forward-mode AD engine
├── scirust-macros/            # #[autodiff] proc-macro
├── scirust-simd/              # Runtime SIMD kernels (AVX2/SSE2/NEON)
├── scirust-simd-macros/       # #[simd] proc-macro with runtime dispatch
├── scirust-gpu/               # GPU/CPU dispatch runtime (rayon fallback)
├── scirust-gpu-macros/        # #[gpu] proc-macro
├── scirust-core/              # Unified facade crate
└── examples/
    ├── demo/                  # Comprehensive demo
    └── benchmarks/            # Criterion benchmarks
```

---

## Features

### 1. Exact AutoDiff (`#[autodiff]`)

Forward-mode automatic differentiation using dual numbers. **Zero approximation** — analytically exact gradients.

```rust
use scirust_core::autodiff;

#[autodiff]
fn rosenbrock(x: f64, y: f64) -> f64 {
    (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
}

fn main() {
    let (dx, dy) = rosenbrock_grad(1.0, 1.0);
    // dx = 0.0, dy = 0.0  (exact)
}
```

**How it works**: The proc-macro generates a `_grad` function that rewrites the
original body using `Dual` numbers (value + derivative pairs). Every arithmetic
operation carries its gradient automatically via operator overloading.

Supported operations: `+`, `-`, `*`, `/`, `.powi()`, `.powf()`, `.sqrt()`,
`.exp()`, `.ln()`, `.sin()`, `.cos()`, `.tan()`, `.abs()`.

### 2. SIMD Auto-Vectorization (`#[simd]`)

Attribute macro that generates architecture-specific variants of a function
with runtime dispatch.

```rust
#[simd]
fn double(x: f32) -> f32 {
    x * 2.0
}
```

The macro emits:
- `__simd_scirust_double_avx2` (x86/x86_64, `#[target_feature(enable = "avx2")]`)
- `__simd_scirust_double_sse2` (x86/x86_64, `#[target_feature(enable = "sse2")]`)
- `__simd_scirust_double_neon` (aarch64, `#[target_feature(enable = "neon")]`)
- `__simd_scirust_double_scalar` (fallback)
- `double` — public dispatcher selecting the best at runtime

Plus stable manual kernels in `scirust-simd::ops`: `add_f32`, `mul_f32`, `add_f64`, `mul_f64`.

### 3. GPU/Parallel Dispatch (`#[gpu]`)

```rust
use scirust_core::dispatch::gpu_or_cpu;

let mut data = vec![1.0f32; 1_000_000];
gpu_or_cpu(&mut data, |chunk| {
    for x in chunk { *x *= 2.0; }
});
```

Uses `rayon::par_chunks_mut` for CPU parallelism. CUDA backend stubbed via `cust`
(activate with `--features cuda`).

---

## Running the Demo

```bash
cd /root/scirust
cargo run --package demo
```

Output:
```
=== SciRust Exact AutoDiff (Forward-Mode Dual) ===
grad of square at 3.0        = 6
grad of square at -2.0       = -4
grad of rosenbrock at (1,1)  = (0, 0)
grad of rosenbrock at (0,0)  = (-2, 0)
grad of neural_activation at 1.0 = -0.533198028611203

=== SciRust SIMD Auto-Vectorization ===
before simd_add_one: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
after  simd_add_one: [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]

=== SciRust GPU/Parallel Dispatch ===
before scale*2: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
after  scale*2: [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]

=== SciRust Dual Number Direct Use ===
f(x) = x^3 + sin(x) at x=2
value     = 8.909297426825681
derivative= 11.583853163452858 (expected: 12 + cos(2) = 11.583853163452858)
```

---

## Benchmarks

```bash
cd /root/scirust/examples/benchmarks
cargo bench
```

Compares scalar loops vs SIMD-vectorized operations on 10,000-element arrays.

---

## Option B — Custom rustc Driver (Compiler-Level)

For users with a **nightly Rust toolchain** + `rustc-dev` component, SciRust
provides a custom compiler driver that performs transformations at the **MIR**
level (Mid-level IR), *before* LLVM codegen:

```
scirust-rustc-driver/          # Custom rustc wrapper
├── src/main.rs                # Injects passes into rustc pipeline
├── src/passes/autodiff.rs     # Dual-number MIR rewrite
├── src/passes/simd.rs         # Loop vectorisation at MIR level
├── src/passes/gpu.rs          # Kernel extraction → SPIR-V/PTX
├── rust-toolchain.toml        # Pins nightly + rustc-dev
└── setup-rustc-dev.sh         # One-command environment setup
```

**Why MIR?** It's after type-checking and borrow-check, but before LLVM.
This means we know every local's type and can inject `Dual`, SIMD vectors,
or GPU kernels natively into the compiler pipeline.

See `ARCHITECTURE-B.md` for the full technical specification.

**Setup:**
```bash
./setup-rustc-dev.sh
cd scirust-rustc-driver && cargo build --release
./target/release/scirust-rustc-driver --edition 2024 myfile.rs
```

---

## Roadmap

- [x] Forward-mode autodiff (exact Dual numbers)
- [x] SIMD auto-vectorization (proc-macro + runtime dispatch)
- [x] GPU/parallel dispatch (rayon fallback)
- [x] Custom rustc driver architecture (MIR passes)
- [ ] Reverse-mode autodiff (tape-based / Wengert list)
- [ ] LLVM IR pass for true compiler-level SIMD
- [ ] CUDA kernel generation via `cust` + PTX
- [ ] BLAS integration (`matrixmultiply`, `ndarray`)
- [ ] JIT compilation cache for repeated kernels

---

## License

MIT OR Apache-2.0
