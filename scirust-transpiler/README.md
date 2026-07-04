# scirust-transpiler

**Inbound scientific transpiler — Python/NumPy → deterministic, safe Rust.**
Phase 0 MVP of the architecture in
[`docs/TRANSPILER_DESIGN.md`](../docs/TRANSPILER_DESIGN.md).

Unlike `scirust-codetrans` (which goes Rust → Python/C), this crate goes the
*other* way — the direction real scientific work needs: prototype in Python,
ship deterministic Rust. Every port is proven against **real NumPy** by a
differential oracle before it is trusted.

## Pipeline

```
Python/NumPy source
   → front_python (hand-written lexer + recursive-descent parser)  → PyModule
   → lower  (type/shape inference, NumPy-intrinsic resolution)      → SIR (typed)
   → emit   (deterministic, std-only Rust; reductions order-pinned) → Rust source
```

Pure Rust, **zero external dependencies** — every line is readable and
auditable, matching the SciRust doctrine.

## Supported subset (contract)

| Category      | Supported |
|---------------|-----------|
| Definitions   | top-level `def`s; params `float` / `int` / `np.ndarray` (hints optional, array-ness inferred from indexing / `np.sum` / `np.dot` / `len`) |
| Types         | scalar `f64`, 1-D array `Vec<f64>` / `&[f64]` |
| Arithmetic    | `+ - * / **`, unary minus; elementwise array ops; scalar↔array broadcasting |
| Intrinsics    | `np.sum`, `np.dot`, `np.zeros`, `np.ones`, `len`, `np.sqrt/exp/sin/cos/abs/tanh` (scalar or elementwise) |
| Routed kernels | `np.linalg.solve(A, b)`, `np.linalg.det(A)` → `scirust-solvers` (verified LU) — the emitted code calls the oracle-validated kernel instead of re-deriving it |
| Control/flow  | `for i in range(...)`, `while cond:`, `if`/`elif`/`else` + comparisons `< <= > >= == !=`, indexing `a[i]`, index-assignment `a[i] = …`, `return` |

Anything outside the subset is **refused with a diagnostic** — never guessed.

## Determinism & safety

* reductions (`sum`, `dot`) emit a **fixed ascending index order**, so results
  are independent of any parallelism (bit-reproducible);
* only `std` is emitted — no FFI, no `unsafe`;
* the emitter produces typed signatures (`&[f64]` vs `f64`), so the output
  compiles as ordinary safe Rust.

## Verification — the differential oracle

`examples/oracle.rs` is the correctness gate. For each case it generates seeded
random inputs (formatted as round-trippable decimals so Python and Rust get
*bit-identical* inputs), transpiles + compiles the Rust with `rustc`, runs the
original source under CPython+NumPy, and compares within tolerance
(`|Δ| ≤ 1e-7 + 1e-9·|numpy|`, 200 trials/case).

```
$ cargo run -p scirust-transpiler --example oracle
  ✓ rk4_step (scalar ODE step)   200/200 trials match
  ✓ dot (vector dot product)     200/200 trials match
  ✓ norm (euclidean)             200/200 trials match
  ✓ weighted_mean                200/200 trials match
  ✓ cumsum (loop + array out)    200/200 trials match
  ✓ saxpy (a*x + y)              200/200 trials match
  ✓ tanh_activation              200/200 trials match
  ✓ relu / clamp / sign          200/200 trials match   (if/elif/else, Phase 1)
  ✓ newton_sqrt / newton_conv    200/200 trials match   (while, Phase 1)
  ✓ linalg.solve / linalg.det    200/200 trials match   (routed to scirust-solvers)
  ✓ sin/cos/abs / exp / ** / ones 200/200 trials match  (full intrinsic coverage)
  ORACLE GREEN — 19/19 cases match NumPy within tolerance
```

Run the whole suite (unit tests + oracle) from one entry point:
`./scripts/test_transpiler.sh`.

The oracle is **dual-mode**: std-only cases compile with bare `rustc`; **routed**
cases (which call verified `scirust-*` kernels, e.g. `np.linalg.solve`) compile
as a tiny standalone cargo project depending on that crate by path — so the
emitted code is exercised against the *real* kernel, not a stand-in.

The gate is non-vacuous: injecting a single wrong operator into the emitter
turns 4/7 cases RED. The oracle requires `python3`, `numpy`, `rustc` (and
`cargo` for routed cases); it is opt-in (not part of `cargo test`). The
library's own unit tests (`cargo test -p scirust-transpiler`) gate CI and need
none of them.

## Honest boundary (not delivered)

* **Not "all of Python".** No `eval`/reflection, no classes, no closures, no
  dynamic typing; only the statically-analysable numeric subset above.
* **No bit-exact equality with CPython.** NumPy's reduction/BLAS order isn't
  specified; we guarantee a *declared tolerance* to NumPy and *internal*
  Rust bit-reproducibility, not bit-identity with CPython.
* **General 2-D arrays** and more routed kernels (`np.fft` → `scirust-signal`,
  `np.linalg.svd`/`eig` → `scirust-solvers`) are the next increments — see the
  roadmap in `docs/TRANSPILER_DESIGN.md`. (`if`/`elif`/`else`, scalar
  comparisons, `while` loops and `np.linalg.solve` routing landed in Phase 1.)
* **Unifying with `codetrans::Expr`** as the shared emission backend is future
  work: its `Function` node has untyped (`Vec<String>`) params, so this MVP
  uses a purpose-built typed emitter to produce compiling Rust.
