# scirust-transpiler

**Inbound scientific transpiler — Python/NumPy _and_ MATLAB/Octave →
deterministic, safe Rust.** Phase 0-2 of the architecture in
[`docs/TRANSPILER_DESIGN.md`](../docs/TRANSPILER_DESIGN.md).

Unlike `scirust-codetrans` (which goes Rust → Python/C), this crate goes the
*other* way — the direction real scientific work needs: prototype in Python or
MATLAB, ship deterministic Rust. Two source front-ends lower into **one** typed
IR and share **one** emitter, so both inherit the same determinism and the same
oracle-validated kernels. Every port is proven against a **real reference
runtime** — Python cases against **NumPy**, MATLAB cases against **Octave** —
by a differential oracle before it is trusted.

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
| Arithmetic    | `+ - * / **`, unary minus, `@` matrix-vector / matrix-matrix product, `A.T` transpose; elementwise array ops; scalar↔array broadcasting |
| Intrinsics    | `np.sum`, `np.dot`, `np.zeros`, `np.ones`, `np.diag`, `len`, `np.sqrt/exp/sin/cos/abs/tanh` (scalar or elementwise) |
| Routed kernels | `np.linalg.solve(A, b)`, `np.linalg.det(A)`, `np.linalg.eigvalsh(A)`, `np.linalg.inv(A)`, `A @ b` → `scirust-solvers` (verified LU / symmetric eigensolver); `np.fft.fft(x)` / `np.fft.rfft(x)` / `np.fft.ifft(...)` / `np.abs(np.fft.fft(x))` → `scirust-signal` (verified FFT, real→complex) — the emitted code calls the oracle-validated kernel instead of re-deriving it |
| Multi-output  | `U, S, Vh = np.linalg.svd(A)` (tuple unpacking) → verified thin SVD in `scirust-solvers`, with `Vh = Vᵀ` to match numpy (square `A`, where thin = full) |
| Composition   | list literals `[a, b, c]` → `Vec<f64>`; **calls to other user functions** defined earlier in the module (define-before-use), with array-ness inferred *across* calls from the callee's signature (no annotation needed) |
| Control/flow  | `for i in range(...)`, `while cond:`, `if`/`elif`/`else` + comparisons `< <= > >= == !=`, indexing `a[i]`, index-assignment `a[i] = …`, `return` |

Anything outside the subset is **refused with a diagnostic** — never guessed.

### MATLAB/Octave subset (second front-end)

A dedicated lexer + parser + lowering (`src/front_matlab/`, `src/lower_matlab.rs`)
maps the MATLAB dialect onto the *same* SIR, handling its distinct semantics:

| MATLAB feature | Lowered to |
|----------------|------------|
| `function y = f(x) … end` / `endfunction` | one `pub fn` returning the output variable's final value |
| 1-based indexing `a(i)` | `a[i-1]` (0-based) |
| inclusive ranges `for i = 1:n` | `for i in 1..(n+1)` |
| element-wise `.*` `./` `.^` (operands inferred as arrays) vs scalar `* / ^` | `EwBin` / broadcast vs scalar op |
| `if`/`elseif`/`else`, `while`, comparisons incl. `~=` | same control-flow IR as Python |
| output/locals first assigned inside a branch | **hoisted** to `let mut y: T;`, validated by Rust's definite-assignment analysis |
| `sqrt/exp/sin/cos/abs/tanh`, `sum`, `length` | scalar/elementwise intrinsics + reductions |

Array-ness is inferred from indexing, `sum`/`length`, and element-wise operands
(MATLAB has no type hints); ambiguous scalar-vs-array uses are refused.

## Determinism & safety

* reductions (`sum`, `dot`) emit a **fixed ascending index order**, so results
  are independent of any parallelism (bit-reproducible);
* only `std` is emitted — no FFI, no `unsafe`;
* the emitter produces typed signatures (`&[f64]` vs `f64`), so the output
  compiles as ordinary safe Rust.

## Verification — the differential oracle

`examples/oracle.rs` is the correctness gate. For each case it generates seeded
random inputs (formatted as round-trippable decimals so the source and the Rust
get *bit-identical* inputs), transpiles + compiles the Rust with `rustc`, runs
the original source under its reference runtime (Python → **CPython+NumPy**,
MATLAB → **Octave**), and compares within tolerance
(`|Δ| ≤ 1e-7 + 1e-9·|ref|`, 200 trials/case).

```
$ cargo run -p scirust-transpiler --example oracle
  Python cases → NumPy · MATLAB cases → Octave
  ✓ rk4_step (scalar ODE step)   200/200 trials match (numpy)
  ✓ dot / norm / weighted_mean   200/200 trials match (numpy)
  ✓ cumsum / saxpy / tanh        200/200 trials match (numpy)
  ✓ relu / clamp / sign          200/200 trials match (numpy)  (if/elif/else, Phase 1)
  ✓ newton_sqrt / newton_conv    200/200 trials match (numpy)  (while, Phase 1)
  ✓ solve/det/eigvalsh/inv/A@b/A@B/A.T 200/200 trials match (numpy)  (routed → scirust-solvers)
  ✓ fft.fft / rfft / ifft        200/200 trials match (numpy)  (routed → scirust-signal, complex)
  ✓ svd singular values + reconstruction 200/200 trials match (numpy)  (tuple unpack, Phase 2)
  ✓ user calls: sumsq / sumdbl / chain 200/200 trials match (numpy)  (function composition, Phase 2)
  ✓ list literal: weighted average 200/200 trials match (numpy)  (Python list → Vec, Phase 2)
  ✓ sin/cos/abs / exp / ** / ones 200/200 trials match (numpy) (full intrinsic coverage)
  ✓ M: norm2 / dot / relu / sign 200/200 trials match (octave) (MATLAB front-end, Phase 2)
  ✓ M: clamp / poly / mysum      200/200 trials match (octave) (1-based idx, for/while, ^)
  ✓ M: newton / ew_scale         200/200 trials match (octave) (while, element-wise array out)
  ORACLE GREEN — 43/43 cases match their reference runtime within tolerance
```

Run the whole suite (unit tests + oracle) from one entry point:
`./scripts/test_transpiler.sh`.

The oracle is **dual-mode**: std-only cases compile with bare `rustc`; **routed**
cases (which call verified `scirust-*` kernels, e.g. `np.linalg.solve`) compile
as a tiny standalone cargo project depending on that crate by path — so the
emitted code is exercised against the *real* kernel, not a stand-in.

The gate is non-vacuous on both front-ends: injecting a single wrong operator
into the emitter turns Python cases RED, and breaking MATLAB's 1-based index
mapping (`i-1` → `i-2`) crashes `mysum` and turns the oracle RED. The oracle
requires `python3`, `numpy`, `rustc` (plus `cargo` for routed cases, and
`octave` for the MATLAB cases — missing `octave` skips those with a notice
rather than failing); it is opt-in (not part of `cargo test`). The library's
own unit tests (`cargo test -p scirust-transpiler`) gate CI and need none of
them.

## Honest boundary (not delivered)

* **Not "all of Python".** No `eval`/reflection, no classes, no closures, no
  dynamic typing; only the statically-analysable numeric subset above.
* **No bit-exact equality with CPython.** NumPy's reduction/BLAS order isn't
  specified; we guarantee a *declared tolerance* to NumPy and *internal*
  Rust bit-reproducibility, not bit-identity with CPython.
* **General 2-D arrays** and more routed kernels (`np.linalg.qr`/`eig` →
  `scirust-solvers`, general tuple *returns*, **recursion / mutual recursion**)
  are the next increments — see the roadmap in `docs/TRANSPILER_DESIGN.md`.
  (`if`/`elif`/`else`, scalar comparisons, `while` loops and `np.linalg.solve`
  routing landed in Phase 1; the MATLAB/Octave front-end, `np.linalg.svd` via
  tuple unpacking, and user-function composition + list literals landed in
  Phase 2.)
* **User calls are define-before-use and non-recursive.** A function may call
  any function defined earlier in the module; forward references and (mutual)
  recursion are refused. Callee parameters must be scalar or array (so argument
  coercion is unambiguous) — matrix/complex parameters can't yet be passed
  between transpiled functions.
* **SVD is proven on square inputs**, where numpy's thin and full SVD coincide
  with `scirust-solvers`' thin SVD; individual singular *vectors* have a sign
  gauge, so U and V are validated only through the gauge-invariant
  reconstruction `U·diag(S)·Vᵀ`, not element-by-element.
* **MATLAB is a scientific subset, not all of MATLAB.** No cell arrays, structs,
  anonymous functions, `end` indexing, or 2-D matrix routing yet; `zeros(n)` is
  not mapped (it is `n×n` in MATLAB, unlike NumPy's 1-D `np.zeros(n)`), and
  element-wise operands are heuristically typed as arrays.
* **Unifying with `codetrans::Expr`** as the shared emission backend is future
  work: its `Function` node has untyped (`Vec<String>`) params, so this MVP
  uses a purpose-built typed emitter to produce compiling Rust.
