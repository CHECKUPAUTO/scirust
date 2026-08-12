# SciRust — Scientific transpiler design (source → Rust)

> Status: **design + Phase 0 delivered**. This document describes the architecture
> of an *inbound* transpiler (Python / MATLAB / Julia / Fortran / C++ → Rust)
> that is deterministic, safe and **oracle-verified**, aligned with the repository's
> doctrine ("no claim without a test"). It rigorously distinguishes what
> **already exists** from what **remains to be built**, and claims no
> undelivered capability.
>
> **Update — Phase 0 (MVP) implemented and proven.** The crate
> [`scirust-transpiler`](../scirust-transpiler) realizes the complete inbound
> pipeline (Python/NumPy front-end → typed SIR → deterministic Rust emission),
> gated by a **differential oracle against real NumPy**:
> `cargo run -p scirust-transpiler --example oracle` → **7/7 cases, 200 trials
> each, conformant** (rk4, dot, norm, weighted-mean, cumsum, saxpy, tanh).
> The oracle is non-vacuous (injecting a wrong operator turns 4/7 cases
> RED). See §9-bis "Implementation status".

---

## 0. Honest summary (read first)

The vision calls for a tool able to **automatically convert** scientific
algorithms written in Python, MATLAB, Julia, Fortran or C++ into
"performant, deterministic and safe" Rust, for 15 regulated sectors.

Real state of the repository today:

| Required brick                              | Status | Where |
|------------------------------------------------|--------|----|
| Language front-ends (source → AST)            | ❌ absent | — |
| Typed scientific IR (shapes, units, effects) | ❌ absent | — |
| Rust emission backend (AST → Rust source)    | 🟡 **reusable** | `scirust-codetrans` (`Expr` + pretty-printer) |
| IR optimization passes                 | 🟡 **reusable** | `scirust-codetrans` (20 rules: CSE, DCE, LICM…) |
| **Oracle-verified target vocabulary**       | ✅ **present** | ~90 `scirust-*` crates (see §5) |
| **Oracle validation doctrine**          | ✅ **present** | whole repository; hash-chained CCOS / MCP audit |
| *Transpilation* oracle harness            | ❌ absent | — |
| SLM / assistant agent                          | ✅ present | `scirust-sciagent`, `scirust-mcp` |

> ⚠️ **Crucial and honest point.** `scirust-codetrans` transpiles **Rust → Python**
> and **Rust → C** (the *outbound* direction), i.e. the **reverse** of what the vision
> asks for. Its `parse_expr` / `parse_pattern` functions read an **internal**
> S-expression AST, not Python/MATLAB/Fortran source code. So today there is
> **no** inbound transpilation capability at all.

**Verdict.** The inbound transpiler is not yet delivered. But two of the three
hardest bricks already are: (1) a **target vocabulary** of numerical
primitives proven bit-exact against a reference oracle, and
(2) the **proof discipline** that distinguishes SciRust from a
"line-by-line LLM" translator. The missing piece is the
*front-end → IR → emission* pipeline and the *transpilation oracle harness*. This
document fixes its architecture and roadmap.

---

## 1. Why a scientific transpiler is NOT a syntactic translator

The obvious trap — an LLM or a regex rule set that "translates line by
line" — produces *plausible* but potentially **wrong, non-deterministic and
unverified** Rust. That is precisely what the target sectors
forbid:

- **DO-178C** (aeronautics) and **IEC 62304 Ed.2** (medical devices)
  require traceability that "assumes deterministic behavior".
- **ISO 26262** (automotive) imposes redundant MIL/SIL/PIL/HIL tests
  *because* the model ⇄ generated-code correspondence is only guaranteed
  "within tolerance" (MathWorks documents this itself).
- Floating-point non-associativity + non-deterministic BLAS threading
  break reproducibility (cf. `docs/DOMAIN_ROADMAP.md`, OpenBLAS bug
  #1844).

SciRust's thesis, applied to transpilation, comes down to three
non-negotiable requirements:

1. **Understand the numerical semantics** (shapes, types, reduction
   order, source of randomness) — not just the syntax.
2. **Emit to already-proven** bit-exact primitives, rather than
   re-deriving the numerics in fresh untested Rust.
3. **Prove source ⇄ Rust equivalence** via an oracle *before* accepting the
   port — exactly the "no claim without a test" rule of the rest of the
   repository.

A port that fails the oracle is **rejected**, not "probably good".

---

## 2. Target architecture (5-stage pipeline)

```
  Scientific source                                          Verified Rust
  (Python/MATLAB/                                              deterministic, safe
   Julia/Fortran/C++)                                          (+ signed report)
        │                                                              ▲
        ▼                                                              │
 ┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────────┐
 │ 1. FRONT-END│──▶│ 2. SIR        │──▶│ 3. ANALYSES   │──▶│ 4. LOWERING       │
 │ (1 per      │   │ Scientific IR │   │ shapes, types,│   │ SIR → codetrans:: │
 │  language)  │   │ typed         │   │ RNG, aliasing,│   │ Expr → Rust src   │
 │  → AST      │   │ (shapes,      │   │ reduction     │   │ (routed to        │
 │             │   │  dtypes,      │   │ order         │   │  scirust-*        │
 │             │   │  units,       │   │               │   │  primitives)      │
 │             │   │  effects)     │   │               │   │                   │
 └─────────────┘   └──────────────┘   └───────────────┘   └──────────────────┘
                                                                    │
                                                                    ▼
                                                        ┌────────────────────────┐
                                                        │ 5. TRANSPILATION       │
                                                        │    ORACLE             │
                                                        │ runs source ⇄ Rust    │
                                                        │ on N inputs, compares │
                                                        │ under declared        │
                                                        │ tolerance             │
                                                        │ → accept / reject     │
                                                        │ → hash-chained report │
                                                        └────────────────────────┘
```

### Stage 1 — Front-ends
One parser per language producing a language-specific AST. We **never** target
"the whole language" but a **contractual scientific subset**,
statically analyzable (see §6). Each front-end explicitly declares what
it accepts and **refuses** (with diagnostics) what it does not understand —
rather than guessing.

### Stage 2 — Scientific IR (SIR)
A typed IR, independent of the source language, where each value carries:
- **shape** and **dtype** (f32/f64/i32/complex…),
- **optional physical unit** (m, s, kg… — useful in aero/space/energy),
- **effects**: purity, I/O, source of randomness (RNG), potential aliasing,
- **required reduction order** (for sums/products).

The SIR is the only place where numerical semantics are reasoned about. It is
also the stable boundary: adding a language = adding a front-end to the
SIR, without touching lowering or the oracle.

### Stage 3 — Analyses
Shape/type inference (essential from dynamic Python/MATLAB),
detection of randomness sources, aliasing detection, fixing the reduction
order. These analyses transform a "possibly dynamic" SIR into a
**statically emittable** SIR.

### Stage 4 — Lowering
Lowering of the SIR to `scirust-codetrans::Expr` (the **already-present**
emission backend, whose `Display` prints Rust), **routing each operation
to a verified `scirust-*` primitive** (see §4). The 20 `codetrans`
optimization rules (constant folding, DCE, CSE, LICM, strength
reduction, inlining, TCO) apply here.

### Stage 5 — Transpilation oracle (core of the trust)
Detailed in §8. Without a green oracle, no port is accepted.

---

## 3. Determinism and safety contract (by construction)

The transpiler does not add determinism *after the fact* — it **only emits
Rust that is deterministic by construction**, relying on guarantees already
held elsewhere in the repository:

- **Fixed reduction order.** Sums/products/means are emitted with a
  pinned order (already guaranteed by `scirust-core`: floating-point
  reductions independent of thread count, identical 64-bit fingerprint).
- **Seeded PRNG.** Any `np.random`, `rand`, `randn`, MATLAB `rand` source is
  mapped onto an explicitly seeded `SplitMix64` stream — never implicit
  system entropy.
- **Anti-aliasing.** The SIR tracks aliasing; emission produces safe
  `&` / `&mut` borrows, or inserts documented explicit copies.
  Goal: **zero unjustified `unsafe`**.
- **Declared tolerance.** Each port carries an explicit numerical tolerance
  (e.g. `rel ≤ 1e-12`); **bit-exact mode** is enabled when the target
  primitive allows it.
- **Optional embedded target.** For embedded AI / NVIDIA Jetson, the
  lowering can target `scirust-edge` / `scirust-embedded` (`no_std`,
  no allocation).

---

## 4. The target: routing operations to proven primitives

This is the central differentiator. **We do not re-derive the numerics in fresh
Rust; we route each source operation to a kernel already validated against a
reference oracle.** Excerpt of the mapping (to be completed as phases
progress):

| Source operation (NumPy/SciPy/MATLAB/BLAS…)         | Target `scirust-*` primitive |
|-----------------------------------------------------|-----------------------------|
| `np.linalg.solve` / MATLAB `\` / LU                 | `scirust-solvers` (LU, QR, Cholesky) |
| `np.linalg.svd` / `eig` / `qr`                      | `scirust-solvers` (Jacobi SVD, Householder+QL eig) |
| `scipy.sparse.linalg` (GMRES/BiCGSTAB)              | `scirust-solvers` (restarted GMRES, BiCGSTAB) |
| `np.fft` / `scipy.signal`                           | `scirust-signal` (FFT, windows, features) |
| `scipy.integrate.odeint` / MATLAB `ode45`           | `scirust-solvers::ode` (RK, autodiff) |
| Kalman/EKF (`filterpy`, MATLAB)                     | `scirust-estimation` (KF/EKF/UD square-root) |
| GNSS/INS, TDOA                                       | `scirust-nav` |
| PID/LQR/MPC                                          | `scirust-control` |
| optimization (`scipy.optimize`, `fmincon`)          | `scirust-solvers`, `scirust-evo` |
| PCA/ICA/K-Means/clustering                           | `scirust-multivariate`, `scirust-unsupervised` |
| neural networks / inference                     | `scirust-core`, `scirust-onnx`, `scirust-sciagent` |
| image processing / CNN / segmentation            | `scirust-vision` |
| rainflow / Palmgren-Miner (fatigue)                 | `scirust-fatigue` |
| power grids / WLS                            | `scirust-grid` |
| biosignals / ECG / dosing                            | `scirust-biomed` |

Where no primitive exists, the transpiler **does not guess**: the lowering
returns an explicit `unsupported ...` error. It never generates source
containing an executable `TODO`.

---

## 5. Coverage of the 15 sectors (honest matrix)

"Target vocabulary" = the verified Rust primitives to emit to.
`✅` = primitives already present; `🟡` = partial; `❌` = to be built.
Python/NumPy and MATLAB transpilation covers the subset verified by the tests
of the crate `scirust-transpiler`; this matrix describes the target primitives
beyond that subset, without presenting them as already lowered.

| # | Sector | Target vocabulary present? | Anchor crates |
|---|---------|------------------------------|------------------|
| 1 | Pharma / biotech (molecular simulation, genomics, PK, bio twins) | 🟡 | `scirust-biomed`, `scirust-solvers`, `scirust-multivariate`, `scirust-tn` |
| 2 | Industrial robotics (trajectory, SLAM, fusion, real-time, vision) | ✅ | `scirust-robotics`, `scirust-fusion`, `scirust-control`, `scirust-vision`, `scirust-estimation` |
| 3 | Aeronautics (guidance, nav, Kalman, flight control, simulation) | ✅ | `scirust-nav`, `scirust-estimation`, `scirust-control`, `scirust-func-safety` |
| 4 | Space (satellite nav, orbit, embedded control, telemetry) | 🟡 | `scirust-nav`, `scirust-estimation`, `scirust-embedded`, `scirust-signal` |
| 5 | Automotive (ADAS, lidar/radar fusion, vision, engine, battery) | ✅ | `scirust-fusion`, `scirust-vision`, `scirust-bms`, `scirust-func-safety`, `scirust-control` |
| 6 | Quantitative finance (pricing, Monte Carlo, risk, portfolio) | 🟡 | `scirust-solvers`, `scirust-evo`, `scirust-trader` |
| 7 | Energy (grids, smart grid, forecasting, wind, nuclear, hydro) | ✅ | `scirust-grid`, `scirust-sis`, `scirust-reliability`, `scirust-seasonal`, `scirust-water` |
| 8 | Geophysics (seismology, exploration, tomography, signals) | 🟡 | `scirust-signal`, `scirust-solvers`, `scirust-shm` |
| 9 | Meteorology (numerical forecasting, assimilation, climate) | 🟡 | `scirust-solvers`, `scirust-estimation` (assimilation ≈ filtering), `scirust-tn` |
| 10 | Embedded AI (preprocessing, ML pipelines, deterministic inference) | ✅ | `scirust-edge`, `scirust-embedded`, `scirust-core`, `scirust-onnx` |
| 11 | Chemical industry (reactors, CFD, thermo, process optimization) | 🟡 | `scirust-solvers`, `scirust-fab`, `scirust-sis`, `scirust-spc` |
| 12 | Medical imaging (CT/MRI reconstruction, segmentation, filtering) | 🟡 | `scirust-vision`, `scirust-signal`, `scirust-solvers` |
| 13 | Defense (simulation, radar, sonar, electronic warfare, fusion) | ✅ | `scirust-signal`, `scirust-fusion`, `scirust-estimation`, `scirust-nav` |
| 14 | Physics (Monte Carlo, quantum, astrophysics, particles) | 🟡 | `scirust-tn`, `scirust-solvers`, `scirust-tensor-*` |
| 15 | Industry 4.0 (digital twins, PdM, prod. optimization, vision) | ✅ | `scirust-pdm`, `scirust-mlops`, `scirust-opcua`, `scirust-mqtt`, `scirust-vision` |

**Reading.** For ~8 sectors out of 15, the target vocabulary is already there and
of oracle quality: the work is the input pipeline, not the numerics. For
the 🟡, a few primitives will need to be completed (CFD, tomographic
reconstruction, financial Monte Carlo…) in parallel with the front-ends.

---

## 6. Front-ends: strategy per language (increasing difficulty)

Priority order guided by (a) the volume of actually relevant scientific code
and (b) the tractability of static analysis.

| Language | Priority | Targeted subset | Difficulty | Parsing approach |
|--------|----------|--------------------|------------|------------------|
| **Python/NumPy** | 1 (MVP) | typed functions, NumPy/SciPy, no `eval`/reflection/monkeypatch | medium | AST via `rustpython-parser` (pure Rust) — to evaluate on license/deps |
| **MATLAB** | 2 | functions, matrices, 1-based indexing, implicit broadcasting | medium-high | dedicated parser (own grammar, copy-on-write) |
| **Fortran** (77/90+) | 3 | numerical routines, column-major arrays | high | dedicated parser; watch `COMMON`/`EQUIVALENCE` |
| **Julia** | 4 | already typed, multiple dispatch | medium | lesser interest (Julia is already fast) |
| **C/C++** | 5 | numerical subset | very high | `c2rust` pre-pass then refinement toward SciRust idioms |

Common principle: **explicit subset contract**, diagnosed refusal
outside scope, never a guessed translation.

Why this order: Python/MATLAB carry research prototyping
(pharma, robotics, finance, medical imaging — "developed in Python then
rewritten"); Fortran carries the legacy code (weather, geophysics, space,
physics — "millions of lines"); C/C++ is the hardest and least
profitable first (UB and templates make provable equivalence
expensive).

---

## 7. Role of the LLM / SLM: **assistant, never oracle**

The repository already has a specialized Rust SLM (`scirust-sciagent`) and an
MCP layer (`scirust-mcp`) drivable by an external LLM. Their place in the
transpiler:

- **Useful for**: filling semantic gaps (ambiguous idioms), proposing
  an operation mapping, **generating the oracle's test cases**, writing the
  port's documentation.
- **Never** a source of truth: **any** assisted output goes through the
  transpilation oracle (§8). This is the posture already held by `scirust-trader`
  ("certified predictions, LLM narration, proof-sealed decisions"),
  transposed to transpilation.

An LLM accelerates the *proposal*; the oracle decides the *acceptance*.

---

## 8. The transpilation oracle harness

This is the brick that turns a "transpiler" into a "*trusted* transpiler".

1. **Differential test.** Run the source in its **real runtime**
   (CPython+NumPy, Octave/MATLAB, `gfortran`, `clang`) and the emitted Rust on
   a corpus of inputs: seeded randomness + edge cases (0, NaN/Inf, singular
   matrices, empty arrays) + possibly property-based. Compare under
   the port's declared tolerance.
2. **Metamorphic test** when no reference runtime is available:
   verify invariants (linearity, energy/mass conservation, symmetry,
   monotonicity) that the port must preserve.
3. **Signed report** hash-chained, reusing the existing audit
   infrastructure (CCOS in `scirust-sciagent::ccos`, SHA-256 chain of
   `scirust-mcp`). Every acceptable port produces a replayable proof.
4. **CI gate.** No port merged without a green oracle; the tolerance and the
   corpus are part of the deliverable, not an afterthought.

---

## 9. Phased roadmap

- **Phase 0 — MVP (thinnest vertical slice). ✅ DELIVERED.** Python/NumPy
  subset → deterministic std-only Rust, **gated by a differential oracle
  against real NumPy**. Goal achieved: **the pipeline is proven end to
  end** (front-end → SIR → lowering → green oracle). Corpus delivered (7 cases,
  200 trials each): **RK4** integrator (scalar), **dot**, Euclidean **norm**,
  **weighted mean**, **cumsum** (loop + output array),
  **saxpy** (broadcast), elementwise **tanh**. *Honest gap vs the initial
  plan:* `np.linalg.solve` and `np.fft` are **not** yet delivered — they
  require routing to `scirust-solvers`/`scirust-signal`, planned for Phase 1.
- **Phase 1 — Broaden Python + route to verified kernels.** ✅ **in progress,
  already delivered:** `if`/`elif`/`else` flow control + `while`, and the first
  **routing `np.linalg.solve` → `scirust-solvers`** (verified LU resolution,
  oracle case compiled via cargo). ⏳ **remaining:** `np.fft` → `scirust-signal`,
  general 2-D arrays, multiple functions. Sectors unlocked by the
  routing: robotics, finance, imaging.
- **Phase 2 — MATLAB + tuples/SVD.** ✅ **delivered:** (1) second front-end (dedicated lexer +
  parser + lowering) on the **same** SIR + emitter as Python, proven against
  **real Octave** (differential oracle, 9 cases × 200 trials) — **1-based**
  indexing (`a(i)` → `a[i-1]`), inclusive `for` ranges (`1:n` → `1..n+1`),
  elementwise operators `.*`/`./`/`.^` vs scalar `*`/`/`, return via
  **output variable**, hoisting of locals assigned in branches (`if`/`else`)
  validated by Rust's definite-assignment analysis; (2) first **multi-output
  kernel**: `U, S, Vh = np.linalg.svd(A)` (tuple destructuring +
  `np.diag`) → verified fine SVD from `scirust-solvers`, proven against NumPy by
  the singular values *and* the `U·diag(S)·Vᵀ` reconstruction; (3) **broadened
  Python**: **user** function calls (one `def` calling another
  defined earlier, with annotation-free inter-function type inference) and
  **literal lists** `[a, b, c]` → `Vec<f64>`. ⏳ **remaining:** matrix routing
  from MATLAB, `zeros(m,n)` 2-D, scalar↔array broadcasting without `.*`,
  other decompositions (`qr`, `eig`), general tuple returns. Target sectors:
  aero, automotive, control, imaging.
- **Phase 3 — Fortran.** Legacy numerical routines; sectors: weather,
  geophysics, space, physics.
- **Phase 4 — C/C++.** Numerical subset via `c2rust` pre-pass.

Each phase delivers: subset contract + oracle corpus + matrix of the
sectors actually unlocked.

---

## 9-bis. Implementation status (measured, not claimed)

| Pipeline brick (§2)                     | Status | Location |
|---------------------------------------------|--------|-------------|
| Python/NumPy front-end (lexer + parser)     | ✅ delivered | `scirust-transpiler/src/front_python/` |
| Typed Scientific IR (scalar/array/int)  | ✅ delivered | `scirust-transpiler/src/sir.rs` |
| Lowering + type/shape inference        | ✅ delivered | `scirust-transpiler/src/lower.rs` |
| Deterministic Rust emission (pinned order)    | ✅ delivered | `scirust-transpiler/src/emit.rs` |
| Differential oracle against real NumPy **and real Octave** | ✅ delivered | `scirust-transpiler/examples/oracle.rs` |
| Unit tests (CI gate, without Python/Octave) | ✅ delivered | `scirust-transpiler/src/lib.rs` (97 tests) |
| `if`/`elif`/`else` flow control + comparisons | ✅ delivered (Phase 1) | `front_python/` + `sir.rs` + `emit.rs` |
| `while` loops (iterative algorithms)     | ✅ delivered (Phase 1) | `front_python/` + `sir.rs` + `emit.rs` |
| Routing `np.linalg.solve`/`det`/`eigvalsh`/`inv` + `A @ b` (matvec) → `scirust-solvers` (2-D matrix return for `inv`) | ✅ delivered (Phase 1) | `sir.rs` (`LinSolve`, `Det`, `Eigvalsh`, `Matvec`, `Inv`, `Ty::MatrixVal`) + `emit.rs` |
| Routing `np.fft.fft`/`rfft`/`ifft` → `scirust-signal` (+ complex type) | ✅ delivered (Phase 1) | `sir.rs` (`Ty::ComplexArray`, `Fft`, `Rfft`, `Ifft`, `ComplexAbs`) + `emit.rs` |
| **Multi-output tuples + `np.linalg.svd`/`qr`** (destructuring `U, S, Vh = …` / `Q, R = …`, `np.diag`) → `scirust-solvers` | ✅ delivered (Phase 2) | `sir.rs` (`TupleExpr`, `SirStmt::LetTuple`, `SirExpr::Diag`) + `emit.rs` |
| **General tuple returns** (`return a, b`, scalar elements) | ✅ delivered (Phase 2) | `sir.rs` (`RetTy`, `SirStmt::ReturnTuple`) + `emit.rs` |
| **User function calls** (composition, inter-function type inference) + **literal lists** | ✅ delivered (Phase 2) | `lower.rs` (`FuncSig`/`Sigs`) + `sir.rs` (`SirExpr::UserCall`, `ArrayLit`) |
| General 2-D arrays                       | ⏳ Phase 1 | — |
| **MATLAB/Octave front-end** (lexer + parser + lowering, proven vs Octave; **multi-output `[a,b]=f(…)`**, math/reduction intrinsics aligned with Python, **linear algebra `det`/`inv`/`\`/`eig` + matrix product `A*b`/`A*B` → `scirust-solvers`**, **`fft`/`ifft` → `scirust-signal`** (complex), `norm`/`dot`, `.^`, vector→vector, `linspace`) | ✅ delivered (Phase 2) | `scirust-transpiler/src/front_matlab/` + `lower_matlab.rs` |
| Fortran / C++ front-ends                     | ⏳ Phases 3-4 | — |

**Oracle result (reproducible).** 140 cases in total: 43 Python proven
against **real NumPy**, 97 MATLAB proven against **real Octave** (200 trials each).

```
$ cargo run -p scirust-transpiler --example oracle
tolerance: |Δ| ≤ 1e-7 + 1e-9·|ref|, 200 trials/case
  Python cases → NumPy · MATLAB cases → Octave
  ✓ rk4_step / dot / norm / weighted_mean / cumsum / saxpy / tanh_activation
  ✓ relu / clamp / sign            (if/elif/else — Phase 1)
  ✓ newton_sqrt / newton_conv      (while — Phase 1)
  ✓ solve/det/eigvalsh/inv/A@b/A@B/A.T (routed to scirust-solvers, cargo-compiled — Phase 1)
  ✓ fft.fft / rfft / ifft / abs(fft) (routed to scirust-signal, complex type — Phase 1)
  ✓ svd singular values + reconstruction U@diag(S)@Vh (tuple unpack → scirust-solvers — Phase 2)
  ✓ qr reconstruction Q@R (tuple unpack → scirust-solvers — Phase 2)
  ✓ user calls: sumsq / sumdbl / chain (function composition, hint-free inference — Phase 2)
  ✓ list literal: weighted average (Python list → Vec — Phase 2)
  ✓ log/log10 / floor/ceil / sinh/cosh/arctan / max-min-mean / prod (broadened vocabulary — Phase 2)
  ✓ sin/cos/abs / exp / ** / ones  (full intrinsic & operator coverage)
  ✓ M: norm2 / dot / relu / sign / clamp / poly / mysum / newton / ew_scale (MATLAB → Octave — Phase 2)
  ✓ M: sumdiff / normstats / stats3 [a,b]=f(…) + mathx (MATLAB multi-output + log/floor/atan/min/max/mean — Phase 2)
  ✓ M: det(A) / inv(A) / A \ b (MATLAB linear algebra → scirust-solvers — Phase 2)
  ✓ M: norm(v) / dot(a,b) / eig(A) (MATLAB vector & symmetric-eigen intrinsics — Phase 2)
  ✓ M: round / fix / mod / rem / sign (MATLAB rounding & modular scalar functions — Phase 2)
  ✓ M: atan2(y,x) / hypot(a,b) (MATLAB two-argument scalar math — Phase 2)
  ✓ M: max(a,b) / min(a,b) (2-arg) / power(a,b) (MATLAB binary max/min & power — Phase 2)
  ✓ M: v.^2 / a.^b / 2.^v (MATLAB elementwise power `.^` on arrays, broadcast — Phase 2)
  ✓ M: cumsum(v) / diff(v) / sort(v) (MATLAB vector→vector builtins — Phase 2)
  ✓ M: cumprod / cummax / cummin / flip (more MATLAB vector→vector builtins — Phase 2)
  ✓ M: var(v) / std(v) / median(v) (MATLAB reduction statistics, N-1 — Phase 2)
  ✓ M: linspace(a,b,n) (MATLAB vector constructor, exact endpoints — Phase 2)
  ✓ M: A*(A\b) / A*inv(A) (MATLAB matrix product `*` → matvec/matmul — Phase 2)
  ✓ M: A' / A'*A (MATLAB transpose operator `'`, Gram matrix — Phase 2)
  ✓ M: trace(A) / cross(a,b) (MATLAB diagonal sum + 3-vector cross product — Phase 2)
  ✓ M: diag(A'*A) extract / diag(cumsum(v)) construct / trapz(v) (overloaded diag + integration — Phase 2)
  ✓ M: kron(a,b) / cumtrapz(v) (MATLAB Kronecker product + cumulative integral — Phase 2)
  ✓ M: conv(a,b) / polyval(p,x) (MATLAB convolution + Horner polynomial eval — Phase 2)
  ✓ M: expm1(x) / log1p(v) (MATLAB accurate-near-zero exp/log — Phase 2)
  ✓ M: atan2/hypot/max/min elementwise & broadcast on arrays (Phase 2)
  ✓ M: deg2rad / rad2deg + sign elementwise (MATLAB angle conversion + vector sign — Phase 2)
  ✓ M: mod(cumsum(v),3) / rem(cumsum(v),3) (MATLAB elementwise modular, broadcast — Phase 2)
  ✓ M: logspace(a,b,6) (MATLAB logarithmic vector constructor, 10^a..10^b — Phase 2)
  ✓ M: norm(v,1) / norm(v,p) (MATLAB general finite vector p-norm — Phase 2)
  ✓ M: tan / asin / acos (scalar & elementwise, MATLAB elementary/inverse trig — Phase 2)
  ✓ M: log2 / asinh / acosh / atanh (scalar & elementwise, base-2 log + inverse hyperbolic — Phase 2)
  ✓ M: gradient(v) (MATLAB unit-spacing numerical gradient, centred + one-sided — Phase 2)
  ✓ M: circshift(v, ±k) (MATLAB circular shift, modular reindex, both signs — Phase 2)
  ✓ M: sind / cosd / tand (scalar & elementwise, MATLAB degree-argument trig — Phase 2)
  ✓ M: asind / acosd / atand (scalar & elementwise, MATLAB inverse degree trig — Phase 2)
  ✓ M: sec / csc / cot (scalar & elementwise, MATLAB reciprocal trig — Phase 2)
  ✓ M: fft / abs(fft) / ifft(fft) (MATLAB FFT routed to scirust-signal, complex — Phase 2)
  ✓ M: fftshift / ifftshift / fftshift(abs(fft)) (MATLAB spectrum centring, floor/ceil — Phase 2)
  ✓ M: range(v) (MATLAB max−min spread reduction — Phase 2)
  ✓ tuple returns: addsub / minmax / stats3 (`return a, b` — Phase 2)
  ORACLE GREEN — 140/140 cases match their reference runtime within tolerance
```

A single entry point runs the whole suite (unit tests + oracle) with
a report and a non-zero exit code at the slightest divergence:

```
$ ./scripts/test_transpiler.sh
```

Non-vacuity check: injecting a wrong operator into the emitter
(`*` → `+`) turns several Python cases RED; on the MATLAB side, breaking
the 1-based indexing (`i-1` → `i-2`) makes `mysum` fail and turns the oracle
RED — the gate really bites on both sides.

> **`codetrans` reuse note.** §10 targets `codetrans::Expr` as the
> emission backend. In practice its `Function` node carries **untyped**
> parameters (`Vec<String>`), which does not allow emitting typed Rust
> signatures (`&[f64]` vs `f64`) that *compile*. The MVP therefore uses a
> dedicated typed emitter; unifying with `codetrans` (by extending its `Function`
> with parameter types) remains future work.

---

## 10. Concrete reuse of the existing (code anchor points)

| Need | Reuse | File |
|--------|-----------|---------|
| Rust emission backend | `codetrans::Expr` + pretty-printer | `scirust-codetrans/src/lib.rs` (`Display for Expr`, l.249) |
| Optimization passes | 20 rules (`optimization_rules`, CSE, DCE, LICM) | `scirust-codetrans/src/lib.rs` (l.1958+) |
| Target vocabulary | solvers, signal, estimation, core, vision… | `scirust-*` crates (§4-5) |
| Proof / audit | CCOS + SHA-256 chain | `scirust-sciagent::ccos`, `scirust-mcp` |
| Agent orchestration | expose the transpiler as an MCP tool | `scirust-mcp` |
| Floating-point determinism | pinned-order reductions, fingerprint | `scirust-core` |

A new crate `scirust-transpiler` (front-ends + SIR + lowering + oracle)
would sit **on top of** these bricks, without duplicating them.

---

## 11. Honest boundary — what will NOT be delivered (short term)

Faithful to the repository's doctrine, the non-goals are stated upfront:

- **No "any language / any program".** Statically analyzable scientific
  subsets only. A Python `eval`, reflection, or
  monkeypatch → **diagnosed refusal**, no guessing.
- **No guaranteed *cross-language* bit-exact reproducibility.** NumPy/BLAS
  operation order is unspecified; we guarantee (a) a **declared tolerance**
  source ⇄ Rust and (b) **internal Rust bit-exactness**
  (independent of thread count, via `scirust-core`). Claiming bit-for-bit
  equality with CPython would be dishonest.
- **No translation of C/C++ UB.** Undefined behavior → reported, never
  "interpreted".
- **Performance comes from routing, not transpilation magic.** The
  emitted Rust targets correctness + determinism first; speed comes from the
  targeted SIMD/GPU `scirust-*` kernels, measured, not assumed.

---

## 12. Acceptance criteria — "how to be sure"

A port is deemed deliverable if and only if:

1. the **oracle is green** on the declared corpus (differential and/or
   metamorphic);
2. the **declared tolerance** is respected over the whole corpus;
3. **internal bit-exactness** is verified (identical fingerprint
   1/2/4/8 threads);
4. **zero unjustified `unsafe`**; aliasing traced;
5. a **signed report** hash-chained is produced and replayable;
6. the **covered subset** is documented, as well as what was refused.

As long as these six gates are not tooled, the honest answer to "does SciRust
know how to transpile my code?" remains **"not yet automatically — here is
the plan and the targeted guarantees"**, not a marketing "yes".

---

*See also: `docs/DOMAIN_ROADMAP.md` (regulated sectors), `docs/ARCHITECTURE.md`
(runtime architecture), `scirust-codetrans` (emission backend),
`scirust-mcp` (agent orchestration + audit).*
