# SciRust algogen IR V2 architecture

Status: implemented contract for `scirust_algogen::tensor::v2`.

Baseline: master commit `d3e435d690e1a2a95e5b4379c681bdb8954cfbb0`.
V1 remains available and byte-stable under `scirust_algogen::tensor`; V2 is an
additive namespace with a V1-to-V2 adapter.

## 1. Scope

IR V2 is a bounded semantic IR for scientific algorithm discovery. It can
represent typed scalar/tensor expressions, reductions, masks, shape algebra,
linear algebra, multiple outputs, and fixed-trip streaming recurrences. It is
not a general-purpose language:

- no arbitrary branches, calls, recursion, pointers, mutation, aliases,
  exceptions, I/O, host callbacks, or dynamic code;
- no loop except the single statically counted recurrence section;
- no symbolic or data-dependent shapes;
- no backend, device, vendor, or scheduling annotations;
- every resource-bearing dimension is checked before execution.

The key design principle is closed semantics plus an explicit trust boundary,
not a large opcode count.

## 2. Implemented module boundary

| Module | Responsibility |
|---|---|
| `types.rs` | `DType`, `ValueType`, `ScalarValue`, static broadcast helpers |
| `ir.rs` | section-scoped refs, operations, `ResearchProgram` |
| `semantics.rs` | rewrite-equivalence regimes |
| `verify.rs` | type/shape/state/resource verification and active maps |
| `interpret.rs` | safe reference execution and runtime float policy |
| `canonical.rs` | versioned canonical bytes, FNV hint, SHA-256 label |
| `simplify.rs` | bounded, ordered, regime-gated rewrite framework |
| `serialization.rs` | explicit versioned JSON envelope and rejection boundary |
| `range.rs` | conservative constant/sign/finiteness/interval facts |
| `cost.rs` | deterministic structural cost and phase liveness |
| `generate.rs` | typed grammar, profiles, seeded generation and rejection stats |
| `evolve.rs` | verified mutation and recurrence-context-aware crossover |
| `search.rs` | counterexamples, multiobjective fitness, Pareto archive, replay |
| `reference.rs` | executable representation fixtures and ADA A1–A10 evidence |
| `compat.rs` | frozen V1 to V2 lift |

Search code does not call `reference.rs`; examples cannot become hidden
generator templates.

## 3. Value and shape model

`ValueType` is a pair of `DType` and a concrete row-major `Vec<usize>` shape.
An empty shape is a rank-zero scalar. Implemented dtypes are true IEEE binary32,
true IEEE binary64, and Boolean. The interpreter uses an `f64` payload carrier,
but casts every binary32 operation to `f32` before computation and back after
rounding; the carrier is storage, not fake f64 evaluation. Boolean payloads
must be exact `+0.0` or `+1.0` bit patterns.

`f16`, `bf16`, integer, and index dtypes are deliberately absent. Adding a
name without implementing its rounding and storage semantics would be false
precision. Integer/index types are the prerequisite for dynamic gather/argmax.

Shapes are static. All element products use checked arithmetic. Broadcasting
uses NumPy-style trailing-axis alignment: equal dimensions or one dimension
equal to one; an absent leading dimension broadcasts. A zero dimension only
broadcasts with zero or one. Implicit broadcast is part of the inferred type
and can be disabled in the generation grammar. `BroadcastTo` is the explicit
form and participates in canonical identity.

Implemented shape/index operations are `Reshape`, `Squeeze`, `Unsqueeze`,
`Transpose` with an explicit permutation, `BroadcastTo`, `Concat`, and
statically bounded `Narrow`. Dynamic gather/scatter is not emulated with float
indices.

## 4. Program and recurrence structure

`ResearchProgram` has three straight-line SSA-flavoured sections:

```text
init once
  state[0..S] := init_state bindings

repeat exactly steps times
  next_state[0..S] := step(previous state, current items)

finalize once
  return outputs in declared order
```

The program declares outer input types, per-step item types, state-component
types, and the static trip count. References are scoped:

| Reference | Legal section |
|---|---|
| `Input(i)` | init, finalize |
| `Item(i)` | step |
| `StatePrev(i)` | step |
| `StateFinal(i)` | finalize |
| `Local(i)` | same section, strictly earlier definition |

State is a typed tuple, not one accumulator. This supports `(m,l)` online
softmax, `(count,mean,M2)` Welford, `(sum,compensation)` Kahan-like updates, and
`(m,l,o)` attention-style folds using ordinary ops. Each next-state binding
must exactly equal its declared component type. The scan is a fold: V2 does
not materialize per-step results and has no optional scan-output sequence.
This prevents an implicit `steps × tensor` allocation. Zero-length recurrence
is rejected; straight-line programs use no state and `steps == 0`.

## 5. Operation inventory

- Constants: typed scalar `Const` nodes.
- Arithmetic: `Add`, `Sub`, `Mul`, `Div`, `Neg`, `MulAdd`, `Pow`.
- Unary scientific: `Abs`, `Exp`, `Exp2`, `Expm1`, `Log`, `Log2`, `Log1p`,
  `Sqrt`, `Rsqrt`, `Sin`, `Cos`, `Tanh`.
- Extrema and selection: `Min`, `Max`, `Clamp`, `Select`.
- Comparisons: `Eq`, `Ne`, `Lt`, `Le`, `Gt`, `Ge`, producing Boolean values.
- Boolean: `And`, `Or`, `Not`.
- Reductions: sum, product, minimum, maximum, mean; an explicit optional axis,
  with `None` meaning all axes and `keep_dim = false`.
- Linear algebra: dot, matrix-vector, vector-matrix, matrix-matrix, one-leading-
  batch matrix multiplication, and outer product.
- Shape/static indexing: the operations listed in §3.

There is no `Softmax`, `OnlineSoftmax`, `Attention`, Entmax, or ADA opcode.
There is no implicit FMA contraction: `MulAdd` is explicitly fused and
`Mul` followed by `Add` remains two roundings.

`Erf` is deferred because Rust `std` provides no native contract and shipping
an approximation would silently choose research semantics. General `Pow` is
implemented; ArgMin/ArgMax await an index dtype.

## 6. Verifier as the trust boundary

`verify_program` is required before interpretation, cost, range analysis,
serialization, mutation/crossover output, and archival. Structured errors
cover:

- section legality, operand existence, causality, and operator arity;
- dtype/operator compatibility and exact/broadcast shape rules;
- reduction axes and forbidden empty max/min/mean domains;
- reshape element count, squeeze axes, permutations, concat and narrow ranges;
- state/step co-occurrence, state binding counts/types, output bounds/uniqueness;
- input/item/state signature rank and element counts;
- maximum inputs, items/step, nodes/section, total nodes, rank, elements/value,
  total register elements, signature elements, conservative host bytes, steps,
  state components, and outputs;
- NaN constants and, under `FiniteOnly`, all non-finite constants.

The result contains inferred types, active-node maps, output types, total
register elements/bytes, signature elements, and conservative resident bytes.
Malformed programs fail before allocation-heavy kernels. Interpreter root
lookups still return structured defensive errors rather than relying on panic.

## 7. Interpreter and effects

The interpreter is a pure function of program, input tensors, item tensors,
execution policy, and limits. It contains no unsafe code, FFI, threads, files,
network, shell, dynamic compilation, callbacks, or host object handles.

It evaluates only verified active nodes. Tensor public fields are revalidated
at entry so serde or struct literals cannot bypass length, binary32 carrier, or
Boolean encoding rules. Accumulation order is ascending row-major flat index;
matrix products use ascending inner index. All execution is deterministic for
a fixed Rust target/toolchain numerical library. Transcendental bit patterns
are not claimed portable across every libm implementation.

The rewrite semantic regime and runtime float policy are related but distinct;
the exact contract is in `SCIRUST_ALGOGEN_IR_V2_NUMERICAL_SEMANTICS.md`.

## 8. Canonicalization, identity, and serialization

The bounded rewrite pipeline uses a fixed order and at most 16 passes. It
performs finite-result constant folding, explicitly classified identities,
regime-gated commutative operand normalization, exact CSE, root rebinding,
stable compaction/value renumbering, and DCE. Every application records pass
number and stable rule id. It never reassociates arithmetic, distributes,
contracts to FMA, or applies `x*0`, `x-x`, or `x/x` rules.

Canonical bytes are authoritative structural identity and begin with:

1. domain-separation magic;
2. canonical format version;
3. IR version;
4. canonicalization version;
5. numerical semantic regime.

They then encode signatures, sections, refs, shapes, and raw float bits using
fixed-width little-endian fields. FNV-128 is only a lookup hint. SHA-256 is a
compact archival label. Neither hash proves equality; exact bytes are retained
and compared by the Pareto archive.

JSON is transport, never identity. The serialization envelope records
serialization, IR, canonicalization, and semantic-regime versions. A bare V2
program or any mismatch is rejected; V1 artifacts are not silently reinterpreted.

## 9. Analyses and search

The structural cost report separates operator classes, logical FLOPs, data
reads/writes, logical bytes, state/output/intermediate footprints, step and
finalize depth, and phase-resolved peak live values/elements/bytes. Step cost is
reported per iteration and scaled exactly once for total cost. No wall-clock
measurement affects fitness.

Range analysis is deliberately lightweight: constant, sign, finiteness, and
simple interval facts. Recurrence propagation reaches a bounded abstract fixed
point (at most eight passes), not a theorem proof.

Generation enumerates verifier-typed proposals from a scoped value pool, then
uses the stable SplitMix64 stream. Its serializable grammar controls operator
classes/dtypes, rank, values, operations, depth, reductions, transcendentals,
expensive ops, linalg, shape ops, comparisons, broadcasting, indexing,
recurrence, state count/trip count, and total proposal enumeration. It records
deterministic rejection statistics. Profiles constrain curricula; none
contains an algorithm AST.

Mutation enumerates bounded same-type operand replacements, same-signature
operator replacements, constant changes, reduction-axis changes, and compatible
state/output rebinding, then returns only a verified child. Crossover requires
identical regime/input/item/state/trip/output contexts and exchanges a complete
section together with its binding vector; it never splices references across
state scopes.

Fitness keeps counterexample error separate from cost. The archive maintains a
deterministic nondominated set across error, logical work, memory, state,
depth, reductions, and expensive operations. Exact canonical bytes protect
against digest collisions. Experiments record source revision, all versions,
grammar/profile, seed, limits, dataset id/digest, candidate seed/bytes/program,
fitness, generation rejection, archive decision/final position, and diagnostics.
Replay reruns the logical pipeline and requires complete archive equality.

## 10. Compatibility and boundaries

V1 public types, canonical bytes, archives, and experiments are unchanged.
`compat::from_v1` lifts a verified V1 expression into a V2 straight-line
program; execution equality is tested. There is intentionally no V2-to-V1
downcast for constructs V1 cannot express.

`scirust-symbolic` remains separate. Its real-algebra expression rules are not
automatically valid for IEEE programs. A future import/export adapter must tag
assumptions and restrict symbolic rewrites to the experimental real-algebraic
regime or independently prove stronger preconditions.

Autodiff is not part of V2. A future differentiability analysis/pass can
consume verified pure sections, but derivative semantics—especially at
min/max/select/masks—must be explicit. CPU/SIMD/GPU/WGSL/SPIR-V/CUDA lowerings
are also separate consumers of verified semantic IR and cannot alter canonical
identity.

## 11. Deliberate limitations

- static dense tensors only; no symbolic dimensions or ragged values;
- no dynamic gather, scatter, segment reduction, or index dtype;
- no scan-output materialization, early exit, or convergence termination;
- no f16/bf16, casts, mixed-precision accumulation, complex values, or `Erf`;
- no formal equivalence proof: bounded differential/metamorphic tests are
  evidence only;
- the generator is a controlled foundation, not an optimizer guaranteed to
  discover large attention algorithms in this phase.

These constraints keep V2 a research IR rather than an unrestricted compiler
or programming language.
