# SciRust algogen IR-V2 — architecture specification

Status: normative for `scirust_algogen::tensor::v2`.
Baseline audited: `docs/SCIRUST_ALGOGEN_IR_V2_AUDIT.md`.
V1 (`scirust_algogen::tensor`) is frozen byte-stable; V2 is additive.

## 1. Goals

A deterministic, verifiable, auditable IR for **automated scientific algorithm
discovery**: able to represent elementwise arithmetic, reductions, broadcasting
shape algebra, linear algebra, Boolean masks, multi-output programs, and
statically bounded recurrences (streaming/online algorithms), while remaining
constrained enough for generation, mutation, canonicalization, cost modeling,
archival and exact replay.

Non-goals: arbitrary control flow, unbounded loops, pointer semantics, unsafe
code, vendor kernels, autodiff (see §13), GPU lowering (see §14).

## 2. Module layout

```
tensor/v2/
  mod.rs         public surface + version constants
  types.rs       DType, ValueType, ScalarValue, shape algebra helpers
  ir.rs          Op, Node, Section, ResearchProgram (the semantic IR)
  verify.rs      VerificationLimits, ProgramError, VerifiedProgram
  interpret.rs   ExecutionPolicy, ExecutionError, execute_program
  canonical.rs   canonical bytes, FNV fingerprint, SHA-256 program digest
  simplify.rs    rewrite framework: DCE, CSE, folding, identities, fixpoint
  cost.rs        structural CostReport + liveness/peak-live analysis
  range.rs       conservative constant/sign/range analysis
  generate.rs    Grammar, GrammarProfile, valid-by-construction generator
  mutate.rs      typed mutation families
  crossover.rs   type-aware crossover
  fitness.rs     FitnessReport, counterexamples, Pareto objectives
  population.rs  deterministic evolution over V2 programs
  dataset.rs     cases with inputs + per-step items + multi-output expected
  problem.rs     serialisable problems, benchmarks, discovery smoke problem
  archive.rs     versioned ExperimentArchive + verification + replay
  compat.rs      V1 -> V2 adapter
  reference_tests.rs   hand-built known-algorithm representations
  ada_tests.rs   ADA A1..A10 capability tests
```

## 3. Value model

- `DType`: `F32`, `F64`, `Bool`. Extension points documented for `F16`, `Bf16`,
  integer/index dtypes; **not** implemented (no fake low-precision arithmetic:
  labelling f32 data bf16 is forbidden). Programs mixing incompatible dtypes
  fail verification.
- Shape: `Vec<usize>` (rank ≤ `max_rank`, default 8). Rank-0 = scalar.
- `ValueType { dtype, shape }`.
- `ScalarValue { F32(f32) | F64(f64) | Bool(bool) }`; non-finite float
  constants are **rejected by the verifier** (illegal-constant rule).
- Symbolic dimensions: deliberately deferred. Concrete shapes keep static
  inference total and deterministic; see §12 (remaining gaps).

### Broadcasting

NumPy-style right-aligned broadcasting, restricted: dimensions pair equal, or
one side is `1`, or absent. Zero-sized dims broadcast only against `0`/`1`.
Used by: arithmetic, comparisons, min/max/clamp/pow, `Select` mask.
`allow_broadcast` can be disabled per grammar profile.

### Shape ops (all statically checked)

`Reshape(target)` (element-count preserving), `Squeeze(axis)` (axis must be 1),
`Unsqueeze(axis)`, `Transpose(perm)` (validated permutation),
`BroadcastTo(target)`, `Concat(lhs, rhs, axis)`,
`Narrow { axis, start, len }` (bounds-checked statically).

## 4. Program structure: three sections and a bounded scan

```rust
pub struct ResearchProgram {
    pub inputs: Vec<ValueType>,   // outer inputs
    pub items: Vec<ValueType>,    // per-step incoming values (stream signature)
    pub state: Vec<ValueType>,    // declared recurrence state components
    pub steps: u32,               // static trip count
    pub init: Section,  pub init_state: Vec<ValueId>,
    pub step: Section,  pub next_state: Vec<ValueId>,
    pub finalize: Section, pub outputs: Vec<ValueId>,
}
```

Execution: `init` once → `state := init_state`;
repeat exactly `steps` times with item vector `items[k]`:
`state := next_state(step(state, items[k]))`; then `finalize` once and read
`outputs`. A program with `state.is_empty()` must have `steps == 0` and an
empty `init` section — i.e. plain straight-line expressions are the degenerate
case. `steps >= 1` requires a non-empty state. There are **no** loops besides
this statically counted scan; the step body cannot recurse, branch, or escape
its declared state signature.

References are section-scoped:

| Ref | Allowed in | Meaning |
|---|---|---|
| `Input(i)` | init, finalize | outer input |
| `Local(j)` | all | earlier node of the same section |
| `Const` | all | `Op::Const(ScalarValue)` node |
| `Item(k)` | step only | k-th incoming value of the current step |
| `StatePrev(s)` | step only | previous value of state slot `s` |
| `StateFinal(s)` | finalize only | final value of state slot `s` |

Every value is defined exactly once (SSA-flavoured); use-before-definition and
cross-section leaks are verifier errors. Multiple outputs and multiple updated
state components are first-class. Dead values unreachable from
`init_state`/`next_state`/`outputs` are removable by DCE.

## 5. Operator inventory (categories)

- Arithmetic (broadcast, float): `Add Sub Mul Div Neg MulAdd` (fused `a*b+c`),
  `Pow`.
- Constants: `Const(ScalarValue)` (zero/one are ordinary constants; CSE makes
  duplicates free).
- Elementary (unary float): `Abs Exp Exp2 Expm1 Log Log2 Log1p Sqrt Rsqrt Sin
  Cos Tanh`.
- Extrema: `Min Max` (broadcast binary), `Clamp(x, lo, hi)`.
- Comparisons (broadcast, float → bool): `Eq Ne Lt Le Gt Ge`.
- Boolean logic: `And Or Not`.
- Selection: `Select(mask, if_true, if_false)` — mask broadcast over branches.
- Reductions (axis-explicit): `ReduceSum ReduceProd ReduceMax ReduceMin
  ReduceMean`, `axis: None` (full → rank 0) or `Some(axis)` (axis removed,
  keep-dim = false). Empty-axis rules: sum/prod defined (0/1);
  max/min/mean over an empty axis are statically rejected (non-finite by
  construction).
- Linear algebra: `Dot`, `MatVec`, `VecMat`, `MatMul`, `BatchedMatMul`
  (leading batch axis), `Outer`.
- No high-level `Softmax`, no `OnlineSoftmax`, no argmax/argmin (needs index
  dtypes — deferred), no gather/scatter (deferred with index dtypes; see
  NUMERICAL_SEMANTICS §7 and ARCHITECTURE §12).

Interpreter semantics are defined directly in safe Rust (checked iteration,
no einsum-backend delegation, no panics on zero-sized dims).

## 6. Verifier (trust boundary)

Statically validates, in one pass per section, then cross-section:

arity/causality/section-legality of every ref; input/item/state-slot bounds;
dtype and shape compatibility (with declared broadcasting rules); reduction
axes; reshape/squeeze/transpose/concat/narrow legality; resource limits
(`max_nodes_per_section`, `max_nodes_total`, `max_rank`,
`max_elements_per_tensor`, `max_total_register_elements`, `max_steps`,
`max_state_components`, `max_outputs`); illegal non-finite constants;
recurrence signature consistency (`init_state`/`next_state` counts and types,
`steps`/state co-occurrence); output existence and uniqueness.

Output: `VerifiedProgram` with per-section inferred types, active maps, output
types, totals. Structured error enum (`ProgramError`, one variant per rule,
each with a negative test).

## 7. Interpreter

`execute_program(program, inputs, items, policy, limits)`. Safe, deterministic,
allocation-checked, no FFI/shell/threads. `ExecutionPolicy { float_policy:
RejectNonFinite (default) | AllowNonFinite }`. Under the default, the first
non-finite intermediate aborts evaluation with a precise error; datasets and
constants are finite by construction. `AllowNonFinite` exists for research
regimes and never participates in default discovery. Dead sections/steps are
skipped exactly as in V1 (liveness is part of the numerical contract).

## 8. Numerical semantics and regimes

Authoritative contract lives in `SCIRUST_ALGOGEN_IR_V2_NUMERICAL_SEMANTICS.md`.
Summary:

- Operator semantics = IEEE-754 as implemented by the corresponding Rust `std`
  operator/function for the operand dtype; the interpreter adds the
  policy-level non-finite gate. Transcendentals inherit toolchain libm ULP
  behaviour; identity/canonicalization never depends on evaluated values.
- Canonicalization operates under the **canonical numeric regime**: IEEE-754
  values restricted to the finite domain, with **signed-zero insensitivity**
  (rewrites may merge value flows that differ only in ±0 propagation). Every
  rewrite states its validity domain; nothing assumes associativity of `+`/`*`;
  `Sub(x,x)→0`, `Mul(x,0)→0` are **rejected** (NaN/±Inf hazards).

## 9. Canonicalization and identity

Deterministic pass pipeline (fixed order, fixed-point with hard step budget):
DCE → CSE (value numbering) → constant folding (only when the folded result is
exactly representable and finite) → identity rewrites → commutative operand
normalisation (Add, Mul, Min, Max, And, Or — sound under IEEE commutativity;
never associative reassociation) → compaction/renumbering → duplicate-output
removal. Idempotent; every rule unit-tested for semantic preservation
(bit-for-bit under the default policy on the rewrite's declared domain).

Identity: `canonical_bytes` (versioned magic `SCIRUST-RIR2`, format version,
stable opcode tags, fixed-width LE integers, raw float bits) is authoritative;
`program_fingerprint` (FNV-1a 128) is a collidable hint; `program_digest`
(SHA-256 hex) is the archival/research identifier. Collisions never imply
equality anywhere in the stack.

Archive records `ir_version`, `canonical_format_version`, and the numerical
policy tag; a change in any of them cannot silently preserve experiment
identity.

## 10. Cost model, liveness, ranges

`CostReport` (structural/logical, saturating, wall-clock-free): per-class op
counts (add/sub, mul, fma, div, pow, exp-family, log-family, sqrt-family,
trig, minmax, compare, select, reductions, linear-algebra), logical scalar
FLOPs with documented weights, elements read/written, temporary elements,
peak live elements (phase-resolved; state counted during the scan),
state elements, steps, per-step logical FLOPs, dependency depth per section
(critical-path proxy). Nothing here claims hardware time.

`range.rs`: conservative, intentionally incomplete analysis of known
constants, sign classes (non-negative / positive / non-positive / negative /
unknown), and finiteness possibilities, propagated through a documented
operator subset. It enables earlier rejection and safer generation; it is
labelled heuristic and never presented as proof.

## 11. Search: grammar, mutation, crossover, selection

- `Grammar` (serialisable): enabled op classes, per-class budgets
  ("max 2 Exp"), node-count bounds per section, depth cap, state/item
  signatures, steps, broadcast toggle. `GrammarProfile` presets:
  `ScalarArithmetic`, `StableReduction`, `OnlineRecurrence`, `LinearAlgebra`,
  `AttentionResearch`, `GeneralScientific`. Profiles configure the generator;
  they never bypass the verifier.
- Generation is valid-by-construction: typed value pools, budget accounting,
  deterministic candidate enumeration, seeded SplitMix64 draws, defensive
  re-verification.
- Mutation families (verified, kind-tagged): replace-operator
  (type-compatible), rewire-source (type-compatible), perturb-constant,
  insert-unary, insert-binary, delete-node, mutate-outputs,
  mutate-reduction-axis.
- Crossover: canonical prune both parents; splice type-compatible subgraphs
  into the finalize section; join matching outputs via a deterministic
  type-compatible elementwise choice; fallback keeps a parent unchanged.
  Offspring always re-verified.
- Selection: Pareto fronts over explicit objective vector (loss, failed cases,
  flops, expensive-op counts, nodes, peak-live, state elements, depth), then
  lexicographic, then canonical bytes, then index — mirroring the proven V1
  total order. Counterexamples: bounded per-report storage (first failing
  cases with ids, absolute/relative errors, first differing elements).

## 12. Deliberately deferred (with reasons)

- Symbolic dims, argmax/argmin, gather/scatter/segment ops, integer/index
  dtypes: require a principled index dtype story first; fake encodings would
  undermine the verifier.
- `Erf`: no std implementation; a custom approximation would bake hidden
  numerics into the IR. Deferred with a documented plan.
- Autodiff: designed-for (pure SSA, typed, closed sections) but not implemented
  in this phase; see §13.
- Lowering backends: boundary defined (semantic IR → passes → execution plan);
  no backend shipped.
- E-graphs: the rewrite set is small and confluent-by-budget; a fixed-point
  pipeline suffices. Revisit if rewrite count grows past ~30 rules.

## 13. Differentiation outlook

Forward-mode AD is straightforward over this IR (every op needs a derivative
rule + broadcasting alignment); reverse-mode needs a tape = the step/init/
finalize sections already are tapes. Planned as a `differentiate()` pass
producing a new `ResearchProgram`, keeping semantic identity untouched.

## 14. Lowering boundary

Semantic IR (this crate) → optimization IR (post-canonicalization view) →
execution/lowering plan (future: scalar Rust, SIMD, GPU, WASM). Backends must
consume verified programs and must never feed back into canonical identity.
No vendor-specific code is part of IR semantics.

## 15. Compatibility strategy

V1 untouched and byte-stable. `compat::from_v1` lifts a V1 `TensorProgram`
into a V2 straight-line program (Scale → Mul·Const); round-trip execution
equality is tested. New experiments target V2 exclusively.
