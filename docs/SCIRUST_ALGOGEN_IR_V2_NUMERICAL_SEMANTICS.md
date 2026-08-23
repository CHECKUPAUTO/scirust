# SciRust algogen IR V2 numerical semantics

This document is normative for `scirust_algogen::tensor::v2`. It separates
three concepts that are often conflated: the value computed by an operation,
the runtime policy for non-finite values, and the equivalence assumptions under
which canonicalization may rewrite a program.

## 1. Dtype evaluation

`F32` operations execute as Rust `f32` operations and round at every IR node.
`F64` operations execute as Rust `f64` operations. The reference tensor stores
both in an `f64` carrier, but every F32 operand is cast to f32 before its kernel
and the f32 result is stored exactly in f64. This does not promote F32
computation.

`Bool` is semantic Boolean data. External Boolean values must be encoded as the
exact f64 bit patterns for positive zero and positive one in the reference
carrier. Comparisons produce that representation. Float and Boolean arithmetic
cannot be mixed.

The rounding mode is the Rust target's normal IEEE round-to-nearest behaviour.
SciRust does not change host rounding modes, flush subnormals, or enable fast
math. Subnormal behaviour therefore follows the Rust target. Transcendental
functions use Rust `std`; deterministic replay is required for the same source,
target, and toolchain, but this version does not claim identical last bits
across every platform libm.

## 2. Core operation contract

- `Add`, `Sub`, `Mul`, `Div`, and `Neg` use the corresponding native operator.
  Overflow yields infinity, underflow may yield a subnormal or signed zero,
  division by zero follows IEEE, and invalid forms such as `0/0` yield NaN.
- `MulAdd(a,b,c)` calls `f32::mul_add` or `f64::mul_add`: one fused operation
  and one final rounding. It is not semantically identical to a `Mul` node
  followed by an `Add` node. No pass contracts or expands FMA.
- `Pow` is native `powf`. It inherits native domain, infinity, signed-zero,
  overflow, underflow, and accuracy behaviour.
- `Abs`, `Exp`, `Exp2`, `Expm1`, `Log`, `Log2`, `Log1p`, `Sqrt`, `Sin`, `Cos`,
  and `Tanh` call the corresponding native function. `Rsqrt(x)` is explicitly
  `1 / sqrt(x)` and therefore has two source-level operations inside its fixed
  opcode semantics; it is not a hardware approximate reciprocal-sqrt promise.
- `Min`/`Max` evaluate the deterministic extrema kernels defined by this
  contract, not unspecified native behaviour (Rust makes no signed-zero or
  payload promise for `f32::min/max`; rust-lang/rust#99640). The kernels are
  swap-symmetric by definition:
  1. if both operands are NaN the result is the canonical quiet NaN;
  2. if exactly one operand is NaN the result is the other operand;
  3. otherwise the numerically smaller (larger) operand is returned, and a
     numerical tie — including `+0` versus `-0` — resolves toward `-0` for
     minimum and toward `+0` for maximum.
  These rules coincide with IEEE-754-2019 `minimumNumber`/`maximumNumber`
  zero handling and with current native behaviour on mainstream targets.
  `Clamp(x,lo,hi)` is evaluated exactly as `max(min(x,hi),lo)` through those
  same kernels; the verifier checks types/shapes but does not require
  `lo <= hi`.
- `Select(mask,t,f)` copies the branch element selected by a Boolean mask. The
  two branch types/shapes are exactly equal; only the mask may broadcast.
- Float comparisons are Rust comparisons. If either operand is NaN, ordered
  comparisons and equality are false and `NotEqual` is true. `+0 == -0` is
  true. Results are Boolean, not floating zero/one values available to float
  arithmetic.

All elementwise float binary operations use the documented trailing-axis
broadcast. Broadcasting changes which stored element is read; it does not add
rounding or dtype conversion.

## 3. Reductions and linear algebra

Reduction order is fixed ascending row-major flat index. `axis=None` maps the
whole tensor to a scalar. `axis=Some(a)` removes axis `a`; keep-dim is false.

- sum starts at positive zero;
- product starts at positive one;
- maximum starts at negative infinity and folds `max(acc, x)` through the
  deterministic extrema kernels of section 2;
- minimum starts at positive infinity and folds `min(acc, x)` likewise;
- mean performs the same ordered sum, then one native division by the reduced
  count.

Because the extrema kernels are associative, commutative, and idempotent on
their value domain, reduce-max/min results are independent of element
encounter order: opposite-signed zeros resolve canonically (`+0` wins max,
`-0` wins min) instead of depending on which came first. NaN elements defer to
numeric operands; a reduction over only NaN values therefore keeps its
±Infinity identity. Empty sum/product are defined by their identities. Empty
max/min/mean are verifier errors.

Dot and every matrix-like product iterate the shared dimension in ascending
order. Each term is an unfused multiply followed by add at the declared dtype.
They do not silently use FMA or reassociate a reduction. Batched matmul has one
exact leading batch dimension; it does not broadcast batches.

## 4. Exceptional values

The value-level IR follows native IEEE behaviour; it does not turn numerical
exceptions into hidden clamps. Runtime handling is selected explicitly:

| `FloatPolicy` | external input/item | intermediate | output |
|---|---|---|---|
| `FiniteOutputs` (default) | must be finite | NaN rejected; infinity may flow | must be finite |
| `RejectNonFinite` | must be finite | any NaN/infinity rejected immediately | must be finite |
| `AllowNonFinite` | permitted | permitted | permitted |

Constants bypass the per-node gate so `-Infinity` can be the running-max
initializer in stable recurrences. NaN constants are always verifier errors.
`FiniteOnly` programs additionally reject infinite constants. Under the default
policy an infinity identity may flow internally, but an observable infinity is
a structured `NonFiniteOutput`; NaN is a structured `NanResult` at its first
active producer. Under `RejectNonFinite`, either is `NonFiniteResult`.

The default policy also rejects non-finite external tensors before executing
any node. Dead nodes are not evaluated: liveness is part of execution semantics
because this is a pure IR with no effects.

Signed zero is never normalized in values or canonical bytes. Constants encode
raw bits and `-0.0` differs structurally from `+0.0`. Operations may naturally
change the sign of zero according to IEEE arithmetic; extrema results are
governed by the deterministic kernels of section 2, so zero-sign outcomes never
depend on operand order.

## 5. Rewrite-equivalence regimes

`ResearchProgram.semantics` is part of canonical identity. It governs which
equivalences canonicalization may assume; it does not replace `FloatPolicy`.

### `StrictIeee`

Preserve IEEE evaluation semantics and every observable distinction: signed
zero, NaN/infinity behaviour, operation order for order-observable operators,
and fused versus unfused rounding. Allowed rules are exact structural rules:
finite-result constant evaluation of the same opcode, Boolean identities,
`Select(mask,v,v)`, Boolean/compare operand normalization, `Min`/`Max` operand
normalization (bit-exact because the extrema kernels are swap-symmetric by
contract), double Boolean negation, exact CSE, stable root renumbering, and
DCE. Floating add/mul/dot operands are not reordered: their results can depend
on operand order once NaN payloads or infinity productions are involved.

### `FiniteOnly`

The experiment asserts that admitted external values and active intermediate
values are finite (normally paired with `RejectNonFinite`). It additionally
permits finite-domain rules such as subtract positive zero, multiply/divide by
one, double float negation, min/max of the same value, and normalization of
commutative floating operands. Signed zero remains significant. Add-zero is not
allowed because `-0 + +0` is `+0`, not `-0`.

### `RealAlgebraicExperimental`

This regime may ignore IEEE distinctions that real arithmetic does not have,
currently adding zero identities including signed-zero-changing forms. The
name is intentionally explicit: a result canonicalized here is not claimed
IEEE-equivalent. Even this version does not enable reassociation,
distributivity, FMA contraction, `x*0`, `x-x`, `x/x`, `log(exp(x))`, or
`sqrt(x*x)`.

## 6. Rewrite audit table

| Rule | Strict | FiniteOnly | Real experimental | Important precondition |
|---|---:|---:|---:|---|
| same-op finite constant fold | yes | yes | yes | result finite; same native opcode |
| Boolean identities / double `Not` | yes | yes | yes | Boolean dtype |
| `Select(mask,v,v) -> v` | yes | yes | yes | exact same value ref/branch type |
| exact CSE / DCE / renumber | yes | yes | yes | same structural node / pure ops |
| `x + 0 -> x` | no | no | yes | real-algebraic signed-zero waiver |
| `x - +0 -> x` | no | yes | yes | finite-domain source |
| `x * 1`, `x / 1` | no | yes | yes | finite-domain source |
| `min(x,x)`, `max(x,x)` | no | yes | yes | finite-domain source (NaN payloads) |
| `-(-x) -> x` | no | yes | yes | finite-domain source |
| reorder `Min`/`Max` operands | yes | yes | yes | kernels swap-symmetric by contract |
| reorder `Add`/`Mul`/`Dot` operands | no | yes | yes | finite active operands; no reassociation |

Every application stores a stable rule id and pass number. Pass order is fixed
and the fixed point is bounded at 16 iterations, preventing rewrite loops.

## 7. Canonical identity and testing standard

Canonical identity contains IR, canonicalization, and semantic-regime versions
plus exact constant bits. A semantic-regime change therefore changes identity
even when syntax does not. SHA-256 equality is not structural proof; archives
compare retained canonical bytes.

Metamorphic tests execute original and canonicalized programs bit-for-bit on
fixtures inside each rule's declared domain. Tests cover signed zero,
overflow/underflow paths, NaN-producing log domains, infinities, division by
zero, FMA versus unfused arithmetic, reductions, and native transcendental
oracles. These tests are finite evidence, not a proof of floating-point
equivalence.
