# SciRust algogen IR V2 ADA readiness

## 1. Claim boundary

IR V2 makes important ADA mechanisms representable from general scientific
building blocks. It does not claim that A1–A10 algorithms have been discovered,
proved, optimized, or integrated into ADA. There are no ADA, attention,
softmax, online-softmax, or Entmax opcodes and no generator templates for these
programs.

Executable evidence lives in `tensor/v2/reference.rs`. The search generator
does not import that module. `ada_readiness_programs(steps, vector_length)`
constructs bounded fixtures only for verifier/interpreter/cost tests.

## 2. Capability matrix

| ADA area | V2 evidence | General primitives exercised | Current boundary |
|---|---|---|---|
| A1 stable online softmax | `online_softmax_recurrence` | two scalar states, max, sub, exp, mul, add, fixed scan | represents `(m,l)` update; not a discovery result or full softmax tensor API |
| A2 indexed/masked/support accumulation | `indexed_masked_accumulation_program` | real Bool mask, Select, ReduceSum, static Narrow | static ranges and masks only; dynamic Gather/Scatter awaits index dtype |
| A3 error-budget expressions | `error_budget_program` | sub, abs, sum/max reductions, multiple outputs | finite fixtures, no proof or interval theorem solver |
| A4 threshold/support comparison | `threshold_support_program` | broadcast compare to Bool, Select | fixed dense shapes; no sparse support container |
| A5 maxima/bounds algebra | `reduction_statistics_program` | sum/min/max/mean with explicit full axis | deterministic reductions, no symbolic bound proof |
| A6 root/iterative update | `bounded_root_recurrence` | typed tuple state, div/add/mul, fixed Newton steps | fixed trip count; no data-dependent convergence/early exit |
| A7 moments/composable recurrence | `welford_recurrence` | `(count,mean,M2)` state and scalar update algebra | count is f64 because integer dtype is deferred |
| A8 attention recurrence discovery substrate | `attention_recurrence` | `(m,l,o)` mixed scalar/vector state, two item components, broadcast weighted accumulation | represents a stable fold; no attention opcode, tiling, causal kernel, or claim of discovery |
| A9 distribution/statistic extraction | `reduction_statistics_program` | ordered multiple outputs for sum/min/max/mean | no quantiles, histograms, arg indices, or dynamic support |
| A10 deterministic numerical oracle style | `two_pass_softmax_building_blocks` | max, broadcast subtraction, exp, sum, multiple outputs | executable deterministic reference computation, not formal oracle correctness |

## 3. A1 and A8 recurrence semantics

The A1 fixture declares two f64 scalar state components. Initialization is
`m=-Infinity`, `l=0`. Each step computes:

```text
m_new = max(m_old, x)
l_new = l_old * exp(m_old - m_new) + exp(x - m_new)
```

Every line is an ordinary V2 node. The default runtime policy allows the
explicit infinity initializer to flow internally, rejects NaN, and requires
finite outputs. State types and next-state bindings are verifier checked.

The A8 fixture extends the same pattern to `(m,l,o)` where `o` is a vector and
each item is `(score,value_vector)`:

```text
alpha = exp(m_old - m_new)
beta  = exp(score - m_new)
l_new = l_old * alpha + beta
o_new = o_old * alpha + value * beta
```

Scalar-to-vector arithmetic uses the documented broadcast rules. The fixture
also exposes `o_new/l_new` as a derived output. Scan intermediates are not
materialized across time: state is replaced once per step and liveness/cost
reports distinguish state from step temporaries.

## 4. Multi-state numerical algorithms

Welford demonstrates three compatible state outputs per step and returns
`(count,mean,M2)` in explicit program-output order. Compensated summation
demonstrates `(sum,compensation)`. Both prove that recurrence is not a hidden
single-accumulator reduction.

The bounded-root fixture keeps the parameter and iterate in two state slots.
It executes exactly the declared number of Newton updates. This is sufficient
to search bounded iterative formulas while preventing an unbounded loop or a
data-dependent nontermination channel.

## 5. Masks, support, and indexing

Comparisons produce true Bool tensors; masks are not f64 predicates. `Select`
requires equal branch types/shapes and permits only the mask to broadcast.
This supports dense masked attention/value updates and threshold support.

`Narrow` provides statically known start/length indexing with verifier bounds.
V2 intentionally does not fake Gather with f64 indices. A future index dtype
must define width, bounds, canonical encoding, interpreter representation, and
cost before Gather/ArgMax. Scatter/ScatterAdd additionally require explicit
duplicate-index ordering/reduction and alias semantics; they remain deferred.

## 6. Required reference-program evidence

The reference test suite verifies and executes:

1. `a*x+b` using separate Mul and Add roundings;
2. reduction sum;
3. reduction maximum;
4. two-pass softmax building blocks `(m,e,l)` without Softmax;
5. stable online-softmax recurrence;
6. Welford `(count,mean,M2)` recurrence;
7. Kahan-like `(sum,compensation)` recurrence;
8. matrix multiplication;
9. masked conditional update;
10. explicit scalar-to-vector broadcast expression.

Additional tests execute bounded root and attention recurrences and verify every
A1–A10 program under the same trust boundary. Exact or independent numerical
oracles are used where suitable; finite execution is asserted where the test's
purpose is representability rather than a completed algorithm specification.

## 7. Search evidence

The deterministic discovery smoke does not use any reference constructor. A
general arithmetic recurrence grammar discovered an exact scalar running sum
from counterexamples within 256 attempts. Candidate seeds, grammar, limits,
dataset digest, canonical program bytes, fitness, cost, Pareto decision, and
diagnostics are archived and replayed exactly.

This result validates the generate→verify→canonicalize→interpret→Pareto→replay
path. It does not establish useful search complexity for the much larger A1 or
A8 grammar. Search-space curriculum, stronger novelty/falsification, and ADA
problem integration remain future work.

## 8. Remaining ADA blockers

- dynamic indices, gather/scatter/segments and sparse storage;
- index/integer/count dtypes and ArgMin/ArgMax;
- symbolic or runtime-dependent sequence lengths and shapes;
- optional materialized scan outputs;
- formal error/range proofs and counterexample synthesis beyond fixed sets;
- differentiability contracts and autodiff for mask/min/max points;
- production lowerings and performance models for CPU/SIMD/GPU;
- empirical discovery studies for stable attention updates under meaningful
  budgets.

These are explicit limitations, not silently encoded assumptions. The current
foundation is ADA-ready at the representation and deterministic-search-smoke
level, not at the solved-algorithm level.
