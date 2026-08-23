# SciRust algogen IR V2 search grammar

The V2 generator is a typed, bounded grammar over verified scientific
operations. Adding an opcode to the semantic IR does not automatically add it
to every search. Operator exposure, frequency, shape growth, recurrence, and
proposal work are explicit configuration.

## 1. Generation contract

For a fixed source revision, `Grammar`, `GenerationRequest`, verification
limits, and seed, `generate_program` returns the same program and the same
rejection statistics. It uses the repository's stable SplitMix64 stream; no OS
entropy, hash-map iteration, time, or thread schedule selects a candidate.

Each section maintains an ordered pool of legal references with statically
inferred `ValueType` and dependency depth. Generation deterministically
enumerates candidate operations over that pool, calls the verifier's operation
type/shape inference, discards invalid proposals, and draws only from the valid
ordered set. The complete program is then reverified defensively. Thus shape
and dtype errors are filtered before random choice rather than becoming most of
the population.

`GenerationRequest` supplies only the problem signature and neutral structural
constraints:

- outer input, per-step item, state, and ordered output types;
- static recurrence length;
- state initializer source (typed constant or compatible outer input);
- independent step/finalize random-node ranges;
- whether a state update may copy `StatePrev` unchanged.

It contains no desired expression, known algorithm AST, fitness oracle, or
privileged operator sequence.

## 2. Operator classes

The stable classes are:

| Class | Operations |
|---|---|
| `Constant` | typed scalar constants |
| `Arithmetic` | add/sub/mul/div/pow, neg/abs, explicit fused mul-add |
| `Transcendental` | exp/log/sqrt/trigonometric families |
| `Extrema` | min/max/clamp |
| `Comparison` | six float comparisons to Boolean |
| `Boolean` | and/or/not |
| `Selection` | Boolean `Select` |
| `Reduction` | sum/product/min/max/mean, explicit axes |
| `LinearAlgebra` | dot, matvec, vecmat, matmul, batched matmul, outer |
| `Shape` | reshape/squeeze/unsqueeze/transpose/broadcast/concat |
| `Indexing` | statically bounded `Narrow` |

The class mapping is exhaustive and public for mutation, reporting, and tests.
Dynamic gather/scatter is not hidden inside `Indexing`.

## 3. Hard search-space controls

`Grammar` serializes all controls that affect candidate structure:

- allowed operator classes and dtypes;
- allowed constant pool;
- maximum rank, values, operations, and dependency depth;
- separate maxima for reductions, transcendental operations, all expensive
  operations, linalg operations, shape/static-index operations, and comparisons;
- recurrence allowed/forbidden, state-component maximum, static-trip maximum;
- implicit broadcasting allowed/forbidden;
- static indexing allowed/forbidden;
- maximum candidate proposals enumerated for the complete program.

The verifier independently enforces maximum inputs, items/step, section/total
nodes, tensor/register/signature elements, conservative host bytes, rank,
steps, states, and outputs. A grammar cannot weaken the verifier.

Proposal enumeration is a hard prefix budget. Reaching it is recorded. If the
remaining prefix contains no valid proposal for a required node, generation
returns a structured failure rather than silently expanding the budget. This
makes candidate-order changes research-visible.

## 4. Profiles as curricula

Profiles provide starting curricula, not semantic modes:

- `ScalarAlgebra`: constants, arithmetic, extrema, comparisons, Boolean,
  selection, and shape plumbing; no recurrence.
- `Reduction`: scalar/tensor core plus reductions and transcendental functions.
- `StreamingRecurrence`: reduction grammar plus bounded recurrence.
- `LinearAlgebra`: core plus linear algebra and reductions.
- `MaskedTensor`: core plus reductions and static indexing.
- `AttentionBuildingBlocks`: core, transcendental, reduction, linalg, masking,
  static indexing, and bounded recurrence.
- `GeneralScientific`: every implemented class under conservative maxima.

Profiles can be narrowed or budgets lowered before a run. They never enable a
magic softmax, attention, Entmax, or ADA production.

Recommended staged use is: start with the narrowest dtype/rank/signature and
cheap class set; measure exact duplicate/rejection rates; then add one class or
raise one budget. This makes grammar expansion auditable and avoids declaring a
dozens-of-ops uniform random enum to be a search strategy.

## 5. Shape-directed candidate construction

Unary candidates use each legal scoped value. Binary and ternary candidates
use ordered reference tuples and are admitted only after dtype/broadcast rules.
Reduction candidates enumerate full reduction and each valid axis. Shape
targets come from declared target types and shapes already present in the pool.
Squeeze, transpose, concat, and narrow attributes are statically enumerated.
Linear algebra proposals are admitted only for exact rank/inner-dimension
contracts.

Depth is `1 + max(operand depth)` and is checked before a proposal is chosen.
Per-class and expensive-operation budgets are checked both during enumeration
and emission. Exact duplicate operations in a section are rejected before
selection; post-generation canonicalization provides stronger CSE/DCE and
cross-candidate exact-byte deduplication.

## 6. Deterministic rejection diagnostics

Every successful generation records:

- proposals considered;
- type/shape rejections;
- depth/rank rejections;
- class/budget rejections;
- exact duplicate-op rejections;
- enumeration truncations;
- valid proposals;
- emitted operations by class.

Experiment diagnostics separately record candidate attempts, generation
failures, canonical duplicates, interpreter case executions, archive
comparisons, and final archive size. These counters diagnose search-engine
overhead but never enter candidate fitness. Wall-clock throughput may be
measured externally, but cannot change ranking.

## 7. Mutation

`MutationConfig` enables stable mutation families and caps proposals. The
current conservative families are:

- replace a constant with an allowed same-dtype constant;
- replace an operation within a same-signature family;
- replace scoped operands with exactly type-compatible refs;
- modify a reduction axis;
- rebind a state component to another exact-type step result;
- rebind an output to another exact-type finalize result.

All variants are deterministically enumerated, whole-program verified,
canonical-byte deduplicated, and then seeded-selected. Invalid proposals are
counted and never interpreted. Insert/remove and arbitrary shape-transform
mutation are intentionally deferred until their local repair rules can avoid a
mostly-invalid population.

## 8. Crossover

Crossover first verifies both parents and requires identical numerical regime,
outer input types, item types, state types, trip count, and ordered output
types. It exchanges one complete `init + init_state`, `step + next_state`, or
`finalize + outputs` unit. This preserves local-id scope and recurrence context;
whole children are reverified. A swap that merely returns a parent is not
reported as a new child. No blind instruction-index splice is performed.

This is intentionally conservative. Future subgraph crossover should use
typed dominance frontiers and explicit state-scope maps, not weaken this
contract.

## 9. Fitness and Pareto archive

Counterexample correctness is an objective independent of structural cost.
The current nondominance vector includes failed cases, mean squared and maximum
absolute error, logical FLOPs, peak live bytes, state bytes, expression/update
depth, reductions, and expensive transcendental counts. All are deterministic
and hardware-independent.

Exact canonical bytes define duplicates and final ordering; SHA-256 is only a
label. The capacity-bounded archive removes dominated entries and uses a fixed
lexicographic order when a nondominated set exceeds capacity. Candidate records
retain admission decision and final Pareto position.

## 10. Discovery smoke evidence

The test `bounded_discovery_finds_sum_recurrence_without_target_ast` requests a
three-step f64 recurrence with one scalar item, one zero-initialized scalar
state, one random step operation, and a scalar output. The grammar exposes only
general constants, arithmetic, and shape operations. It evaluates adversarial
positive, signed-zero, mixed-sign, and cancellation fixtures. Within a hard
256-candidate cap it discovers an exact `Add(StatePrev, Item)` recurrence.

The experiment is rerun from its recorded seed and required to produce the
same candidate records, canonical bytes, Pareto archive, diagnostics, and
content digest. This is a small smoke result, not evidence that larger
attention recurrences are already tractable.
