# SciRust-Hypermemory — relational structure discrimination ("F1 for relations")

> **Status:** deterministic experiment results for the relation-usefulness
> question. Companion to
> [`SCIRUST_HYPERMEMORY_PHASE1.md`](SCIRUST_HYPERMEMORY_PHASE1.md) (F1) and
> [`SCIRUST_HYPERMEMORY_F2_F6.md`](SCIRUST_HYPERMEMORY_F2_F6.md) (F2/F6).
>
> These are **reproducible facts** (pure `f32` algebra, fixed-seed LCG, fixed
> index-order reductions), not timing benchmarks.

## The question

Phase 1 confirmed **F1**: the sedenion algebra adds nothing to *retrieval* over
the same 16 real components. F2/F6 showed the relation codes *can* discriminate
structure for generic operands. This experiment asks the usefulness question
directly:

> Encoding a structured triple through the non-commutative, non-associative
> sedenion product — does it discriminate the triple's **structure** (operand
> order and grouping) better than a plain real-vector encoding of the same 16
> components?

## Method

A structured triple `(a, b, c)` with a parenthesization (`(a·b)·c` or `a·(b·c)`)
is encoded to a 16-lane code by one of four `Encoding`s (module
`scirust_hypermemory::binding`):

| encoding | operation | order-sensitive? | grouping-sensitive? |
|---|---|:--:|:--:|
| `Sedenion` | parenthesized sedenion product | — (measured) | — (measured) |
| `Real(Sum)` | `a + b + c` | no (commutative) | no (associative) |
| `Real(Hadamard)` | `a ⊙ b ⊙ c` | no (commutative) | no (associative) |
| `Real(PositionWeighted)` | `1·a + 2·b + 3·c` | yes (weights) | no (associative) |

Three deterministic metrics:

* **order sensitivity** — mean relative code distance `‖·‖/(‖·‖+‖·‖)` between
  `enc(a,b,c)` and `enc(b,a,c)`;
* **grouping sensitivity** — mean relative code distance between the two
  parenthesizations of the same operands;
* **structure retrieval** — nearest-neighbour (cosine) accuracy recovering
  *which* of the 12 structures (6 orderings × 2 parenthesizations, over 3 fixed
  atoms) a **noisy** query came from. Chance = `1/12 ≈ 0.0833`.

## Reproduce

```bash
cd scirust-hypermemory
HYPERMEMORY_GIT_COMMIT=$(git rev-parse HEAD) \
  cargo +nightly-2026-07-02 run --release --bin hypermemory-relations
```

Below: `SENS_SAMPLES = 50_000`; retrieval over `200` atom sets × `200`
trials/set.

## Results

### Order & grouping sensitivity (mean relative code distance)

| encoding | order (swap a,b) | grouping (L vs R) |
|---|---:|---:|
| Sedenion | **0.897544** | **0.662896** |
| Real(Sum) | 0.000000 | 0.000000 |
| Real(Hadamard) | 0.000000 | 0.000000 |
| Real(PositionWeighted) | 0.193686 | 0.000000 |

### Noisy structure retrieval — nearest-neighbour accuracy (chance ≈ 0.0833)

| encoding | noise 0 | noise 0.1 | noise 0.25 | noise 0.5 |
|---|---:|---:|---:|---:|
| Sedenion | **1.0000** | **0.9990** | **0.9901** | **0.8690** |
| Real(Sum) | 0.0829 | 0.0829 | 0.0829 | 0.0829 |
| Real(Hadamard) | 0.0811 | 0.0824 | 0.0809 | 0.0834 |
| Real(PositionWeighted) | 0.5000 | 0.5000 | 0.4969 | 0.4239 |

## Reading

* **The sedenion product discriminates both order and grouping** (order 0.90,
  grouping 0.66), and this translates into **near-perfect, noise-robust
  structure retrieval** (99.9% at noise 0.1, still 87% at the aggressive noise
  0.5).
* **Commutative real baselines are at chance.** `Sum` and `Hadamard` collapse
  all 12 structures over the same atoms to one code (order- and grouping-blind
  by construction), so retrieval is exactly chance — a plain "16 real numbers,
  no algebra" bag *cannot* recover structure.
* **Position-weighting recovers order but not grouping.** It distinguishes the 6
  orderings but maps the two parenthesizations of each ordering to the same code,
  so it caps at ≈50% (it always confuses `(a·b)·c` with `a·(b·c)`).

So the sedenion algebra provides a **genuine capacity advantage over a plain
real-vector encoding for representing structure** — the first place in this
program where the algebra does something a comparable real encoding does not.

## Honest caveats (this is capacity, not proven usefulness)

* **The grouping advantage is redundant with Phase 1's explicit tree.** The
  `S16Expr` tree already stores grouping exactly, without the algebra. The
  algebra's contribution is packing order + grouping into a *single fixed-width
  16-lane code* — useful only if a fixed-width structural code is what you need.
* **The real baselines here are deliberately simple.** Stronger real structural
  encodings exist (e.g. HRR / VSA circular convolution with role vectors, or a
  learned bilinear binding); this experiment does **not** claim the sedenion beats
  those — only the elementary Sum / Hadamard / position-weighted baselines at
  the same 16 dims.
* **Synthetic task.** Retrieving a structure from a 12-entry codebook over 3
  fixed atoms is a probe, not a downstream application. Whether the advantage
  survives a real task — and justifies the algebra's cost and its structured-input
  collapse risks (F2) — is the next phase's burden (structured / relation-aware
  queries, gated on beating a real baseline per F1).

## Verdict

Against elementary real 16-D encodings, the sedenion parenthesized product is
the first component in this program to show a **measurable, robust advantage for
structure discrimination**. It establishes *capacity*; it does **not** establish
usefulness on a real task, nor superiority over stronger structural baselines.
