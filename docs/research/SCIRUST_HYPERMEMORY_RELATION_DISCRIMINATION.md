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
| `Hrr` | HRR/VSA tree: nested circular convolution with fixed L/R role vectors — `L⊛(L⊛a+R⊛b)+R⊛c` etc. | yes (roles) | yes (nesting) |

`Hrr` is the **strong** baseline: a purpose-built structural encoding that, like
the sedenion, discriminates *both* order and grouping. It is the fair opponent —
the `Real` bindings are deliberately naive.

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
| Hrr | 0.657844 | 0.645989 |

### Noisy structure retrieval — nearest-neighbour accuracy (chance ≈ 0.0833)

| encoding | noise 0 | noise 0.1 | noise 0.25 | noise 0.5 |
|---|---:|---:|---:|---:|
| Sedenion | 1.0000 | 0.9990 | 0.9901 | 0.8690 |
| Real(Sum) | 0.0829 | 0.0829 | 0.0829 | 0.0829 |
| Real(Hadamard) | 0.0811 | 0.0824 | 0.0809 | 0.0834 |
| Real(PositionWeighted) | 0.5000 | 0.5000 | 0.4969 | 0.4239 |
| **Hrr** | **1.0000** | **1.0000** | **0.9979** | **0.9457** |

## Reading

* **The sedenion product discriminates both order and grouping** (order 0.90,
  grouping 0.66) → **near-perfect, noise-robust structure retrieval** (99.9% at
  noise 0.1, 87% at the aggressive noise 0.5). Against the naive real baselines
  this is a clear win.
* **Naive real baselines fail by construction.** `Sum` / `Hadamard` are at chance
  (they collapse all 12 structures to one code); `PositionWeighted` recovers
  order but not grouping, so it caps at ≈50% (always confuses `(a·b)·c` with
  `a·(b·c)`).
* **But HRR — the strong baseline — matches or beats the sedenion.** HRR
  discriminates order (0.66) and grouping (0.65) too, and its retrieval is *at
  least as good at every noise level* and **more robust under heavy noise**
  (noise 0.5: **HRR 0.9457 vs Sedenion 0.8690**). A purpose-built real
  structural encoding (circular convolution + role vectors) does the job as well
  or better — without the sedenion's zero-divisor collapse risk (F2).

So the sedenion's structural capacity is **real but not superior**: it beats
naive 16-real encodings, and it *loses to* an established structural method on
its own turf. This is the honest bound the program was built to find.

## Honest caveats

* **The grouping advantage is redundant with Phase 1's explicit tree.** The
  `S16Expr` tree already stores grouping exactly, without the algebra. The
  algebra's only distinct contribution is packing order + grouping into a
  *single fixed-width 16-lane code* — and HRR does that too, better.
* **HRR uses auxiliary role vectors; the sedenion uses none.** One could argue
  the sedenion is "cheaper" (no role vectors, binding is intrinsic to the
  product). But cheaper *and losing* is not a winning position: HRR's two fixed
  role vectors are a trivial cost, and it is both more accurate under noise and
  free of zero-divisor collapse.
* **Synthetic task.** Retrieving a structure from a 12-entry codebook over 3
  fixed atoms is a probe, not a downstream application; a real task could shift
  the ranking either way. But the burden of proof is now clearly on the sedenion
  to show an advantage it did *not* show here.

## Verdict

Against **naive** real 16-D encodings, the sedenion parenthesized product is a
clear structure-discrimination win (they are order/grouping-blind by
construction). Against a **strong** structural baseline — HRR/VSA (circular
convolution + role vectors) — it is **not**: HRR matches it at low noise and is
**more robust under heavy noise** (0.946 vs 0.869 at noise 0.5), while carrying
*none* of the sedenion's zero-divisor collapse risk (F2).

The honest conclusion of this arc: the sedenion algebra has **real structural
capacity but no demonstrated superiority** over established methods at the same
16 dimensions. Combined with F1 (no retrieval advantage), the evidence so far
**does not justify** the algebra over ordinary real-vector techniques — it
bounds its value rather than vindicating it. That is exactly the kind of
conservative, falsification-first result the program was designed to surface.
