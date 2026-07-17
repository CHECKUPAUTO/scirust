# SciRust-Hypermemory — F2 / F6 falsification results

> **Status:** deterministic experiment results for falsification criteria **F2**
> (zero-divisor / norm collapse) and **F6** (structure discrimination), defined
> in [`SCIRUST_HYPERMEMORY_PHASE1.md`](SCIRUST_HYPERMEMORY_PHASE1.md) §10.
>
> These are **reproducible facts**, not timing benchmarks: the harness is pure
> `f32` algebra over a fixed-seed LCG with fixed index-order reductions, so a
> given `(seed, samples, distribution)` yields the same numbers on any target
> where IEEE-754 `f32` is not reassociated.

## Why these experiments

Phase 1 already **confirmed F1**: the sedenion algebra adds nothing to
*retrieval* over the same 16 real components. The only place the non-associative
algebra does anything a real vector cannot is *relation composition*
(`S16Expr`). Before building later phases on that, two questions must be
answered honestly:

* **F2** — do products `a·b` collapse toward zero often enough that relation
  codes cannot carry information?
* **F6** — do the two parenthesizations `(a·b)·c` and `a·(b·c)` evaluate to
  *distinguishable* codes, or do they frequently coincide?

## Method

`scirust_hypermemory::experiments` (public API) provides deterministic surveys:

* **F2** — for `samples` products, the **norm-defect ratio**
  `r = ‖a·b‖² / (‖a‖²·‖b‖²)` (a composition algebra has `r ≡ 1`), plus the
  near-zero-divisor count (`‖a·b‖² ≤ 1e-12`, both operands non-zero).
* **F6** — for `samples` triples, the **relative associator**
  `ρ = ‖(a·b)·c − a·(b·c)‖ / (‖(a·b)·c‖ + ‖a·(b·c)‖)`; a triple is
  *indistinguishable* when `ρ ≤ 1e-3`.

Operands are drawn from three regimes: `DenseUniform` (generic, all 16 lanes),
`Sparse{k}` (k random non-zero lanes), and `Subalgebra{d}` (confined to lanes
`e₀..e_{d-1}`; `d ∈ {2,4}` are the **associative** complex / quaternion
subalgebras).

## Reproduce

```bash
cd scirust-hypermemory
HYPERMEMORY_GIT_COMMIT=$(git rev-parse HEAD) \
  cargo +nightly-2026-07-02 run --release --bin hypermemory-falsify
```

The values below were produced with `samples = 100_000` per distribution.

## F2 — zero-divisor / norm-collapse

| distribution        | near-0 | exact-0 | min r | mean r | max r | r < 0.01 |
|---------------------|-------:|--------:|------:|-------:|------:|---------:|
| DenseUniform        |      0 |       0 | 0.2170 | 0.9999 | 1.7492 |   0.00% |
| Sparse{2}           |      0 |       0 | 0.0003 | 1.0002 | 1.9999 |   0.01% |
| Sparse{4}           |      0 |       0 | 0.0527 | 0.9996 | 1.9299 |   0.00% |
| Subalgebra{4} (ℍ)   |      0 |       0 | 1.0000 | 1.0000 | 1.0000 |   0.00% |

Reading:

* **Generic operands never collapse.** Zero divisors are a measure-zero set;
  over 100 000 random products, zero near-zero divisors and zero exact zeros.
  Norm collapse is a **structured-input risk, not a generic one**.
* **The norm is not sub-multiplicative on 𝕊.** `max r > 1` (up to ≈1.75 dense,
  ≈2.0 sparse): `‖a·b‖` can *exceed* `‖a‖·‖b‖`. This is a measured property, not
  an assumption — 𝕊 is simply not a composition algebra.
* **Sparse operands can shrink hard** (`min r ≈ 3e-4` for `Sparse{2}`), i.e. the
  cancellation risk concentrates on low-rank / structured inputs — exactly where
  a future design must be careful.
* **The quaternion subalgebra is a composition algebra** (`r ≡ 1.0000`): a clean
  cross-validation of the reused Cayley–Dickson algebra.

## F6 — structure discrimination

| distribution        | discriminable | min ρ | mean ρ | max ρ |
|---------------------|--------------:|------:|-------:|------:|
| DenseUniform        |     100.0000% | 0.166802 | 0.663532 | 0.975659 |
| Sparse{2}           |      99.9510% | 0.000005 | 0.693429 | 1.000000 |
| Subalgebra{2} (ℂ)   |       0.0000% | 0.000000 | 0.000000 | 0.000000 |
| Subalgebra{4} (ℍ)   |       0.0000% | 0.000000 | 0.000000 | 0.000000 |

Reading:

* **Generic parenthesizations are always distinguishable** (100% at `ρ > 1e-3`,
  `min ρ ≈ 0.17`): `(a·b)·c` and `a·(b·c)` carry genuinely different codes.
* **F6 fires exactly where it must.** Inside the complex or quaternion
  **associative** subalgebra the associator vanishes (`ρ ≡ 0`), so the two
  structures are identical — 0% discriminable. Any future "the code encodes the
  tree" shortcut is therefore falsified for associative-subspace operands; Phase 1
  already refuses to reconstruct from the code (it reads the stored tree), so it
  is robust to this by design.
* **Sparse operands occasionally coincide** (`Sparse{2}` ≈ 0.05%
  indistinguishable): a few 2-lane triples land in an effectively associative
  configuration.

## Verdict (this harness, these distributions)

* **F2 is not triggered for generic operands** (no collapse); the risk is real
  but confined to sparse / structured inputs.
* **F6 is not triggered for generic operands** (100% discriminable); it triggers
  precisely and only inside associative subalgebras.

**This does not establish usefulness.** It establishes that the *necessary
discriminative capacity exists* for generic operands, and it maps exactly where
it breaks down (low-rank and associative-subspace inputs). Whether that capacity
translates into a measurable advantage over real-vector baselines for a real
task remains open — and is what a later phase (structured / relation-aware
queries, F1 gate) must demonstrate.

## Limitations

* The regimes are synthetic; real workloads may concentrate operands very
  differently (the sparse and subalgebra cases show the outcome is
  input-dependent).
* Metrics are `f32`; `ρ` and `r` inherit `f32` rounding, hence the `1e-3`
  thresholds rather than exact zero tests.
* F6 covers the minimal non-associative case (3 atoms, 2 parenthesizations).
  Larger trees have Catalan-many structures; their pairwise discrimination is
  not surveyed here.
