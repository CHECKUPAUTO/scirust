# scirust-algebra

`scirust-algebra` is SciRust's deterministic, pure-Rust algebra and group-theory layer.
It is designed around fixed-size values, caller-owned workspaces and static dispatch so
hot paths do not require heap allocation or dynamic dispatch.

## Compatibility

The crate follows the root SciRust workspace baseline: Rust 1.89, edition 2021,
`#![forbid(unsafe_code)]`, and no external dependencies. The reference kernels are kept
portable across the stable x86_64, ARM64 and MSRV validation lanes used by the workspace.
The root lockfile includes the crate as a dependency-free workspace package so standard
`--locked` CI validation can build it without re-resolving the dependency graph.

## Architecture

- `core`: algebraic law traits (`Magma`, `Semigroup`, `Monoid`, `Group`,
  `AbelianGroup`, `Ring`, `Field`, `LieAlgebra`), quotient representatives, direct
  products, semidirect products and group actions.
- `discrete`: compact fixed-degree permutations, cycle/signature operations,
  deterministic finite-group enumeration/reference membership, a public automatic
  Schreier-Sims facade, and free-group word reduction/ordered rewriting.
- `schreier`: fixed-storage orbit/transversal construction, deterministic Schreier
  generator production, strong-generator completion, stabilizer-chain sifting,
  orbit-stabilizer order and exact membership.
- `presented`: fixed-capacity deterministic Todd-Coxeter coset enumeration with compact
  inverse-paired tables and union-find coincidence handling.
- `representation`: fixed matrices, character tables, finite-group irrep projectors,
  reference group Fourier transform and spherical-harmonic primitives.
- `lie`: `so(3)`, `SO(3)`, `se(3)`, `SE(3)`, the `SU(2)` quaternion model,
  second-order BCH and fixed-storage Clifford algebras `Cl(p,q)`.
- `equivariant`: statically typed representation actions, typed irrep labels,
  Clebsch-Gordan contractions and invariant bilinear contractions.

## Performance policy

The core value types use arrays and `const` generics. Algorithms whose state can grow
accept storage from the caller or use caller-selected compile-time capacities. The crate
has no dependencies and forbids `unsafe`. This makes the reference kernels deterministic
and suitable as correctness oracles for future SIMD/GPU specialisations.

## Algorithmic status

Permutation groups now have an automatic deterministic Schreier-Sims path. Starting from
ordinary generators, SciRust builds orbit transversals, produces Schreier generators,
sifts residues into deeper stabilizers and inserts missing strong generators until the
natural-base chain closes. The resulting BSGS provides exact order and membership without
full subgroup enumeration. Storage is bounded by the caller-selected strong-generator
capacity, and exhaustion is reported explicitly.

`PermutationGroup::enumerate_into` remains available intentionally as a small-group
reference oracle. `group_fourier` is still the direct finite-group transform rather than a
subgroup-factorised non-abelian FFT for `S_n`. Knuth-Bendix completion, general `SU(N)`,
Wigner-D matrices and algorithmic Clebsch-Gordan coefficient generation remain follow-up
implementation work.

## Example

```rust
use scirust_algebra::discrete::{Permutation, PermutationGroup, SchreierSims};

let swap = Permutation::new([1, 0, 2]).unwrap();
let cycle = Permutation::new([1, 2, 0]).unwrap();
let group = PermutationGroup::new([swap, cycle]);
let ss = SchreierSims::<3, 2, 8>::build(group).unwrap();
assert_eq!(ss.order(), Some(6));
assert!(ss.contains(&Permutation::new([0, 2, 1]).unwrap()));
```
