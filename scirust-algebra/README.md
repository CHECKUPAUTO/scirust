# scirust-algebra

`scirust-algebra` is SciRust's deterministic, pure-Rust algebra and group-theory layer.
It is designed around fixed-size values, caller-owned workspaces and static dispatch so
hot paths do not require heap allocation or dynamic dispatch.

## Compatibility

The crate follows the root SciRust workspace baseline: Rust 1.89, edition 2021,
`#![forbid(unsafe_code)]`, and no external dependencies. The reference kernels are kept
portable across the stable x86_64, ARM64 and MSRV validation lanes used by the workspace.

## Architecture

- `core`: algebraic law traits (`Magma`, `Semigroup`, `Monoid`, `Group`,
  `AbelianGroup`, `Ring`, `Field`, `LieAlgebra`), quotient representatives, direct
  products, semidirect products and group actions.
- `discrete`: compact fixed-degree permutations, cycle/signature operations,
  deterministic finite-group enumeration/reference membership, and free-group word
  reduction/ordered rewriting.
- `representation`: fixed matrices, character tables, finite-group irrep projectors,
  reference group Fourier transform and spherical-harmonic primitives.
- `lie`: `so(3)`, `SO(3)`, `se(3)`, `SE(3)`, the `SU(2)` quaternion model,
  second-order BCH and fixed-storage Clifford algebras `Cl(p,q)`.
- `equivariant`: statically typed representation actions, typed irrep labels,
  Clebsch-Gordan contractions and invariant bilinear contractions.

## Performance policy

The core value types use arrays and `const` generics. Algorithms whose state can grow
accept storage from the caller. The crate has no dependencies and forbids `unsafe`.
This makes the reference kernels deterministic and suitable as correctness oracles for
future SIMD/GPU specialisations.

## Important algorithmic status

This first integration provides exact small-group reference algorithms and stable APIs.
The `SchreierSims` facade currently uses deterministic exact closure enumeration for
membership/order; it is **not yet the polynomial-time stabilizer-chain implementation**
required for very large permutation groups. Likewise, `group_fourier` is the direct
finite-group transform, not yet a subgroup-factorised FFT for `S_n`. Todd-Coxeter,
Knuth-Bendix completion, general `SU(N)` matrix exponentials, Wigner-D matrices and
algorithmic Clebsch-Gordan coefficient generation remain follow-up implementation work.
The crate intentionally states these limits rather than presenting reference algorithms
as asymptotically stronger algorithms.

## Example

```rust
use scirust_algebra::discrete::{Permutation, PermutationGroup, SchreierSims};

let swap = Permutation::new([1, 0, 2]).unwrap();
let cycle = Permutation::new([1, 2, 0]).unwrap();
let group = PermutationGroup::new([swap, cycle]);
let ss = SchreierSims::new(group);
let mut scratch = [Permutation::<3>::identity_array(); 6];
assert_eq!(ss.order(&mut scratch), Ok(6));
```
