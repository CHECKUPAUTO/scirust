#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Deterministic, allocation-conscious algebra and group-theory primitives for SciRust.
//!
//! The crate deliberately separates algebraic laws (traits) from algorithms. The
//! hot-path data structures use fixed-size arrays and caller-owned storage; algorithms
//! that intrinsically require growing state expose explicit workspace objects.

#[rustfmt::skip]
mod scalar;

/// Integer-angular-momentum coupling coefficients for SO(3).
#[rustfmt::skip]
pub mod angular_momentum;
/// Decomposition of real finite-group characters into irreducible multiplicities.
#[rustfmt::skip]
pub mod character_decomposition;
/// Certification and inter-reduction helpers for bounded rewriting systems.
#[rustfmt::skip]
pub mod completion;
/// Deterministic conjugacy-class decomposition for finite permutation groups.
#[rustfmt::skip]
pub mod conjugacy;
/// Algebraic laws, products, quotients and actions.
#[rustfmt::skip]
pub mod core;
/// Finite/discrete and combinatorial group algorithms.
#[rustfmt::skip]
pub mod discrete;
/// Equivariant maps and symmetry-aware tensor coupling.
#[rustfmt::skip]
pub mod equivariant;
/// Normalized spherical harmonics and integer-spin Wigner rotation primitives.
#[rustfmt::skip]
pub mod harmonics;
/// Deterministic fixed-capacity Knuth-Bendix completion.
#[rustfmt::skip]
pub mod knuth_bendix;
/// Lie groups, Lie algebras and Clifford/geometric algebra.
#[rustfmt::skip]
pub mod lie;
/// Presented groups and fixed-capacity coset enumeration.
#[rustfmt::skip]
pub mod presented;
/// Deterministic orbit/transversal and stabilizer-chain algorithms.
#[rustfmt::skip]
pub mod schreier;
/// General semiring abstractions and adapters.
#[rustfmt::skip]
pub mod semiring;
/// Finite-group representation helpers and harmonic-analysis primitives.
#[rustfmt::skip]
pub mod representation;
/// Complex Wigner D elements and complex spherical harmonics.
#[rustfmt::skip]
pub mod wigner;

pub use crate::core::{AbelianGroup, Field, Group, Magma, Monoid, Ring, Semigroup};
pub use crate::semiring::{BooleanSemiring, CommutativeSemiring, RingSemiring, Semiring};
