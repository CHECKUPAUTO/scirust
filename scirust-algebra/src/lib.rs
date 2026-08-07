#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Deterministic, allocation-conscious algebra and group-theory primitives for SciRust.
//!
//! The crate deliberately separates algebraic laws (traits) from algorithms.  The
//! hot-path data structures use fixed-size arrays and caller-owned storage; algorithms
//! that intrinsically require growing state expose explicit workspace objects.

/// Algebraic laws, products, quotients and actions.
pub mod core;
/// Finite/discrete and combinatorial group algorithms.
pub mod discrete;
/// Finite-group representation helpers and harmonic-analysis primitives.
pub mod representation;
/// Lie groups, Lie algebras and Clifford/geometric algebra.
pub mod lie;
/// Equivariant maps and symmetry-aware tensor coupling.
pub mod equivariant;

pub use core::{AbelianGroup, Field, Group, Magma, Monoid, Ring, Semigroup};
