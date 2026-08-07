#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Deterministic, allocation-conscious algebra and group-theory primitives for SciRust.
//!
//! The crate deliberately separates algebraic laws (traits) from algorithms. The
//! hot-path data structures use fixed-size arrays and caller-owned storage; algorithms
//! that intrinsically require growing state expose explicit workspace objects.

#[rustfmt::skip]
mod scalar;

/// Algebraic laws, products, quotients and actions.
#[rustfmt::skip]
pub mod core;
/// Finite/discrete and combinatorial group algorithms.
#[rustfmt::skip]
pub mod discrete;
/// Equivariant maps and symmetry-aware tensor coupling.
#[rustfmt::skip]
pub mod equivariant;
/// Lie groups, Lie algebras and Clifford/geometric algebra.
#[rustfmt::skip]
pub mod lie;
/// Finite-group representation helpers and harmonic-analysis primitives.
#[rustfmt::skip]
pub mod representation;

pub use core::{AbelianGroup, Field, Group, Magma, Monoid, Ring, Semigroup};
