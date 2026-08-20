//! General semiring abstractions.
//!
//! A semiring carries two binary operations with distinct neutral elements:
//! addition-like `plus` with zero and multiplication-like `times` with one.
//! The mathematical semiring laws are semantic requirements of an implementation;
//! Rust's trait system does not prove associativity or distributivity.
//!
//! This module deliberately uses operation names that do not collide with the
//! existing [`crate::core::Ring`] API. Rings can be viewed as semirings through
//! [`RingSemiring`] without changing the public `Ring` trait.

use crate::core::{Field, Ring};

/// A set equipped with additive and multiplicative monoid structures where
/// multiplication distributes over addition and zero is multiplicatively
/// absorbing.
///
/// Implementors are responsible for satisfying the semiring laws.
pub trait Semiring: Sized {
    /// Additive identity `0`.
    fn additive_identity() -> Self;

    /// Multiplicative identity `1`.
    fn multiplicative_identity() -> Self;

    /// Semiring addition.
    fn plus(&self, rhs: &Self) -> Self;

    /// Semiring multiplication.
    fn times(&self, rhs: &Self) -> Self;
}

/// Marker trait for semirings whose multiplication is commutative.
///
/// Implementors are responsible for the commutativity law in addition to the
/// laws required by [`Semiring`].
pub trait CommutativeSemiring: Semiring {}

/// Transparent adapter exposing any existing [`Ring`] as a [`Semiring`].
///
/// A ring already satisfies all semiring laws after forgetting additive
/// inverses, so no arithmetic conversion is required.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct RingSemiring<T>(pub T);

impl<T> RingSemiring<T> {
    /// Wrap a ring value without changing its representation.
    #[inline]
    pub const fn new(value: T) -> Self {
        Self(value)
    }

    /// Borrow the underlying ring value.
    #[inline]
    pub const fn inner(&self) -> &T {
        &self.0
    }

    /// Consume the adapter and return the underlying ring value.
    #[inline]
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T: Ring> Semiring for RingSemiring<T> {
    #[inline]
    fn additive_identity() -> Self {
        Self(T::zero())
    }

    #[inline]
    fn multiplicative_identity() -> Self {
        Self(T::one())
    }

    #[inline]
    fn plus(&self, rhs: &Self) -> Self {
        Self(self.0.add(&rhs.0))
    }

    #[inline]
    fn times(&self, rhs: &Self) -> Self {
        Self(self.0.mul(&rhs.0))
    }
}

/// Fields in SciRust are defined as commutative rings, therefore their
/// [`RingSemiring`] adapters are commutative semirings.
impl<T: Field> CommutativeSemiring for RingSemiring<T> {}

/// Boolean semiring with logical OR as addition and logical AND as
/// multiplication.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct BooleanSemiring(pub bool);

impl Semiring for BooleanSemiring {
    #[inline]
    fn additive_identity() -> Self {
        Self(false)
    }

    #[inline]
    fn multiplicative_identity() -> Self {
        Self(true)
    }

    #[inline]
    fn plus(&self, rhs: &Self) -> Self {
        Self(self.0 || rhs.0)
    }

    #[inline]
    fn times(&self, rhs: &Self) -> Self {
        Self(self.0 && rhs.0)
    }
}

impl CommutativeSemiring for BooleanSemiring {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boolean_semiring_has_expected_operations() {
        let zero = BooleanSemiring::additive_identity();
        let one = BooleanSemiring::multiplicative_identity();
        let t = BooleanSemiring(true);
        let f = BooleanSemiring(false);

        assert_eq!(zero, f);
        assert_eq!(one, t);
        assert_eq!(t.plus(&f), t);
        assert_eq!(t.times(&f), f);
        assert_eq!(t.times(&t), t);
    }

    #[test]
    fn ring_adapter_for_f64_preserves_ring_arithmetic() {
        let a = RingSemiring::new(2.0_f64);
        let b = RingSemiring::new(3.0_f64);

        assert_eq!(a.plus(&b).into_inner(), 5.0);
        assert_eq!(a.times(&b).into_inner(), 6.0);
        assert_eq!(RingSemiring::<f64>::additive_identity().into_inner(), 0.0);
        assert_eq!(RingSemiring::<f64>::multiplicative_identity().into_inner(), 1.0);
    }
}
