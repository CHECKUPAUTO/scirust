//! Partial orders, lattices, product orders, and minimal antichains.
//!
//! These abstractions model order relations that are not necessarily total.
//! They are useful in distributed progress tracking, causal/version orders,
//! multi-objective optimization, dependency analysis, and algebraic algorithms.
//!
//! The mathematical laws documented by the traits are semantic requirements of
//! implementations; Rust's type system does not prove reflexivity, transitivity,
//! antisymmetry, or lattice identities.

/// A mathematical partial order.
///
/// Implementors must provide a reflexive, transitive, and antisymmetric
/// `less_equal` relation.
pub trait PartiallyOrdered: Sized {
    /// Returns `true` exactly when `self <= other` in the represented order.
    fn less_equal(&self, other: &Self) -> bool;

    /// Returns `true` exactly when `self < other` in the represented order.
    #[inline]
    fn less_than(&self, other: &Self) -> bool {
        self.less_equal(other) && !other.less_equal(self)
    }

    /// Returns `true` when both values denote the same element of the order.
    #[inline]
    fn order_equivalent(&self, other: &Self) -> bool {
        self.less_equal(other) && other.less_equal(self)
    }
}

/// A partial order in which every pair has a least upper bound.
pub trait JoinSemilattice: PartiallyOrdered {
    /// Returns the least upper bound of `self` and `other`.
    fn join(&self, other: &Self) -> Self;
}

/// A partial order in which every pair has a greatest lower bound.
pub trait MeetSemilattice: PartiallyOrdered {
    /// Returns the greatest lower bound of `self` and `other`.
    fn meet(&self, other: &Self) -> Self;
}

/// A partial order supporting both joins and meets.
pub trait Lattice: JoinSemilattice + MeetSemilattice {}

impl<T: JoinSemilattice + MeetSemilattice> Lattice for T {}

/// Adapter that gives any ordinary [`Ord`] type its natural total order as a
/// SciRust partial order and lattice.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct TotalOrder<T>(pub T);

impl<T> TotalOrder<T> {
    /// Wraps a value in its ordinary total order.
    #[inline]
    pub const fn new(value: T) -> Self {
        Self(value)
    }

    /// Borrows the wrapped value.
    #[inline]
    pub const fn inner(&self) -> &T {
        &self.0
    }

    /// Consumes the wrapper and returns the value.
    #[inline]
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T: Ord> PartiallyOrdered for TotalOrder<T> {
    #[inline]
    fn less_equal(&self, other: &Self) -> bool {
        self.0 <= other.0
    }
}

impl<T: Ord + Clone> JoinSemilattice for TotalOrder<T> {
    #[inline]
    fn join(&self, other: &Self) -> Self {
        if self.0 >= other.0 {
            self.clone()
        } else {
            other.clone()
        }
    }
}

impl<T: Ord + Clone> MeetSemilattice for TotalOrder<T> {
    #[inline]
    fn meet(&self, other: &Self) -> Self {
        if self.0 <= other.0 {
            self.clone()
        } else {
            other.clone()
        }
    }
}

/// Coordinate-wise product partial order on two ordered components.
///
/// `(a1, b1) <= (a2, b2)` exactly when `a1 <= a2` and `b1 <= b2`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ProductOrder2<A, B> {
    /// First coordinate.
    pub first: A,
    /// Second coordinate.
    pub second: B,
}

impl<A, B> ProductOrder2<A, B> {
    /// Constructs a two-coordinate product-order value.
    #[inline]
    pub const fn new(first: A, second: B) -> Self {
        Self { first, second }
    }
}

impl<A: PartiallyOrdered, B: PartiallyOrdered> PartiallyOrdered for ProductOrder2<A, B> {
    #[inline]
    fn less_equal(&self, other: &Self) -> bool {
        self.first.less_equal(&other.first) && self.second.less_equal(&other.second)
    }
}

impl<A: JoinSemilattice, B: JoinSemilattice> JoinSemilattice for ProductOrder2<A, B> {
    #[inline]
    fn join(&self, other: &Self) -> Self {
        Self {
            first: self.first.join(&other.first),
            second: self.second.join(&other.second),
        }
    }
}

impl<A: MeetSemilattice, B: MeetSemilattice> MeetSemilattice for ProductOrder2<A, B> {
    #[inline]
    fn meet(&self, other: &Self) -> Self {
        Self {
            first: self.first.meet(&other.first),
            second: self.second.meet(&other.second),
        }
    }
}

/// A deterministic set of pairwise-incomparable minimal elements.
///
/// Inserting an element already dominated by an existing element is a no-op.
/// Inserting a new element removes any existing elements that it dominates.
/// Element order follows surviving insertion order and is therefore stable for
/// identical insertion sequences.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Antichain<T> {
    elements: Vec<T>,
}

impl<T> Default for Antichain<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Antichain<T> {
    /// Constructs an empty antichain.
    #[inline]
    pub const fn new() -> Self {
        Self {
            elements: Vec::new(),
        }
    }

    /// Returns the number of minimal elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Returns whether the antichain is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Borrows the stored minimal elements.
    #[inline]
    pub fn elements(&self) -> &[T] {
        &self.elements
    }

    /// Consumes the antichain and returns its minimal elements.
    #[inline]
    pub fn into_elements(self) -> Vec<T> {
        self.elements
    }
}

impl<T: PartiallyOrdered> Antichain<T> {
    /// Inserts `element` while preserving the minimal-antichain invariant.
    ///
    /// Returns `true` when the frontier changed.
    pub fn insert(&mut self, element: T) -> bool {
        if self
            .elements
            .iter()
            .any(|existing| existing.less_equal(&element))
        {
            return false;
        }

        self.elements
            .retain(|existing| !element.less_equal(existing));
        self.elements.push(element);
        true
    }

    /// Returns `true` when some frontier element is `<= value`.
    ///
    /// This is the standard frontier-membership predicate used when an antichain
    /// represents the minimal elements of an upward-closed set.
    #[inline]
    pub fn less_equal(&self, value: &T) -> bool {
        self.elements
            .iter()
            .any(|frontier| frontier.less_equal(value))
    }
}

impl<T: PartiallyOrdered> FromIterator<T> for Antichain<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut antichain = Self::new();
        for element in iter {
            antichain.insert(element);
        }
        antichain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type U = TotalOrder<u32>;
    type P = ProductOrder2<U, U>;

    #[test]
    fn total_order_forms_expected_lattice() {
        let a = U::new(3);
        let b = U::new(7);
        assert!(a.less_than(&b));
        assert_eq!(a.join(&b), b);
        assert_eq!(a.meet(&b), a);
    }

    #[test]
    fn product_order_preserves_incomparability() {
        let left = P::new(U::new(1), U::new(3));
        let right = P::new(U::new(3), U::new(1));
        assert!(!left.less_equal(&right));
        assert!(!right.less_equal(&left));
        assert_eq!(left.join(&right), P::new(U::new(3), U::new(3)));
        assert_eq!(left.meet(&right), P::new(U::new(1), U::new(1)));
    }

    #[test]
    fn antichain_keeps_only_minimal_elements() {
        let mut frontier = Antichain::new();
        assert!(frontier.insert(P::new(U::new(1), U::new(3))));
        assert!(frontier.insert(P::new(U::new(3), U::new(1))));
        assert!(!frontier.insert(P::new(U::new(4), U::new(4))));
        assert_eq!(frontier.len(), 2);

        assert!(frontier.insert(P::new(U::new(1), U::new(1))));
        assert_eq!(frontier.elements(), &[P::new(U::new(1), U::new(1))]);
        assert!(frontier.less_equal(&P::new(U::new(5), U::new(2))));
    }
}
