//! Core algebraic laws and zero-allocation constructions.

use core::marker::PhantomData;

/// A set equipped with a closed binary operation.
pub trait Magma: Sized {
    /// The binary operation.
    fn op(&self, rhs: &Self) -> Self;
}

/// An associative magma.
pub trait Semigroup: Magma {}

/// A semigroup with a neutral element.
pub trait Monoid: Semigroup {
    /// Neutral element.
    fn identity() -> Self;
}

/// A monoid whose elements are invertible.
pub trait Group: Monoid {
    /// Multiplicative/group inverse.
    fn inverse(&self) -> Self;

    /// Group division `self * rhs^-1`.
    #[inline]
    fn div(&self, rhs: &Self) -> Self {
        self.op(&rhs.inverse())
    }

    /// Integer power using binary exponentiation without allocation.
    fn powi(&self, exponent: i64) -> Self {
        if exponent == 0 {
            return Self::identity();
        }
        let mut base = if exponent < 0 { self.inverse() } else { self.op(&Self::identity()) };
        let mut e = exponent.unsigned_abs();
        let mut acc = Self::identity();
        while e != 0 {
            if e & 1 == 1 {
                acc = acc.op(&base);
            }
            e >>= 1;
            if e != 0 {
                base = base.op(&base);
            }
        }
        acc
    }
}

/// A commutative group.
pub trait AbelianGroup: Group {}

/// A ring with additive and multiplicative structure.
pub trait Ring: Sized {
    /// Additive identity.
    fn zero() -> Self;
    /// Multiplicative identity.
    fn one() -> Self;
    /// Addition.
    fn add(&self, rhs: &Self) -> Self;
    /// Additive inverse.
    fn neg(&self) -> Self;
    /// Multiplication.
    fn mul(&self, rhs: &Self) -> Self;

    /// Subtraction.
    #[inline]
    fn sub(&self, rhs: &Self) -> Self {
        self.add(&rhs.neg())
    }
}

/// A field: a commutative ring with inverses for non-zero elements.
pub trait Field: Ring + PartialEq {
    /// Multiplicative inverse, or `None` for zero.
    fn reciprocal(&self) -> Option<Self>;

    /// Division, or `None` when `rhs == 0`.
    #[inline]
    fn checked_div(&self, rhs: &Self) -> Option<Self> {
        rhs.reciprocal().map(|inv| self.mul(&inv))
    }
}

/// A Lie algebra over a scalar field.
pub trait LieAlgebra: Sized {
    /// Scalar type.
    type Scalar: Field;
    /// Lie bracket `[self, rhs]`.
    fn bracket(&self, rhs: &Self) -> Self;
    /// Scalar multiplication.
    fn scale(&self, scalar: &Self::Scalar) -> Self;
    /// Vector addition.
    fn add(&self, rhs: &Self) -> Self;
}

/// A left group action of `G` on `X`.
pub trait GroupAction<G: Group, X> {
    /// Apply a group element to a value.
    fn act(g: &G, value: &X) -> X;
}

/// Product group `G × H`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct DirectProduct<G, H> {
    /// First component.
    pub left: G,
    /// Second component.
    pub right: H,
}

impl<G: Magma, H: Magma> Magma for DirectProduct<G, H> {
    #[inline]
    fn op(&self, rhs: &Self) -> Self {
        Self { left: self.left.op(&rhs.left), right: self.right.op(&rhs.right) }
    }
}
impl<G: Semigroup, H: Semigroup> Semigroup for DirectProduct<G, H> {}
impl<G: Monoid, H: Monoid> Monoid for DirectProduct<G, H> {
    #[inline]
    fn identity() -> Self {
        Self { left: G::identity(), right: H::identity() }
    }
}
impl<G: Group, H: Group> Group for DirectProduct<G, H> {
    #[inline]
    fn inverse(&self) -> Self {
        Self { left: self.left.inverse(), right: self.right.inverse() }
    }
}
impl<G: AbelianGroup, H: AbelianGroup> AbelianGroup for DirectProduct<G, H> {}

/// A stateless homomorphism `H -> Aut(G)` used by a semidirect product.
pub trait AutomorphismAction<G: Group, H: Group> {
    /// Apply the automorphism represented by `h` to `g`.
    fn apply(h: &H, g: &G) -> G;
}

/// Semidirect product `G ⋊ H` represented without dynamic dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct SemidirectProduct<G, H, A> {
    /// Normal-factor component.
    pub normal: G,
    /// Acting-factor component.
    pub acting: H,
    marker: PhantomData<A>,
}

impl<G, H, A> SemidirectProduct<G, H, A> {
    /// Construct a semidirect-product element.
    #[inline]
    pub const fn new(normal: G, acting: H) -> Self {
        Self { normal, acting, marker: PhantomData }
    }
}

impl<G, H, A> Magma for SemidirectProduct<G, H, A>
where
    G: Group,
    H: Group,
    A: AutomorphismAction<G, H>,
{
    #[inline]
    fn op(&self, rhs: &Self) -> Self {
        let twisted = A::apply(&self.acting, &rhs.normal);
        Self::new(self.normal.op(&twisted), self.acting.op(&rhs.acting))
    }
}
impl<G, H, A> Semigroup for SemidirectProduct<G, H, A>
where G: Group, H: Group, A: AutomorphismAction<G, H> {}
impl<G, H, A> Monoid for SemidirectProduct<G, H, A>
where G: Group, H: Group, A: AutomorphismAction<G, H> {
    fn identity() -> Self { Self::new(G::identity(), H::identity()) }
}
impl<G, H, A> Group for SemidirectProduct<G, H, A>
where G: Group, H: Group, A: AutomorphismAction<G, H> {
    fn inverse(&self) -> Self {
        let h_inv = self.acting.inverse();
        let g_inv = A::apply(&h_inv, &self.normal.inverse());
        Self::new(g_inv, h_inv)
    }
}

/// Equivalence relation suitable for quotient groups.
pub trait Equivalence<T> {
    /// Whether `lhs ~ rhs`.
    fn equivalent(lhs: &T, rhs: &T) -> bool;
}

/// A representative of a quotient `G / ~`.
///
/// This type does not allocate and does not attempt to canonicalize a class.  The
/// caller supplies an equivalence relation whose compatibility with the group law
/// is a mathematical precondition.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Quotient<G, R> {
    representative: G,
    marker: PhantomData<R>,
}

impl<G, R> Quotient<G, R> {
    /// Wrap a representative.
    #[inline]
    pub const fn new(representative: G) -> Self {
        Self { representative, marker: PhantomData }
    }
    /// Borrow the stored representative.
    #[inline]
    pub const fn representative(&self) -> &G { &self.representative }
    /// Consume the quotient value and return its representative.
    #[inline]
    pub fn into_representative(self) -> G { self.representative }
}

impl<G: PartialEq, R: Equivalence<G>> PartialEq for Quotient<G, R> {
    fn eq(&self, rhs: &Self) -> bool { R::equivalent(&self.representative, &rhs.representative) }
}
impl<G: Eq, R: Equivalence<G>> Eq for Quotient<G, R> {}
impl<G: Magma, R> Magma for Quotient<G, R> {
    fn op(&self, rhs: &Self) -> Self { Self::new(self.representative.op(&rhs.representative)) }
}
impl<G: Semigroup, R> Semigroup for Quotient<G, R> {}
impl<G: Monoid, R> Monoid for Quotient<G, R> {
    fn identity() -> Self { Self::new(G::identity()) }
}
impl<G: Group, R> Group for Quotient<G, R> {
    fn inverse(&self) -> Self { Self::new(self.representative.inverse()) }
}
