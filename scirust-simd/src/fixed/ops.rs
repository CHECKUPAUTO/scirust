// scirust-simd/src/fixed/ops.rs
//
// # Surcharge d'opérateurs
//
// Les opérateurs `+ − * / -x` et leurs variantes `*Assign` délèguent aux
// méthodes **enveloppantes** de [`Fixed`], avec troncature vers zéro pour
// `*`/`/` (cf. [`super::OverflowMode`]/[`super::RoundingMode`] pour le
// pourquoi). Pour toute autre politique, utiliser les méthodes explicites.
//
// `/` **panique** si le diviseur est nul (comme la division entière ;
// déterministe). Toutes les impls sont génériques sur `I: FixedStorage`, donc
// écrites une seule fois pour `FixedI32` et `FixedI64`.

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::repr::FixedStorage;
use super::types::Fixed;

impl<I: FixedStorage, const FRAC: u32> Add for Fixed<I, FRAC> {
    type Output = Self;
    /// Addition enveloppante (voir [`Fixed::wrapping_add`]).
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}

impl<I: FixedStorage, const FRAC: u32> Sub for Fixed<I, FRAC> {
    type Output = Self;
    /// Soustraction enveloppante (voir [`Fixed::wrapping_sub`]).
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        self.wrapping_sub(rhs)
    }
}

impl<I: FixedStorage, const FRAC: u32> Neg for Fixed<I, FRAC> {
    type Output = Self;
    /// Négation enveloppante (`−MIN` ↦ `MIN`, voir [`Fixed::wrapping_neg`]).
    #[inline(always)]
    fn neg(self) -> Self {
        self.wrapping_neg()
    }
}

impl<I: FixedStorage, const FRAC: u32> Mul for Fixed<I, FRAC> {
    type Output = Self;
    /// Multiplication enveloppante, troncature vers zéro
    /// (voir [`Fixed::wrapping_mul`]).
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl<I: FixedStorage, const FRAC: u32> Div for Fixed<I, FRAC> {
    type Output = Self;
    /// Division enveloppante, troncature vers zéro (voir [`Fixed::wrapping_div`]).
    ///
    /// # Panics
    /// Panique si `rhs == 0`.
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        self.wrapping_div(rhs)
    }
}

impl<I: FixedStorage, const FRAC: u32> AddAssign for Fixed<I, FRAC> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl<I: FixedStorage, const FRAC: u32> SubAssign for Fixed<I, FRAC> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl<I: FixedStorage, const FRAC: u32> MulAssign for Fixed<I, FRAC> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl<I: FixedStorage, const FRAC: u32> DivAssign for Fixed<I, FRAC> {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
