//! Exact prime-field values for the locally bounded toy domain.

use std::error::Error;
use std::fmt;

use scirust_modalg::numtheory::{inv_mod, is_prime, mulmod, pow_mod};

/// Minimum prime accepted by the toy-curve domain.
pub const MIN_TOY_PRIME: u64 = 5;

/// Maximum prime accepted by the toy-curve domain.
pub const MAX_TOY_PRIME: u64 = 4093;

/// A verified odd prime in the bounded local-research domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ToyPrime(u64);

impl ToyPrime {
    /// Validates a prime for the bounded toy-curve domain.
    pub fn new(value: u64) -> Result<Self, PrimeError> {
        if value < MIN_TOY_PRIME
        {
            return Err(PrimeError::BelowMinimum { value });
        }
        if value > MAX_TOY_PRIME
        {
            return Err(PrimeError::AboveMaximum { value });
        }
        if !is_prime(value)
        {
            return Err(PrimeError::Composite { value });
        }
        Ok(Self(value))
    }

    /// Returns the prime as an unsigned integer.
    pub const fn value(self) -> u64 {
        self.0
    }
}

/// An invalid request to create a toy prime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimeError {
    /// The candidate is too small for the short-Weierstrass toy domain.
    BelowMinimum {
        /// Rejected candidate.
        value: u64,
    },
    /// The candidate exceeds the deliberate local-research limit.
    AboveMaximum {
        /// Rejected candidate.
        value: u64,
    },
    /// The candidate is not prime.
    Composite {
        /// Rejected candidate.
        value: u64,
    },
}

impl fmt::Display for PrimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::BelowMinimum { value } => write!(
                formatter,
                "{value} is below the minimum toy prime {MIN_TOY_PRIME}"
            ),
            Self::AboveMaximum { value } => write!(
                formatter,
                "{value} exceeds the maximum toy prime {MAX_TOY_PRIME}"
            ),
            Self::Composite { value } => write!(formatter, "{value} is not prime"),
        }
    }
}

impl Error for PrimeError {}

/// An exact canonical residue modulo one toy prime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fp {
    prime: ToyPrime,
    value: u64,
}

impl Fp {
    /// Reduces value to its canonical representative modulo prime.
    pub const fn new(prime: ToyPrime, value: u64) -> Self {
        Self {
            prime,
            value: value % prime.value(),
        }
    }

    /// Returns the modulus of this field value.
    pub const fn prime(self) -> ToyPrime {
        self.prime
    }

    /// Returns the canonical representative in the interval from zero to p.
    pub const fn value(self) -> u64 {
        self.value
    }

    /// Returns whether this value is zero.
    pub const fn is_zero(self) -> bool {
        self.value == 0
    }

    /// Adds two values after checking that they belong to the same field.
    pub fn checked_add(self, other: Self) -> Result<Self, FieldError> {
        self.ensure_same_prime(other)?;
        Ok(self.add_same(other))
    }

    /// Subtracts two values after checking that they belong to the same field.
    pub fn checked_sub(self, other: Self) -> Result<Self, FieldError> {
        self.ensure_same_prime(other)?;
        Ok(self.sub_same(other))
    }

    /// Multiplies two values after checking that they belong to the same field.
    pub fn checked_mul(self, other: Self) -> Result<Self, FieldError> {
        self.ensure_same_prime(other)?;
        Ok(self.mul_same(other))
    }

    /// Divides two values after checking their field and nonzero denominator.
    pub fn checked_div(self, other: Self) -> Result<Self, FieldError> {
        self.ensure_same_prime(other)?;
        let inverse = other.inverse().ok_or(FieldError::ZeroHasNoInverse)?;
        Ok(self.mul_same(inverse))
    }

    /// Returns the additive inverse.
    pub const fn neg(self) -> Self {
        if self.value == 0
        {
            self
        }
        else
        {
            Self {
                prime: self.prime,
                value: self.prime.value() - self.value,
            }
        }
    }

    /// Returns the multiplicative inverse, if this value is nonzero.
    pub fn inverse(self) -> Option<Self> {
        inv_mod(self.value, self.prime.value()).map(|value| Self {
            prime: self.prime,
            value,
        })
    }

    /// Raises this value to a nonnegative integer power exactly.
    pub fn pow(self, exponent: u64) -> Self {
        Self {
            prime: self.prime,
            value: pow_mod(self.value, exponent, self.prime.value()),
        }
    }

    /// Adds values known to belong to the same verified field.
    #[inline(always)]
    pub(crate) fn add_same(self, other: Self) -> Self {
        debug_assert_eq!(self.prime, other.prime);
        Self::new(self.prime, self.value + other.value)
    }

    /// Subtracts values known to belong to the same verified field using constant-time subtraction.
    #[inline(always)]
    pub(crate) fn sub_same(self, other: Self) -> Self {
        debug_assert_eq!(self.prime, other.prime);
        let p = self.prime.value();
        // Since both self.value and other.value are < p, self.value + p - other.value
        // is guaranteed to be in [1, 2*p - 1], avoiding any negative values or branching.
        let value = (self.value + p - other.value) % p;
        Self {
            prime: self.prime,
            value,
        }
    }

    /// Multiplies values known to belong to the same verified field.
    #[inline(always)]
    pub(crate) fn mul_same(self, other: Self) -> Self {
        debug_assert_eq!(self.prime, other.prime);
        Self {
            prime: self.prime,
            value: mulmod(self.value, other.value, self.prime.value()),
        }
    }

    /// Squares this value exactly.
    #[inline(always)]
    pub(crate) fn square(self) -> Self {
        self.mul_same(self)
    }

    fn ensure_same_prime(self, other: Self) -> Result<(), FieldError> {
        if self.prime == other.prime
        {
            Ok(())
        }
        else
        {
            Err(FieldError::DifferentPrimes)
        }
    }
}

/// An invalid cross-field arithmetic operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldError {
    /// The operands have different moduli.
    DifferentPrimes,
    /// Zero has no multiplicative inverse.
    ZeroHasNoInverse,
}

impl fmt::Display for FieldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::DifferentPrimes => write!(formatter, "field values have different prime moduli"),
            Self::ZeroHasNoInverse => write!(formatter, "zero has no multiplicative inverse"),
        }
    }
}

impl Error for FieldError {}
